import os, json, gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from medpy.metric.binary import hd95

from dataset import Glioma3DDataset, build_cv_loaders
from evaluations import dice_coefficient_score, iou_score
from main_foundation_model import get_bbox, run_medsam_seg_layer
from medsam import get_medsam_predictor

def conv_block(in_channels, out_channels, kernel_size=3, num_convs=2):
    layers = []
    for i in range(num_convs):
        layers.append(
            nn.Conv3d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
        )
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SegNet3D(nn.Module):
    def __init__(self, in_channels, n_labels=1, kernel=3, pool_size=2):
        super(SegNet3D, self).__init__()

        # -------- Encoder --------
        self.enc1 = conv_block(in_channels, 32, kernel, num_convs=2)
        self.enc2 = conv_block(32, 64, kernel, num_convs=2)
        self.enc3 = conv_block(64, 128, kernel, num_convs=3)
        self.enc4 = conv_block(128, 256, kernel, num_convs=3)
        self.enc5 = conv_block(256, 256, kernel, num_convs=3)

        self.pool = nn.MaxPool3d(pool_size, stride=pool_size, return_indices=True)
        self.unpool = nn.MaxUnpool3d(pool_size, stride=pool_size)

        # -------- Decoder --------
        self.dec5 = conv_block(256, 256, kernel, num_convs=3)
        self.dec4 = conv_block(256, 128, kernel, num_convs=3)
        self.dec3 = conv_block(128, 64, kernel, num_convs=3)
        self.dec2 = conv_block(64, 32, kernel, num_convs=2)
        self.dec1 = conv_block(32, 32, kernel, num_convs=1)

        self.final_conv = nn.Conv3d(32, n_labels, kernel_size=1)

        self._initialize_weights()

    def forward(self, x):

        # -------- Encoder --------
        x1 = self.enc1(x)
        size1 = x1.size()
        x1p, idx1 = self.pool(x1)

        x2 = self.enc2(x1p)
        size2 = x2.size()
        x2p, idx2 = self.pool(x2)

        x3 = self.enc3(x2p)
        size3 = x3.size()
        x3p, idx3 = self.pool(x3)

        x4 = self.enc4(x3p)
        size4 = x4.size()
        x4p, idx4 = self.pool(x4)

        x5 = self.enc5(x4p)
        size5 = x5.size()
        x5p, idx5 = self.pool(x5)

        # -------- Decoder --------
        d5 = self.unpool(x5p, idx5, output_size=size5)
        d5 = self.dec5(d5)

        d4 = self.unpool(d5, idx4, output_size=size4)
        d4 = self.dec4(d4)

        d3 = self.unpool(d4, idx3, output_size=size3)
        d3 = self.dec3(d3)

        d2 = self.unpool(d3, idx2, output_size=size2)
        d2 = self.dec2(d2)

        d1 = self.unpool(d2, idx1, output_size=size1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2 * intersection + 1e-6) / (
            probs.sum() + targets.sum() + 1e-6
        )

        return self.bce_weight * bce + self.dice_weight * (1 - dice)

def build_segnet_model(device, in_channels=1, state_dict=None):
    model = SegNet3D(in_channels=in_channels)
    if state_dict is not None:
        dict_model = torch.load(state_dict, map_location=device, weights_only=False)
        model.load_state_dict(dict_model['model_state_dict'])
    return model

def build_training_components(model, lr=5e-3):
    criterion = DiceBCELoss(
        dice_weight=5.0,
        bce_weight=0.5
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    return criterion, optimizer

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    print("Training...")

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if y.dim() == 4:# (B, D, H, W)
            y = y.unsqueeze(1)# (B, 1, D, H, W)
        y = y.float()

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print("Train avg loss", running_loss / len(loader))
    return running_loss / len(loader)

def soft_dice_from_logits(logits, targets):
    probs = torch.sigmoid(logits)
    probs = probs.view(-1)
    targets = targets.view(-1)

    intersection = (probs * targets).sum()
    return (2 * intersection) / (probs.sum() + targets.sum() + 1e-8)

def evaluate(model, loader, device):
    model.eval()

    scores = {'dice': 0.0, 'iou': 0.0, 'softdice': 0.0}#, 'hd': 0.0, 'hd95': 0.0}
    count = 0

    print("Evaluating...")
    with torch.no_grad():
        for x, y in loader:
            count += 1
            x = x.to(device)
            y = y.to(device)

            y = y.squeeze().float()

            logits = model(x)
            # print(
            #     type(logits),
            #     logits.shape,
            #     logits.min().item(),
            #     logits.max().item(),
            #     logits.mean().item()
            # )

            scores['softdice'] += soft_dice_from_logits(logits, y).item()

            preds = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(bool)
            y_np = y.cpu().numpy().astype(bool)
            # print(y_np.shape)#240,240,155
            # print(preds.shape)#240,240,155

            scores['dice'] += dice_coefficient_score(y_np, preds)
            scores['iou'] += iou_score(y_np, preds)

            del x, y, logits, preds
            gc.collect()

    for k in scores:
        scores[k] /= count

    print("Evaluation scores:", scores)
    return scores

def run_cv_training(
    dataset_root,
    device,
    num_folds=5,
    num_epochs=100,
    batch_size=1,
    num_workers=4
):
    fold_results = {}
    best_fold = 0
    best_fold_dice = 0

    for fold in range(num_folds):
        print(f"\n===== Fold {fold + 1}/{num_folds} =====")

        train_loader, val_loader = build_cv_loaders(dataset_root, fold, num_folds=num_folds, batch_size=batch_size, num_workers=num_workers)

        model = build_segnet_model(device)
        model.to(device)
        criterion, optimizer = build_training_components(model)

        best_dice = 0.0
        fold_log = []
        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_scores = evaluate(model, val_loader, device)

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {train_loss:.4f} | "
                f"Val Dice {val_scores['dice']:.4f}"
            )
            fold_log.append({'loss': train_loss, 'val_scores': val_scores})

            if val_scores['dice'] > best_dice:
                best_dice = val_scores['dice']
                os.makedirs(os.path.join('model_output', 'SegNet'), exist_ok=True)
                save_path = os.path.join('model_output', 'SegNet', f"segnet3d_fold{fold}.pt")
                torch.save(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_scores": val_scores,
                    },
                    save_path,
                )

        fold_results[fold] = fold_log
        print(f"Best Dice (fold {fold}): {best_dice:.4f}")
        if best_dice > best_fold_dice:
            best_fold_dice = best_dice
            best_fold = fold
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.join('output', 'SegNet'), exist_ok=True)
    with open(os.path.join('output', 'SegNet', 'training_result.json'), 'w') as f:
        json.dump(fold_results, f)
    return best_fold

def run_test(ds_root, device, fold):
    dataset = Glioma3DDataset(ds_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_segnet_model(device, state_dict=os.path.join('model_output', 'SegNet', f"segnet3d_fold{fold}.pt"))
    model.to(device)
    model = model.float()
    model.eval()

    scores = {'dice': 0, 'iou': 0, 'hd95': 0}#, 'hd': 0, 'hd95': 0}
    count = 0

    with torch.no_grad():
        for x, y in loader:
            count += 1
            x = x.to(device, dtype=torch.float32)
            y = y.to('cpu', dtype=torch.float32)
            if y.dim() == 4:  # (B, D, H, W)
                y = y.squeeze()  # → (D, H, W)
            y = y.numpy()
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).detach().int().squeeze().cpu().numpy()# binary mask

            dice, iou, h95 = dice_coefficient_score(y, preds), iou_score(y, preds), hd95(y, preds)#, hausdorff(y, preds), hd95(y, preds)
            scores['dice']+= dice
            scores['iou'] += iou
            # scores['hd'] += hd
            scores['hd95'] += h95

            del x, y, logits, probs, preds
            if device == 'cuda':
                torch.cuda.empty_cache()

        scores['dice'] /= count
        scores['iou'] /= count
        # scores['hd'] /= count
        scores['hd95'] /= count
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return scores

def run_test_with_medsam(ds_root, device, fold):
    dataset = Glioma3DDataset(ds_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_segnet_model(device, state_dict=os.path.join('model_output', 'SegNet', f"segnet3d_fold{fold}.pt"))
    model.to(device)
    model = model.float()
    model.eval()

    predictor = get_medsam_predictor(device)

    scores = {'dice': 0, 'iou': 0, 'hd95': 0}
    count = 0

    with torch.no_grad():
        for x, y in loader:
            count += 1
            x = x.to(device, dtype=torch.float32)
            y = y.to('cpu', dtype=torch.int)
            if y.dim() == 4:  # (B, D, H, W)
                y = y.squeeze()
            y = y.numpy()
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).detach().int().squeeze().cpu().numpy()# binary mask

            # start medsam task
            x = x.squeeze().numpy()

            H, W, C = x.shape

            for segment_layer in range(C):
                segment_bbox = get_bbox(preds[:, :, segment_layer].astype(bool), model='medsam')
                if segment_bbox is not None:
                    mask = run_medsam_seg_layer(predictor, x, segment_layer, segment_bbox)
                    # print(mask)
                    preds[:, :, segment_layer] = mask.astype(preds.dtype)

            dice, iou, h95 = dice_coefficient_score(y, preds), iou_score(y, preds), hd95(y, preds)
            scores['dice']+= dice
            scores['iou'] += iou
            scores['hd95'] += h95
        
        scores['dice'] /= count
        scores['iou'] /= count
        scores['hd95'] /= count
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return scores

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: " + device)

    best_fold = run_cv_training(
        device=device,
        num_folds=5,
        num_epochs=50,
        batch_size=1,
        num_workers=4,
    )

    test_score = run_test(
        device,
        best_fold
    )
    os.makedirs(os.path.join('output', 'SegNet'), exist_ok=True)
    with open(os.path.join('output', 'SegNet', 'test_result.json'), 'w') as f:
        json.dump(test_score, f)

    med_score = run_test_with_medsam(
        device,
        best_fold
    )
    with open(os.path.join('output', 'SegNet', 'test_medsam_result.json'), 'w') as f:
        json.dump(med_score, f)


if __name__ == "__main__":
    main()
