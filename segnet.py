import os, json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary

from dataset import Glioma3DDataset, build_cv_loaders
from evaluations import dice_coefficient_score, iou_score, hausdorff, hd95
from main_foundation_model import get_bbox, run_medsam_seg_layer
from medsam import get_medsam_predictor

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SegNet3D(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2, stride=2, return_indices=True)

        self.enc2 = ConvBlock3D(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2, stride=2, return_indices=True)

        self.enc3 = ConvBlock3D(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2, stride=2, return_indices=True)

        # Bottleneck (NO channel change!)
        self.bottleneck = ConvBlock3D(base_filters * 4, base_filters * 4)

        # Decoder
        self.unpool3 = nn.MaxUnpool3d(2, stride=2)
        self.dec3 = ConvBlock3D(base_filters * 4, base_filters * 2)

        self.unpool2 = nn.MaxUnpool3d(2, stride=2)
        self.dec2 = ConvBlock3D(base_filters * 2, base_filters)

        self.unpool1 = nn.MaxUnpool3d(2, stride=2)
        self.dec1 = ConvBlock3D(base_filters, base_filters)

        # Output
        self.final = nn.Conv3d(base_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x1p, idx1 = self.pool1(x1)

        x2 = self.enc2(x1p)
        x2p, idx2 = self.pool2(x2)

        x3 = self.enc3(x2p)
        x3p, idx3 = self.pool3(x3)

        # Bottleneck
        x4 = self.bottleneck(x3p)

        # Decoder (channels now MATCH indices)
        x = self.unpool3(x4, idx3, output_size=x3.size())
        x = self.dec3(x)

        x = self.unpool2(x, idx2, output_size=x2.size())
        x = self.dec2(x)

        x = self.unpool1(x, idx1, output_size=x1.size())
        x = self.dec1(x)

        return self.final(x)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


def build_segnet_model(device, in_channels=1, f_maps=32, state_dict=None):
    # f_maps=32#16 for small GPU VRAM, 32 for large memory
    model = SegNet3D(in_channels=in_channels, base_filters=f_maps)
    if state_dict is not None:
        dict = torch.load(state_dict, map_location=device)
        model.load_state_dict(dict['model_state_dict'])
    return model

def combined_loss(logits, targets):
    dice = DiceLoss()(logits, targets)
    bce = nn.BCEWithLogitsLoss()(logits, targets)
    return dice + bce

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if y.dim() == 4:  # (B, D, H, W)
            y = y.unsqueeze(1)  # → (B, 1, D, H, W)

        optimizer.zero_grad()
        logits = model(x)
        loss = combined_loss(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    scores = {'dice': 0, 'iou': 0, 'hd': 0, 'hd95': 0}
    count = 0
    with torch.no_grad():
        for x, y in loader:
            count += 1
            x, y = x.to(device), y.to('cpu')
            if y.dim() == 4:  # (B, D, H, W)
                y = y.unsqueeze(1)  # → (B, 1, D, H, W)
            y = y.numpy()
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).detach().float().cpu().numpy()

            dice, iou, hd, h95 = dice_coefficient_score(y, preds), iou_score(y, preds), hausdorff(y, preds), hd95(y, preds)
            scores['dice']+= dice
            scores['iou'] += iou
            scores['hd'] += hd
            scores['hd95'] += h95

        scores['dice'] /= count
        scores['iou'] /= count
        scores['hd'] /= count
        scores['hd95'] /= count

        return scores

def run_cv_training(
    dataset_root,
    device,
    num_folds=5,
    num_epochs=100,
    batch_size=1,
    num_workers=4,
):
    fold_results = {}
    best_fold = 0
    best_fold_dice = 0

    for fold in range(num_folds):
        print(f"\n===== Fold {fold + 1}/{num_folds} =====")

        train_loader, val_loader = build_cv_loaders(dataset_root, fold, num_folds=num_folds, batch_size=batch_size, num_workers=num_workers)

        model = build_segnet_model(device, f_maps=32)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_dice = 0.0
        fold_log = []
        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device
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

    model = build_segnet_model(device, state_dict=os.path.join('.', 'model_output', 'SegNet', f"segnet3d_fold{fold}.pt"))
    model.to(device)
    model = model.float()
    model.eval()

    scores = {'dice': 0, 'iou': 0, 'hd': 0, 'hd95': 0}
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to('cpu', dtype=torch.float32)
            if y.dim() == 4:  # (B, D, H, W)
                y = y.unsqueeze(1)  # → (B, 1, D, H, W)
            y = y.numpy()
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).detach().int().cpu().numpy()# binary mask
            dice, iou, hd, h95 = dice_coefficient_score(y, preds), iou_score(y, preds), hausdorff(y, preds), hd95(y, preds)
            scores['dice']+= dice
            scores['iou'] += iou
            scores['hd'] += hd
            scores['hd95'] += h95
        
        scores['dice'] /= count
        scores['iou'] /= count
        scores['hd'] /= count
        scores['hd95'] /= count
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return scores

def run_test_with_medsam(ds_root, device, fold):
    dataset = Glioma3DDataset(ds_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_segnet_model(device, state_dict=os.path.join('.', 'model_output', 'SegNet', f"segnet3d_fold{fold}.pt"))
    model.to(device)
    model = model.float()
    model.eval()

    predictor = get_medsam_predictor(device)

    scores = {'dice': 0, 'iou': 0, 'hd': 0, 'hd95': 0}
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to('cpu', dtype=torch.float32)
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

            dice, iou, hd, h95 = dice_coefficient_score(y, preds), iou_score(y, preds), hausdorff(y, preds), hd95(y, preds)
            scores['dice']+= dice
            scores['iou'] += iou
            scores['hd'] += hd
            scores['hd95'] += h95
        
        scores['dice'] /= count
        scores['iou'] /= count
        scores['hd'] /= count
        scores['hd95'] /= count
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return scores

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: " + device)

    best_fold = run_cv_training(
    )

    run_test(
    )

    run_test_with_medsam(
    )

if __name__ == "__main__":
    main()
