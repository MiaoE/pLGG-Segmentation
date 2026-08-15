import os, json
import random, gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch3dunet.unet3d.model import UNet3D
from torchinfo import summary
from medpy.metric.binary import hd95

from data.dataset import Glioma3DDataset, build_cv_loaders
from training.evaluations import dice_coefficient_score, iou_score
from main_foundation_model import get_bbox, run_medsam_seg_layer
from medsam import get_medsam_predictor

def build_unet_model(device, state_dict=None, f_maps=32):
    # f_maps=32#16 for small GPU VRAM, 32 for large memory
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        final_sigmoid=False,   # for CrossEntropyLoss
        f_maps=f_maps,
        layer_order="gcr",
        # num_groups=8,
        is_segmentation=True,
    ).to(device)
    if state_dict is not None:
        mydict = torch.load(state_dict, map_location=device, weights_only=False)
        model.load_state_dict(mydict['model_state_dict'])
    return model

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

def build_training_components(model, lr=1e-4):
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

        _, logits = model(x, return_logits=True)
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

            _, logits = model(x, return_logits=True)

            scores['softdice'] += soft_dice_from_logits(logits, y).item()

            # preds = (probs > 0.5).squeeze().cpu().numpy().astype(bool)
            preds = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(bool)
            y_np = y.cpu().numpy().astype(bool)
            # print(y_np.shape)#240,240,155
            # print(preds.shape)#240,240,155

            scores['dice'] += dice_coefficient_score(y_np, preds)
            scores['iou'] += iou_score(y_np, preds)
            # hd100, hd_95 = hausdorff(y_np, preds), hd95(y_np, preds)
            # scores['hd'] += hd100
            # scores['hd95'] += hd_95

            del x, y, logits, preds#, hd100, hd_95
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

        model = build_unet_model(device, state_dict=os.path.join('model_pretrained', '3dunet', 'best_checkpoint.pytorch'))
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
                os.makedirs(os.path.join('model_output', 'UNet'), exist_ok=True)
                save_path = os.path.join('model_output', 'UNet', f"unet3d_fold{fold}.pt")
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
    os.makedirs(os.path.join('output', 'UNet'), exist_ok=True)
    with open(os.path.join('output', 'UNet', 'training_result.json'), 'w') as f:
        json.dump(fold_results, f)
    return best_fold


def run_test(ds_root, device, fold):
    dataset = Glioma3DDataset(ds_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_unet_model(device, state_dict=os.path.join('model_output', 'UNet', f"unet3d_fold{fold}.pt"))
    model.to(device)
    model = model.float()
    model.eval()

    scores = {'dice': 0, 'iou': 0, 'hd95': 0}#, 'hd95': 0}
    count = 0

    with torch.no_grad():
        for x, y in loader:
            count += 1
            x = x.to(device, dtype=torch.float32)
            y = y.to('cpu')
            if y.dim() == 4:  # (B, D, H, W)
                y = y.squeeze()  # → (D, H, W)
            y = y.numpy().astype(int)
            _, logits = model(x, return_logits=True)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).detach().int().squeeze().cpu().numpy()# binary mask

            print(y.shape)
            print(preds.shape)
            dice, iou, h95 = dice_coefficient_score(y, preds), iou_score(y, preds), hd95(y, preds)#, hausdorff(y, preds), hd95(y, preds)
            scores['dice']+= dice
            scores['iou'] += iou
            # scores['hd'] += hd
            scores['hd95'] += h95

            del x, y, logits, probs, preds

        scores['dice'] /= count
        scores['iou'] /= count
        # scores['hd'] /= count
        scores['hd95'] /= count
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return scores

'''def run_test(ds_root, device, fold):
    dataset = Glioma3DDataset(ds_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_unet_model(device, state_dict=os.path.join('.', 'model_output', 'UNet', f"unet3d_fold{fold}.pt"))
    model.to(device)
    model = model.float()
    model.eval()

    scores = {'dice': 0, 'iou': 0, 'hd': 0, 'hd95': 0}
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            if y.dim() == 4:  # (B, D, H, W)
                y = y.unsqueeze(1)  # → (B, 1, D, H, W)
            y = y.numpy()
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().numpy()# binary mask

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
    return scores'''

def run_test_with_medsam(ds_root, device, fold):
    dataset = Glioma3DDataset(ds_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_unet_model(device, state_dict=os.path.join('.', 'model_output', 'UNet', f"unet3d_fold{fold}.pt"))
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
                y = y.squeeze()  # → (D, H, W)
            y = y.numpy()
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).detach().cpu().int().squeeze().numpy()# binary mask

            # start medsam task, all volumes in NumPy
            x = x.squeeze().numpy()

            H, W, C = x.shape

            for segment_layer in range(C):
                segment_bbox = get_bbox(preds[:, :, segment_layer].astype(bool), model='medsam')
                if segment_bbox is not None:
                    mask = run_medsam_seg_layer(predictor, x, segment_layer, segment_bbox)
                    # print(mask)
                    preds[:, :, segment_layer] = mask.astype(preds.dtype)
            preds = preds.astype(int)
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
    print("Using device:", device)

    best_fold = run_cv_training(
        device=device,
        num_folds=5,
        num_epochs=50,
        batch_size=1,
    )

    test_score = run_test(
        device,
        best_fold
    )
    os.makedirs(os.path.join('output', 'UNet'), exist_ok=True)
    with open(os.path.join('output', 'UNet', 'test_result.json'), 'w') as f:
        json.dump(test_score, f)
    
    med_score = run_test_with_medsam(
        device,
        best_fold
    )
    with open(os.path.join('output', 'UNet', 'test_medsam_result.json'), 'w') as f:
        json.dump(med_score, f)


if __name__ == "__main__":
    main()
