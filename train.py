"""
train.py – Stage 2: Train U-Net on Pseudo-Labels
SNUC GLOFeagles 2026 Challenge

Training loop with:
  - Combined Dice + BCE loss
  - AdamW optimiser with cosine annealing LR schedule
  - Early stopping based on validation Dice
  - Checkpoint saving (best and final)
  - Progress logging to CSV
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from tqdm import tqdm
import config
from model import build_model, CombinedLoss
from dataset import build_dataloaders
from metrics import compute_metrics_batch


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# One Epoch – Train
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    use_cuda = str(device) == "cuda"

    for images, masks in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type="cuda" if use_cuda else "cpu",
            enabled=use_cuda
        ):
            logits = model(images)
            loss   = criterion(logits, masks)

        if use_cuda:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > config.SIGMOID_THRESHOLD).float()
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics_batch(
        torch.cat(all_preds), torch.cat(all_targets)
    )
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# One Epoch – Validate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for images, masks in tqdm(loader, desc="  Valid", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, masks)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > config.SIGMOID_THRESHOLD).float()
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics_batch(
        torch.cat(all_preds), torch.cat(all_targets)
    )
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    epochs:       int   = config.NUM_EPOCHS,
    patience:     int   = config.PATIENCE,
    batch_size:   int   = config.BATCH_SIZE,
    lr:           float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    device_str:   str   = config.DEVICE,
):
    set_seed()
    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"  GLACIAL LAKE U-NET TRAINING")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"{'='*60}\n")

    # ── Dataloaders ───────────────────────────────────────────────────────
    train_loader, val_loader, train_ids, val_ids = build_dataloaders(
        batch_size=batch_size
    )

    # ── Model, Loss, Optimiser ────────────────────────────────────────────
    model     = build_model(pretrained=config.PRETRAINED)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    scaler    = torch.amp.GradScaler('cuda') if device_str == "cuda" else None

    # ── Training State ────────────────────────────────────────────────────
    best_val_dice  = 0.0
    patience_count = 0
    history        = []
    val_dice       = 0.0
    t0             = time.time()

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch:02d}/{epochs}  (lr={current_lr:.2e})")

        tr_loss, tr_m = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        va_loss, va_m = validate(model, val_loader, criterion, device)

        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────
        row = dict(
            epoch=epoch, lr=current_lr,
            tr_loss=tr_loss, va_loss=va_loss,
            tr_dice=tr_m["f1"],        va_dice=va_m["f1"],
            tr_iou=tr_m["iou"],        va_iou=va_m["iou"],
            tr_prec=tr_m["precision"], va_prec=va_m["precision"],
            tr_rec=tr_m["recall"],     va_rec=va_m["recall"],
            tr_acc=tr_m["accuracy"],   va_acc=va_m["accuracy"],
            tr_kappa=tr_m["kappa"],    va_kappa=va_m["kappa"],
        )
        history.append(row)

        print(f"  Train → Loss: {tr_loss:.4f}  Dice: {tr_m['f1']:.4f}  "
              f"IoU: {tr_m['iou']:.4f}  κ: {tr_m['kappa']:.4f}")
        print(f"  Valid → Loss: {va_loss:.4f}  Dice: {va_m['f1']:.4f}  "
              f"IoU: {va_m['iou']:.4f}  κ: {va_m['kappa']:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────
        val_dice = va_m["f1"]
        if val_dice > best_val_dice:
            best_val_dice  = val_dice
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "opt_state":   optimizer.state_dict(),
                "val_dice":    val_dice,
                "val_iou":     va_m["iou"],
                "val_kappa":   va_m["kappa"],
                "config": {
                    "encoder":    "resnet18",
                    "img_size":   config.IMG_SIZE,
                    "num_classes":config.NUM_CLASSES,
                }
            }, config.BEST_CKPT)
            print(f"  [OK] Best model saved (dice={val_dice:.4f})")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{patience})")

        if patience_count >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break

    # ── Save Final Model ──────────────────────────────────────────────────
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_dice":    val_dice,
    }, config.FINAL_CKPT)

    # ── Save Training History ─────────────────────────────────────────────
    hist_df   = pd.DataFrame(history)
    hist_path = config.CHECKPOINT_DIR / "training_history.csv"
    hist_df.to_csv(hist_path, index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Best Val Dice (F1) : {best_val_dice:.4f}")
    print(f"  Checkpoints saved to: {config.CHECKPOINT_DIR}")
    print(f"{'='*60}\n")

    # ── Final summary JSON ────────────────────────────────────────────────
    best_row = max(history, key=lambda r: r["va_dice"])
    summary = {
        "best_epoch":   best_row["epoch"],
        "val_dice_f1":  round(best_row["va_dice"], 4),
        "val_iou":      round(best_row["va_iou"], 4),
        "val_precision":round(best_row["va_prec"], 4),
        "val_recall":   round(best_row["va_rec"], 4),
        "val_accuracy": round(best_row["va_acc"], 4),
        "val_kappa":    round(best_row["va_kappa"], 4),
        "train_images": len(train_ids),
        "val_images":   len(val_ids),
    }
    with open(config.CHECKPOINT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Training Summary:")
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")

    return model, summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
