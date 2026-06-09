"""
metrics.py – Evaluation Metrics for Binary Segmentation
SNUC GLOFeagles 2026 Challenge

Implements all competition-required metrics:
  (i)  Mean IoU (mIoU)
  (ii) F1 Score (= Dice)
  (iii) Precision
  (iv) Recall
  (v)  Accuracy
  (vi) Cohen's Kappa (κ)
"""

import torch
import numpy as np
from typing import Dict
import json
from pathlib import Path
import config


# ─────────────────────────────────────────────────────────────────────────────
# Core Metric Computation (Tensor-based, batched)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_batch(
    preds:   torch.Tensor,   # (N,1,H,W) float 0/1
    targets: torch.Tensor,   # (N,1,H,W) float 0/1
    eps:     float = 1e-6,
) -> Dict[str, float]:
    """
    Compute all metrics over a batch of predictions.

    Parameters
    ----------
    preds   : binary predictions (already thresholded), float32
    targets : binary ground-truth masks, float32
    eps     : small constant for numerical stability

    Returns
    -------
    dict with keys: iou, f1, precision, recall, accuracy, kappa, specificity
    """
    p = preds.view(-1).float()
    t = targets.view(-1).float()

    tp = (p * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    fn = ((1 - p) * t).sum().item()
    tn = ((1 - p) * (1 - t)).sum().item()

    total = tp + fp + fn + tn + eps

    precision   = tp / (tp + fp + eps)
    recall      = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1          = 2 * tp / (2 * tp + fp + fn + eps)       # Dice coefficient
    iou         = tp / (tp + fp + fn + eps)               # Jaccard index
    accuracy    = (tp + tn) / total

    # Cohen's Kappa
    p_o  = accuracy                                         # observed agreement
    p_e  = ((tp + fp) / total) * ((tp + fn) / total) + \
           ((tn + fn) / total) * ((tn + fp) / total)       # expected agreement
    kappa = (p_o - p_e) / (1.0 - p_e + eps)

    return {
        "iou":         round(iou,         4),
        "f1":          round(f1,          4),
        "precision":   round(precision,   4),
        "recall":      round(recall,      4),
        "accuracy":    round(accuracy,    4),
        "kappa":       round(kappa,       4),
        "specificity": round(specificity, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-Image Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_single(
    pred:   np.ndarray,   # (H,W) binary uint8 or float (0/1)
    target: np.ndarray,   # (H,W) binary uint8 or float (0/1)
    eps:    float = 1e-6,
) -> Dict[str, float]:
    """Compute metrics for a single image pair (numpy arrays)."""
    p = torch.from_numpy(pred.astype(np.float32)).view(-1)
    t = torch.from_numpy(target.astype(np.float32)).view(-1)
    return compute_metrics_batch(
        p.unsqueeze(0).unsqueeze(0),
        t.unsqueeze(0).unsqueeze(0),
        eps
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate Over Dataset
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_metrics(per_image_metrics: list) -> Dict[str, float]:
    """
    Compute mean metrics over a list of per-image metric dicts.
    Uses macro-average (mean of per-image scores).
    """
    keys = ["iou", "f1", "precision", "recall", "accuracy", "kappa", "specificity"]
    agg  = {}
    for k in keys:
        vals    = [m[k] for m in per_image_metrics if k in m]
        agg[k]  = round(float(np.mean(vals)), 4)
        agg[f"{k}_std"] = round(float(np.std(vals)), 4)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate Masks on Disk
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_mask_directories(
    pred_dir:   Path,
    target_dir: Path,
    save_path:  Path = None,
) -> Dict[str, float]:
    """
    Compare predicted masks in pred_dir against target masks in target_dir.
    Both directories must contain matching {id}.png files.
    """
    import cv2
    from tqdm import tqdm

    pred_files   = sorted(pred_dir.glob("*.png"),   key=lambda p: int(p.stem))
    target_files = {int(p.stem): p for p in target_dir.glob("*.png")}

    if not pred_files:
        raise FileNotFoundError(f"No prediction masks found in {pred_dir}")

    per_image = []
    for pred_path in tqdm(pred_files, desc="Evaluating"):
        img_id = int(pred_path.stem)
        if img_id not in target_files:
            continue

        pred   = cv2.imread(str(pred_path),              cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(str(target_files[img_id]),   cv2.IMREAD_GRAYSCALE)

        pred   = (pred   > 127).astype(np.float32)
        target = (target > 127).astype(np.float32)

        m = compute_metrics_single(pred, target)
        m["image_id"] = img_id
        per_image.append(m)

    agg = aggregate_metrics(per_image)

    if save_path:
        result = {"aggregate": agg, "per_image": per_image}
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Metrics saved to {save_path}")

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Pretty Print
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    print(f"\n{'─'*45}")
    print(f"  {title}")
    print(f"{'─'*45}")
    display = [
        ("Mean IoU (mIoU)",   "iou"),
        ("F1 Score (Dice)",   "f1"),
        ("Precision",         "precision"),
        ("Recall",            "recall"),
        ("Accuracy",          "accuracy"),
        ("Cohen's Kappa (κ)", "kappa"),
        ("Specificity",       "specificity"),
    ]
    for label, key in display:
        val = metrics.get(key, "N/A")
        std = metrics.get(f"{key}_std", "")
        std_str = f" ± {std:.4f}" if std != "" else ""
        print(f"  {label:<25}: {val:.4f}{std_str}" if isinstance(val, float)
              else f"  {label:<25}: {val}")
    print(f"{'─'*45}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Perfect prediction
    pred   = torch.ones(4, 1, 512, 512)
    target = torch.ones(4, 1, 512, 512)
    m      = compute_metrics_batch(pred, target)
    print("Perfect prediction:")
    print_metrics(m)
    assert m["f1"] > 0.999, "F1 should be 1.0 for perfect prediction"

    # All wrong
    pred   = torch.zeros(4, 1, 512, 512)
    target = torch.ones(4, 1, 512, 512)
    m      = compute_metrics_batch(pred, target)
    print("All wrong prediction:")
    print_metrics(m)

    print("Metrics sanity check PASSED [OK]")
