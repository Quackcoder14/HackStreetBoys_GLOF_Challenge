"""
utils.py – Utility functions for GLOFeagles challenge
SNUC GLOFeagles 2026 Challenge

Contains helper functions for image processing, metrics calculation, and visualization.
"""

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple

def load_image(path: str) -> np.ndarray:
    """Loads an image in RGB format."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_mask(path: str, mask: np.ndarray) -> None:
    """Saves a binary segmentation mask (0 or 255)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, mask.astype(np.uint8))

def overlay_mask(img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), alpha: float = 0.4) -> np.ndarray:
    """Overlays a semi-transparent colored mask on an RGB image."""
    mask_bool = mask > 0
    overlay = img.copy()
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay

def compute_binary_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Computes basic binary segmentation metrics at pixel level."""
    pred_b = (pred > 0).astype(bool)
    target_b = (target > 0).astype(bool)
    
    tp = np.logical_and(pred_b, target_b).sum()
    fp = np.logical_and(pred_b, np.logical_not(target_b)).sum()
    fn = np.logical_and(np.logical_not(pred_b), target_b).sum()
    tn = np.logical_and(np.logical_not(pred_b), np.logical_not(target_b)).sum()
    
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    # Cohen's Kappa
    total = tp + tn + fp + fn
    po = (tp + tn) / (total + 1e-8)
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total * total + 1e-8)
    kappa = (po - pe) / (1 - pe + 1e-8)
    
    return {
        "mIoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "Accuracy": float(accuracy),
        "Kappa": float(kappa)
    }
