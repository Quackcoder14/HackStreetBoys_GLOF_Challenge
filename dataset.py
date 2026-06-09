"""
dataset.py – PyTorch Dataset for Glacial Lake Segmentation
SNUC GLOFeagles 2026 Challenge

Uses torchvision transforms only (no albumentations required).
Applies consistent augmentation to both image and mask.
"""

import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple
import config

# ─────────────────────────────────────────────────────────────────────────────
# ImageNet normalisation constants
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Paired augmentation helpers (image + mask must get identical spatial transforms)
# ─────────────────────────────────────────────────────────────────────────────

def paired_augment(img_rgb: np.ndarray, mask: np.ndarray,
                   img_size: int = config.IMG_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply identical spatial transforms to both image and mask.
    Colour jitter is applied to image only.
    Returns augmented img_rgb (H,W,3 uint8) and mask (H,W float32).
    """
    h, w = img_rgb.shape[:2]

    # ── Resize to target ──────────────────────────────────────────────────
    img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask    = cv2.resize(mask.astype(np.float32), (img_size, img_size),
                         interpolation=cv2.INTER_NEAREST)

    # ── Horizontal flip ───────────────────────────────────────────────────
    if random.random() < 0.5:
        img_rgb = img_rgb[:, ::-1, :].copy()
        mask    = mask   [:, ::-1   ].copy()

    # ── Vertical flip ─────────────────────────────────────────────────────
    if random.random() < 0.5:
        img_rgb = img_rgb[::-1, :, :].copy()
        mask    = mask   [::-1, :   ].copy()

    # ── Random 90° rotation ───────────────────────────────────────────────
    if random.random() < 0.5:
        k       = random.randint(1, 3)
        img_rgb = np.rot90(img_rgb, k).copy()
        mask    = np.rot90(mask,    k).copy()

    # ── Random crop then resize ───────────────────────────────────────────
    if random.random() < 0.3:
        crop_frac = random.uniform(0.75, 0.95)
        ch = int(img_size * crop_frac)
        cw = int(img_size * crop_frac)
        y0 = random.randint(0, img_size - ch)
        x0 = random.randint(0, img_size - cw)
        img_rgb = img_rgb[y0:y0+ch, x0:x0+cw, :]
        mask    = mask   [y0:y0+ch, x0:x0+cw]
        img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        mask    = cv2.resize(mask,    (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    # ── Brightness / contrast jitter (image only) ─────────────────────────
    if random.random() < 0.5:
        alpha = 1.0 + random.uniform(-config.CONTRAST_LIMIT,   config.CONTRAST_LIMIT)
        beta  =       random.uniform(-config.BRIGHTNESS_LIMIT * 255,
                                      config.BRIGHTNESS_LIMIT * 255)
        img_rgb = np.clip(img_rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # ── Gaussian noise (image only) ───────────────────────────────────────
    if random.random() < 0.2:
        noise   = np.random.normal(0, random.uniform(3, 12), img_rgb.shape).astype(np.float32)
        img_rgb = np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img_rgb, mask


def to_tensor_normalise(img_rgb: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 RGB → CHW float32 tensor, ImageNet-normalised."""
    img_f = img_rgb.astype(np.float32) / 255.0
    mean  = np.array(IMAGENET_MEAN, dtype=np.float32)
    std   = np.array(IMAGENET_STD,  dtype=np.float32)
    img_f = (img_f - mean) / std
    return torch.from_numpy(img_f.transpose(2, 0, 1))   # (3, H, W)


def val_preprocess(img_rgb: np.ndarray,
                   img_size: int = config.IMG_SIZE) -> torch.Tensor:
    """Resize + normalise only (no augmentation)."""
    img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return to_tensor_normalise(img_rgb)


# ─────────────────────────────────────────────────────────────────────────────
# Training Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GlacialLakeDataset(Dataset):
    """
    PyTorch dataset for glacial lake segmentation with pseudo-labels.

    Parameters
    ----------
    image_ids   : list of integer image IDs (filename stems)
    image_dir   : directory containing RGB .png images
    mask_dir    : directory containing binary pseudo-mask .png
    augment     : if True, apply training augmentation
    """

    def __init__(
        self,
        image_ids: List[int],
        image_dir: Path  = None,
        mask_dir:  Path  = None,
        augment:   bool  = False,
    ):
        self.image_ids = image_ids
        self.image_dir = image_dir or config.DATASET_DIR
        self.mask_dir  = mask_dir  or config.PSEUDO_MASK_DIR
        self.augment   = augment

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id   = self.image_ids[idx]

        # Load image (BGR → RGB)
        img_bgr  = cv2.imread(str(self.image_dir / f"{img_id}.png"))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load pseudo-mask (0/255 → 0/1 float)
        msk_path = self.mask_dir / f"{img_id}.png"
        if msk_path.exists():
            mask_raw = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
            mask     = (mask_raw > 127).astype(np.float32)
        else:
            mask = np.zeros(img_rgb.shape[:2], dtype=np.float32)

        # Augment (training) or just resize (validation)
        if self.augment:
            img_rgb, mask = paired_augment(img_rgb, mask)
        else:
            img_rgb = cv2.resize(img_rgb, (config.IMG_SIZE, config.IMG_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
            mask    = cv2.resize(mask, (config.IMG_SIZE, config.IMG_SIZE),
                                 interpolation=cv2.INTER_NEAREST)

        img_tensor  = to_tensor_normalise(img_rgb)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)   # (1, H, W)

        return img_tensor, mask_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Inference-only Dataset
# ─────────────────────────────────────────────────────────────────────────────

class InferenceDataset(Dataset):
    """Returns (image_tensor, image_id) — no masks needed."""

    def __init__(self, image_ids: List[int], image_dir: Path = None):
        self.image_ids = image_ids
        self.image_dir = image_dir or config.DATASET_DIR

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_id  = self.image_ids[idx]
        img_bgr = cv2.imread(str(self.image_dir / f"{img_id}.png"))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor  = val_preprocess(img_rgb)
        return tensor, img_id


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    image_dir:   Path  = None,
    mask_dir:    Path  = None,
    train_split: float = config.TRAIN_SPLIT,
    batch_size:  int   = config.BATCH_SIZE,
    seed:        int   = config.SEED,
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    """Build train/val DataLoaders from pseudo-mask directory."""
    image_dir = image_dir or config.DATASET_DIR
    mask_dir  = mask_dir  or config.PSEUDO_MASK_DIR

    all_ids = sorted(
        [int(p.stem) for p in mask_dir.glob("*.png")]
    )
    if not all_ids:
        raise RuntimeError(
            f"No pseudo-masks in {mask_dir}. Run classical_detector.py first."
        )

    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(len(all_ids))
    n_tr   = int(len(all_ids) * train_split)
    train_ids = [all_ids[i] for i in idx[:n_tr]]
    val_ids   = [all_ids[i] for i in idx[n_tr:]]

    train_ds = GlacialLakeDataset(train_ids, image_dir, mask_dir, augment=True)
    val_ds   = GlacialLakeDataset(val_ids,   image_dir, mask_dir, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True,  num_workers=0, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False
    )

    print(f"[Dataset] Train: {len(train_ids)} | Val: {len(val_ids)} images")
    return train_loader, val_loader, train_ids, val_ids


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from classical_detector import run_classical_detection
    if not any(config.PSEUDO_MASK_DIR.glob("*.png")):
        print("Generating pseudo-masks first …")
        run_classical_detection()

    train_loader, val_loader, _, _ = build_dataloaders()
    imgs, masks = next(iter(train_loader))
    print(f"Image batch : {imgs.shape}   dtype={imgs.dtype}")
    print(f"Mask  batch : {masks.shape}  dtype={masks.dtype}")
    print(f"Mask unique : {masks.unique().tolist()}")
    print("Dataset sanity check PASSED [OK]")
