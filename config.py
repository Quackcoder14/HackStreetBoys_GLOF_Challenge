"""
config.py – Central configuration for the Glacial Lake Detection Pipeline
SNUC GLOFeagles 2026 Challenge
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "SNUC GLOFeagles 2026 challenge datasets"
OUTPUT_DIR = BASE_DIR / "outputs"

PSEUDO_MASK_DIR   = OUTPUT_DIR / "pseudo_masks"
FINAL_MASK_DIR    = OUTPUT_DIR / "masks"
OVERLAY_DIR       = OUTPUT_DIR / "overlays"
CHECKPOINT_DIR    = OUTPUT_DIR / "checkpoints"
REPORT_DIR        = BASE_DIR  / "report"

for d in [PSEUDO_MASK_DIR, FINAL_MASK_DIR, OVERLAY_DIR, CHECKPOINT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Image Settings
# ─────────────────────────────────────────────
IMG_SIZE        = 512          # Input/output spatial dimension
NUM_CHANNELS    = 3            # RGB

# ─────────────────────────────────────────────
# Classical Detector Hyperparameters
# ─────────────────────────────────────────────
# Brightness threshold (0-255): pixels darker than this are lake candidates
BRIGHTNESS_THRESHOLD    = 80   # mean(R,G,B) < 80 → very dark
# HSV-Value threshold: lakes are very dark
HSV_V_THRESHOLD         = 0.35
# HSV-Saturation: lakes have very low colour saturation
HSV_S_MAX               = 0.30
# Hue exclusion: warm hues (debris/rock) in degrees [0,180] (OpenCV scale)
HUE_WARM_LOW            = 5    # orange/brown start
HUE_WARM_HIGH           = 35   # orange/brown end
# Local texture threshold: std-dev of neighbourhood must be low inside lakes
TEXTURE_STD_MAX         = 25.0
TEXTURE_WINDOW          = 9
# Minimum connected component area to keep (pixels²)
MIN_LAKE_AREA           = 200
# Morphological kernel sizes
MORPH_OPEN_SIZE         = 3
MORPH_CLOSE_SIZE        = 7

# ─────────────────────────────────────────────
# U-Net / Training Hyperparameters
# ─────────────────────────────────────────────
ENCODER_NAME    = "efficientnet_b0"   # timm backbone
PRETRAINED      = True
NUM_CLASSES     = 1                   # binary segmentation
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
BATCH_SIZE      = 4
NUM_EPOCHS      = 30
PATIENCE        = 7                   # early stopping patience
TRAIN_SPLIT     = 0.80
DICE_WEIGHT     = 0.6                 # weight for Dice loss vs BCE
BCE_WEIGHT      = 0.4
GRAD_CLIP       = 1.0
BEST_CKPT       = str(CHECKPOINT_DIR / "best_model.pth")
FINAL_CKPT      = str(CHECKPOINT_DIR / "final_model.pth")

# ─────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────
AUG_PROB        = 0.5
BRIGHTNESS_LIMIT = 0.2
CONTRAST_LIMIT   = 0.2

# ─────────────────────────────────────────────
# Inference / Post-processing
# ─────────────────────────────────────────────
SIGMOID_THRESHOLD   = 0.45    # Sigmoid output threshold for binary mask
INFERENCE_BATCH_SIZE = 8

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Random Seed
# ─────────────────────────────────────────────
SEED = 42
