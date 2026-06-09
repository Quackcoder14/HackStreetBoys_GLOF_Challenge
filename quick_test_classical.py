"""
quick_test_classical.py – Fast validation of Stage 1 classical detector
No PyTorch/GPU required. Uses only opencv + scikit-image + numpy.

Run this first to verify the environment and generate pseudo-labels.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── Dependency check (lightweight only) ───────────────────────────────────
missing = []
for pkg in ["cv2", "numpy", "skimage", "scipy", "tqdm"]:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"Missing packages: {missing}")
    print("Install with:  pip install opencv-python scikit-image scipy tqdm numpy")
    sys.exit(1)

import cv2
import numpy as np
from pathlib import Path
import time
import config

print("=" * 60)
print("  STAGE 1: CLASSICAL GLACIAL LAKE DETECTOR")
print("  Quick Test + Pseudo-Label Generation")
print("=" * 60)
print(f"\n  Dataset : {config.DATASET_DIR}")
print(f"  Output  : {config.PSEUDO_MASK_DIR}\n")

# ── Verify dataset ─────────────────────────────────────────────────────────
image_paths = sorted(config.DATASET_DIR.glob("*.png"), key=lambda p: int(p.stem))
print(f"  Found {len(image_paths)} images in dataset")
assert len(image_paths) > 0, f"No images found in {config.DATASET_DIR}"

# ── Run classical detector ─────────────────────────────────────────────────
from classical_detector import run_classical_detection, detect_lakes

# Quick test on 5 sample images
sample_ids = [1, 2, 48, 116, 400]
print(f"\n  Quick test on {len(sample_ids)} sample images:")
print(f"  {'ID':<6} {'Lake px':<12} {'Lake %':<10} {'Time(ms)'}")
print(f"  {'-'*42}")

for img_id in sample_ids:
    p = config.DATASET_DIR / f"{img_id}.png"
    img = cv2.imread(str(p))
    if img is None:
        print(f"  {img_id:<6} Could not load image")
        continue
    t0   = time.perf_counter()
    mask = detect_lakes(img)
    ms   = (time.perf_counter() - t0) * 1000
    lake_px  = (mask > 127).sum()
    lake_pct = lake_px / mask.size * 100
    print(f"  {img_id:<6} {lake_px:<12,} {lake_pct:<10.2f} {ms:.1f}ms")

# ── Run on all 575 images ──────────────────────────────────────────────────
print(f"\n  Running on all {len(image_paths)} images ...")
t0    = time.time()
stats = run_classical_detection(
    image_dir=config.DATASET_DIR,
    out_dir=config.PSEUDO_MASK_DIR,
    verbose=True,
)
elapsed = time.time() - t0

# ── Summary ────────────────────────────────────────────────────────────────
fractions   = list(stats.values())
n_with_lake = sum(f > 0.005 for f in fractions)
n_large     = sum(f > 0.05  for f in fractions)

print(f"\n{'='*60}")
print(f"  STAGE 1 COMPLETE  ({elapsed:.1f}s  |  {len(fractions)/elapsed:.1f} img/s)")
print(f"{'='*60}")
print(f"  Images processed          : {len(fractions)}")
print(f"  With detectable lakes     : {n_with_lake} ({100*n_with_lake/len(fractions):.1f}%)")
print(f"  Large lakes (>5% cover)   : {n_large}")
print(f"  Mean lake coverage        : {np.mean(fractions):.4f}")
print(f"  Median lake coverage      : {np.median(fractions):.4f}")
print(f"  Max lake coverage         : {np.max(fractions):.4f}")
print(f"\n  Pseudo-masks saved to: {config.PSEUDO_MASK_DIR}")
print(f"  Total masks: {len(list(config.PSEUDO_MASK_DIR.glob('*.png')))}")
print()
print("  [OK] Stage 1 passed. Ready to run Stage 2 (train.py) or")
print("    full pipeline (run_pipeline.py --skip-train for inference only).")
