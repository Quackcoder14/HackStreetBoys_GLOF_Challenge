"""
inference.py – Stage 3: Run Inference on All 575 Images
SNUC GLOFeagles 2026 Challenge

Loads the best trained U-Net checkpoint and:
  1. Generates binary segmentation masks for all images
  2. Saves masks as PNG (0=background, 255=lake)
  3. Saves colour overlay visualisations (red = detected lake)
  4. Exports per-image statistics to CSV
  5. Optionally applies morphological post-processing refinement
"""

import cv2
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import config
from model import build_model, GlacialLakeUNet
from dataset import InferenceDataset
from torch.utils.data import DataLoader
from metrics import compute_metrics_single, aggregate_metrics, print_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str = None, device: str = None) -> GlacialLakeUNet:
    """
    Load U-Net from checkpoint.
    Falls back to classical-only mode if no checkpoint exists.
    """
    checkpoint_path = checkpoint_path or config.BEST_CKPT
    device          = device          or config.DEVICE

    model = build_model(pretrained=False)

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        val_dice  = ckpt.get("val_dice",  "N/A")
        val_iou   = ckpt.get("val_iou",   "N/A")
        val_kappa = ckpt.get("val_kappa", "N/A")
        epoch     = ckpt.get("epoch",     "N/A")
        print(f"[Inference] Loaded checkpoint: {ckpt_path.name}")
        print(f"            Epoch={epoch}  ValDice={val_dice}  "
              f"ValIoU={val_iou}  ValKappa={val_kappa}")
    else:
        print(f"[Inference] WARNING: No checkpoint at {ckpt_path}. "
              f"Using random weights – classical fallback recommended.")

    model.eval()
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Morphological Post-Processing
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_mask(mask_prob: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    Convert probability map → refined binary mask.

    Steps:
      1. Threshold at sigmoid_threshold
      2. Morphological opening  (remove small noise)
      3. Morphological closing  (fill holes in lakes)
      4. Remove tiny components (area < MIN_LAKE_AREA)
    """
    from skimage import measure

    threshold = threshold or config.SIGMOID_THRESHOLD
    binary    = (mask_prob > threshold).astype(np.uint8)

    # Open → removes isolated noise pixels
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open)

    # Close → fills holes within lakes
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

    # Remove components smaller than MIN_LAKE_AREA
    labels  = measure.label(closed, connectivity=2)
    props   = measure.regionprops(labels)
    cleaned = np.zeros_like(closed)
    for prop in props:
        if prop.area >= config.MIN_LAKE_AREA:
            cleaned[labels == prop.label] = 1

    return (cleaned * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Overlay Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def create_overlay(img_bgr: np.ndarray, mask: np.ndarray,
                   alpha: float = 0.40) -> np.ndarray:
    """
    Blend original image with a semi-transparent red lake mask.

    Parameters
    ----------
    img_bgr : Original BGR image (H, W, 3)
    mask    : Binary mask (H, W) values 0 or 255
    alpha   : Transparency of overlay (0=transparent, 1=opaque)

    Returns
    -------
    overlay : BGR image with lake regions highlighted in red
    """
    overlay = img_bgr.copy()
    lake_px = mask > 127

    # Apply red tint on lake pixels
    overlay[lake_px, 0]  = np.clip(
        img_bgr[lake_px, 0] * (1 - alpha), 0, 255).astype(np.uint8)   # B
    overlay[lake_px, 1]  = np.clip(
        img_bgr[lake_px, 1] * (1 - alpha), 0, 255).astype(np.uint8)   # G
    overlay[lake_px, 2]  = np.clip(
        img_bgr[lake_px, 2] * (1 - alpha) + 255 * alpha, 0, 255
    ).astype(np.uint8)                                                   # R

    # Draw contour border around lakes
    contours, _ = cv2.findContours(
        (lake_px * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)   # Cyan border

    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Batch Inference Loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    checkpoint_path:  str  = None,
    image_dir:        Path = None,
    mask_out_dir:     Path = None,
    overlay_out_dir:  Path = None,
    device_str:       str  = None,
    save_overlays:    bool = True,
    batch_size:       int  = config.INFERENCE_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Run full inference on all images.

    Returns
    -------
    DataFrame with per-image statistics (lake_fraction, area_px, etc.)
    """
    image_dir       = image_dir       or config.DATASET_DIR
    mask_out_dir    = mask_out_dir    or config.FINAL_MASK_DIR
    overlay_out_dir = overlay_out_dir or config.OVERLAY_DIR
    device_str      = device_str      or config.DEVICE

    mask_out_dir.mkdir(parents=True, exist_ok=True)
    overlay_out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str)

    # ── Load Model ────────────────────────────────────────────────────────
    model  = load_model(checkpoint_path, device_str)

    # ── Collect All Image IDs ─────────────────────────────────────────────
    all_ids = sorted(
        [int(p.stem) for p in image_dir.glob("*.png")],
        key=lambda x: x
    )
    print(f"[Inference] Running on {len(all_ids)} images …")

    # ── Dataset & Loader ──────────────────────────────────────────────────
    inf_ds     = InferenceDataset(all_ids, image_dir)
    inf_loader = DataLoader(
        inf_ds, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False
    )

    # ── Inference Loop ────────────────────────────────────────────────────
    records = []

    for batch_imgs, batch_ids in tqdm(inf_loader, desc="Inference"):
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        # Forward pass
        with torch.amp.autocast(device_type='cuda' if device_str == 'cuda' else 'cpu',
                                  enabled=(device_str == 'cuda')):
            logits = model(batch_imgs)
        probs  = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)

        for i, img_id in enumerate(batch_ids.tolist()):
            prob_map = probs[i, 0]              # (H, W) float [0,1]

            # Post-process → binary mask (0 or 255)
            mask_uint8 = postprocess_mask(prob_map)

            # Save mask
            cv2.imwrite(str(mask_out_dir / f"{img_id}.png"), mask_uint8)

            # Lake statistics
            lake_pixels   = (mask_uint8 > 127).sum()
            total_pixels  = mask_uint8.size
            lake_fraction = lake_pixels / total_pixels

            records.append({
                "image_id":      img_id,
                "lake_pixels":   int(lake_pixels),
                "total_pixels":  int(total_pixels),
                "lake_fraction": round(float(lake_fraction), 6),
                "has_lake":      lake_fraction > 0.005,
                "mean_prob":     round(float(prob_map.mean()), 6),
                "max_prob":      round(float(prob_map.max()),  6),
            })

            # Overlay visualisation
            if save_overlays:
                img_bgr = cv2.imread(str(image_dir / f"{img_id}.png"))
                if img_bgr is not None:
                    # Resize mask to original image size if needed
                    if mask_uint8.shape[:2] != img_bgr.shape[:2]:
                        mask_uint8 = cv2.resize(
                            mask_uint8, (img_bgr.shape[1], img_bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                    overlay = create_overlay(img_bgr, mask_uint8)
                    cv2.imwrite(str(overlay_out_dir / f"{img_id}.png"), overlay)

    # ── Save Statistics ───────────────────────────────────────────────────
    df = pd.DataFrame(records).sort_values("image_id")
    stats_path = config.OUTPUT_DIR / "inference_statistics.csv"
    df.to_csv(stats_path, index=False)

    n_with_lake = df["has_lake"].sum()
    print(f"\n[Inference] Complete:")
    print(f"  Images with detectable lakes : {n_with_lake}/{len(df)}")
    print(f"  Mean lake fraction           : {df['lake_fraction'].mean():.4f}")
    print(f"  Max  lake fraction           : {df['lake_fraction'].max():.4f}")
    print(f"  Masks saved to               : {mask_out_dir}")
    print(f"  Overlays saved to            : {overlay_out_dir}")
    print(f"  Statistics saved to          : {stats_path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Self-Evaluation Against Pseudo-Labels
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_against_pseudo_labels(
    pred_dir:   Path = None,
    pseudo_dir: Path = None,
) -> dict:
    """
    Compare final model predictions against Stage 1 pseudo-labels.
    Provides a proxy quality measure (how well the DL model generalised
    beyond the pseudo-labels).
    """
    pred_dir   = pred_dir   or config.FINAL_MASK_DIR
    pseudo_dir = pseudo_dir or config.PSEUDO_MASK_DIR

    pred_files  = sorted(pred_dir.glob("*.png"),   key=lambda p: int(p.stem))
    pseudo_dict = {int(p.stem): p for p in pseudo_dir.glob("*.png")}

    per_image = []
    for pred_path in tqdm(pred_files, desc="Comparing to pseudo-labels"):
        img_id = int(pred_path.stem)
        if img_id not in pseudo_dict:
            continue
        pred   = cv2.imread(str(pred_path),           cv2.IMREAD_GRAYSCALE)
        pseudo = cv2.imread(str(pseudo_dict[img_id]), cv2.IMREAD_GRAYSCALE)
        pred   = (pred   > 127).astype(np.float32)
        pseudo = (pseudo > 127).astype(np.float32)
        m      = compute_metrics_single(pred, pseudo)
        m["image_id"] = img_id
        per_image.append(m)

    agg = aggregate_metrics(per_image)
    print_metrics(agg, title="DL vs Pseudo-Label Agreement")

    out_path = config.OUTPUT_DIR / "dl_vs_pseudolabel_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"aggregate": agg, "n_images": len(per_image)}, f, indent=2)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Classical-Only Fallback Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_classical_inference_only(
    image_dir:       Path = None,
    mask_out_dir:    Path = None,
    overlay_out_dir: Path = None,
    save_overlays:   bool = True,
) -> pd.DataFrame:
    """
    Fallback: copy pseudo-masks as final output (no DL model needed).
    Used when no GPU / time is available for training.
    """
    from classical_detector import run_classical_detection
    import shutil

    image_dir       = image_dir       or config.DATASET_DIR
    mask_out_dir    = mask_out_dir    or config.FINAL_MASK_DIR
    overlay_out_dir = overlay_out_dir or config.OVERLAY_DIR

    # Run or reuse classical detection
    pseudo_dir = config.PSEUDO_MASK_DIR
    if not any(pseudo_dir.glob("*.png")):
        run_classical_detection(image_dir, pseudo_dir, verbose=True)

    records = []
    for src in tqdm(sorted(pseudo_dir.glob("*.png"), key=lambda p: int(p.stem)),
                    desc="Classical Fallback"):
        dst = mask_out_dir / src.name
        shutil.copy2(str(src), str(dst))

        mask_uint8    = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        lake_pixels   = (mask_uint8 > 127).sum()
        lake_fraction = lake_pixels / mask_uint8.size
        img_id        = int(src.stem)

        records.append({
            "image_id":      img_id,
            "lake_pixels":   int(lake_pixels),
            "total_pixels":  int(mask_uint8.size),
            "lake_fraction": round(float(lake_fraction), 6),
            "has_lake":      lake_fraction > 0.005,
        })

        if save_overlays:
            img_bgr = cv2.imread(str(image_dir / src.name))
            if img_bgr is not None:
                overlay = create_overlay(img_bgr, mask_uint8)
                cv2.imwrite(str(overlay_out_dir / src.name), overlay)

    df = pd.DataFrame(records).sort_values("image_id")
    df.to_csv(config.OUTPUT_DIR / "classical_statistics.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Glacial Lake Inference")
    parser.add_argument("--classical-only", action="store_true",
                        help="Use only classical detector (no DL model)")
    parser.add_argument("--no-overlays",    action="store_true",
                        help="Skip saving overlay visualisations")
    parser.add_argument("--checkpoint",     type=str, default=None,
                        help="Path to model checkpoint")
    args = parser.parse_args()

    if args.classical_only:
        print("[Mode] Classical-only inference")
        run_classical_inference_only(save_overlays=not args.no_overlays)
    else:
        print("[Mode] Deep Learning inference")
        df = run_inference(
            checkpoint_path=args.checkpoint,
            save_overlays=not args.no_overlays,
        )
        # Also compare vs pseudo-labels for quality estimation
        evaluate_against_pseudo_labels()
