"""
run_pipeline.py – Master Script: End-to-End Glacial Lake Detection
SNUC GLOFeagles 2026 Challenge

Runs all three stages in sequence:
  Stage 1 → Classical detector  → pseudo-label masks
  Stage 2 → U-Net training      → model checkpoint
  Stage 3 → DL inference        → final masks + overlays

  Optional: --ablation to run ablation study
  Optional: --classical-only to skip DL (useful without GPU)

Usage:
  python run_pipeline.py                  # full pipeline
  python run_pipeline.py --classical-only # no DL, classical masks only
  python run_pipeline.py --ablation       # also run ablation study
  python run_pipeline.py --skip-train     # use existing checkpoint
"""

import argparse
import time
import json
import sys
from pathlib import Path
import config


def banner(text: str, char: str = "═", width: int = 65):
    line = char * width
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def check_dependencies():
    """Verify all required packages are installed."""
    required = {
        "cv2":           "opencv-python",
        "torch":         "torch",
        "sklearn":       "scikit-learn",
        "skimage":       "scikit-image",
        "scipy":         "scipy",
        "tqdm":          "tqdm",
        "pandas":        "pandas",
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("⚠  Missing packages. Install with:")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)
    print("  All dependencies satisfied [OK]")


def stage1_classical(verbose: bool = True) -> dict:
    """Stage 1: Classical detector → pseudo-label masks."""
    from classical_detector import run_classical_detection

    banner("STAGE 1 – Classical Glacial Lake Detector")

    # Skip if already done
    existing = list(config.PSEUDO_MASK_DIR.glob("*.png"))
    if len(existing) >= 500:
        print(f"  Found {len(existing)} existing pseudo-masks. Skipping re-generation.")
        print(f"  (Delete {config.PSEUDO_MASK_DIR} to force regeneration)")
        return {}

    t0    = time.time()
    stats = run_classical_detection(
        image_dir=config.DATASET_DIR,
        out_dir=config.PSEUDO_MASK_DIR,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    n_total     = len(stats)
    n_with_lake = sum(1 for v in stats.values() if v > 0.005)
    mean_frac   = sum(stats.values()) / max(n_total, 1)

    print(f"\n  Stage 1 Complete  ({elapsed:.1f}s)")
    print(f"  Images processed       : {n_total}")
    print(f"  Images with lakes      : {n_with_lake} ({100*n_with_lake/max(n_total,1):.1f}%)")
    print(f"  Mean lake coverage     : {mean_frac:.4f}")
    print(f"  Pseudo-masks saved to  : {config.PSEUDO_MASK_DIR}")
    return stats


def stage2_train(
    epochs:      int = config.NUM_EPOCHS,
    skip:        bool = False,
) -> dict:
    """Stage 2: Train U-Net on pseudo-labels."""
    from train import train

    banner("STAGE 2 – U-Net Training on Pseudo-Labels")

    if skip:
        ckpt = Path(config.BEST_CKPT)
        if ckpt.exists():
            print(f"  Skipping training. Using existing checkpoint: {ckpt}")
            summary_path = config.CHECKPOINT_DIR / "training_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    return json.load(f)
            return {}
        else:
            print("  No checkpoint found — training is required.")

    t0     = time.time()
    _, summary = train(epochs=epochs)
    elapsed = time.time() - t0

    print(f"\n  Stage 2 Complete  ({elapsed/60:.1f} min)")
    print(f"  Best checkpoint    : {config.BEST_CKPT}")
    return summary


def stage3_inference(
    checkpoint_path: str  = None,
    save_overlays:   bool = True,
    classical_only:  bool = False,
):
    """Stage 3: Run inference → final masks."""
    from inference import run_inference, run_classical_inference_only, \
                          evaluate_against_pseudo_labels

    banner("STAGE 3 – Final Inference")

    t0 = time.time()

    if classical_only:
        print("  Mode: Classical-only (no DL model)")
        df = run_classical_inference_only(
            image_dir=config.DATASET_DIR,
            mask_out_dir=config.FINAL_MASK_DIR,
            overlay_out_dir=config.OVERLAY_DIR,
            save_overlays=save_overlays,
        )
    else:
        df = run_inference(
            checkpoint_path=checkpoint_path or config.BEST_CKPT,
            image_dir=config.DATASET_DIR,
            mask_out_dir=config.FINAL_MASK_DIR,
            overlay_out_dir=config.OVERLAY_DIR,
            save_overlays=save_overlays,
        )
        # Self-evaluation against pseudo-labels
        evaluate_against_pseudo_labels()

    elapsed = time.time() - t0
    print(f"\n  Stage 3 Complete  ({elapsed:.1f}s)")
    print(f"  Final masks saved to : {config.FINAL_MASK_DIR}")
    if save_overlays:
        print(f"  Overlays saved to    : {config.OVERLAY_DIR}")
    return df


def stage4_ablation(n_samples: int = 80):
    """Optional Stage 4: Ablation study."""
    from ablation_study import run_full_ablation

    banner("STAGE 4 – Ablation Study")
    run_full_ablation(n_samples=n_samples)
    print(f"  Ablation reports saved to: {config.REPORT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Final Summary Report
# ─────────────────────────────────────────────────────────────────────────────

def write_final_report(
    stage1_stats:  dict,
    train_summary: dict,
    inference_df,
    classical_only: bool,
):
    """Write a combined JSON summary of the entire pipeline run."""
    import pandas as pd

    n_total    = len(inference_df) if inference_df is not None else 0
    n_lake     = int(inference_df["has_lake"].sum()) if inference_df is not None else 0
    mean_cov   = float(inference_df["lake_fraction"].mean()) if inference_df is not None else 0

    report = {
        "pipeline_mode": "classical_only" if classical_only else "hybrid_dl",
        "dataset": {
            "total_images":         575,
            "images_processed":     n_total,
            "images_with_lakes":    n_lake,
            "mean_lake_coverage":   round(mean_cov, 6),
        },
        "stage1_classical": {
            "method":               "Multi-feature brightness/HSV/texture/edge/shape",
            "pseudo_masks_created": len(list(config.PSEUDO_MASK_DIR.glob("*.png"))),
        },
        "stage2_training": train_summary if not classical_only else "Skipped",
        "outputs": {
            "masks_dir":    str(config.FINAL_MASK_DIR),
            "overlays_dir": str(config.OVERLAY_DIR),
            "checkpoints":  str(config.CHECKPOINT_DIR),
            "reports":      str(config.REPORT_DIR),
        },
        "metrics_reported": [
            "mIoU", "F1 (Dice)", "Precision", "Recall", "Accuracy", "Cohen's Kappa"
        ],
        "model": {
            "architecture":  "U-Net with EfficientNet-B0 encoder" if not classical_only else "N/A",
            "encoder":       config.ENCODER_NAME if not classical_only else "N/A",
            "input_size":    f"{config.IMG_SIZE}x{config.IMG_SIZE}",
            "loss":          "0.6×Dice + 0.4×BCE" if not classical_only else "N/A",
            "parameters":    "~4.7M" if not classical_only else "N/A",
        },
    }

    out_path = config.OUTPUT_DIR / "pipeline_summary.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Glacial Lake Detection Pipeline – SNUC GLOFeagles 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Full pipeline
  python run_pipeline.py --classical-only   # No DL model (fast)
  python run_pipeline.py --skip-train       # Use existing checkpoint
  python run_pipeline.py --ablation         # Include ablation study
  python run_pipeline.py --epochs 20        # Fewer training epochs
        """
    )
    parser.add_argument("--classical-only", action="store_true",
                        help="Use classical detector only (no deep learning)")
    parser.add_argument("--skip-train",    action="store_true",
                        help="Skip training, use existing checkpoint")
    parser.add_argument("--ablation",      action="store_true",
                        help="Run ablation study after main pipeline")
    parser.add_argument("--no-overlays",   action="store_true",
                        help="Skip saving overlay visualisations")
    parser.add_argument("--epochs",        type=int,
                        default=config.NUM_EPOCHS,
                        help=f"Training epochs (default: {config.NUM_EPOCHS})")
    parser.add_argument("--ablation-samples", type=int, default=80,
                        help="Images to use for ablation study (default: 80)")
    args = parser.parse_args()

    t_start = time.time()

    banner("GLACIAL LAKE DETECTION PIPELINE\nSNUC GLOFeagles 2026 Challenge", "█")
    print(f"  Dataset   : {config.DATASET_DIR}")
    print(f"  Output    : {config.OUTPUT_DIR}")
    print(f"  Device    : {config.DEVICE}")
    print(f"  Mode      : {'Classical-only' if args.classical_only else 'Hybrid DL'}")
    print()

    # Dependency check
    check_dependencies()

    # ── Stage 1: Classical Pseudo-Labels ──────────────────────────────────
    stage1_stats = stage1_classical()

    # ── Stage 2: U-Net Training ───────────────────────────────────────────
    train_summary = {}
    if not args.classical_only:
        train_summary = stage2_train(
            epochs=args.epochs,
            skip=args.skip_train,
        )

    # ── Stage 3: Inference ────────────────────────────────────────────────
    inference_df = stage3_inference(
        save_overlays=not args.no_overlays,
        classical_only=args.classical_only,
    )

    # ── Stage 4: Ablation (optional) ─────────────────────────────────────
    if args.ablation:
        stage4_ablation(n_samples=args.ablation_samples)

    # ── Final Report ──────────────────────────────────────────────────────
    report = write_final_report(
        stage1_stats, train_summary, inference_df, args.classical_only
    )

    total_time = time.time() - t_start

    banner("PIPELINE COMPLETE", "═")
    print(f"  Total time            : {total_time/60:.1f} min")
    print(f"  Final masks           : {config.FINAL_MASK_DIR}")
    print(f"  Model checkpoint      : {config.BEST_CKPT}")
    print(f"  Pipeline report       : {config.OUTPUT_DIR / 'pipeline_summary.json'}")
    if args.ablation:
        print(f"  Ablation reports      : {config.REPORT_DIR}")
    print()


if __name__ == "__main__":
    main()
