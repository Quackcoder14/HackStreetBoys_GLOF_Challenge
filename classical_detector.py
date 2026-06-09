"""
classical_detector.py – Stage 1: Multi-feature Classical Glacial Lake Detector
Generates pseudo-label masks used to supervise the U-Net.

Key insight: Glacial lakes appear as:
  - Very dark (low brightness across all RGB channels)
  - Low saturation (nearly achromatic dark)
  - Smooth / homogeneous interiors (low local texture variance)
  - Not warm-hued (distinguishes from debris/brown rock)
  - Compact and rounded (distinguishes from elongated shadows)

SNUC GLOFeagles 2026 Challenge
"""

import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage import morphology, measure
from tqdm import tqdm
import config


# ─────────────────────────────────────────────────────────────────────────────
# Individual Feature Maps
# ─────────────────────────────────────────────────────────────────────────────

def compute_brightness(img_bgr: np.ndarray) -> np.ndarray:
    """
    Mean brightness across BGR channels, normalised to [0,1].
    Glacial lakes: brightness < BRIGHTNESS_THRESHOLD/255
    """
    return img_bgr.mean(axis=2).astype(np.float32) / 255.0


def compute_hsv_features(img_bgr: np.ndarray):
    """Return H, S, V planes (OpenCV scale: H∈[0,180], S,V∈[0,255])."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H = hsv[:, :, 0]          # Hue  0-180
    S = hsv[:, :, 1] / 255.0  # Saturation 0-1
    V = hsv[:, :, 2] / 255.0  # Value 0-1
    return H, S, V


def compute_local_texture(img_bgr: np.ndarray, window: int = None) -> np.ndarray:
    """
    Local standard deviation of grayscale intensity.
    Low texture → smooth water surface. High texture → rock / debris / shadow.
    """
    if window is None:
        window = config.TEXTURE_WINDOW
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Local mean
    mean_sq = cv2.blur(gray ** 2, (window, window))
    sq_mean = cv2.blur(gray, (window, window)) ** 2
    variance = np.maximum(mean_sq - sq_mean, 0.0)
    return np.sqrt(variance)   # std-dev map


def compute_blue_dominance(img_bgr: np.ndarray) -> np.ndarray:
    """
    B - max(R, G): positive for bluish/neutral dark, negative for warm hues.
    Helps distinguish deep water from brown debris.
    Returned in [-255, 255], normalised to [-1, 1].
    """
    b = img_bgr[:, :, 0].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    bd = b - np.maximum(r, g)
    return bd / 255.0


def compute_ndwi_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """
    Approximate NDWI using RGB only:
      NDWI_rgb = (G - R) / (G + R + 1e-6)
    Without NIR, this is a rough proxy. Positive values hint at water.
    """
    r = img_bgr[:, :, 2].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    ndwi = (g - r) / (g + r + 1e-6)
    return ndwi  # range [-1, 1]


def compute_edge_density(img_bgr: np.ndarray, window: int = 15) -> np.ndarray:
    """
    Local edge density via Canny. Water surfaces are smooth (low edge density).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100).astype(np.float32) / 255.0
    kernel = np.ones((window, window), dtype=np.float32) / (window * window)
    return cv2.filter2D(edges, -1, kernel)  # local edge fraction


# ─────────────────────────────────────────────────────────────────────────────
# Candidate Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidate_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Combine multiple features into a candidate water mask using adaptive thresholds.
    Returns binary uint8 mask (0 or 255).
    """
    brightness = compute_brightness(img_bgr)          # 0-1 low=dark
    H, S, V    = compute_hsv_features(img_bgr)
    texture    = compute_local_texture(img_bgr)        # 0-255 low=smooth
    blue_dom   = compute_blue_dominance(img_bgr)       # -1..1 positive=blue
    ndwi       = compute_ndwi_rgb(img_bgr)             # -1..1 positive=water
    edge_dens  = compute_edge_density(img_bgr)         # 0..1 low=smooth

    # ── Condition 1: Very dark pixels ──────────────────────────────────────
    dark_mask = V < config.HSV_V_THRESHOLD

    # ── Condition 2: Low saturation (achromatic dark, not coloured debris) ──
    low_sat_mask = S < config.HSV_S_MAX

    # ── Condition 3: Not warm-hued (exclude brown/orange rock & debris) ──
    # Hue < 5 or Hue > 35 to keep only non-warm pixels
    not_warm = np.logical_or(H < config.HUE_WARM_LOW, H > config.HUE_WARM_HIGH)

    # ── Condition 4: Low local texture (smooth surface) ────────────────────
    smooth_mask = texture < config.TEXTURE_STD_MAX

    # ── Condition 5: Low edge density ──────────────────────────────────────
    low_edge = edge_dens < 0.08

    # ── Combine: primary (strict) and secondary (relaxed) ──────────────────
    # Primary: must satisfy darkness + low saturation + not warm
    primary   = dark_mask & low_sat_mask & not_warm
    # Secondary: primary + (smooth OR low edge)
    secondary = primary & (smooth_mask | low_edge)

    # Use secondary as candidate
    candidate = secondary.astype(np.uint8) * 255
    return candidate


# ─────────────────────────────────────────────────────────────────────────────
# Shadow Discrimination
# ─────────────────────────────────────────────────────────────────────────────

def reject_shadow_regions(mask: np.ndarray, img_bgr: np.ndarray) -> np.ndarray:
    """
    Terrain cast shadows are:
      - Elongated (high eccentricity)
      - Near high-gradient terrain edges
      - Narrow compared to typical lake shapes
    Lakes are:
      - Compact / sub-circular (low eccentricity)
      - Isolated from strong gradient edges
    """
    labels = measure.label(mask > 0, connectivity=2)
    props  = measure.regionprops(labels)

    out_mask = np.zeros_like(mask)

    for prop in props:
        area = prop.area
        # Skip tiny blobs
        if area < config.MIN_LAKE_AREA:
            continue

        eccentricity = prop.eccentricity   # 0=circle, 1=line
        solidity     = prop.solidity       # filled area ratio

        # Shadow heuristic: very elongated + low area
        if eccentricity > 0.97 and area < 2000:
            continue  # likely shadow stripe

        # Very low solidity → fragmented, likely shadow texture
        if solidity < 0.25:
            continue

        # Accept as lake
        out_mask[labels == prop.label] = 255

    return out_mask


# ─────────────────────────────────────────────────────────────────────────────
# Morphological Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """
    Opening (removes noise), Closing (fills holes), then keep largest blobs.
    """
    k_open  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.MORPH_OPEN_SIZE,) * 2)
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.MORPH_CLOSE_SIZE,) * 2)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
    return closed


# ─────────────────────────────────────────────────────────────────────────────
# Per-Image Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_lakes(img_bgr: np.ndarray) -> np.ndarray:
    """
    Full classical detection pipeline for one image.
    Returns binary uint8 mask (0 or 255), same HxW as input.
    """
    # Step 1: Generate candidates
    candidate = generate_candidate_mask(img_bgr)

    # Step 2: Morphological cleanup
    cleaned   = morphological_cleanup(candidate)

    # Step 3: Shadow / shape-based rejection
    refined   = reject_shadow_regions(cleaned, img_bgr)

    # Step 4: Final morphological closing to fill remaining holes
    k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    final  = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k_fill)

    return final


# ─────────────────────────────────────────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────────────────────────────────────────

def run_classical_detection(
        image_dir: Path = None,
        out_dir:   Path = None,
        verbose:   bool = True
) -> dict:
    """
    Run classical detector on all images in image_dir.
    Saves masks to out_dir as PNG (binary: 0=background, 255=lake).
    Returns dict of {img_id: lake_pixel_fraction}.
    """
    if image_dir is None:
        image_dir = config.DATASET_DIR
    if out_dir is None:
        out_dir = config.PSEUDO_MASK_DIR

    image_paths = sorted(image_dir.glob("*.png"),
                         key=lambda p: int(p.stem))
    if verbose:
        print(f"[Classical Detector] Processing {len(image_paths)} images …")

    stats = {}
    for img_path in tqdm(image_paths, desc="Classical Detection", disable=not verbose):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  WARNING: Could not read {img_path}")
            continue

        mask = detect_lakes(img_bgr)
        lake_fraction = (mask > 0).sum() / mask.size
        stats[int(img_path.stem)] = lake_fraction

        # Save pseudo-label mask
        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), mask)

    if verbose:
        total_with_lakes = sum(1 for v in stats.values() if v > 0.005)
        print(f"[Classical Detector] Done. "
              f"{total_with_lakes}/{len(stats)} images contain detectable lakes.")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stats = run_classical_detection(verbose=True)
    # Show summary
    fractions = list(stats.values())
    print(f"\nLake coverage statistics:")
    print(f"  Mean  lake fraction : {np.mean(fractions):.4f}")
    print(f"  Median lake fraction: {np.median(fractions):.4f}")
    print(f"  Max   lake fraction : {np.max(fractions):.4f}")
    print(f"  Images with >1% lake: {sum(f > 0.01 for f in fractions)}")
