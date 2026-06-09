# Glacial Lake Detection from Satellite Imagery

### SNUC GLOFeagles 2026 Challenge Submission

This repository contains the complete technical solution for the **GLOFeagles '26 Challenge**. Our framework leverages a **3-stage hybrid approach** that combines a classical multi-feature detector to generate high-quality pseudo-labels with a custom Deep Learning U-Net for noise filtering, generalization, and final boundary segmentation.

---

## 📺 Explanation Video (Max 5 Minutes)

- **YouTube Link:**

---

## 🚀 Key Features & Methodology

1. **Stage 1 (Classical Multi-Feature Detector):** Combines HSV Value/Saturation bands, Warm-hue exclusions, local standard deviation texture filtering, and shape boundaries (eccentricity, solidity) to output high-precision pseudo-masks.
2. **Stage 2 (Custom U-Net Training):** Trained on pseudo-masks using a custom PyTorch architecture, bypassing external dependency constraints. Regularized via a compound loss function: $0.6 \times \text{Dice Loss} + 0.4 \times \text{Weighted BCE}$.
3. **Stage 3 (Morphological Post-Processing):** Filters high-frequency noise and terrain shadows using structuring elements (opening/closing operations) and connected components filters.

### 📊 Performance Summary

- **Validation Dice (F1-score):** **0.5980**
- **Validation mIoU:** **0.4266**
- **Validation Recall:** **0.9940** (Highly sensitive to identifying water bodies)
- **Validation Accuracy:** **83.66%**
- **Cohen's Kappa ($\kappa$):** **0.5151** (Substantial agreement)

---

## 📂 Repository Structure

```
TeamName_GLOF_Challenge/
├── run_pipeline.py          <- End-to-end pipeline run script
├── config.py                <- Hyperparameters and configurations
├── classical_detector.py    <- Stage 1: rule-based detector (pseudo-labels)
├── model_architecture.py    <- U-Net model architecture definition
├── model.py                 <- Model factories and custom losses
├── dataset.py               <- PyTorch dataset and loaders
├── train.py                 <- Stage 2: U-Net training loop
├── inference.py             <- Stage 3: Inference, post-processing & overlays
├── metrics.py               <- Evaluation script containing all 6 metrics
├── utils.py                 <- Utility functions (saving, overlays, load_image)
├── requirements.txt         <- Python dependencies
├── pipeline_notebook.ipynb  <- Step-by-step notebook for visualization
├── report/
│   └── methodology.md       <- Full technical methodology report
└── outputs/
    ├── checkpoints/
    │   └── best_model.pth   <- Trained U-Net model checkpoint (SUBMIT)
    ├── masks/               <- Final predicted binary masks (SUBMIT)
    ├── overlays/            <- Visual check overlays (original + mask)
    └── pipeline_summary.json<- Execution summary log
```

---

## 🛠️ Reproduction Instructions

### 1. Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline

To run the full detector, train the U-Net for 3 epochs, and generate masks/overlays, execute:

```bash
python run_pipeline.py --epochs 3
```

_Outputs will be written to the `outputs/` folder._

### 3. Step-by-Step Visualization

Open the interactive Jupyter Notebook to visualize predictions on sample images:

```bash
jupyter notebook pipeline_notebook.ipynb
```

---

## 📝 Challenge Submission Standard

- Submitted to: `glofeagles@snuchennai.edu.in`
- Repository name: `HackStreetBoys_GLOF_Challenge`
- Final predictions location: `outputs/masks/*.png`
- Model weights location: `outputs/checkpoints/best_model.pth`
