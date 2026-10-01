# Final Evaluation Metrics - IEEE TMI Revision

This document contains all the extracted metrics from the server logs to be copied directly into the manuscript tables.

---

## 🟢 Table 3: Missing Classification Metrics (Mod-SE(2))
*(Values extracted from `metrics_output.log`)*

- **Accuracy:** 0.9986
- **Precision:** 0.8230
- **Recall (Sensitivity):** 0.8875
- **Dice:** 0.8541
- **IoU:** 0.7453
- **3D HD95:** 0.71 mm (IQR: 0.71 - 4.60)

---

## 🟢 Table 11 / Figure 8: Lesion-wise HD95 Analysis
*(Values extracted from `generate_table11_metrics.py`)*

| Lesion | GT Area (px) | Pred Area (px) | HD95 (mm) |
|--------|--------------|----------------|-----------|
| **1**  | 419          | 415            | 0.45      |
| **2**  | 237          | 229            | 0.45      |
| **3**  | 204          | 205            | 0.45      |
| **4**  | 80           | 70             | 0.52      |
| **TOTAL** | - | - | **0.45 mm** |

---

## 🟢 Table 10: Full Cohort Lesion Tracking & Volumetric Agreement
*(Values extracted from `generate_table10_metrics.py`)*

### 1. Mod-Seg-SE(2) [Old Version / Baseline]
* **Detection & False Positives:**
  - Lesion-wise Sensitivity: `54.05%`
  - Lesion-wise Precision: `56.18%`
  - Lesion-wise F1 Score: `0.5510`
  - False-positive lesions per scan: `25.33`
  - Slice Accuracy: `80.04%`
* **Cross-Slice Tracking:**
  - Ref tracks recovered as 1 track: `52.97%`
  - Fragmented tracks: `1.08%`
  - Merged tracks: `0.00%`
* **Volumetric Error & Bland-Altman:**
  - Bland-Altman Bias: `-0.9367 mL` (95% LoA: `[-11.7947, 9.9213]`)
  - Abs Error for Lesions < 1 mL: `0.0437 mL`
  - Abs Error for Lesions 1-10 mL: `0.0577 mL`
  - Abs Error for Lesions > 10 mL: `11.2191 mL`

### 2. CT-SE(2) [Proposed Version / Ablation A8]
* **Detection & False Positives:**
  - Lesion-wise Sensitivity: `87.57%`
  - Lesion-wise Precision: `84.37%`
  - Lesion-wise F1 Score: `0.8594`
  - False-positive lesions per scan: `8.67`
  - Slice Accuracy: `91.80%`
* **Cross-Slice Tracking:**
  - Ref tracks recovered as 1 track: `84.32%`
  - Fragmented tracks: `2.70%`
  - Merged tracks: `0.54%`
* **Volumetric Error & Bland-Altman:**
  - Bland-Altman Bias: `-0.0641 mL` (95% LoA: `[-3.8787, 3.7506]`)
  - Abs Error for Lesions < 1 mL: `0.0149 mL`
  - Abs Error for Lesions 1-10 mL: `0.0787 mL`
  - Abs Error for Lesions > 10 mL: `4.6684 mL`

---

## 🟢 Table 4: Baseline Models (CECT NTUH Dataset)
*(Values extracted from `remaining_hd95_v2.log`)*

- **Mod-Seg-SE(2):** `0.45 mm`
- **HarmonicNet:** `0.95 mm`
- **nnU-Net:** `33.83 mm`
- **Standard U-Net:** `73.84 mm`
- **Attention U-Net:** `60.75 mm`
- **TransUNet:** `313.48 mm`

---

## 🟢 Table 5: Baseline Models (Kaggle Stroke Dataset)
*(Values extracted from `remaining_hd95_v2.log`)*

- **Mod-Seg-SE(2):** `0.64 mm`
- **HarmonicNet:** *(N/A - Model not evaluated)*
- **nnU-Net:** `1.01 mm`
- **Standard U-Net:** `1.52 mm`
- **Attention U-Net:** `1.35 mm`
- **TransUNet:** `1.41 mm`

---

*Note: Please provide the remaining screenshots for **Table 3 (NCCT Baseline)**, **Table 6 (Kaggle Hemorrhage)**, and **Table 7-9 (Ablations)** whenever they are ready!*
