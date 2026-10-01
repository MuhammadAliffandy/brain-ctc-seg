# 📄 IEEE TMI Manuscript - Missing Metrics (XXXX) Tracker

This document tracks all the missing quantitative values (`XXXX`) in the manuscript across various tables and text sections. It serves as our ultimate checklist before final submission.

## 🔴 1. The Global HD95 Bug (Tables 3, 4, 5, 6, 7, 8, 9, 11)
**Issue:** The previous evaluation scripts used a `surface_distances` algorithm that failed (maxing out at `486.001 mm`) when encountering empty mask slices (slices without tumors). 
**Action Required:** Rewrite the HD95 evaluation logic to safely ignore or penalize empty slices without breaking the mean calculation, then recalculate for **ALL** datasets and models.

- [ ] **Table 3 (NCCT NTUH):** HD95 for CT-SE(2), HarmonicNet, Mod-SE(2), nnU-Net, U-Net, Att U-Net, TransUNet.
- [ ] **Table 4 (CECT NTUH):** HD95 for all evaluated models.
- [ ] **Table 5 (Kaggle Stroke):** HD95 for all evaluated models. *(Requires Public Dataset Path)*
- [ ] **Table 6 (Kaggle Hemorrhage):** HD95 for all evaluated models. *(Requires Public Dataset Path)*
- [ ] **Table 7 (Component Ablation):** HD95 for models A1 through A8.
- [ ] **Table 8 (Loss Ablation):** HD95 for Dice, Focal+Dice, Dice+Bound, CT-SE(2).
- [ ] **Table 9 (Context Ablation):** HD95 for 1-slice, 3-slices, 5-slices.
- [x] **Table 11 (Fig 8 Lesion-wise):** HD95 specifically for Lesions 1, 2, 3, 4, and Total.

## 🟡 2. Mod-SE(2) Missing Basic Metrics (Table 3)
**Issue:** The ablation evaluation script only saved Dice, IoU, ASSD, and Surface Dice. Table 3 requires full classification metrics.
**Action Required:** Evaluate the `A8_ct_best.pth` (Mod-SE(2)) model to extract missing classification metrics.

- [x] **Table 3:** Accuracy for Mod-SE(2).
- [x] **Table 3:** Precision for Mod-SE(2).
- [x] **Table 3:** Recall for Mod-SE(2).

## 🔵 3. Full Cohort Lesion Tracking & Volumetric (Table 10 & Abstract)
**Issue:** The previous tracking script only evaluated 3 pilot scans. Table 10 requires a massive evaluation across the **complete NTUH NCCT test set**. Furthermore, ICC and Volume metrics were never calculated.
**Action Required:** Build a full-scale tracking script comparing `CT-SE(2)` vs `Mod-SE(2)` across the entire test set.

- [x] **Detection:** Lesion-wise Sensitivity (%), Precision (%), F1 Score.
- [x] **False Positives:** False-positive lesions per scan.
- [x] **Cross-Slice Tracking:** Reference tracks recovered as one track (%), Fragmented tracks (%), Merged tracks (%).
- [x] **Volumetric Agreement:** Volume ICC(2,1) [95% CI].
- [x] **Bland-Altman:** Bias, mL [95% LoA].
- [x] **Stratified Error:** Absolute volume error for lesions <1 mL, 1–10 mL, and >10 mL.

## 🟢 4. Dataset Paths Setup
Ensure that any new comprehensive evaluation script maps dynamically to these paths:
- **NCCT / CECT (Private):** `~/Clara/new_drive/CT Brain Data/MyDrive/Dataset_CT_Report.csv`
- **Kaggle Stroke (Public):** Ensure script logic resolves public dataset paths (can be reset/configured first).
- **Kaggle Hemorrhage (Public):** Ensure script logic resolves public dataset paths (can be reset/configured first).

---
*Target: Create one master script `generate_missing_manuscript_metrics.py` to automate all unchecked boxes above.*
