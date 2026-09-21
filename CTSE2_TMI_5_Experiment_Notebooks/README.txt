CT-SE(2) IEEE TMI EXPERIMENT PACKAGE
====================================

Files:
01_component_ablation.ipynb
02_prior_ModSE2_vs_CTSE2.ipynb
03_cohort_lesion_tracking.ipynb
04_boundary_specific_evaluation.ipynb
05_2D_vs_2p5D_context.ipynb

HOW TO USE
1. Prepare manifest.csv with patient_id, split, image_path, mask_path,
   prev1_path, next1_path, and preferably pixel_spacing_x/y + slice_index.
   Add prev2_path/next2_path for the optional 5-slice experiment.
2. Open each notebook.
3. Edit USER CONFIG paths/hyperparameters.
4. Replace MODEL FACTORY with imports for your actual models.
5. Ensure build_model(variant) returns logits [B,1,H,W].
6. Run All.

IMPORTANT SCIENTIFIC NOTES
- Use the exact preprocessing, focal loss, Dice loss, edge-boundary loss,
  optimizer settings, and SE(2) implementation from the final manuscript.
- The included Sobel boundary loss is a runnable reference implementation,
  not a claim that it is identical to your existing code.
- Keep patient-level splits fixed across all variants.
- For fair architecture comparisons, control parameter capacity where feasible.
- The tracking notebook uses connected-component + Hungarian IoU matching.
  Predefine the matching threshold before reporting final results.
- The tracking analysis is cross-slice tracking. True longitudinal tracking
  requires serial examinations from the same patients and an additional
  time-point matching design.
