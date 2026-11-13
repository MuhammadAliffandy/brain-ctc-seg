# Fixed ensemble coefficients

`Weighted Soft Voting` combines each model probability map as
`sum(weight[model] * probability[model]) / sum(weights)`, then applies the 0.5 threshold.
Checkpoint parameters (`.pth`) and ensemble coefficients are different quantities.

The default registry is `config/ensemble_weights.json`. Override its location with
`BRAIN_ENSEMBLE_WEIGHTS`. Profiles are separate for `ncct`, `cect`, `stroke`, and
`hemorrhage`. The committed registry is empty because a complete validation
evaluation for all six models has not been supplied. Missing or invalid profiles
use equal coefficients and report `equal-fallback` in Technical review and the
orchestrator evidence. Manual coefficients in Settings are labelled `experimental`.
Simulated outputs still block the automatic review workflow.

## Building a profile

Export one Dice score per validation patient and model, using the same patients
for all six checkpoints. Required CSV columns:

```text
patient_id,dataset,model,split,dice
```

Use model keys `se2,harmonic,nn,attn,trans,std` and split `validation`.
Aggregate slices to patient-level scores before export. Preserve the original
checkpoint versions, patient split manifests and preprocessing configuration with
the evaluation. Training and final test patients must be excluded. The builder
checks CSV labels and balanced patient coverage; it cannot independently establish
patient disjointness or checkpoint provenance from these five columns.

From the repository root:

```bash
python -m training.build_ensemble_weights /path/to/validation.csv --output config/ensemble_weights.json
```

The builder normalizes each model's mean patient validation Dice into a coefficient.
This is a validation-derived baseline, not a search for the globally optimal weights,
not probability calibration, and not adaptive selection per scan. It rejects test
rows, duplicate patient/model rows, missing models, unequal patient sets and invalid
scores. It records the input CSV hash, patient count, mean scores and method.

Reload the app after updating the registry. A changed profile invalidates prior
review results. Rebuild the profile when checkpoints, dataset split or preprocessing
change. Evaluate the chosen ensemble once on the untouched patient-level test set
before making any performance claim.
