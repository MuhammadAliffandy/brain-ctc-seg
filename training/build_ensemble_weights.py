"""Derive fixed coefficients from a balanced patient-level validation Dice CSV."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from src.services.ensemble_weights import MODEL_KEYS, normalize_weights


def build_profiles(csv_path):
    path = Path(csv_path)
    groups = defaultdict(lambda: defaultdict(dict))
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"].strip().lower() != "validation":
                raise ValueError("Only validation rows are allowed; never tune coefficients on the test set")
            dataset, model, patient = (row[k].strip() for k in ("dataset", "model", "patient_id"))
            score = float(row["dice"])
            if not dataset or not patient or model not in MODEL_KEYS or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid dataset, patient, model key or Dice score")
            if patient in groups[dataset][model]:
                raise ValueError("Duplicate patient/model row; aggregate slices to patient-level Dice first")
            groups[dataset][model][patient] = score
    if not groups:
        raise ValueError("Validation CSV is empty")
    profiles = {}
    for dataset, models in groups.items():
        if set(models) != set(MODEL_KEYS):
            raise ValueError(f"{dataset}: all six model keys are required")
        patients = set(models[MODEL_KEYS[0]])
        if any(set(rows) != patients for rows in models.values()):
            raise ValueError("Every model must be evaluated on the same validation patients")
        means = {k: sum(rows.values()) / len(rows) for k, rows in models.items()}
        profiles[dataset] = {"split": "validation", "source": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "method": "normalized_mean_patient_validation_dice", "patient_count": len(patients),
            "mean_dice": means, "weights": normalize_weights(means, MODEL_KEYS)}
    return {"profiles": profiles}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="patient_id,dataset,model,split,dice columns")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_profiles(args.csv)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
