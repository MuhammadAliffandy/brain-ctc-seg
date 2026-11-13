import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.services.ensemble_weights import MODEL_KEYS, normalize_weights, resolve_weights, weighted_probability
from training.build_ensemble_weights import build_profiles


class EnsembleWeightsTests(unittest.TestCase):
    def test_weighted_probability_changes_result(self):
        maps = {"a": np.array([[.9, .2]]), "b": np.array([[.1, .8]])}
        np.testing.assert_allclose(weighted_probability(maps, {"a": 3, "b": 1}), [[.7, .35]])
        np.testing.assert_allclose(weighted_probability(maps, {"a": 1, "b": 1}), [[.5, .5]])

    def test_invalid_coefficients_are_rejected(self):
        for coefficients in ({"a": 0}, {"a": -1}, {"a": float("nan")}, {"b": 1}):
            with self.assertRaises(ValueError):
                normalize_weights(coefficients, ["a"])

    def test_invalid_probability_maps_are_rejected(self):
        with self.assertRaises(ValueError):
            weighted_probability({"a": np.array([[float("nan")]])}, {"a": 1})

    def test_missing_profile_falls_back_equally(self):
        result = resolve_weights("/nonexistent/ensemble.json", "ncct")
        self.assertEqual(result["mode"], "equal-fallback")
        self.assertAlmostEqual(sum(result["weights"].values()), 1)

    def test_validation_profiles_are_dataset_specific(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profiles.json"
            path.write_text(json.dumps({"profiles": {"ncct": {"split": "validation", "source": "val.csv",
                "method": "validation Dice", "weights": dict(zip(MODEL_KEYS, [6, 5, 4, 3, 2, 1]))}}}))
            self.assertEqual(resolve_weights(path, "ncct")["mode"], "validation-derived")
            self.assertEqual(resolve_weights(path, "cect")["mode"], "equal-fallback")
            data = json.loads(path.read_text()); data["profiles"]["ncct"]["split"] = "test"
            path.write_text(json.dumps(data))
            self.assertEqual(resolve_weights(path, "ncct")["mode"], "equal-fallback")

    def test_manual_weights_are_labelled_experimental(self):
        result = resolve_weights("", "ncct", manual=dict.fromkeys(MODEL_KEYS, 1))
        self.assertEqual(result["mode"], "experimental")

    def test_csv_builder_and_test_set_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "validation.csv"
            rows = [{"patient_id": p, "dataset": "ncct", "model": k, "split": "validation", "dice": (i + 1) / 10}
                    for p in ["p1", "p2"] for i, k in enumerate(MODEL_KEYS)]

            def write_rows():
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

            write_rows()
            profile = build_profiles(path)["profiles"]["ncct"]
            self.assertEqual(profile["patient_count"], 2)
            self.assertAlmostEqual(profile["weights"]["std"], 6 / 21)
            rows[0]["split"] = "test"; write_rows()
            with self.assertRaises(ValueError):
                build_profiles(path)
            rows[0]["split"] = "validation"; rows.pop(); write_rows()
            with self.assertRaises(ValueError):
                build_profiles(path)


if __name__ == "__main__":
    unittest.main()
