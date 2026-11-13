"""Fixed ensemble coefficients with explicit provenance and equal-weight fallback."""

import json
from pathlib import Path

import numpy as np


MODEL_KEYS = ("se2", "harmonic", "nn", "attn", "trans", "std")


def normalize_weights(weights, keys):
    if set(weights) != set(keys):
        raise ValueError("Weights must cover exactly the configured model panel")
    values = np.asarray([weights[k] for k in keys], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative and have a positive total")
    return {k: float(v / values.sum()) for k, v in zip(keys, values)}


def resolve_weights(path, dataset, keys=MODEL_KEYS, manual=None):
    if manual is not None:
        return {"mode": "experimental", "dataset": dataset, "method": "manual",
                "weights": normalize_weights(manual, keys), "source": "Session settings",
                "reason": "Manual fixed weights; performance has not been established."}
    try:
        profiles = json.loads(Path(path).read_text())["profiles"]
        if not isinstance(profiles, dict):
            raise ValueError("Profiles must be an object keyed by dataset")
        profile = profiles.get(dataset)
        if not isinstance(profile, dict):
            raise ValueError(f"No validation profile for {dataset}")
        if profile.get("split") != "validation" or not profile.get("source") or not profile.get("method"):
            raise ValueError("Profile must identify its validation split, source and method")
        weights = normalize_weights(profile["weights"], keys)
        return {**profile, "mode": "validation-derived", "dataset": dataset, "weights": weights,
                "reason": "Fixed coefficients derived from validation data; not adaptive per scan."}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"mode": "equal-fallback", "dataset": dataset, "method": "equal",
                "weights": normalize_weights(dict.fromkeys(keys, 1.0), keys), "source": str(path),
                "reason": f"{exc}. Using equal coefficients."}


def weighted_probability(probabilities, weights):
    normalized = normalize_weights(weights, list(probabilities))
    arrays = [np.asarray(p, dtype=np.float64) for p in probabilities.values()]
    if not arrays or any(a.shape != arrays[0].shape or not np.isfinite(a).all()
                         or np.any(a < 0) or np.any(a > 1) for a in arrays):
        raise ValueError("Probability maps must share a shape and contain finite values in [0, 1]")
    return sum(normalized[k] * p for k, p in zip(probabilities, arrays)).astype(np.float32)
