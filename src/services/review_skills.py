"""Bounded, inspectable review procedures; these are not diagnostic classifiers."""

import hashlib
from datetime import datetime, timezone

import cv2
import numpy as np


SKILLS = {
    1: "Check contour evidence",
    2: "Assess model availability",
    3: "Inspect boundary ambiguity",
    4: "Propose contour cleanup",
    5: "Check a contour change",
    6: "Measure contour variability",
}


def fingerprint(image, original, active, probabilities):
    digest = hashlib.sha256()
    for array in [image, original, active, *probabilities.values()]:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def geometry(mask):
    mask = (mask > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    area = int(mask.sum())
    return {"area_px": area, "components": int(count - 1),
            "single_pixel_islands": int(np.sum(areas == 1)),
            "compactness": float(4 * np.pi * area / perimeter ** 2) if perimeter else 0.0}


def run_review_skill(skill_id, image, original, active, probabilities, simulated_keys,
                     pixel_spacing, volume, slice_thickness):
    image = np.asarray(image, dtype=np.float32)
    original = (original > 0).astype(np.uint8)
    active = (active > 0).astype(np.uint8)
    result = {"skill_id": skill_id, "name": SKILLS[skill_id],
              "time": datetime.now(timezone.utc).isoformat(), "status": "Review required",
              "evidence": {}, "limitations": [], "next_action": "Review the contour against the CT.",
              "tools": [], "candidate": None}
    simulated_keys = set(simulated_keys)
    real = {k: p for k, p in probabilities.items() if k not in simulated_keys}
    if not np.isfinite(image).all() or image.shape != active.shape:
        result.update(status="Blocked", summary="Image and contour must have matching, finite data.")
        return result
    if skill_id not in SKILLS:
        raise ValueError("Unknown review skill")
    if simulated_keys:
        result["limitations"].append("Simulated model masks are present. Results are exploratory and cannot verify a model prediction.")
    result["limitations"].append("Physical values use configured display spacing, not verified acquisition geometry.")
    shape = geometry(active)
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.morphologyEx(active, cv2.MORPH_GRADIENT, kernel) > 0
    outside = (cv2.dilate(active, np.ones((7, 7), np.uint8)) > 0) & (active == 0)
    inside = image[active > 0]
    ring = image[outside]
    cnr = float(abs(inside.mean() - ring.mean()) / max(float(ring.std()), 1e-6)) if inside.size and ring.size else None

    if skill_id == 1:
        result["tools"] = ["Sobel gradient", "Boundary sampling", "Mask overlap comparison"]
        gradient = cv2.magnitude(cv2.Sobel(image, cv2.CV_32F, 1, 0), cv2.Sobel(image, cv2.CV_32F, 0, 1))
        nonzero_gradient = gradient[gradient > 0]
        threshold = float(np.percentile(nonzero_gradient, 75)) if nonzero_gradient.size else float("inf")
        support = float(np.mean(gradient[boundary] >= threshold)) if boundary.any() else None
        result["evidence"] = {"boundary_pixels": int(boundary.sum()), "boundary_in_top_nonzero_gradient_quartile": support,
                              "real_model_outputs": len(real), "simulated_outputs": len(simulated_keys)}
        result["summary"] = "Image-edge evidence sampled along the current contour." if boundary.any() else "No contour boundary is available to inspect."
        result["status"] = "Review required" if boundary.any() else "Blocked"
        result["limitations"].append("Edge strength is a normalized image cue. It cannot establish lesion presence or rule out a shared miss.")
    elif skill_id == 2:
        result["tools"] = ["Model output inventory", "Probability-map validation"]
        valid = [k for k, p in real.items() if np.isfinite(p).all() and p.shape == image.shape and np.min(p) >= 0 and np.max(p) <= 1]
        result["evidence"] = {"valid_real_models": valid, "simulated_models": sorted(simulated_keys)}
        result.update(status="Blocked", summary="Model availability checked. Adaptive expert selection is not configured.",
                      next_action="Use available model comparisons or supply patient-disjoint model utility evidence before adaptive selection.")
        result["limitations"].append("No validated utility predictor or OOD detector is connected; this procedure does not invent model rankings.")
    elif skill_id == 3:
        result["tools"] = ["Local contrast-to-noise calculation", "Connected-component analysis"]
        result["evidence"] = {**shape, "local_contrast_to_noise": cnr, "inside_samples": int(inside.size), "outside_samples": int(ring.size)}
        result["summary"] = "Local contrast and contour fragmentation measured." if cnr is not None else "Insufficient contour or surrounding image samples."
        result["status"] = "Review required" if cnr is not None else "Blocked"
        result["limitations"].append("These cues do not diagnose an artifact, disease, or failure mechanism. No calibrated clinical cutoff is applied.")
    elif skill_id == 4:
        result["tools"] = ["3×3 morphological closing", "Mask difference", "Boundary-distance guardrail"]
        candidate = cv2.morphologyEx(active, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        changed = candidate != active
        change_pct = float(changed.sum() / max(int(active.sum()), 1) * 100)
        distance = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 3)
        distance_px = float(distance[changed].max()) if changed.any() else 0.0
        bounded = active.any() and change_pct <= 20 and distance_px <= 2
        result["evidence"] = {"changed_pixels": int(changed.sum()), "changed_area_percent": change_pct,
                              "maximum_change_distance_px": distance_px, "limit_area_percent": 20, "limit_distance_px": 2}
        result.update(status="Proposal ready" if bounded and changed.any() else ("No change" if not changed.any() else "Blocked"),
                      summary="A small contour cleanup is available for inspection." if bounded and changed.any() else "No bounded cleanup proposal is available.",
                      next_action="Inspect the before/after preview, run the change check, then explicitly apply or discard the proposal.")
        result["candidate"] = candidate if bounded and changed.any() else None
        result["limitations"].append("Engineering bounds are 2 display pixels and 20% changed area; they are not validated clinical safety limits.")
    elif skill_id == 5:
        result["tools"] = ["Mask difference", "Connected-component analysis", "Compactness calculation"]
        baseline = geometry(original)
        drift = float(abs(shape["area_px"] - baseline["area_px"]) / max(baseline["area_px"], 1) * 100)
        changed = int(np.sum(original != active))
        checks = {"area_drift_at_most_20_percent": drift <= 20,
                  "no_single_pixel_islands": shape["single_pixel_islands"] == 0,
                  "no_additional_components": shape["components"] <= baseline["components"],
                  "nonempty_contour": shape["area_px"] > 0}
        result["evidence"] = {**shape, "changed_pixels": changed, "area_drift_percent": drift, **checks}
        result.update(status="No change" if not changed else ("Checks passed" if all(checks.values()) else "Review required"),
                      summary="No contour change to verify." if not changed else "Geometric change checks completed.",
                      next_action="A clinician must inspect image alignment and approve any change; passing geometric checks is not clinical acceptance.")
        result["limitations"].append("This is a geometric guardrail, not an independent trained quality estimator or ground-truth Dice/IoU assessment.")
    elif skill_id == 6:
        result["tools"] = ["Pixel count", "Area conversion", "Voxel count", "Per-model area spread"]
        areas = {k: float(np.sum(p >= 0.5) * pixel_spacing ** 2) for k, p in real.items()}
        result["evidence"] = {"active_slice_area_mm2": float(active.sum() * pixel_spacing ** 2),
                              "current_volume_mask_ml": float(np.sum(volume > 0) * pixel_spacing ** 2 * slice_thickness / 1000),
                              "real_model_slice_areas_mm2": areas}
        if len(areas) >= 2:
            result["evidence"].update(minimum_model_area_mm2=min(areas.values()), maximum_model_area_mm2=max(areas.values()))
        result.update(status="Measured", summary="Contour measurements calculated from the supplied masks.")
        result["limitations"].append("Model min/max spread is not a confidence interval. Full-volume inference is required for model-derived lesion volumetry.")
    return result
