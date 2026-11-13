"""Deterministic routing for the CT review procedures, with no automatic mask writes."""

from datetime import datetime, timezone

import cv2
import numpy as np

from .review_skills import run_review_skill


POLICY_VERSION = "review-workflow-2"


def run_review(image, original, active, probabilities, simulated_keys, pixel_spacing,
               volume, slice_thickness, expected_models, fusion_policy=None):
    report = {"policy": POLICY_VERSION, "time": datetime.now(timezone.utc).isoformat(),
              "status": "Blocked", "summary": "Review could not start.", "steps": [],
              "skills": {}, "next_action": "Inspect the scan and model status.",
              "clinician_approval_required": True, "fusion_policy": fusion_policy}

    def event(stage, status, reason):
        report["steps"].append({"stage": stage, "status": status, "reason": reason})

    def execute(skill_id, baseline=original, target=active):
        try:
            result = run_review_skill(skill_id, image, baseline, target, probabilities,
                                      simulated_keys, pixel_spacing, volume, slice_thickness,
                                      **({"fusion_policy": fusion_policy} if fusion_policy else {}))
        except Exception as exc:
            event(f"S{skill_id:02d}", "Failed", f"{type(exc).__name__}: {exc}")
            report["summary"] = "A review procedure failed; the original contour was retained."
            return None
        report["skills"][skill_id] = result
        event(f"S{skill_id:02d}", result["status"], result["summary"])
        return result

    arrays = [image, original, active, volume]
    valid = (bool(expected_models) and isinstance(probabilities, dict)
             and all(isinstance(a, np.ndarray) for a in arrays)
             and image.ndim == 2 and volume.ndim == 3 and volume.size > 0
             and original.shape == image.shape == active.shape and image.size > 0
             and all(np.isfinite(a).all() for a in arrays)
             and np.isfinite(pixel_spacing) and pixel_spacing > 0
             and np.isfinite(slice_thickness) and slice_thickness > 0)
    if not valid:
        event("Input", "Blocked", "Image, masks, volume or configured geometry are invalid.")
        report["summary"] = "Input validation failed."
        return report
    event("Input", "Checked", "Array shapes and configured spacing checked; acquisition geometry is not independently verified.")
    inventory = execute(2)
    if inventory is None:
        return report
    available = set(inventory["evidence"]["valid_real_models"])
    if simulated_keys or not set(expected_models).issubset(available):
        event("Model gate", "Blocked", "The configured model panel includes simulated, missing or invalid outputs.")
        report["summary"] = "Real model outputs are required before automatic review can continue."
        report["next_action"] = "Resolve missing model weights or inference errors, then run review again. Manual contour review remains available."
        return report
    event("Model gate", "Checked", "All configured probability maps are real, finite and in range. The recorded fusion policy is fixed; adaptive expert ranking remains unavailable.")
    evidence = execute(1)
    ambiguity = execute(3)
    if evidence is None or ambiguity is None or evidence["status"] == "Blocked" or ambiguity["status"] == "Blocked":
        report["summary"] = "Insufficient contour evidence for automatic review."
        report["next_action"] = "Inspect or draw a contour manually; absence of a contour does not establish absence of a lesion."
        return report
    if np.any(original != active):
        verification = execute(5)
        if verification is None or verification["status"] != "Checks passed":
            report.update(status="Review required", summary="The current contour change needs clinician inspection.",
                          next_action="Inspect the edit and its geometric check results before making further changes.")
            return report
    # A small enclosed hole is a concrete trigger for the existing closing tool.
    _, hierarchy = cv2.findContours((active > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = int(np.sum(hierarchy[0, :, 3] >= 0)) if hierarchy is not None else 0
    proposal_ready = False
    if holes:
        event("Cleanup route", "Selected", f"Found {holes} enclosed contour hole(s).")
        proposal = execute(4)
        if proposal is None:
            return report
        if proposal["candidate"] is not None:
            verification = execute(5, active, proposal["candidate"])
            proposal_ready = verification is not None and verification["status"] == "Checks passed"
            if not proposal_ready:
                proposal["candidate"] = None
                event("Proposal", "Rejected", "Geometric verification did not pass; no change was applied.")
    else:
        event("S04", "Skipped", "No enclosed contour holes; no cleanup was requested.")
    measurements = execute(6)
    if measurements is None:
        # A failed workflow must not leave an applicable candidate behind.
        if 4 in report["skills"]:
            report["skills"][4]["candidate"] = None
        return report
    report.update(status="Proposal ready" if proposal_ready else "Awaiting clinician review",
                  summary="A checked cleanup proposal is ready for inspection." if proposal_ready else "Review checks completed; the contour is awaiting clinician review.",
                  next_action="Inspect the before/after images and explicitly apply or discard the proposal." if proposal_ready else "Review the contour and measurement evidence. No contour change was applied.")
    event("Clinician", "Pending", "Workflow completion does not approve a contour or establish diagnostic correctness.")
    return report


def serializable_report(report):
    return {**report, "skills": {str(k): {name: value for name, value in result.items() if name != "candidate"}
                               for k, result in report["skills"].items()}}
