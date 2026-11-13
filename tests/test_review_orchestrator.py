import json
import unittest
from unittest.mock import patch

import numpy as np

from src.services.review_orchestrator import run_review, serializable_report
from src.services.review_skills import run_review_skill


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((32, 32), np.float32)
        self.image[8:24, 8:24] = .6
        self.mask = np.zeros((32, 32), np.uint8)
        self.mask[10:22, 10:22] = 1

    def run_case(self, mask=None, simulated=(), active=None, probabilities=None):
        mask = self.mask if mask is None else mask
        return run_review(self.image, mask, mask if active is None else active,
                          {"m": mask.astype(float)} if probabilities is None else probabilities,
                          simulated, .5, np.stack([mask] * 8), 5., ["m"])

    def test_normal_route_skips_cleanup_and_does_not_approve(self):
        result = self.run_case()
        self.assertEqual(result["status"], "Awaiting clinician review")
        self.assertNotIn(4, result["skills"])
        self.assertIn(6, result["skills"])
        self.assertTrue(result["clinician_approval_required"])

    def test_simulated_panel_stops_at_inventory(self):
        result = self.run_case(simulated=["m"])
        self.assertEqual(result["status"], "Blocked")
        self.assertEqual(list(result["skills"]), [2])

    def test_invalid_probability_map_stops(self):
        result = self.run_case(probabilities={"m": np.full((32, 32), np.nan)})
        self.assertEqual(result["status"], "Blocked")
        self.assertNotIn(1, result["skills"])

    def test_empty_contour_escalates_without_refinement(self):
        result = self.run_case(mask=np.zeros_like(self.mask))
        self.assertEqual(result["status"], "Blocked")
        self.assertNotIn(4, result["skills"])

    def test_hole_triggers_checked_proposal_without_mutation(self):
        mask = self.mask.copy()
        mask[15, 15] = 0
        before = mask.copy()
        result = self.run_case(mask=mask)
        self.assertEqual(result["status"], "Proposal ready")
        self.assertEqual(result["skills"][5]["status"], "Checks passed")
        self.assertEqual(result["skills"][4]["candidate"][15, 15], 1)
        np.testing.assert_array_equal(mask, before)
        self.assertNotIn("candidate", serializable_report(result)["skills"]["4"])
        json.dumps(serializable_report(result), allow_nan=False)

    def test_large_existing_edit_stops_before_cleanup(self):
        edited = np.ones_like(self.mask)
        result = self.run_case(active=edited)
        self.assertIn(result["status"], ["Blocked", "Review required"])
        self.assertNotIn(4, result["skills"])

    def test_tool_failure_stops_and_preserves_input(self):
        with patch("src.services.review_orchestrator.run_review_skill", side_effect=RuntimeError("tool failed")):
            result = self.run_case()
        self.assertEqual(result["status"], "Blocked")
        self.assertTrue(any(s["status"] == "Failed" for s in result["steps"]))

    def test_rejected_proposal_is_not_available_to_apply(self):
        mask = self.mask.copy()
        mask[15, 15] = 0

        def reject_verification(skill_id, *args):
            result = run_review_skill(skill_id, *args)
            if skill_id == 5:
                result["status"] = "Review required"
            return result

        with patch("src.services.review_orchestrator.run_review_skill", side_effect=reject_verification):
            result = self.run_case(mask=mask)
        self.assertIsNone(result["skills"][4]["candidate"])
        self.assertNotEqual(result["status"], "Proposal ready")

    def test_missing_model_panel_stops(self):
        result = self.run_case(probabilities={})
        self.assertEqual(result["status"], "Blocked")
        self.assertNotIn(1, result["skills"])

    def test_invalid_input_stops_before_tools(self):
        self.image[0, 0] = np.nan
        result = self.run_case()
        self.assertEqual(result["status"], "Blocked")
        self.assertEqual(result["skills"], {})


if __name__ == "__main__":
    unittest.main()
