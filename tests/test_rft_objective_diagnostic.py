"""Independent invariants for exact online-RFT length-bias diagnostics."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_rft_objective.py"
SPEC = importlib.util.spec_from_file_location("rft_objective_diagnostic", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class ObjectiveDiagnosticTests(unittest.TestCase):
    def test_all_success_legacy_drift_matches_analytic_identity(self):
        for probability in (0.1, 0.5, 0.9):
            for group_size in (2, 8, 32):
                value = diagnostic.expected_gradient(probability=probability, group_size=group_size)
                self.assertAlmostEqual(value, probability * (1 - probability) * (1 / 8 - 1 / 2), places=13)
                self.assertLess(value, 0)

    def test_fixed_sum_has_no_expected_all_success_drift(self):
        for probability in (0.1, 0.37, 0.9):
            for lengths in ((2, 8), (400, 8), (2, 4000)):
                for group_size in (2, 8, 32):
                    for scale in (1, 128, 4096):
                        value = diagnostic.expected_gradient(probability, lengths, (1, 1), group_size,
                                                             scale, "rft_fixed")
                        self.assertAlmostEqual(value, 0, places=13)

    def test_fixed_rft_prefers_success_even_when_success_is_long(self):
        # This is still positive-only RFT: failed responses get weight zero.
        p, m, scale = 0.37, 8, 4096
        for lengths in ((2, 8), (400, 8)):
            short = diagnostic.expected_gradient(p, lengths, (1, 0), m, scale, "rft_fixed")
            long = diagnostic.expected_gradient(p, lengths, (0, 1), m, scale, "rft_fixed")
            self.assertAlmostEqual(short, -(1 - (1 - p) ** m) * (1 - p) / scale, places=13)
            self.assertAlmostEqual(long, (1 - p ** m) * p / scale, places=13)
            self.assertLess(short, 0)
            self.assertGreater(long, 0)

    def test_finite_batch_token_mean_does_not_remove_drift(self):
        for group_size in (2, 8, 32):
            self.assertLess(diagnostic.expected_gradient(group_size=group_size,
                                                        objective="rft_batch_token_mean"), 0)

    def test_equal_lengths_remove_spurious_all_success_drift(self):
        for objective in diagnostic.OBJECTIVES:
            self.assertAlmostEqual(diagnostic.expected_gradient(lengths=(8, 8), objective=objective), 0, places=13)

    def test_grpo_and_all_failure_invariants(self):
        for objective in diagnostic.OBJECTIVES:
            self.assertEqual(diagnostic.expected_gradient(rewards=(0, 0), objective=objective), 0)
        for objective in ("grpo_per_response", "grpo_fixed"):
            self.assertEqual(diagnostic.expected_gradient(rewards=(1, 1), objective=objective), 0)
            self.assertLess(diagnostic.expected_gradient(rewards=(1, 0), objective=objective), 0)
            self.assertGreater(diagnostic.expected_gradient(rewards=(0, 1), objective=objective), 0)

    def test_fixed_scale_changes_magnitude_without_changing_direction(self):
        unscaled = diagnostic.expected_gradient(rewards=(0, 1), fixed_scale=1, objective="rft_fixed")
        scaled = diagnostic.expected_gradient(rewards=(0, 1), fixed_scale=4096, objective="rft_fixed")
        self.assertAlmostEqual(unscaled, scaled * 4096, places=13)


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in ("torch", "verl")),
                     "Actual CPU autograd verification needs the pinned training environment")
class RegisteredObjectiveDiagnosticTests(unittest.TestCase):
    def test_actual_registered_losses_match_enumerated_gradients(self):
        result = diagnostic.verify_registered_losses()
        self.assertTrue(result["all_batches_match"])
        self.assertLess(result["maximum_batch_gradient_error"], 1e-7)
        all_correct = result["expected_loss_gradients"]["both_correct"]
        self.assertAlmostEqual(all_correct["rft_fixed"], 0, places=13)
        # Upstream per-response reduction adds 1e-8 to each token count.
        self.assertAlmostEqual(all_correct["rft_per_response"], -0.09375, places=8)


if __name__ == "__main__":
    unittest.main()
