"""Exercise the pinned trainer's entropy path on CPU without importing verl/Ray."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import textwrap
import unittest
from unittest.mock import Mock

from math_rft.entropy_control import adapt_source, entropy_requested, old_log_prob_metrics


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "vendor/verl/verl/trainer/ppo/ray_trainer.py"


def upstream_method(name):
    source = TRAINER.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    trainer = next(node for node in ast.parse(source).body
                   if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return textwrap.dedent("".join(lines[method.lineno - 1:method.end_lineno]))


class Tensor:
    """Minimal CPU value with only methods used by the adapted trainer sections."""
    def __init__(self, value):
        self.value = value

    def float(self):
        return self

    def detach(self):
        return self

    def item(self):
        return self.value


class Proto:
    def __init__(self, batch):
        self.batch = batch

    def to_tensordict(self):
        return dict(self.batch)

    @classmethod
    def from_tensordict(cls, batch):
        return cls(batch)


def actor_config(enabled, coefficient=0.0, sum_pi_squared=False):
    class Config(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)
    return Config(calculate_entropy=enabled, entropy_coeff=coefficient,
                  calculate_sum_pi_squared=sum_pi_squared,
                  loss_agg_mode="seq-mean-token-mean", loss_scale_factor=None)


class EntropyMetricTests(unittest.TestCase):
    def test_disabled_entropy_omits_metric_and_does_not_aggregate(self):
        log_prob = Proto({"old_log_probs": Tensor(-0.3)})
        aggregate = Mock(side_effect=AssertionError("No entropy reduction is expected"))
        metrics = old_log_prob_metrics(log_prob, Proto({}), actor_config(False), 0.8, aggregate)
        self.assertEqual(metrics, {"perf/mfu/actor_infer": 0.8})
        self.assertEqual(set(log_prob.batch), {"old_log_probs"})
        aggregate.assert_not_called()

    def test_enabled_entropy_preserves_real_metric_and_removes_auxiliary_tensor(self):
        entropy, mask = Tensor(2.5), object()
        log_prob = Proto({"old_log_probs": Tensor(-0.3), "entropys": entropy})
        aggregate = Mock(return_value=entropy)
        metrics = old_log_prob_metrics(log_prob, Proto({"response_mask": mask}), actor_config(True), 0.8, aggregate)
        self.assertEqual(metrics, {"actor/entropy": 2.5, "perf/mfu/actor_infer": 0.8})
        aggregate.assert_called_once_with(loss_mat=entropy, loss_mask=mask,
                                         loss_agg_mode="seq-mean-token-mean", loss_scale_factor=None)
        self.assertNotIn("entropys", log_prob.batch)

    def test_nonzero_regularization_still_requires_entropy(self):
        self.assertTrue(entropy_requested(actor_config(False, coefficient=0.01)))
        with self.assertRaises(KeyError):
            old_log_prob_metrics(Proto({}), Proto({}), actor_config(False, coefficient=0.01), 0.8, Mock())

    def test_unexpected_entropy_is_not_silently_discarded(self):
        with self.assertRaises(RuntimeError):
            old_log_prob_metrics(Proto({"entropys": Tensor(1)}), Proto({}), actor_config(False), 0.8, Mock())

    def test_untrusted_source_is_refused(self):
        for name in ("fit", "_compute_old_log_prob", "unknown_method"):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "Unaudited"):
                adapt_source(name, "def fit(self):\n    return None\n")


@unittest.skipUnless(TRAINER.is_file(), "Pinned verl checkout is needed to test its actual methods")
class PinnedTrainerEntropyTests(unittest.TestCase):
    def execute_old_log_prob(self, config, adapted=True):
        source = upstream_method("_compute_old_log_prob")
        if adapted:
            source = adapt_source("_compute_old_log_prob", source)
        log_probs, entropy, sum_pi_squared = Tensor(-0.7), Tensor(2.4), Tensor(0.2)
        routed_experts = object()
        metadata, outputs_read = {}, []

        def assign_non_tensor(data, **values):
            metadata.update(values)

        def worker(data):
            result = {"log_probs": log_probs, "routed_experts": routed_experts, "metrics": {"mfu": 0.9}}
            if metadata["calculate_entropy"]:
                result["entropy"] = entropy
            if metadata["calculate_sum_pi_squared"]:
                result["sum_pi_squared"] = sum_pi_squared
            return result

        def get_output(data, key):
            outputs_read.append(key)
            if key == "entropy":
                return data[key]  # Fail if an omitted entropy tensor is read.
            return data.get(key)

        def restore_padding(value, batch):
            self.assertIsNotNone(value)
            return value

        namespace = {
            "DataProto": Proto,
            "left_right_2_no_padding": lambda data: data,
            "no_padding_2_padding": restore_padding,
            "tu": SimpleNamespace(assign_non_tensor=assign_non_tensor, get=get_output,
                                  get_tensordict=lambda values: values),
            "_math_rft_entropy_requested": entropy_requested,
        }
        exec(compile(source, "<audited-old-log-prob-test>", "exec"), namespace)
        trainer = SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=config)),
                                  actor_rollout_wg=SimpleNamespace(compute_log_prob=worker))
        result, mfu = namespace["_compute_old_log_prob"](trainer, Proto({"input_ids": [1, 2]}))
        self.assertIs(result.batch["old_log_probs"], log_probs)
        self.assertIs(result.batch["routed_experts"], routed_experts)
        self.assertEqual(mfu, 0.9)
        if config.calculate_sum_pi_squared:
            self.assertIs(result.batch["sum_pi_squared"], sum_pi_squared)
        return result, metadata, outputs_read

    def test_disabled_entropy_worker_flag_is_false_and_log_probs_survive(self):
        for sum_pi_squared in (False, True):
            with self.subTest(sum_pi_squared=sum_pi_squared):
                result, metadata, reads = self.execute_old_log_prob(actor_config(False, sum_pi_squared=sum_pi_squared))
                self.assertFalse(metadata["calculate_entropy"])
                self.assertNotIn("entropy", reads)
                self.assertNotIn("entropys", result.batch)

    def test_enabled_path_matches_upstream_outputs(self):
        original, original_metadata, original_reads = self.execute_old_log_prob(actor_config(True), adapted=False)
        adapted, adapted_metadata, adapted_reads = self.execute_old_log_prob(actor_config(True))
        self.assertEqual(original_metadata, adapted_metadata)
        self.assertEqual(original_reads, adapted_reads)
        self.assertEqual(set(original.batch), set(adapted.batch))
        for key in ("old_log_probs", "entropys"):
            self.assertEqual(original.batch[key].value, adapted.batch[key].value)

    def test_nonzero_coefficient_keeps_worker_entropy_on(self):
        result, metadata, reads = self.execute_old_log_prob(actor_config(False, coefficient=0.01))
        self.assertTrue(metadata["calculate_entropy"])
        self.assertIn("entropys", result.batch)
        self.assertIn("entropy", reads)

    def test_fit_uses_optional_metrics_without_reading_omitted_entropy(self):
        adapted = adapt_source("fit", upstream_method("fit"))
        compile(adapted, "<audited-fit-test>", "exec")
        tree = ast.parse(adapted)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "_math_rft_old_log_prob_metrics"]
        self.assertEqual(len(calls), 1)
        self.assertNotIn('old_log_prob.batch["entropys"]', adapted)
        self.assertIn('batch = batch.union(old_log_prob)', adapted)
        statement = next(node for node in ast.walk(tree) if isinstance(node, ast.Expr)
                         and calls[0] in ast.walk(node))
        module = ast.fix_missing_locations(ast.Module(body=[statement], type_ignores=[]))
        metrics = {}
        aggregate = Mock(side_effect=AssertionError("Entropy is disabled"))
        namespace = {
            "metrics": metrics, "old_log_prob": Proto({"old_log_probs": Tensor(-0.7)}),
            "batch": Proto({}), "old_log_prob_mfu": 0.9, "agg_loss": aggregate,
            "self": SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=actor_config(False)))),
            "_math_rft_old_log_prob_metrics": old_log_prob_metrics,
        }
        exec(compile(module, "<audited-fit-metrics-test>", "exec"), namespace)
        self.assertEqual(metrics, {"perf/mfu/actor_infer": 0.9})
        aggregate.assert_not_called()

    def test_any_upstream_source_change_is_refused(self):
        for name in ("fit", "_compute_old_log_prob"):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "Unaudited"):
                adapt_source(name, upstream_method(name) + "\n# changed upstream\n")


if __name__ == "__main__":
    unittest.main()
