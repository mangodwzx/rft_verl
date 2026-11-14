"""CPU checks for the high-k estimator, trend summaries and paired analysis."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest

from math_rft.analyze import assert_comparable, compare, load_run
from math_rft.evaluate import artifact_paths, canonical_hash, sample_seed
from math_rft.metrics import describe_gap_curve, hierarchical_delta_ci, pass_at_k
from math_rft.plot import comparison_summaries


class MetricTests(unittest.TestCase):
    def test_estimator_matches_exact_combinations(self):
        for n in (8, 256, 1024):
            for c in (0, 1, n // 4, n):
                for k in (1, min(128, n), min(256, n)):
                    expected = 1 - math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 1
                    self.assertAlmostEqual(pass_at_k(n, c, k), expected, places=13)

    def test_invalid_sample_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            pass_at_k(128, 10, 256)

    def test_curve_is_descriptive_and_signed_gap_is_distinct(self):
        trend = describe_gap_curve([1, 64, 128, 256], [0.01, 0.08, 0.04, 0.02])
        self.assertEqual(trend["signed_delta"]["shape_on_reported_grid"], "rise_then_fall")
        self.assertEqual(trend["signed_delta"]["peak_k"], 64)
        self.assertAlmostEqual(trend["signed_delta"]["drop_from_peak_to_last"], 0.06)
        negative = describe_gap_curve([1, 128, 256], [-0.01, -0.03, -0.04])
        self.assertEqual(negative["signed_delta"]["shape_on_reported_grid"], "nonincreasing")
        self.assertEqual(negative["absolute_gap"]["shape_on_reported_grid"], "nondecreasing")
        self.assertEqual(describe_gap_curve([1, 2], [0.0, 0.0])["signed_delta"]["shape_on_reported_grid"], "flat")
        with self.assertRaises(ValueError):
            describe_gap_curve([2, 1], [0.0, 0.0])

    def test_common_question_resampling_preserves_seed_pairing(self):
        # Opposite question effects cancel exactly across the two paired seeds.
        result = hierarchical_delta_ci([[0.1, -0.1], [-0.1, 0.1]], repeats=100, seed=3)
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["seed_means"], [0, 0])
        self.assertEqual(result["bootstrap_unit"], "paired_training_seed_and_common_question")


def ledger(directory: Path, method: str, training_seed: int | None, correct: tuple[int, int],
           training_objective: dict | None = None) -> Path:
    n, evaluation_seed = 256, 19
    path = directory / f"{method}_{training_seed}.jsonl"
    rows = [{"id": f"q{index}", "subject": "algebra", "level": level} for index, level in enumerate((2, 5))]
    config = {"dataset_sha256": canonical_hash(rows), "question_ids": [row["id"] for row in rows],
              "sampling": {"n": n, "seed": evaluation_seed}, "grader": "synthetic", "role": "development",
              "versions": {}, "base_model": "synthetic/base", "method": method, "training_seed": training_seed,
              "engine": {"dtype": "bfloat16", "max_model_len": 64, "max_prompt_tokens": 16, "generation_config": {}}}
    if training_objective is not None:
        config["training_objective"] = training_objective
    paths = artifact_paths(path)
    paths["meta"].write_text(json.dumps({"config": config, "manifest_hash": canonical_hash(config)}), encoding="utf-8")
    paths["prompts"].write_text("".join(json.dumps({**row, "prompt_token_ids": [1]}) + "\n" for row in rows), encoding="utf-8")
    with path.open("w", encoding="utf-8") as stream:
        for row, count in zip(rows, correct, strict=True):
            for index in range(n):
                stream.write(json.dumps({"id": row["id"], "sample_index": index,
                                         "seed": sample_seed(evaluation_seed, row["id"], index),
                                         "score": int(index < count), "response_token_ids": [2],
                                         "finish_reason": "stop", "grader_status": "ok"}) + "\n")
    return path


class AnalysisIntegrationTests(unittest.TestCase):
    def test_training_objective_mismatch_rejected_but_base_comparison_allowed(self):
        fixed = {"loss_normalization": "fixed", "loss_scale_factor": 4096}
        for other in (None, {"loss_normalization": "per_response", "loss_scale_factor": None},
                      {"loss_normalization": "fixed", "loss_scale_factor": 2048}):
            with self.subTest(other=other), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                left = load_run(ledger(root, "rft", 42, (1, 1), fixed))
                right = load_run(ledger(root, "grpo", 42, (1, 1), other))
                with self.assertRaisesRegex(ValueError, "training_objective"):
                    assert_comparable(left, right)
                base = load_run(ledger(root, "base", None, (1, 1)))
                assert_comparable(left, base)
                assert_comparable(base, left)

    def test_objective_is_preserved_in_report_and_consistent_across_seeds(self):
        fixed = {"loss_normalization": "fixed", "loss_scale_factor": 4096}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = [ledger(root, "rft", 42, (1, 1), fixed)]
            right = [ledger(root, "grpo", 42, (1, 1), fixed)]
            args = argparse.Namespace(left=left, right=right, output_dir=root / "analysis", bootstrap=100, seed=7)
            compare(args)
            report = json.loads((args.output_dir / "comparison.json").read_text())
            self.assertEqual(report["training_objectives"], {"left": fixed, "right": fixed})
            left.append(ledger(root, "rft", 43, (1, 1)))
            right.append(ledger(root, "grpo", 43, (1, 1)))
            args.output_dir = root / "mixed"
            with self.assertRaisesRegex(ValueError, "Cross-seed.*training_objective"):
                compare(args)
            self.assertFalse(args.output_dir.exists())

    def test_two_seed_complete_ledgers_produce_trends_and_mean_curves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = [ledger(root, "rft", 42, (10, 1)), ledger(root, "rft", 43, (8, 2))]
            right = [ledger(root, "grpo", 42, (5, 2)), ledger(root, "grpo", 43, (4, 3))]
            output = root / "analysis"
            compare(argparse.Namespace(left=left, right=right, output_dir=output, bootstrap=100, seed=7))
            comparison = json.loads((output / "comparison.json").read_text())
            self.assertEqual(comparison["paired_training_seeds"], [42, 43])
            summary = comparison["trends"]
            self.assertIn("level_4_5", summary["aggregate"])
            # At n=k all four runs covered both questions: delta256 is zero.
            delta128 = comparison["aggregate"]["all"]["128"]["mean"]
            change = summary["fixed_k_changes"]["all"]["256_minus_128"]
            self.assertAlmostEqual(change["mean"], -delta128)
            low, high = comparison["aggregate"]["all"]["128"]["ci95"]
            self.assertAlmostEqual(change["ci95"][0], -high)
            self.assertAlmostEqual(change["ci95"][1], -low)
            summaries, labels = comparison_summaries(comparison)
            self.assertEqual(len(labels), 2)
            self.assertEqual(summaries[0]["ks"][-1], 256)
            self.assertEqual(summaries[0]["groups"]["all"]["pass_at_k"]["256"], 1.0)
            self.assertTrue((output / "trend_summary.csv").is_file())
            self.assertTrue((output / "high_k_changes.csv").is_file())
            with self.assertRaises(ValueError):
                comparison_summaries(comparison, {"n": 128, "questions": []})

    def test_incomplete_ledger_is_not_silently_analyzed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = ledger(Path(temporary), "rft", 42, (1, 1))
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "Incomplete evaluation"):
                load_run(path)


if __name__ == "__main__":
    unittest.main()
