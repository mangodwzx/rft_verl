"""Offline diagnostics must preserve source files and expose sampling failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from math_rft.diagnose import diagnose, evaluation_metrics, rollout_metrics


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def evaluation(path: Path) -> None:
    rows = [{"id": q, "sample_index": i, "score": score, "text": text,
             "response_token_ids": tokens, "grader_status": "correct" if score else "incorrect",
             "finish_reason": "stop"}
            for q, i, score, text, tokens in [
                ("q1", 0, 1, "same", [1, 2]), ("q1", 1, 0, "same", [1, 2]),
                ("q2", 0, 0, "a", [3]), ("q2", 1, 0, "b", [4])]]
    write_jsonl(path, rows)
    path.with_name(path.name + ".meta.json").write_text(json.dumps({"config": {
        "question_ids": ["q1", "q2"], "sampling": {"n": 2}, "execution_profile": "synthetic"}}))


class DiagnosticsTests(unittest.TestCase):
    def test_lengths_use_characters_and_empty_success_group_is_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "20.jsonl"
            write_jsonl(path, [{"input": "q", "output": "中文", "score": 0},
                               {"input": "q", "output": "abc", "score": 0}])
            result = rollout_metrics(path, "rft", "train")
            self.assertEqual(result["mean_response_chars"], 2.5)
            self.assertIsNone(result["correct_mean_response_chars"])
            self.assertEqual(result["observed_question_coverage"], 0)
            self.assertEqual(result["samples_per_question_min"], 2)

    def test_evaluation_recomputes_coverage_and_uniqueness(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dev.jsonl"
            evaluation(path)
            result = evaluation_metrics(path, "base")
            self.assertEqual(result["covered_questions"], 1)
            self.assertEqual(result["pass_at_k"], {"1": 0.25, "2": 0.5})
            self.assertEqual(result["mean_unique_texts_per_question"], 1.5)
            self.assertEqual(result["mean_most_frequent_text_fraction"], 0.75)
            self.assertEqual(result["mean_response_tokens"], 1.5)

    def test_incomplete_or_duplicate_samples_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dev.jsonl"
            evaluation(path)
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                evaluation_metrics(path, "rft")
            path.write_text("\n".join(lines + [lines[0]]) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                evaluation_metrics(path, "rft")

    def test_run_preserves_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs/model/rft/pilot/seed_42"
            rollout = run / "rollouts/1.jsonl"
            write_jsonl(rollout, [{"input": "q1", "output": "answer", "score": 1}])
            ledger = root / "eval/model/rft/pilot/seed_42/dev.jsonl"
            evaluation(ledger)
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            args = argparse.Namespace(run_root=root / "runs", eval_root=root / "eval", baseline=None,
                                      model="model", stage="pilot", seed=42, methods=["rft"],
                                      tensorboard_root=None, output_dir=root / "report", no_plots=True)
            result = diagnose(args)
            self.assertEqual(result["training"][0]["samples"], 1)
            self.assertTrue((root / "report/evaluation_metrics.csv").is_file())
            for path, original in before.items():
                self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                diagnose(args)


if __name__ == "__main__":
    unittest.main()
