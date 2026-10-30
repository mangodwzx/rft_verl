"""Check the experiment CLI plan without model, dataset or GPU dependencies."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

from math_rft import evaluate, pipeline


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class PipelinePlanTests(unittest.TestCase):
    def plan(self, directory: Path, *extra: str) -> list[list[str]]:
        stream = io.StringIO()
        output = directory / "new_runs"
        data = directory / "missing_data"
        argv = ["pipeline", "--data-dir", str(data), "--output-dir", str(output), *extra]
        with patch.object(sys, "argv", argv), redirect_stdout(stream), \
                patch.object(pipeline.subprocess, "run") as launch, \
                patch.object(pipeline, "frozen_revisions") as resolve:
            pipeline.main()
            launch.assert_not_called()
            resolve.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(data.exists())
        return [shlex.split(line)[2:] for line in stream.getvalue().splitlines() if not line.startswith("PLAN ONLY")]

    def test_pilot_evaluates_all_three_curves_on_dev_at_256(self):
        with tempfile.TemporaryDirectory() as temporary:
            commands = self.plan(Path(temporary), "--models", "qwen3_1p7b", "--seeds", "42")
        evaluations = [command for command in commands if command[0] == "math_rft.evaluate"]
        self.assertEqual(len(evaluations), 3)
        self.assertEqual({option(command, "--method") for command in evaluations}, {"base", "rft", "grpo"})
        for command in evaluations:
            parsed = evaluate.parser().parse_args(command[1:])
            self.assertEqual(parsed.role, "development")
            self.assertEqual(parsed.n, 256)
            self.assertEqual(parsed.data.name, "dev.jsonl")
        training = [command for command in commands if command[0] == "math_rft.train"]
        self.assertEqual(len(training), 2)
        self.assertEqual({option(command, "--method") for command in training}, {"rft", "grpo"})

    def test_full_defaults_to_four_models_and_three_training_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            commands = self.plan(Path(temporary), "--stage", "full")
        training = [command for command in commands if command[0] == "math_rft.train"]
        self.assertEqual(len(training), 24)
        self.assertEqual({option(command, "--seed") for command in training}, {"42", "43", "44"})
        self.assertEqual(len({option(command, "--model") for command in training}), 4)
        evaluations = [command for command in commands if command[0] == "math_rft.evaluate"]
        self.assertEqual(len(evaluations), 28)
        for command in evaluations:
            parsed = evaluate.parser().parse_args(command[1:])
            self.assertEqual((parsed.role, parsed.n, parsed.data.name), ("formal", 256, "math500.jsonl"))
        self.assertEqual(sum(command[0] == "math_rft.plot" for command in commands), 4)

    def test_analysis_and_plot_paths_match_evaluation_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commands = self.plan(root, "--models", "qwen3_1p7b", "--seeds", "42", "43")
        evaluations = [command for command in commands if command[0] == "math_rft.evaluate"]
        baseline = next(command for command in evaluations if option(command, "--method") == "base")
        analysis = next(command for command in commands if command[0] == "math_rft.analyze")
        plot = next(command for command in commands if command[0] == "math_rft.plot")
        self.assertEqual(option(plot, "--baseline"), option(baseline, "--output") + ".summary.json")
        self.assertEqual(Path(option(plot, "--comparison")), Path(option(analysis, "--output-dir")) / "comparison.json")
        self.assertEqual(Path(option(plot, "--output-dir")), Path(option(analysis, "--output-dir")) / "figures")
        self.assertEqual(analysis[analysis.index("--left") + 1:analysis.index("--right")],
                         [option(command, "--output") for command in evaluations if option(command, "--method") == "rft"])
        self.assertEqual(analysis[analysis.index("--right") + 1:analysis.index("--output-dir")],
                         [option(command, "--output") for command in evaluations if option(command, "--method") == "grpo"])

    def test_optional_single_method_does_not_request_nonexistent_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            commands = self.plan(Path(temporary), "--models", "qwen3_1p7b", "--seeds", "42", "--methods", "rft")
        self.assertFalse(any(command[0] in {"math_rft.analyze", "math_rft.plot"} for command in commands))

    def test_larger_sample_budget_is_forwarded_consistently(self):
        with tempfile.TemporaryDirectory() as temporary:
            commands = self.plan(Path(temporary), "--models", "qwen3_1p7b", "--seeds", "42", "--n", "1024")
        evaluations = [command for command in commands if command[0] == "math_rft.evaluate"]
        self.assertTrue(all(option(command, "--n") == "1024" for command in evaluations))

    def test_small_budget_and_removed_methods_are_rejected(self):
        for arguments in (("--n", "128"), ("--methods", "grpo_fixed"), ("--seeds", "42", "42")):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as temporary:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    self.plan(Path(temporary), *arguments)
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(list(Path(temporary).iterdir()), [])


class EvaluationProtocolTests(unittest.TestCase):
    def rows(self, count: int) -> list[dict]:
        return [{"id": str(index), "problem": "1 + 1", "prompt": "Solve 1 + 1", "answer": "2",
                 "subject": "algebra", "level": 4, "split": "math500", "data_source": "math500"}
                for index in range(count)]

    def test_formal_requires_full_math500_and_support_for_k256(self):
        for count, n, valid in ((500, 256, True), (499, 256, False), (500, 128, False)):
            with self.subTest(count=count, n=n), patch.object(evaluate, "read_jsonl", return_value=iter(self.rows(count))):
                if valid:
                    self.assertEqual(len(evaluate.load_data(Path("unused"), "formal", n)), 500)
                else:
                    with self.assertRaises(ValueError):
                        evaluate.load_data(Path("unused"), "formal", n)

    def test_development_refuses_formal_questions(self):
        with patch.object(evaluate, "read_jsonl", return_value=iter(self.rows(1))):
            with self.assertRaises(ValueError):
                evaluate.load_data(Path("unused"), "development", 256)


if __name__ == "__main__":
    unittest.main()
