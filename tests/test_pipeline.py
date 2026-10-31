"""Check the experiment CLI plan without model, dataset or GPU dependencies."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
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

    def test_lowmem_preserves_full_matrix_and_scientific_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            default = self.plan(root, "--stage", "full")
            lowmem = self.plan(root, "--profile", "lowmem", "--stage", "full")
        self.assertEqual(len(default), len(lowmem))
        for original, adjusted in zip(default, lowmem, strict=True):
            without_profile = adjusted.copy()
            if original[0] in {"math_rft.evaluate", "math_rft.train"}:
                self.assertEqual(option(adjusted, "--profile"), "lowmem")
                index = without_profile.index("--profile")
                del without_profile[index:index + 2]
            self.assertEqual(original, without_profile)
        training = [command for command in lowmem if command[0] == "math_rft.train"]
        self.assertEqual(len(training), 24)
        self.assertEqual({option(command, "--seed") for command in training}, {"42", "43", "44"})
        self.assertEqual(len({option(command, "--model") for command in training}), 4)
        evaluations = [command for command in lowmem if command[0] == "math_rft.evaluate"]
        self.assertEqual(len(evaluations), 28)
        for command in evaluations:
            parsed = evaluate.parser().parse_args(command[1:])
            evaluate.resolve_execution_options(parsed)
            self.assertEqual((parsed.n, parsed.role, parsed.data.name), (256, "formal", "math500.jsonl"))
            self.assertEqual((parsed.max_tokens, parsed.max_prompt_tokens, parsed.max_model_len), (4096, 4096, 8192))
            self.assertEqual((parsed.batch_size, parsed.gpu_memory_utilization), (8, 0.3))
            self.assertEqual((parsed.max_num_seqs, parsed.max_num_batched_tokens), (16, 2048))
            self.assertTrue(parsed.enable_chunked_prefill)
            self.assertTrue(parsed.enforce_eager)

    def test_lowmem_keeps_pilot_sampling_and_does_not_shorten_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            commands = self.plan(Path(temporary), "--profile", "lowmem", "--models", "qwen3_1p7b",
                                 "--seeds", "42", "--set", "trainer.n_gpus_per_node=2")
        training = [command for command in commands if command[0] == "math_rft.train"]
        self.assertEqual(len(training), 2)
        self.assertEqual({option(command, "--method") for command in training}, {"rft", "grpo"})
        self.assertTrue(all(option(command, "--set") == "trainer.n_gpus_per_node=2" for command in training))
        evaluations = [command for command in commands if command[0] == "math_rft.evaluate"]
        self.assertEqual(len(evaluations), 3)
        self.assertTrue(all(option(command, "--n") == "256" for command in evaluations))
        self.assertTrue(all(option(command, "--data").endswith("dev.jsonl") for command in evaluations))
        self.assertTrue(all("--set" not in command for command in evaluations))

    def test_lowmem_still_rejects_insufficient_large_k_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.plan(Path(temporary), "--profile", "lowmem", "--n", "128")
            commands = self.plan(Path(temporary), "--profile", "lowmem", "--models", "qwen3_1p7b",
                                 "--seeds", "42", "--n", "512")
        evaluations = [command for command in commands if command[0] == "math_rft.evaluate"]
        self.assertTrue(all(option(command, "--n") == "512" for command in evaluations))

    def test_lowmem_output_default_is_separate_and_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(sys, "argv", ["pipeline", "--profile", "lowmem"]), \
                    patch("os.getcwd", return_value=str(root)), \
                    patch.object(pipeline, "invoke") as invoke, \
                    patch.object(pipeline, "frozen_revisions") as resolve, redirect_stdout(io.StringIO()):
                pipeline.main()
            resolve.assert_not_called()
            self.assertTrue(all(not call.args[1] for call in invoke.call_args_list))
            training = [call.args[0] for call in invoke.call_args_list if call.args[0][0] == "math_rft.train"]
            self.assertTrue(all(Path(option(command, "--output-dir")) == root / "runs_lowmem_pipeline"
                                for command in training))
            self.assertEqual(list(root.iterdir()), [])

    def test_completed_trial_from_another_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            for name in ("pilot.parquet", "dev.parquet", "dev.jsonl", "manifest.json"):
                (data / name).write_text("{}", encoding="utf-8")
            output = root / "runs"
            run = output / "qwen3_1p7b/rft/pilot/seed_42"
            run.mkdir(parents=True)
            (run / "training_complete.json").write_text(json.dumps({
                "model": "qwen3_1p7b", "method": "rft", "stage": "pilot", "seed": 42,
            }), encoding="utf-8")
            argv = ["pipeline", "--profile", "lowmem", "--models", "qwen3_1p7b", "--seeds", "42",
                    "--methods", "rft", "--execute", "--data-dir", str(data), "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), patch.object(pipeline, "invoke"), \
                    patch.object(pipeline, "frozen_revisions", return_value={"qwen3_1p7b": "a" * 40}), \
                    self.assertRaisesRegex(ValueError, "Completed trial profile mismatch"):
                pipeline.main()
            request = json.loads((run / "matrix_request.json").read_text())
            self.assertEqual(request["profile"], "lowmem")


class PipelineEnvironmentTests(unittest.TestCase):
    def test_lowmem_sampler_workaround_is_confined_to_child_environment(self):
        with patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "1"}), \
                patch.object(pipeline.subprocess, "run") as launch, redirect_stdout(io.StringIO()):
            pipeline.invoke(["math_rft.evaluate"], True, profile="lowmem")
            self.assertEqual(launch.call_args.kwargs["env"]["VLLM_USE_FLASHINFER_SAMPLER"], "0")
            self.assertEqual(os.environ["VLLM_USE_FLASHINFER_SAMPLER"], "1")

    def test_default_environment_and_lowmem_plan_launch_behavior_are_unchanged(self):
        with patch.object(pipeline.subprocess, "run") as launch, redirect_stdout(io.StringIO()):
            pipeline.invoke(["math_rft.evaluate"], True)
            self.assertNotIn("env", launch.call_args.kwargs)
            launch.reset_mock()
            pipeline.invoke(["math_rft.evaluate"], False, profile="lowmem")
            launch.assert_not_called()


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
