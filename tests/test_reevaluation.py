"""Check independent re-evaluation planning and protection of completed trials."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from math_rft import reevaluate
from math_rft.io_utils import sha256_file


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class FakeChild:
    def __init__(self, pid, args, events, polls=(0,)):
        self.pid, self.args, self.events = pid, args, events
        self.polls = iter(polls)
        self.returncode = None

    def poll(self):
        self.events.append(("poll", self.pid))
        result = next(self.polls, self.returncode)
        if isinstance(result, BaseException):
            raise result
        self.returncode = result
        return result

    def wait(self, timeout=None):
        self.events.append(("wait", self.pid, timeout))
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class ReevaluationTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path]:
        source, data, output = root / "training", root / "data", root / "reevaluated"
        data.mkdir()
        for filename in ("pilot.parquet", "dev.jsonl", "math500.jsonl"):
            (data / filename).write_text("{}\n", encoding="utf-8")
        for method in ("rft", "grpo"):
            run = source / "qwen3_1p7b" / method / "pilot/seed_42"
            export = run / "checkpoints/global_step_96/actor/huggingface"
            export.mkdir(parents=True)
            (export / "config.json").write_text("{}", encoding="utf-8")
            (export / "model.safetensors").write_bytes(b"fake weights")
            (run / "resolved_config.yaml").write_text("trainer: {}\n", encoding="utf-8")
            identity = {"model": "qwen3_1p7b", "method": method, "stage": "pilot", "seed": 42,
                        "profile": "lowmem", "source_sha256": {"old_source.py": "original-hash"},
                        "resolved_config_sha256": sha256_file(run / "resolved_config.yaml")}
            (run / "run_manifest.json").write_text(json.dumps({
                **identity, "expected_steps": 96,
                "model_provenance": {"repo_id": "Qwen/Qwen3-1.7B-Base", "revision": "a" * 40},
                "data_sha256": {str(data / "pilot.parquet"): sha256_file(data / "pilot.parquet")},
            }), encoding="utf-8")
            (run / "training_complete.json").write_text(json.dumps({
                **identity, "global_step": 96, "final_hf_model": str(export),
            }), encoding="utf-8")
        return source, data, output

    def arguments(self, source: Path, data: Path, output: Path, *extra: str) -> list[str]:
        return ["--run-root", str(source), "--data-dir", str(data), "--output-dir", str(output), *extra]

    def test_plan_resolves_original_final_checkpoints_without_writes_or_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data, output = self.fixture(root)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            stream = io.StringIO()
            with patch.object(reevaluate.subprocess, "run") as launch, redirect_stdout(stream):
                reevaluate.main(self.arguments(source, data, output))
            launch.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})
            commands = [shlex.split(line)[2:] for line in stream.getvalue().splitlines()[1:]]
            self.assertEqual([c[0] for c in commands],
                             ["math_rft.evaluate", "math_rft.evaluate", "math_rft.analyze", "math_rft.plot"])
            for method, command in zip(("rft", "grpo"), commands[:2], strict=True):
                self.assertEqual(option(command, "--profile"), "throughput")
                self.assertEqual(option(command, "--model"),
                                 str(source / "qwen3_1p7b" / method / "pilot/seed_42/checkpoints/global_step_96/actor/huggingface"))
                self.assertEqual(option(command, "--n"), "256")
                self.assertEqual(option(command, "--seed"), "20261025")
                self.assertEqual(option(command, "--training-seed"), "42")
                self.assertNotIn("--training-loss-normalization", command)
                self.assertNotIn("--training-loss-scale-factor", command)
                self.assertEqual(option(command, "--base-model"), "Qwen/Qwen3-1.7B-Base")
                self.assertEqual(option(command, "--role"), "development")
                self.assertEqual(option(command, "--output"),
                                 str(output / "qwen3_1p7b" / method / "pilot/seed_42/dev.jsonl"))
            comparison = output / "comparisons/pilot/qwen3_1p7b/seed_42/rft_vs_grpo"
            self.assertEqual(option(commands[2], "--output-dir"), str(comparison))
            self.assertEqual(option(commands[3], "--comparison"), str(comparison / "comparison.json"))

    def write_objective(self, source: Path, method: str, normalization, factor):
        run = source / "qwen3_1p7b" / method / "pilot/seed_42"
        for filename in ("run_manifest.json", "training_complete.json"):
            path = run / filename
            payload = json.loads(path.read_text())
            payload.update(loss_normalization=normalization, loss_scale_factor=factor)
            path.write_text(json.dumps(payload))

    def test_recorded_training_objective_is_forwarded_to_all_shards(self):
        for normalization, factor in (("fixed", 4096), ("per_response", None)):
            with self.subTest(normalization=normalization), tempfile.TemporaryDirectory() as temporary:
                source, data, output = self.fixture(Path(temporary))
                for method in ("rft", "grpo"):
                    self.write_objective(source, method, normalization, factor)
                commands = reevaluate.plan(reevaluate.parser().parse_args(
                    self.arguments(source, data, output, "--gpus", "2", "6")))
                evaluations = [command for command in commands if command[0] == "math_rft.evaluate"]
                self.assertEqual(len(evaluations), 4)
                for command in evaluations:
                    self.assertEqual(option(command, "--training-loss-normalization"), normalization)
                    if factor is None:
                        self.assertNotIn("--training-loss-scale-factor", command)
                    else:
                        self.assertEqual(option(command, "--training-loss-scale-factor"), str(factor))
                self.assertFalse(output.exists())

    def test_include_base_shards_pinned_model_on_same_alfworld_test_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            (data / "test.jsonl").write_text("{}\n", encoding="utf-8")
            for method in ("rft", "grpo"):
                run = source / "qwen3_1p7b" / method / "pilot/seed_42"
                for filename in ("run_manifest.json", "training_complete.json"):
                    path = run / filename
                    payload = json.loads(path.read_text())
                    payload.update(benchmark="alfworld", alfworld={"max_steps": 37, "max_action_tokens": 91},
                                   loss_normalization="fixed", loss_scale_factor=4096)
                    path.write_text(json.dumps(payload))
            args = self.arguments(source, data, output, "--split", "test", "--gpus", "0", "6",
                                  "--profile", "lowmem", "--batch-size", "16", "--env-workers", "8")
            commands = reevaluate.plan(reevaluate.parser().parse_args([*args, "--include-base"]))
            evaluations = [c for c in commands if c[0] == "math_rft.evaluate"]
            base_shards = []
            expected = {"--benchmark": "alfworld", "--role": "formal", "--data": str(data / "test.jsonl"),
                        "--max-steps": "37", "--max-action-tokens": "91", "--profile": "lowmem",
                        "--batch-size": "16", "--env-workers": "8", "--n": "256", "--seed": "20261025",
                        "--num-shards": "2"}
            self.assertCountEqual([(option(c, "--method"), option(c, "--shard-index")) for c in evaluations],
                                  [(method, index) for method in ("base", "rft", "grpo") for index in ("0", "1")])
            for command in evaluations:
                self.assertEqual({name: option(command, name) for name in expected}, expected)
                if option(command, "--method") == "base":
                    self.assertEqual(option(command, "--model"), "Qwen/Qwen3-1.7B-Base")
                    self.assertEqual(option(command, "--revision"), "a" * 40)
                    for name in ("--training-seed", "--training-loss-normalization", "--training-loss-scale-factor"):
                        self.assertNotIn(name, command)
                    base_shards.append(option(command, "--output"))
            merged = output / "baselines/pilot/qwen3_1p7b/test.jsonl"
            self.assertIn(["math_rft.merge_evaluations", "--inputs", *base_shards, "--data", str(data / "test.jsonl"),
                           "--output", str(merged)], commands)

    def test_mixed_or_unknown_training_objectives_fail_before_any_job(self):
        for other in (None, ("per_response", None), ("fixed", 2048)):
            with self.subTest(other=other), tempfile.TemporaryDirectory() as temporary:
                source, data, output = self.fixture(Path(temporary))
                self.write_objective(source, "rft", "fixed", 4096)
                if other is not None:
                    self.write_objective(source, "grpo", *other)
                with patch.object(reevaluate.subprocess, "run") as launch, \
                        self.assertRaisesRegex(ValueError, "training objective"):
                    reevaluate.main(self.arguments(source, data, output, "--execute"))
                launch.assert_not_called()
                self.assertFalse(output.exists())

    def test_incomplete_mismatched_or_invalid_objective_markers_are_rejected(self):
        for change in ("missing", "mismatch", "zero", "boolean", "per_response_scale", "null"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                source, data, output = self.fixture(Path(temporary))
                self.write_objective(source, "rft", "fixed", 4096)
                run = source / "qwen3_1p7b/rft/pilot/seed_42"
                if change in ("missing", "mismatch"):
                    path = run / "training_complete.json"
                    payload = json.loads(path.read_text())
                    if change == "missing":
                        payload.pop("loss_normalization")
                    else:
                        payload["loss_scale_factor"] = 2048
                    path.write_text(json.dumps(payload))
                else:
                    normalization, factor = {"zero": ("fixed", 0), "boolean": ("fixed", True),
                                             "per_response_scale": ("per_response", 4096),
                                             "null": (None, None)}[change]
                    self.write_objective(source, "rft", normalization, factor)
                with self.assertRaises(ValueError):
                    reevaluate.plan(reevaluate.parser().parse_args(self.arguments(source, data, output)))

    def test_execute_uses_ordered_checked_subprocesses_and_shared_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            with patch.object(reevaluate.subprocess, "run") as launch, redirect_stdout(io.StringIO()):
                reevaluate.main(self.arguments(
                    source, data, output, "--execute", "--batch-size", "128", "--max-num-seqs", "32",
                    "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.5", "--enforce-eager",
                ))
            self.assertEqual(launch.call_count, 4)
            for call in launch.call_args_list:
                self.assertEqual(call.kwargs, {"check": True})
            for call in launch.call_args_list[:2]:
                command = call.args[0]
                self.assertEqual(option(command, "--batch-size"), "128")
                self.assertEqual(option(command, "--max-num-seqs"), "32")
                self.assertEqual(option(command, "--max-num-batched-tokens"), "4096")
                self.assertEqual(option(command, "--gpu-memory-utilization"), "0.5")
                self.assertIn("--enforce-eager", command)

    def test_missing_second_checkpoint_fails_before_first_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            (source / "qwen3_1p7b/grpo/pilot/seed_42/training_complete.json").unlink()
            with patch.object(reevaluate.subprocess, "run") as launch, self.assertRaises(FileNotFoundError):
                reevaluate.main(self.arguments(source, data, output, "--execute"))
            launch.assert_not_called()
            self.assertFalse(output.exists())

    def test_different_base_revisions_or_training_data_fail_before_any_job(self):
        for change, expected in (("revision", "different frozen base model revisions"),
                                 ("data", "different original training dataset hashes")):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                source, data, output = self.fixture(Path(temporary))
                manifest = source / "qwen3_1p7b/grpo/pilot/seed_42/run_manifest.json"
                prior = json.loads(manifest.read_text())
                if change == "revision":
                    prior["model_provenance"]["revision"] = "b" * 40
                else:
                    different_data = data / "different_pilot.parquet"
                    different_data.write_text("different training examples")
                    prior["data_sha256"] = {str(different_data): sha256_file(different_data)}
                manifest.write_text(json.dumps(prior))
                with patch.object(reevaluate.subprocess, "run") as launch, self.assertRaisesRegex(ValueError, expected):
                    reevaluate.main(self.arguments(source, data, output, "--execute"))
                launch.assert_not_called()
                self.assertFalse(output.exists())

    def test_missing_or_mutable_base_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            manifest = source / "qwen3_1p7b/rft/pilot/seed_42/run_manifest.json"
            prior = json.loads(manifest.read_text())
            for revision in (None, "main", "g" * 40, "a" * 39):
                with self.subTest(revision=revision):
                    if revision is None:
                        prior["model_provenance"].pop("revision", None)
                    else:
                        prior["model_provenance"]["revision"] = revision
                    manifest.write_text(json.dumps(prior))
                    with self.assertRaisesRegex(ValueError, "immutable 40-character"):
                        reevaluate.plan(reevaluate.parser().parse_args(self.arguments(source, data, output)))

    def test_failed_evaluation_does_not_launch_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            with patch.object(reevaluate.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "evaluate")) as launch, \
                    redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                reevaluate.main(self.arguments(source, data, output, "--execute"))
            self.assertEqual(launch.call_count, 1)

    def test_single_method_needs_only_its_checkpoint_and_has_no_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            (source / "qwen3_1p7b/rft/pilot/seed_42/training_complete.json").unlink()
            args = reevaluate.parser().parse_args(self.arguments(
                source, data, output, "--methods", "grpo", "--split", "math500", "--no-enforce-eager",
            ))
            commands = reevaluate.plan(args)
            self.assertEqual(len(commands), 1)
            self.assertEqual(option(commands[0], "--role"), "formal")
            self.assertEqual(option(commands[0], "--data"), str(data / "math500.jsonl"))
            self.assertIn("--no-enforce-eager", commands[0])

    def test_overlapping_roots_are_rejected_in_both_directions(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, _ = self.fixture(Path(temporary))
            for output in (source, source / "new_eval", source.parent):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "non-overlapping"):
                    reevaluate.plan(reevaluate.parser().parse_args(self.arguments(source, data, output)))

    def test_output_symlink_cannot_write_into_original_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            output.mkdir()
            (output / "qwen3_1p7b").symlink_to(source / "qwen3_1p7b", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                reevaluate.plan(reevaluate.parser().parse_args(self.arguments(source, data, output)))

    def test_modified_training_artifacts_and_wrong_final_step_are_rejected(self):
        for change, expected in (("identity", "identity mismatch"), ("config", "configuration was modified"),
                                 ("source", "source manifest"), ("data", "Training data changed"),
                                 ("step", "planned final step"), ("base", "different base model"),
                                 ("export", "final export"), ("shard", "Missing model shard")):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                source, data, output = self.fixture(Path(temporary))
                run = source / "qwen3_1p7b/rft/pilot/seed_42"
                complete_path = run / "training_complete.json"
                complete = json.loads(complete_path.read_text())
                if change == "identity":
                    complete["seed"] = 43
                elif change == "config":
                    (run / "resolved_config.yaml").write_text("modified")
                elif change == "source":
                    complete["source_sha256"] = {"modified": "hash"}
                elif change == "data":
                    (data / "pilot.parquet").write_text("modified")
                elif change == "step":
                    complete["global_step"] = 40
                elif change == "base":
                    path = run / "run_manifest.json"
                    prior = json.loads(path.read_text())
                    prior["model_provenance"]["repo_id"] = "different/base"
                    path.write_text(json.dumps(prior))
                elif change == "export":
                    complete["final_hf_model"] = str(run / "checkpoints/global_step_40/actor/huggingface")
                elif change == "shard":
                    export = Path(complete["final_hf_model"])
                    (export / "model.safetensors.index.json").write_text(json.dumps({
                        "weight_map": {"layer": "missing.safetensors"},
                    }))
                complete_path.write_text(json.dumps(complete))
                with self.assertRaisesRegex((ValueError, FileNotFoundError), expected):
                    reevaluate.plan(reevaluate.parser().parse_args(self.arguments(source, data, output)))

    def test_invalid_protocol_and_execution_budgets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            for extra in (("--n", "128"), ("--seed", "-1"), ("--evaluation-seed", "-1"),
                          ("--batch-size", "0"), ("--max-num-seqs", "-1"),
                          ("--max-num-batched-tokens", "0"), ("--gpu-memory-utilization", "nan"),
                          ("--gpu-memory-utilization", "1"), ("--methods", "rft", "rft"),
                          ("--env-workers", "-1"), ("--env-workers", "8")):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    reevaluate.plan(reevaluate.parser().parse_args(self.arguments(source, data, output, *extra)))

    def test_gpu_plan_has_disjoint_shards_and_ordered_merge_without_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            argv = self.arguments(source, data, output, "--gpus", "2", "6")
            commands = reevaluate.plan(reevaluate.parser().parse_args(argv))
            self.assertEqual([command[0] for command in commands], [
                "math_rft.evaluate", "math_rft.evaluate", "math_rft.merge_evaluations",
                "math_rft.evaluate", "math_rft.evaluate", "math_rft.merge_evaluations",
                "math_rft.analyze", "math_rft.plot",
            ])
            for start, method in ((0, "rft"), (3, "grpo")):
                paths = []
                for index, command in enumerate(commands[start:start + 2]):
                    self.assertEqual(option(command, "--num-shards"), "2")
                    self.assertEqual(option(command, "--shard-index"), str(index))
                    self.assertEqual(option(command, "--n"), "256")
                    self.assertEqual(option(command, "--data"), str(data / "dev.jsonl"))
                    path = output / "qwen3_1p7b" / method / "pilot/seed_42/shards" / f"shard_{index:05d}_of_00002/dev.jsonl"
                    self.assertEqual(option(command, "--output"), str(path))
                    paths.append(str(path))
                merge = commands[start + 2]
                self.assertEqual(merge[merge.index("--inputs") + 1:merge.index("--data")], paths)
                self.assertEqual(option(merge, "--output"),
                                 str(output / "qwen3_1p7b" / method / "pilot/seed_42/dev.jsonl"))
            stream = io.StringIO()
            with patch.object(reevaluate.subprocess, "Popen") as spawn, \
                    patch.object(reevaluate.subprocess, "run") as launch, redirect_stdout(stream):
                reevaluate.main(argv)
            spawn.assert_not_called()
            launch.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(stream.getvalue().count("CUDA_VISIBLE_DEVICES=2 "), 2)
            self.assertEqual(stream.getvalue().count("CUDA_VISIBLE_DEVICES=6 "), 2)

    def test_gpu_shards_launch_together_with_isolated_environment_and_finish_before_merge(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            events, spawned = [], []

            def spawn(command, **kwargs):
                pid = 1000 + len(spawned)
                events.append(("spawn", pid))
                spawned.append((command, kwargs))
                return FakeChild(pid, command, events)

            def run(command, **kwargs):
                events.append(("run", command[2]))

            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7", "PARENT_TEST_VALUE": "kept"}), \
                    patch.object(reevaluate.subprocess, "Popen", side_effect=spawn), \
                    patch.object(reevaluate.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                reevaluate.main(self.arguments(source, data, output, "--gpus", "2", "6", "--execute"))
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")
            self.assertEqual([kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs in spawned], ["2", "6", "2", "6"])
            for _, kwargs in spawned:
                self.assertEqual(kwargs["env"]["PARENT_TEST_VALUE"], "kept")
                self.assertTrue(kwargs["start_new_session"])
                self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
                self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
            self.assertEqual(events[:2], [("spawn", 1000), ("spawn", 1001)])
            self.assertLess(events.index(("wait", 1000, None)), events.index(("run", "math_rft.merge_evaluations")))
            self.assertLess(events.index(("wait", 1001, None)), events.index(("run", "math_rft.merge_evaluations")))
            self.assertLess(events.index(("run", "math_rft.merge_evaluations")), events.index(("spawn", 1002)))
            self.assertEqual([event[1] for event in events if event[0] == "run"],
                             ["math_rft.merge_evaluations", "math_rft.merge_evaluations", "math_rft.analyze", "math_rft.plot"])
            logs = list(output.rglob("*.log"))
            self.assertEqual(len(logs), 4)
            self.assertTrue(all("CUDA_VISIBLE_DEVICES=" in path.read_text() for path in logs))

    def test_gpu_failure_or_interrupt_cancels_all_groups_and_skips_merge(self):
        for failure in (1, KeyboardInterrupt()):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                source, data, output = self.fixture(Path(temporary))
                events, children = [], []

                def spawn(command, children=children, failure=failure, events=events, **kwargs):
                    pid = 1000 + len(children)
                    polls = (None,) if not children else (failure,)
                    child = FakeChild(pid, command, events, polls=polls)
                    children.append(child)
                    return child

                expected = KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else subprocess.CalledProcessError
                with patch.object(reevaluate.subprocess, "Popen", side_effect=spawn), \
                        patch.object(reevaluate.subprocess, "run") as launch, \
                        patch.object(reevaluate.os, "killpg") as kill, redirect_stdout(io.StringIO()), \
                        self.assertRaises(expected):
                    reevaluate.main(self.arguments(source, data, output, "--gpus", "2", "6", "--execute"))
                self.assertEqual(len(children), 2)
                launch.assert_not_called()
                self.assertEqual([call.args for call in kill.call_args_list], [
                    (1000, signal.SIGINT), (1001, signal.SIGINT),
                    (1000, signal.SIGKILL), (1001, signal.SIGKILL),
                ])
                waits = [event[2] for event in events if event[0] == "wait" and event[2] is not None]
                self.assertTrue(waits)
                self.assertTrue(all(0 <= timeout <= 10 for timeout in waits))

    def test_second_gpu_spawn_failure_cleans_first_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            child = FakeChild(1000, ["evaluator"], [])
            with patch.object(reevaluate.subprocess, "Popen", side_effect=[child, OSError("cannot spawn")]), \
                    patch.object(reevaluate.subprocess, "run") as launch, \
                    patch.object(reevaluate.os, "killpg") as kill, redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(OSError, "cannot spawn"):
                reevaluate.main(self.arguments(source, data, output, "--gpus", "2", "6", "--execute"))
            launch.assert_not_called()
            self.assertEqual([call.args for call in kill.call_args_list],
                             [(1000, signal.SIGINT), (1000, signal.SIGKILL)])

    def test_gpu_failure_removes_owned_locks_only_after_process_cleanup(self):
        for kind in ("owned", "foreign", "preexisting"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                source, data, output = self.fixture(Path(temporary))
                argv = self.arguments(source, data, output, "--gpus", "2", "6", "--execute")
                commands = reevaluate.plan(reevaluate.parser().parse_args(argv))[:2]
                locks = [Path(option(command, "--output") + ".lock") for command in commands]
                if kind == "preexisting":
                    locks[1].parent.mkdir(parents=True)
                    # Even a matching PID does not authorize removing a lock
                    # which predated our child (for example, after PID reuse).
                    locks[1].write_text("pid=1001\n")
                children, events = [], []

                def spawn(command, children=children, events=events, kind=kind, **kwargs):
                    pid = 1000 + len(children)
                    lock = Path(option(command, "--output") + ".lock")
                    if not lock.exists():
                        owner = 7777 if kind == "foreign" and children else pid
                        lock.write_text(f"pid={owner}\n")
                    child = FakeChild(pid, command, events, polls=(1,) if children else (None,))
                    children.append(child)
                    return child

                def killpg(pid, sig, locks=locks):
                    # Lock cleanup must follow all process-group signals.
                    self.assertTrue(all(lock.exists() for lock in locks))

                with patch.object(reevaluate.subprocess, "Popen", side_effect=spawn), \
                        patch.object(reevaluate.subprocess, "run") as launch, \
                        patch.object(reevaluate.os, "killpg", side_effect=killpg), \
                        redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                    reevaluate.main(argv)
                launch.assert_not_called()
                self.assertFalse(locks[0].exists())
                self.assertEqual(locks[1].exists(), kind != "owned")
                if kind != "owned":
                    self.assertEqual(locks[1].read_text(), "pid=7777\n" if kind == "foreign" else "pid=1001\n")

    def test_lock_cleanup_retains_live_or_malformed_or_symlink_locks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live, malformed, target, link = [root / name for name in ("live", "malformed", "target", "link")]
            live.write_text("pid=1000\n")
            malformed.write_text("pid=1001\npid=9999\n")
            target.write_text("pid=1002\n")
            link.symlink_to(target)
            owners = [(FakeChild(1000, [], [], polls=(None,)), live, False),
                      (FakeChild(1001, [], []), malformed, False),
                      (FakeChild(1002, [], []), link, False)]
            reevaluate.remove_interrupted_locks(owners)
            self.assertTrue(all(path.exists() for path in (live, malformed, target, link)))

    def test_single_explicit_gpu_uses_full_ledger_without_merge(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            commands = reevaluate.plan(reevaluate.parser().parse_args(
                self.arguments(source, data, output, "--gpus", "2", "--methods", "grpo"),
            ))
            self.assertEqual(len(commands), 1)
            self.assertNotIn("--num-shards", commands[0])
            self.assertEqual(option(commands[0], "--output"),
                             str(output / "qwen3_1p7b/grpo/pilot/seed_42/dev.jsonl"))

    def test_invalid_gpu_lists_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            for gpus in (("2", "2"), ("-1",), ("02",), ("+2",), ("2,6",), ("gpu2",)):
                with self.subTest(gpus=gpus), self.assertRaisesRegex(ValueError, "canonical GPU IDs"):
                    reevaluate.plan(reevaluate.parser().parse_args(self.arguments(source, data, output, "--gpus", *gpus)))

    def test_shard_log_symlink_cannot_write_into_training_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, data, output = self.fixture(Path(temporary))
            log = output / "qwen3_1p7b/rft/pilot/seed_42/shards/shard_00000_of_00002/dev.jsonl.log"
            log.parent.mkdir(parents=True)
            log.symlink_to(source / "original.log")
            with self.assertRaisesRegex(ValueError, "escapes"):
                reevaluate.plan(reevaluate.parser().parse_args(self.arguments(source, data, output, "--gpus", "2", "6")))


if __name__ == "__main__":
    unittest.main()
