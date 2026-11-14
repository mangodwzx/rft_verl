"""End-to-end episode evaluation, provenance, sharding and resume without GPUs."""
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from math_rft import alfworld_env, alfworld_episode, evaluate, merge_evaluations, reevaluate
from math_rft.analyze import assert_comparable, load_run


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(token) for token in tokens if token)


@contextmanager
def runtime(fail_step=False):
    events, initializations = [], []

    class Environment:
        def __init__(self, task, max_steps, seed):
            self.task, self.max_steps, self.seed = task, max_steps, seed
            self.steps = 0
            events.append(("create", id(self)))

        def reset(self):
            events.append(("reset", id(self)))
            return self.task["initial_observation"]

        def step(self, action):
            if fail_step:
                raise RuntimeError("environment infrastructure failed")
            self.steps += 1
            success = self.steps == 2 and self.task.get("can_succeed", True)
            done = success or self.steps == self.max_steps
            return "You see an apple on the table.", success, done, {"truncated": done and not success}

        def close(self):
            events.append(("close", id(self)))

    class Engine:
        def __init__(self, **kwargs):
            initializations.append(kwargs)

        def get_tokenizer(self):
            return Tokenizer()

        def generate(self, prompts, sampling_params, use_tqdm):
            requests = []
            for prompt, params in zip(prompts, sampling_params, strict=True):
                if params.stop != ["\n"] or not params.include_stop_str_in_output:
                    raise AssertionError("Action boundary changed")
                text = Tokenizer().decode(prompt["prompt_token_ids"])
                action = "take apple from table\n" if "You see an apple on the table." in text else "look\n"
                ids = Tokenizer().encode(action)
                if len(ids) > params.max_tokens:
                    ids = ids[:params.max_tokens]
                    finish = "length"
                else:
                    finish = "stop"
                requests.append(SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], outputs=[
                    SimpleNamespace(token_ids=ids, text=action, finish_reason=finish, stop_reason="\n")]))
            return requests

    vllm = ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = Engine, SimpleNamespace
    identity = {"kind": "huggingface", "repo_id": "fake/base", "revision": "a" * 40}
    with patch.dict(sys.modules, {"vllm": vllm}), \
            patch.object(evaluate, "model_identity", return_value=("fake/base", "a" * 40, identity)), \
            patch.object(evaluate.importlib.metadata, "version", return_value="fake"), \
            patch.object(alfworld_env, "environment_identity", return_value={"adapter": "fake"}), \
            patch.object(alfworld_env, "AlfWorldEnvironment", Environment), redirect_stdout(io.StringIO()):
        yield events, initializations


class AlfWorldEvaluationTests(unittest.TestCase):
    def dataset(self, root, count=3, split="dev"):
        game, trajectory = root / "game.tw-pddl", root / "traj_data.json"
        game.write_text("frozen game")
        trajectory.write_text("{}")
        source = "valid_unseen" if split == "test" else "valid_seen"
        rows = []
        for index in range(count):
            task = {"game_file": str(game), "game_sha256": evaluate.file_hash(game),
                    "trajectory_file": str(trajectory), "trajectory_sha256": evaluate.file_hash(trajectory),
                    "initial_observation": f"Room {index}. Your task is to take an apple.",
                    "source_split": source, "task_type": "pick_and_place_simple"}
            rows.append({"id": f"alfworld:{source}:{index}", "problem": "Take an apple.",
                         "prompt": alfworld_episode.render_prompt(task["initial_observation"]),
                         "subject": task["task_type"], "level": None, "split": split,
                         "data_source": "alfworld", "extra_info": {"alfworld_task": task}})
        path = root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        (root / "manifest.json").write_text(json.dumps({"benchmark": "alfworld", "environment": {"adapter": "fake"}, "splits": {
            split: {"ids": [row["id"] for row in rows], "count": count, "jsonl_sha256": evaluate.file_hash(path)}}}))
        return path, rows

    def args(self, data, output, *extra):
        return ["--model", "fake/base", "--benchmark", "alfworld", "--data", str(data),
                "--output", str(output), "--role", "debug", "--n", "2", "--batch-size", "4",
                "--max-steps", "3", *extra]

    def test_full_episodes_mask_observations_and_resume_without_regeneration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, rows = self.dataset(root)
            output = root / "samples.jsonl"
            with runtime() as (events, engines):
                evaluate.main(self.args(data, output))
                run = load_run(output)
                before = output.read_bytes()
                evaluate.main(self.args(data, output))
            self.assertEqual(len(engines), 1)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(sum(event[0] == "reset" for event in events), 6)
            self.assertEqual(sum(event[0] == "close" for event in events), 6)
            self.assertEqual(run["config"]["benchmark"], "alfworld")
            self.assertEqual(run["summary"]["groups"]["all"]["pass_at_k"]["2"], 1)
            self.assertEqual(run["summary"]["groups"]["level_4_5"]["questions"], 0)
            for record in evaluate.read_jsonl(output):
                self.assertEqual(record["score"], 1)
                self.assertEqual(record["termination"], "success")
                self.assertEqual(len(record["turns"]), 2)
                self.assertIn(0, record["response_mask"])
                count = run["counts"][record["id"]]
                self.assertEqual(count["generated_tokens"], 2 * sum(record["response_mask"]))
                self.assertGreater(count["observation_tokens"], 0)

    def test_protocol_change_and_mutated_assets_reject_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _ = self.dataset(root)
            output = root / "samples.jsonl"
            with runtime() as (_, engines):
                evaluate.main(self.args(data, output))
                before = output.read_bytes()
                with self.assertRaisesRegex(ValueError, "Resume rejected"):
                    evaluate.main(self.args(data, output, "--max-steps", "4"))
                self.assertEqual(len(engines), 1)
                (root / "game.tw-pddl").write_text("modified state")
                with self.assertRaisesRegex(ValueError, "asset changed"):
                    evaluate.main(self.args(data, output))
                self.assertEqual(output.read_bytes(), before)

    def test_formal_checks_whole_unseen_split_and_development_excludes_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, rows = self.dataset(root, split="test")
            self.assertEqual(evaluate.load_data(data, "formal", 256, "alfworld"), rows)
            with self.assertRaisesRegex(ValueError, "development"):
                evaluate.load_data(data, "development", 256, "alfworld")
            with self.assertRaisesRegex(ValueError, "n>=256"):
                evaluate.load_data(data, "formal", 8, "alfworld")
            data.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
            with self.assertRaisesRegex(ValueError, "full prepared test split"):
                evaluate.load_data(data, "formal", 256, "alfworld")

    def test_infrastructure_errors_are_not_committed_as_failed_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _ = self.dataset(root, count=1)
            output = root / "samples.jsonl"
            with runtime(fail_step=True) as (events, _), self.assertRaisesRegex(RuntimeError, "infrastructure"):
                evaluate.main(self.args(data, output))
            self.assertEqual(output.read_text(), "")
            self.assertEqual(sum(event[0] == "close" for event in events), 2)
            self.assertFalse(evaluate.artifact_paths(output)["lock"].exists())

    def test_action_budget_records_failure_and_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _ = self.dataset(root, count=1)
            output = root / "samples.jsonl"
            with runtime():
                evaluate.main(self.args(data, output, "--max-action-tokens", "2"))
            run = load_run(output)
            self.assertEqual(run["summary"]["groups"]["all"]["truncation_rate"], 1)
            self.assertEqual(run["summary"]["groups"]["all"]["pass_at_k"]["2"], 0)
            self.assertEqual(run["summary"]["groups"]["all"]["environment_steps"], 0)

    def test_environment_version_change_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _ = self.dataset(root, count=1)
            output = root / "samples.jsonl"
            with runtime() as (_, engines), \
                    patch.object(alfworld_env, "environment_identity", return_value={"adapter": "changed"}), \
                    self.assertRaisesRegex(ValueError, "prepared dataset environment"):
                evaluate.main(self.args(data, output))
            self.assertEqual(engines, [])
            self.assertFalse(output.exists())

    def test_shards_merge_and_protocol_mismatch_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, rows = self.dataset(root)
            outputs = [root / f"shard{index}.jsonl" for index in range(2)]
            with runtime():
                for index, output in enumerate(outputs):
                    evaluate.main(self.args(data, output, "--num-shards", "2", "--shard-index", str(index)))
            merged = root / "merged.jsonl"
            with redirect_stdout(io.StringIO()):
                merge_evaluations.merge(outputs, data, merged)
            run = load_run(merged)
            self.assertEqual(run["rows"], rows)
            changed = {**run, "config": {**run["config"], "episode_protocol": {"max_steps": 99}}}
            with self.assertRaisesRegex(ValueError, "episode_protocol"):
                assert_comparable(run, changed)

    def test_reevaluation_infers_benchmark_and_forwards_episode_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _ = self.dataset(root)
            source, output = root / "training", root / "reevaluated"
            protocol = {"max_steps": 7, "max_action_tokens": 31}
            for method in ("rft", "grpo"):
                run = source / "qwen3_1p7b" / method / "pilot/seed_42"
                run.mkdir(parents=True)
                (run / "run_manifest.json").write_text(json.dumps({"benchmark": "alfworld"}))
            args = reevaluate.parser().parse_args([
                "--run-root", str(source), "--output-dir", str(output), "--data-dir", str(data.parent)])
            fake_protocol = {"base_revision": "a" * 40, "data_sha256": {}, "training_objective": None,
                             "benchmark": "alfworld", "alfworld": protocol}
            with patch.object(reevaluate, "final_model", return_value=(root / "export", fake_protocol)):
                commands = reevaluate.plan(args)
            for command in commands[:2]:
                self.assertEqual(command[command.index("--benchmark") + 1], "alfworld")
                self.assertEqual(command[command.index("--max-steps") + 1], "7")
                self.assertEqual(command[command.index("--max-action-tokens") + 1], "31")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
