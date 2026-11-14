"""ALFWorld uses the same objectives with independent full-episode rollouts."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from math_rft import pipeline, train

HAS_CONFIG = all(importlib.util.find_spec(name) is not None for name in ("verl", "hydra", "omegaconf"))


@unittest.skipUnless(HAS_CONFIG, "Requires pinned verl/Hydra configuration")
class AlfWorldTrainingTests(unittest.TestCase):
    def compose(self, *options):
        args = train.parser().parse_args([
            "--benchmark", "alfworld", "--model", "qwen3_1p7b", "--method", "rft", *options,
        ])
        return train.compose_config(args, "unused-model")

    def test_episode_runner_and_rewards_preserve_group_objective(self):
        for method in ("rft", "grpo"):
            with self.subTest(method=method):
                config, output = self.compose("--method", method, "--profile", "lowmem")
                rollout, actor = config.actor_rollout_ref.rollout, config.actor_rollout_ref.actor
                self.assertEqual(rollout.agent.default_agent_loop, "alfworld")
                self.assertTrue(rollout.multi_turn.enable)
                self.assertEqual((rollout.n, config.data.train_batch_size, actor.ppo_mini_batch_size), (8, 64, 64))
                self.assertEqual((actor.loss_agg_mode, actor.loss_scale_factor), ("seq-mean-token-sum-norm", 4096))
                self.assertEqual((config.alfworld.max_steps, config.alfworld.max_action_tokens), (50, 128))
                self.assertEqual(Path(config.reward.custom_reward_function.path).name, "alfworld_reward.py")
                self.assertEqual(Path(config.data.train_files[0]).parent.name, "data_alfworld")
                self.assertEqual(output.parts[-5], "runs_alfworld_lowmem_fixed")

    def test_explicit_horizon_is_shared_with_configuration(self):
        config, _ = self.compose("--max-steps", "20", "--max-action-tokens", "64")
        self.assertEqual((config.alfworld.max_steps, config.alfworld.max_action_tokens), (20, 64))

    def test_invalid_episode_settings_fail_before_runtime(self):
        for options in (("--max-steps", "0"), ("--max-action-tokens", "4097"),
                        ("--stage", "deepmath_8192"),
                        ("--set", "actor_rollout_ref.rollout.full_determinism=true"),
                        ("--set", "actor_rollout_ref.rollout.multi_turn.enable=false"),
                        ("--set", "alfworld.max_steps=30"),
                        ("--set", "actor_rollout_ref.rollout.agent.default_agent_loop=raw_math")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.compose(*options)

    def test_test_tasks_cannot_be_used_as_training_or_validation(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from math_rft import alfworld_env

        for filename, source in (("test", "valid_unseen"), ("pilot", "valid_unseen")):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                training, dev = root / f"{filename}.parquet", root / "dev.parquet"
                for path, origin in ((training, source), (dev, "valid_seen")):
                    pq.write_table(pa.Table.from_pylist([{"extra_info": {
                        "split": path.stem, "alfworld_task": {"source_split": origin},
                    }}]), path)
                (root / "manifest.json").write_text(json.dumps({
                    "benchmark": "alfworld", "environment": {},
                    "prompt_sha256": train.benchmark_prompt_fingerprint("alfworld"),
                    "splits": {path.stem: {"parquet_sha256": train._sha256(path)} for path in (training, dev)},
                }))
                config = SimpleNamespace(
                    data=SimpleNamespace(train_files=[str(training)], val_files=[str(dev)], train_max_samples=-1,
                                         gen_batch_size=None, train_batch_size=1),
                    trainer=SimpleNamespace(total_epochs=1, total_training_steps=None),
                    actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(loss_scale_factor=4096)),
                    alfworld=SimpleNamespace(max_steps=50, max_action_tokens=128),
                )
                argv = ["train", "--benchmark", "alfworld", "--model", "qwen3_1p7b", "--method", "rft"]
                with patch.object(sys, "argv", argv), patch.object(train, "check_verl_revision", return_value={}), \
                        patch.object(train, "_resolve_model", return_value=("unused", {})), \
                        patch.object(train, "compose_config", return_value=(config, root / "run")), \
                        patch.object(train, "_source_hashes", return_value={}), \
                        patch.object(train, "environment_versions", return_value={}), \
                        patch.object(alfworld_env, "environment_identity", return_value={}), \
                        self.assertRaisesRegex(ValueError, "held-out test|incorrect source split"):
                    train.main()
                self.assertFalse((root / "run").exists())


class AlfWorldPipelineTests(unittest.TestCase):
    def plan(self, *options):
        argv = ["pipeline", "--benchmark", "alfworld", "--models", "qwen3_1p7b", "--seeds", "42", *options]
        with patch.object(sys, "argv", argv), patch.object(pipeline, "invoke") as invoke, \
                patch.object(pipeline, "frozen_revisions") as resolve, redirect_stdout(io.StringIO()):
            pipeline.main()
        resolve.assert_not_called()
        self.assertTrue(all(not call.args[1] for call in invoke.call_args_list))
        return [call.args[0] for call in invoke.call_args_list]

    def test_full_uses_unseen_test_and_same_episode_budgets(self):
        from math_rft import evaluate
        commands = self.plan("--stage", "full", "--max-steps", "30", "--max-action-tokens", "64")
        for command in commands:
            if command[0] == "math_rft.evaluate":
                args = evaluate.parser().parse_args(command[1:])
                self.assertEqual((args.benchmark, args.role, args.data.name, args.n),
                                 ("alfworld", "formal", "test.jsonl", 256))
                self.assertEqual((args.max_steps, args.max_action_tokens), (30, 64))
            elif command[0] == "math_rft.train":
                args = train.parser().parse_args(command[1:])
                self.assertEqual((args.benchmark, args.max_steps, args.max_action_tokens), ("alfworld", 30, 64))
                self.assertEqual(Path(args.output_dir).name, "runs_alfworld_fixed")

    def test_pilot_uses_seen_dev(self):
        commands = self.plan()
        for command in commands:
            if command[0] == "math_rft.evaluate":
                self.assertEqual(command[command.index("--role") + 1], "development")
                self.assertEqual(Path(command[command.index("--data") + 1]).name, "dev.jsonl")

    def test_wrong_benchmark_split_and_unpaired_budgets_fail(self):
        for options in (("--evaluation-split", "math500"), ("--stage", "deepmath_8192"),
                        ("--max-steps", "0"), ("--set", "data.max_response_length=1024")):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.plan(*options)


if __name__ == "__main__":
    unittest.main()
