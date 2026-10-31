"""Check low-memory execution without changing the scientific training protocol."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from math_rft import train


HAS_CONFIG_STACK = all(importlib.util.find_spec(name) is not None for name in ("verl", "hydra", "omegaconf"))


def leaves(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(leaves(item, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


@unittest.skipUnless(HAS_CONFIG_STACK, "Requires the pinned verl environment for real Hydra composition")
class TrainingProfileTests(unittest.TestCase):
    def compose(self, *options):
        args = train.parser().parse_args(["--model", "qwen3_1p7b", "--method", "rft", *options])
        return train.compose_config(args, "Qwen/Qwen3-1.7B-Base")

    def test_lowmem_changes_only_allowlisted_execution_settings(self):
        from omegaconf import OmegaConf
        allowed = {
            "actor_rollout_ref.actor.calculate_entropy",
            "actor_rollout_ref.actor.fsdp_config.param_offload",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.standalone_gpu_memory_utilization",
            "actor_rollout_ref.rollout.max_num_seqs",
            "actor_rollout_ref.rollout.max_num_batched_tokens",
            "actor_rollout_ref.rollout.enable_chunked_prefill",
            "actor_rollout_ref.rollout.enforce_eager",
            "ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_FLASHINFER_SAMPLER",
        }
        for method in ("rft", "grpo"):
            with self.subTest(method=method):
                # Equal destinations avoid unrelated output-path differences.
                default, _ = self.compose("--method", method, "--output-dir", "same_output")
                lowmem, _ = self.compose("--method", method, "--output-dir", "same_output", "--profile", "lowmem")
                before = leaves(OmegaConf.to_container(default, resolve=True))
                after = leaves(OmegaConf.to_container(lowmem, resolve=True))
                missing = object()
                changed = {key for key in before.keys() | after.keys()
                           if before.get(key, missing) != after.get(key, missing)}
                self.assertTrue(changed)
                self.assertLessEqual(changed, allowed, changed - allowed)

    def test_full_training_budget_and_loss_are_preserved(self):
        for method, advantage, loss in (("rft", "math_rft_question", "math_rft_nll"),
                                        ("grpo", "grpo", "math_grpo_clip")):
            with self.subTest(method=method):
                config, _ = self.compose("--profile", "lowmem", "--method", method)
                actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
                self.assertEqual((config.data.train_batch_size, actor.ppo_mini_batch_size, rollout.n), (64, 64, 8))
                self.assertEqual((config.data.max_prompt_length, config.data.max_response_length,
                                  rollout.max_model_len), (4096, 4096, 8192))
                self.assertEqual((config.trainer.n_gpus_per_node, config.trainer.total_epochs), (8, 3))
                self.assertIsNone(config.trainer.total_training_steps)
                self.assertTrue(config.trainer.val_before_train)
                self.assertEqual((config.trainer.test_freq, config.trainer.save_freq, rollout.val_kwargs.n), (20, 20, 16))
                self.assertEqual((rollout.temperature, rollout.top_p, rollout.top_k), (1.0, 1.0, -1))
                self.assertEqual((actor.fsdp_config.model_dtype, actor.fsdp_config.dtype), ("fp32", "bfloat16"))
                self.assertEqual(actor.optim.lr, 1e-6)
                self.assertEqual(actor.ppo_micro_batch_size_per_gpu, 1)
                self.assertEqual(config.actor_rollout_ref.model.lora.rank, 0)
                self.assertEqual((config.algorithm.adv_estimator, actor.policy_loss.loss_mode), (advantage, loss))
                self.assertTrue(actor.fsdp_config.param_offload)
                self.assertTrue(actor.fsdp_config.optimizer_offload)
                self.assertFalse(actor.calculate_entropy)
                self.assertEqual(actor.entropy_coeff, 0)
                self.assertEqual((rollout.gpu_memory_utilization, rollout.max_num_seqs,
                                  rollout.max_num_batched_tokens), (0.3, 16, 2048))
                self.assertTrue(rollout.enforce_eager)
                self.assertTrue(rollout.enable_chunked_prefill)
                self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_FLASHINFER_SAMPLER, "0")

    def test_lowmem_is_available_for_full_and_deepmath_stages(self):
        for stage in ("full", "deepmath_8192", "deepmath_16384"):
            with self.subTest(stage=stage):
                config, output = self.compose("--profile", "lowmem", "--stage", stage)
                self.assertEqual(Path(config.data.train_files[0]).name, f"{stage}.parquet")
                self.assertEqual(output.parent.name, stage)
                self.assertIsNone(config.trainer.total_training_steps)

    def test_profile_does_not_mutate_defaults_and_has_distinct_output(self):
        _, lowmem_path = self.compose("--profile", "lowmem")
        default, default_path = self.compose()
        self.assertEqual(lowmem_path, Path("runs_lowmem/qwen3_1p7b/rft/pilot/seed_42").resolve())
        self.assertEqual(default_path, Path("runs/qwen3_1p7b/rft/pilot/seed_42").resolve())
        self.assertTrue(default.actor_rollout_ref.actor.calculate_entropy)
        self.assertFalse(default.actor_rollout_ref.actor.fsdp_config.param_offload)
        self.assertFalse(default.actor_rollout_ref.actor.fsdp_config.optimizer_offload)
        self.assertEqual(default.actor_rollout_ref.rollout.gpu_memory_utilization, 0.65)

    def test_explicit_overrides_win_without_changing_global_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "custom"
            config, run_dir = self.compose(
                "--profile", "lowmem", "--output-dir", str(output),
                "--set", "trainer.n_gpus_per_node=2",
                "--set", "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
            )
            self.assertEqual(config.trainer.n_gpus_per_node, 2)
            self.assertEqual(config.data.train_batch_size, 64)
            self.assertEqual(config.actor_rollout_ref.rollout.gpu_memory_utilization, 0.4)
            self.assertTrue(run_dir.is_relative_to(output))
            self.assertFalse(output.exists())

    def test_profile_keeps_algorithm_safeguards(self):
        for override in ("actor_rollout_ref.rollout.n=1", "data.train_batch_size=4",
                         "actor_rollout_ref.rollout.max_model_len=4096",
                         "actor_rollout_ref.model.lora.rank=8"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.compose("--profile", "lowmem", "--set", override)

    def test_print_config_never_resolves_weights_or_writes_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not_created"
            argv = ["train", "--model", "qwen3_1p7b", "--method", "grpo", "--profile", "lowmem",
                    "--output-dir", str(output), "--print-config"]
            stream = io.StringIO()
            with patch.object(sys, "argv", argv), patch.object(train, "check_verl_revision"), \
                    patch.object(train, "_resolve_model") as resolve, redirect_stdout(stream):
                train.main()
            resolve.assert_not_called()
            self.assertFalse(output.exists())
            from omegaconf import OmegaConf
            config = OmegaConf.create(stream.getvalue())
            self.assertIsNone(config.trainer.total_training_steps)
            self.assertEqual(config.actor_rollout_ref.actor.policy_loss.loss_mode, "math_grpo_clip")


class TrainingProfileCliTests(unittest.TestCase):
    def test_unknown_profile_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train.parser().parse_args(["--model", "qwen3_1p7b", "--method", "rft", "--profile", "smoke"])


if __name__ == "__main__":
    unittest.main()
