"""Test the lightweight ALFWorld adapter contract without importing the GPU stack."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from math_rft import alfworld_reward
from math_rft.alfworld_episode import EpisodeResult, render_prompt


def load_adapter():
    interface = ModuleType("verl.experimental.agent_loop.agent_loop")
    interface.AgentLoopBase = object
    interface.AgentLoopOutput = SimpleNamespace
    path = Path(__file__).resolve().parents[1] / "src/math_rft/alfworld_loop.py"
    spec = importlib.util.spec_from_file_location("math_rft._test_alfworld_loop", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {interface.__name__: interface}):
        spec.loader.exec_module(module)
    return module


class LoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = load_adapter()
        self.loop = self.module.AlfWorldAgentLoop()
        self.loop.config = SimpleNamespace(alfworld=SimpleNamespace(max_steps=50, max_action_tokens=128))
        self.loop.rollout_config = SimpleNamespace(prompt_length=4096, response_length=4096)
        self.loop.tokenizer = object()
        self.loop.server_manager = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(
            token_ids=[10, 20], log_probs=[-0.5, -0.8], stop_reason="completed")))
        self.task = {"initial_observation": "Task"}
        self.kwargs = {"extra_info": {"alfworld_task": self.task},
                       "raw_prompt": [{"role": "user", "content": render_prompt("Task")}]}

    async def test_live_score_masks_and_sampler_boundaries_reach_verl(self):
        async def episode(task, tokenizer, generate, **kwargs):
            generated = await generate([1, 2], 16, None)
            self.assertEqual(generated.token_ids, [10, 20])
            self.assertIsNone(kwargs["seed"])
            return EpisodeResult([1, 2], [10, 30, 20], [1, 0, 1], [-0.5, 0, -0.8],
                                 1, [{"action": "look"}], "success", "trace",
                                 {"environment_steps": 1, "generate_sequences": 1.0, "tool_calls": 0.5})
        with patch.object(self.module, "run_episode", side_effect=episode):
            result = await self.loop.run({"temperature": 1.0}, **self.kwargs)
        self.assertEqual(result.reward_score, 1.0)
        self.assertEqual(result.response_mask, [1, 0, 1])
        params = self.loop.server_manager.generate.call_args.kwargs["sampling_params"]
        self.assertEqual(params["stop"], ["\n"])
        self.assertTrue(params["include_stop_str_in_output"])
        self.assertFalse(params["ignore_eos"])
        self.assertEqual(params["max_tokens"], 16)
        self.assertNotIn("seed", params)

    async def test_prompt_mismatch_is_rejected_before_rollout(self):
        self.kwargs["raw_prompt"][0]["content"] = "Different task"
        with self.assertRaisesRegex(ValueError, "raw_prompt"):
            await self.loop.run({}, **self.kwargs)
        self.loop.server_manager.generate.assert_not_called()

    async def test_text_cannot_claim_environment_success(self):
        with self.assertRaisesRegex(RuntimeError, "live environment"):
            await alfworld_reward.compute_score("alfworld", "You win!", "success", {"alfworld_success": 1})


if __name__ == "__main__":
    unittest.main()
