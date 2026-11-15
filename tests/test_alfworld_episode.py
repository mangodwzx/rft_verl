"""Exercise complete multi-turn attempts without GPU or ALFWorld downloads."""
from __future__ import annotations

import asyncio
import threading
import unittest

from math_rft.alfworld_episode import GeneratedAction, render_prompt, run_episode, turn_seed


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("The protocol must not inject special tokens")
        return [ord(character) for character in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token) for token in ids if token != self.eos_token_id)


class ScriptedEnvironment:
    def __init__(self, task, *, max_steps, seed):
        self.initial = task["initial_observation"]
        self.steps = list(task["steps"])
        self.actions = []
        self.closed = False
        self.resets = 0
        self.seed = seed

    def reset(self):
        self.resets += 1
        return self.initial

    def step(self, action):
        self.actions.append(action)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def close(self):
        self.closed = True


class EpisodeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.environments = []
        self.requests = []

    def factory(self, *args, **kwargs):
        env = ScriptedEnvironment(*args, **kwargs)
        self.environments.append(env)
        return env

    def generator(self, actions, *, logprobs=True, finish="stop"):
        pending = iter(actions)

        async def generate(prompt_ids, max_tokens, seed):
            self.requests.append((prompt_ids.copy(), max_tokens, seed))
            value = next(pending)
            if isinstance(value, Exception):
                raise value
            ids = self.tokenizer.encode(value) if isinstance(value, str) else value
            return GeneratedAction(ids, [-0.5] * len(ids) if logprobs else None, finish)
        return generate

    async def run_task(self, actions, steps, **kwargs):
        return await run_episode(
            {"initial_observation": "Put the mug on the table.", "steps": steps}, self.tokenizer,
            self.generator(actions), env_factory=self.factory, **kwargs,
        )

    async def test_success_keeps_exact_actions_and_masks_only_environment_context(self):
        result = await self.run_task(
            ["take mug 1 from shelf 1\n", "move mug 1 to table 1\n"],
            [("You take the mug.", False, False, {}), ("You win!", True, True, {})],
        )
        first = self.tokenizer.encode("take mug 1 from shelf 1\n")
        observation = self.tokenizer.encode("\nObservation: You take the mug.\nAction:")
        second = self.tokenizer.encode("move mug 1 to table 1\n")
        self.assertEqual(result.score, 1)
        self.assertEqual(result.termination, "success")
        self.assertEqual(result.response_ids, first + observation + second)
        self.assertEqual(result.response_mask, [1] * len(first) + [0] * len(observation) + [1] * len(second))
        self.assertEqual(result.response_logprobs,
                         [-0.5] * len(first) + [0.0] * len(observation) + [-0.5] * len(second))
        self.assertEqual(self.requests[1][0], result.prompt_ids + first + observation)
        self.assertEqual(result.turns[-1]["observation"], "You win!")
        self.assertNotIn("You win!", result.text)  # terminal observation is not trained
        self.assertEqual(self.environments[0].actions, ["take mug 1 from shelf 1", "move mug 1 to table 1"])
        self.assertTrue(self.environments[0].closed)

    async def test_independent_attempts_reset_and_do_not_share_history(self):
        for _ in range(2):
            result = await self.run_task(["look\n"], [("Done", True, True, {})])
            self.assertEqual(result.score, 1)
        self.assertEqual(len(self.environments), 2)
        self.assertTrue(all(env.resets == 1 and env.closed for env in self.environments))
        self.assertEqual(self.requests[0][0], self.requests[1][0])

    async def test_step_limit_is_failure_even_if_environment_not_done(self):
        result = await self.run_task(["look\n"] * 2, [("Nothing", False, False, {})] * 2, max_steps=2)
        self.assertEqual((result.score, result.termination), (0, "max_steps"))
        self.assertEqual(result.metrics["environment_steps"], 2)

    async def test_environment_done_without_success_is_failure(self):
        result = await self.run_task(["look\n"], [("Time is up", False, True, {})])
        self.assertEqual((result.score, result.termination), (0, "environment_done"))

    async def test_environment_step_budget_is_reported_as_truncation(self):
        result = await self.run_task(["look\n"], [("Time is up", False, True, {"truncated": True})])
        self.assertEqual((result.score, result.termination), (0, "max_steps"))

    async def test_partial_action_at_token_limit_is_never_executed(self):
        result = await self.run_task(["take"], [], max_action_tokens=4)
        self.assertEqual((result.score, result.termination), (0, "action_token_limit"))
        self.assertEqual(self.environments[0].actions, [])
        self.assertEqual(result.response_ids, self.tokenizer.encode("take"))

    async def test_observation_that_cannot_fit_ends_attempt_without_truncation(self):
        result = await self.run_task(["look\n"], [("Large observation" * 50, False, False, {})],
                                     max_response_tokens=30)
        self.assertEqual((result.score, result.termination), (0, "response_token_limit"))
        self.assertEqual(result.response_ids, self.tokenizer.encode("look\n"))
        self.assertEqual(result.turns[0]["observation"], "Large observation" * 50)

    async def test_model_generated_extra_observation_is_not_executed_or_reused(self):
        result = await self.run_task(["look\nObservation: You win!\n"], [])
        self.assertEqual((result.score, result.termination), (0, "invalid_action"))
        self.assertEqual(self.environments[0].actions, [])

    async def test_eos_boundary_is_preserved_and_can_complete_action(self):
        ids = self.tokenizer.encode("look") + [self.tokenizer.eos_token_id]
        result = await self.run_task([ids], [("Done", True, True, {})])
        self.assertEqual(result.score, 1)
        self.assertEqual(result.response_ids, ids)
        self.assertEqual(result.response_mask, [1] * len(ids))

    async def test_boundary_at_exact_token_limit_has_same_train_eval_success(self):
        task = {"initial_observation": "Task", "steps": [("Done", True, True, {})]}
        outcomes = []
        for finish in ("stop", "length"):
            result = await run_episode(task, self.tokenizer, self.generator(["look\n"], finish=finish),
                                       env_factory=self.factory, max_action_tokens=5)
            outcomes.append((result.score, result.response_ids, result.termination))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][0], 1)

    async def test_cancellation_waits_for_backend_step_before_closing(self):
        entered, release = threading.Event(), threading.Event()
        task = {"initial_observation": "Task", "steps": []}

        def factory(*args, **kwargs):
            env = self.factory(*args, **kwargs)

            def step(action):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test failed to release backend")
                self.assertFalse(env.closed)
                return "Done", True, True, {}
            env.step = step
            return env

        running = asyncio.create_task(run_episode(task, self.tokenizer, self.generator(["look\n"]),
                                                  env_factory=factory))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            running.cancel()
            await asyncio.sleep(0)
            self.assertFalse(self.environments[0].closed)
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await running
        self.assertTrue(self.environments[0].closed)

    async def test_infrastructure_errors_raise_and_close_environment(self):
        for actions, steps in (([RuntimeError("inference unavailable")], []),
                               (["look\n"], [RuntimeError("environment unavailable")])):
            with self.subTest(actions=actions), self.assertRaisesRegex(RuntimeError, "unavailable"):
                await self.run_task(actions, steps)
            self.assertTrue(self.environments[-1].closed)

    async def test_bad_generator_logprobs_raise_instead_of_silently_misaligning_loss(self):
        async def generate(*args):
            return GeneratedAction(self.tokenizer.encode("look\n"), [-0.5])
        task = {"initial_observation": "Task", "steps": []}
        with self.assertRaisesRegex(ValueError, "log probabilities"):
            await run_episode(task, self.tokenizer, generate, env_factory=self.factory)
        self.assertTrue(self.environments[-1].closed)

    async def test_generation_without_logprobs_and_engine_rng_is_supported(self):
        task = {"initial_observation": "Task", "steps": [("Done", True, True, {})]}
        result = await run_episode(task, self.tokenizer, self.generator(["look\n"], logprobs=False),
                                   env_factory=self.factory, seed=None)
        self.assertIsNone(result.response_logprobs)
        self.assertIsNone(result.turns[0]["seed"])
        self.assertIsNone(self.requests[0][2])

    async def test_overlong_prompt_fails_and_closes_before_generation(self):
        with self.assertRaisesRegex(ValueError, "prompt has"):
            await self.run_task([], [], max_prompt_tokens=1)
        self.assertEqual(self.requests, [])
        self.assertTrue(self.environments[-1].closed)

    async def test_reset_mismatch_is_an_infrastructure_error(self):
        def factory(*args, **kwargs):
            env = self.factory(*args, **kwargs)
            env.initial = "Changed state"
            return env
        task = {"initial_observation": "Task", "steps": []}
        with self.assertRaisesRegex(ValueError, "reset observation"):
            await run_episode(task, self.tokenizer, self.generator([]), env_factory=factory)
        self.assertTrue(self.environments[-1].closed)

    def test_seed_schedule_and_prompt_are_stable(self):
        self.assertEqual(turn_seed(42, 0), turn_seed(42, 0))
        self.assertNotEqual(turn_seed(42, 0), turn_seed(42, 1))
        self.assertNotEqual(turn_seed(42, 0), turn_seed(43, 0))
        self.assertTrue(render_prompt("Task").endswith("Observation: Task\nAction:"))


if __name__ == "__main__":
    unittest.main()
