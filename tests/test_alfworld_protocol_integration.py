"""Opt-in CPU regression checks against frozen ALFWorld TextWorld games.

Run with RUN_ALFWORLD_INTEGRATION=1 and this project's alfworld extra installed.
The prepared debug split and its local game assets must already be available;
these checks do not download assets or load model weights.
"""
from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
from importlib import metadata
import io
import json
import os
from pathlib import Path
import re
import unittest

from math_rft.alfworld_env import AlfWorldEnvironment
from math_rft.alfworld_episode import GeneratedAction, INSTRUCTION, run_episode


class CharacterTokenizer:
    """Exercise episode boundaries without loading a model tokenizer."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("The raw action protocol must not inject special tokens")
        return [ord(character) for character in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token) for token in ids if token != self.eos_token_id)


@unittest.skipUnless(os.environ.get("RUN_ALFWORLD_INTEGRATION") == "1",
                     "Set RUN_ALFWORLD_INTEGRATION=1 for real CPU ALFWorld checks")
class AlfWorldProtocolIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for package in ("alfworld", "textworld", "fast-downward-textworld"):
            try:
                metadata.version(package)
            except metadata.PackageNotFoundError as exc:
                raise unittest.SkipTest(f"Requires installed {package}") from exc
        data_path = Path(__file__).resolve().parents[1] / "data_alfworld" / "debug.jsonl"
        if not data_path.is_file():
            raise unittest.SkipTest("Requires prepared data_alfworld/debug.jsonl")

        candidates = []
        for line in data_path.read_text().splitlines():
            row = json.loads(line)
            task = row["extra_info"]["alfworld_task"]
            game_path = Path(task["game_file"])
            if not game_path.is_file() or not Path(task["trajectory_file"]).is_file():
                continue
            walkthrough = json.loads(game_path.read_text()).get("walkthrough", [])
            if walkthrough and re.fullmatch(r"move (.+) to (.+)", walkthrough[-1]):
                candidates.append((task, walkthrough))
        if not candidates:
            raise unittest.SkipTest("Requires a local debug game with a placement walkthrough")
        cls.task, cls.walkthrough = min(candidates, key=lambda item: len(item[1]))
        cls.object_name, cls.location_name = re.fullmatch(
            r"move (.+) to (.+)", cls.walkthrough[-1],
        ).groups()

    def advertised_placement(self):
        # Derive the action from the actual model instruction: an independent
        # hard-coded walkthrough alone would miss a regression in the prompt.
        templates = re.findall(r"(?:move|put) OBJECT (?:to|in/on|in|on) LOCATION", INSTRUCTION)
        self.assertEqual(len(templates), 1, "Advertise one unambiguous placement command")
        return templates[0].replace("OBJECT", self.object_name).replace("LOCATION", self.location_name)

    def test_advertised_command_completes_real_episode_with_positive_reward(self):
        tokenizer = CharacterTokenizer()
        actions = self.walkthrough[:-1] + [self.advertised_placement()]
        pending = iter(actions)

        async def generate(prompt_ids, max_tokens, seed):
            ids = tokenizer.encode(next(pending) + "\n")
            return GeneratedAction(ids, [-0.5] * len(ids), "stop")

        with redirect_stdout(io.StringIO()):
            result = asyncio.run(run_episode(
                self.task, tokenizer, generate, max_steps=len(actions),
                max_prompt_tokens=32768, max_response_tokens=32768,
            ))
        self.assertEqual((result.score, result.termination), (1, "success"))
        self.assertEqual(result.metrics["environment_steps"], len(actions))
        self.assertEqual([turn["action"] for turn in result.turns], actions)
        self.assertTrue(result.turns[-1]["success"])
        self.assertTrue(result.turns[-1]["info"]["won"])
        self.assertFalse(any(turn["info"]["invalid_action"] for turn in result.turns))

    def test_thor_placement_variants_fail_where_advertised_textworld_action_succeeds(self):
        # All probes share the final placement state. Rejected commands must
        # leave it unsolved, while the command taught to the policy must win.
        with redirect_stdout(io.StringIO()), AlfWorldEnvironment(self.task, max_steps=50) as env:
            env.reset()
            for action in self.walkthrough[:-1]:
                _, success, done, info = env.step(action)
                self.assertFalse(success or done)
                self.assertFalse(info["invalid_action"])
            for preposition in ("on", "in", "in/on"):
                action = f"put {self.object_name} {preposition} {self.location_name}"
                with self.subTest(action=action):
                    _, success, done, info = env.step(action)
                    self.assertFalse(success or done)
                    self.assertTrue(info["invalid_action"])
                    self.assertFalse(info["won"])
            _, success, done, info = env.step(self.advertised_placement())
            self.assertTrue(success and done and info["won"])
            self.assertFalse(info["invalid_action"])


if __name__ == "__main__":
    unittest.main()
