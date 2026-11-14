"""Episode boundaries, terminal scoring, and frozen ALFWorld asset checks."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import Mock, patch

from math_rft.alfworld_env import AlfWorldEnvironment, _BackendLease, validate_task
from math_rft.io_utils import sha256_file


def fixture_task(root: Path) -> dict:
    game, trajectory = root / "game.tw-pddl", root / "traj_data.json"
    game.write_text(json.dumps({"solvable": True, "pddl_problem": "example"}))
    trajectory.write_text(json.dumps({"task_type": "pick_and_place_simple"}))
    return {"game_file": str(game), "game_sha256": sha256_file(game),
            "trajectory_file": str(trajectory), "trajectory_sha256": sha256_file(trajectory),
            "initial_observation": "You are in a room."}


def backend():
    env = Mock()
    env.reset.return_value = {"feedback": "You are in a room.", "won": False,
                              "admissible_commands": ["look", "take apple 1 from table 1"]}
    return env


class AlfWorldEnvironmentTests(unittest.TestCase):
    def test_pool_releases_each_exclusive_lease_once_without_native_unloading(self):
        runtime = backend()
        idle = []
        with patch("math_rft.alfworld_env._IDLE_BACKENDS", idle):
            lease = _BackendLease(runtime)
            lease.close()
            lease.close()
            self.assertEqual(idle, [runtime])
            runtime.close.assert_not_called()

    def test_pool_quarantines_backend_after_step_failure(self):
        runtime = backend()
        runtime.step.side_effect = RuntimeError("planner failed")
        idle = []
        with patch("math_rft.alfworld_env._IDLE_BACKENDS", idle):
            lease = _BackendLease(runtime)
            with self.assertRaisesRegex(RuntimeError, "planner failed"):
                lease.step("look")
            lease.close()
            self.assertEqual(idle, [])

    def test_independent_attempts_create_fresh_backends_and_close(self):
        with tempfile.TemporaryDirectory() as folder:
            task = fixture_task(Path(folder))
            first, second = backend(), backend()
            with patch("math_rft.alfworld_env._start_environment", side_effect=[first, second]) as start:
                with AlfWorldEnvironment(task, seed=42) as env:
                    self.assertEqual(env.reset(), task["initial_observation"])
                    env.reset()
                    first.close.assert_called_once()
                    second.seed.assert_called_once_with(42)
                second.close.assert_called_once()
                self.assertEqual(start.call_count, 2)

    def test_step_limit_is_failure_even_when_backend_reward_is_positive(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = backend()
            runtime.step.return_value = ({"feedback": "Nothing happens.", "won": False}, 0.5, False)
            with patch("math_rft.alfworld_env._start_environment", return_value=runtime):
                with AlfWorldEnvironment(fixture_task(Path(folder)), max_steps=1) as env:
                    env.reset()
                    observation, success, done, info = env.step("nonexistent command")
                    self.assertEqual(observation, "Nothing happens.")
                    self.assertFalse(success)
                    self.assertTrue(done)
                    self.assertTrue(info["truncated"])
                    self.assertTrue(info["invalid_action"])
                    with self.assertRaisesRegex(RuntimeError, "ended"):
                        env.step("look")

    def test_success_at_last_step_takes_precedence_over_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = backend()
            runtime.step.return_value = ({"feedback": "You won!", "won": True}, 1, True)
            with patch("math_rft.alfworld_env._start_environment", return_value=runtime):
                with AlfWorldEnvironment(fixture_task(Path(folder)), max_steps=1) as env:
                    env.reset()
                    _, success, done, info = env.step("look")
                    self.assertTrue(success and done)
                    self.assertFalse(info["truncated"])
                    self.assertTrue(info["terminated"])

    def test_backend_done_is_not_equivalent_to_success(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = backend()
            runtime.step.return_value = ({"feedback": "You lost.", "won": False}, 0, True)
            with patch("math_rft.alfworld_env._start_environment", return_value=runtime):
                with AlfWorldEnvironment(fixture_task(Path(folder))) as env:
                    env.reset()
                    _, success, done, _ = env.step("look")
                    self.assertTrue(done)
                    self.assertFalse(success)

    def test_modified_asset_rejected_before_backend_load(self):
        with tempfile.TemporaryDirectory() as folder:
            task = fixture_task(Path(folder))
            Path(task["game_file"]).write_text("modified")
            with patch("math_rft.alfworld_env._start_environment") as start:
                with self.assertRaisesRegex(ValueError, "changed since preparation"):
                    AlfWorldEnvironment(task).reset()
                start.assert_not_called()

    def test_changed_initial_observation_closes_backend_and_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            task = fixture_task(Path(folder))
            task["initial_observation"] = "Another room"
            runtime = backend()
            with patch("math_rft.alfworld_env._start_environment", return_value=runtime):
                with self.assertRaisesRegex(ValueError, "initial observation"):
                    AlfWorldEnvironment(task).reset()
                runtime.close.assert_called_once()

    def test_upstream_translator_argv_mutation_is_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = backend()
            state = runtime.reset.return_value
            original = sys.argv

            def reset():
                sys.argv = ["translate.py", "domain", "task"]
                return state

            runtime.reset.side_effect = reset
            with patch("math_rft.alfworld_env._start_environment", return_value=runtime):
                with AlfWorldEnvironment(fixture_task(Path(folder))) as env:
                    env.reset()
                    self.assertIs(sys.argv, original)

    def test_backend_error_is_not_converted_into_a_failed_task(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = backend()
            runtime.step.side_effect = OSError("backend failed")
            with patch("math_rft.alfworld_env._start_environment", return_value=runtime):
                with self.assertRaisesRegex(OSError, "backend failed"):
                    with AlfWorldEnvironment(fixture_task(Path(folder))) as env:
                        env.reset()
                        env.step("look")
                runtime.close.assert_called_once()

    def test_missing_success_flag_and_multiline_actions_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = backend()
            runtime.step.return_value = ({"feedback": "Unknown result"}, 1, True)
            with patch("math_rft.alfworld_env._start_environment", return_value=runtime):
                with AlfWorldEnvironment(fixture_task(Path(folder))) as env:
                    env.reset()
                    with self.assertRaises(ValueError):
                        env.step("look\ninventory")
                    runtime.step.assert_not_called()
                    with self.assertRaisesRegex(RuntimeError, "success flag"):
                        env.step("look")

    def test_relative_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            task = fixture_task(Path(folder))
            task["game_file"] = "game.tw-pddl"
            with self.assertRaisesRegex(ValueError, "absolute"):
                validate_task(task)


if __name__ == "__main__":
    unittest.main()
