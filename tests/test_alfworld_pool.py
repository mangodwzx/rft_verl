"""Exercise real spawned workers without importing a model or TextWorld."""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import unittest

from math_rft.alfworld_pool import EnvironmentPool


class FakeEnvironment:
    """Top-level factory remains importable in a fresh spawned interpreter."""

    def __init__(self, task, *, max_steps, seed):
        self.task, self.max_steps, self.seed = task, max_steps, seed
        self.steps = 0

    def reset(self):
        self.steps = 0
        if self.task.get("fail_reset"):
            raise RuntimeError("reset failed in worker")
        return self.task["name"]

    def step(self, action):
        if action == "fail":
            raise RuntimeError("step failed in worker")
        started = time.monotonic()
        if action == "wait":
            time.sleep(0.25)
        self.steps += 1
        return self.task["name"], False, self.steps == self.max_steps, {
            "steps": self.steps, "seed": self.seed, "pid": os.getpid(),
            "start_method": multiprocessing.get_start_method(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started": started, "finished": time.monotonic(),
        }

    def close(self):
        if self.task.get("closed_path"):
            Path(self.task["closed_path"]).write_text(str(os.getpid()))


class EnvironmentPoolTests(unittest.TestCase):
    def test_worker_affinity_state_isolation_and_reuse(self):
        parent_pid = os.getpid()
        with EnvironmentPool(2, env_factory=FakeEnvironment) as pool:
            first = pool.factory({"name": "first"}, max_steps=3, seed=7)
            second = pool.factory({"name": "second"}, max_steps=2, seed=11)
            third = pool.factory({"name": "third"}, max_steps=2, seed=13)
            for env, name in [(first, "first"), (second, "second"), (third, "third")]:
                self.assertEqual(env.reset(), name)
            a, b, c = first.step("look"), second.step("look"), third.step("look")
            self.assertNotEqual(a[3]["pid"], parent_pid)
            self.assertNotEqual(a[3]["pid"], b[3]["pid"])
            self.assertEqual(a[3]["pid"], c[3]["pid"])
            self.assertEqual(a[3]["start_method"], "spawn")
            self.assertEqual(a[3]["cuda_visible_devices"], "")
            self.assertEqual(first.step("look")[3]["steps"], 2)
            self.assertEqual(third.step("look")[3]["steps"], 2)
            self.assertEqual(a[3]["seed"], 7)
            first.reset()
            self.assertEqual(first.step("look")[3]["steps"], 1)
            first.close()
            first.close()
            with self.assertRaisesRegex(RuntimeError, "lease is closed"):
                first.reset()
            later = pool.factory({"name": "later"}, max_steps=1, seed=0)
            later.reset()
            self.assertEqual(later.step("look")[3]["pid"], b[3]["pid"])

    def test_environment_calls_overlap_across_processes(self):
        with EnvironmentPool(2, env_factory=FakeEnvironment) as pool:
            envs = [pool.factory({"name": str(i)}) for i in range(2)]
            for env in envs:
                env.reset()  # Pay spawn startup before measuring execution.
            with ThreadPoolExecutor(max_workers=2) as threads:
                results = list(threads.map(lambda env: env.step("wait")[3], envs))
            self.assertLess(max(r["started"] for r in results), min(r["finished"] for r in results))

    def test_errors_propagate_and_shutdown_closes_every_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = EnvironmentPool(2, env_factory=FakeEnvironment)
            one = pool.factory({"name": "one", "closed_path": str(root / "one")})
            two = pool.factory({"name": "two", "closed_path": str(root / "two"), "fail_reset": True})
            one.reset()
            with self.assertRaisesRegex(RuntimeError, "reset failed in worker"):
                two.reset()
            with self.assertRaisesRegex(RuntimeError, "step failed in worker"):
                one.step("fail")
            pool.close()
            pool.close()
            one.close()
            two.close()
            self.assertTrue((root / "one").is_file())
            self.assertTrue((root / "two").is_file())
            pids = {int((root / name).read_text()) for name in ["one", "two"]}
            self.assertFalse(pids & {child.pid for child in multiprocessing.active_children()})
            with self.assertRaisesRegex(RuntimeError, "pool is closed"):
                pool.factory({"name": "late"})

    def test_context_exception_still_cleans_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            closed = Path(temporary) / "closed"
            with self.assertRaisesRegex(ValueError, "parent failed"):
                with EnvironmentPool(1, env_factory=FakeEnvironment) as pool:
                    env = pool.factory({"name": "one", "closed_path": str(closed)})
                    env.reset()
                    raise ValueError("parent failed")
            self.assertTrue(closed.is_file())

    def test_worker_count_validation(self):
        for workers in [0, -1, True, 1.5]:
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                EnvironmentPool(workers)


if __name__ == "__main__":
    unittest.main()
