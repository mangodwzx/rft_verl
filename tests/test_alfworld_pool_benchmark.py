"""Validate comparable benchmark sample coverage without loading a model."""
import argparse
import unittest

from scripts import benchmark_alfworld_pool as benchmark


class PoolBenchmarkTests(unittest.TestCase):
    def test_default_configuration_preserves_thread_process_pair(self):
        options = benchmark.parser().parse_args([])
        self.assertEqual(benchmark.configurations(options), {
            "threads": {"batch_size": 8, "env_workers": 0},
            "processes": {"batch_size": 8, "env_workers": 4},
        })

    def test_mixed_batches_cover_identical_sample_indices_and_rotate(self):
        options = benchmark.parser().parse_args([
            "--process-configs", "8:4", "8:8", "16:8", "--sample-index-start", "231"])
        configs = benchmark.configurations(options)
        for task in range(3):
            batches = list(benchmark.batch_schedule(configs, samples=25, start=231, task_position=task))
            self.assertEqual(batches[0][0], list(configs)[task])
            for mode, config in configs.items():
                indices = [index for name, batch in batches if name == mode for index in batch]
                self.assertEqual(indices, list(range(231, 256)))
                self.assertTrue(all(len(batch) <= config["batch_size"]
                                    for name, batch in batches if name == mode))

    def test_invalid_and_duplicate_configurations_are_rejected(self):
        for value in ("8", "8:4:2", "0:4", "8:-1", "eight:4"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                benchmark.process_config(value)
        options = benchmark.parser().parse_args(["--process-configs", "8:4", "8:4"])
        with self.assertRaisesRegex(ValueError, "distinct"):
            benchmark.configurations(options)

    def test_speedups_use_aggregate_time_and_normalize_for_episode_lengths(self):
        def batch(mode, episodes, seconds, steps):
            return {"mode": mode, "episodes": episodes, "wall_seconds": seconds,
                    "mean_episode_metrics": {"environment_steps": steps}, "successes": 0}

        rows = [batch("baseline", 8, 20, 50), batch("baseline", 8, 40, 50),
                batch("candidate", 16, 30, 25)]
        result = benchmark.summarize(rows, ("baseline", "candidate"))
        self.assertEqual(result["baseline"]["episodes_per_hour"], 960)
        self.assertEqual(result["candidate"]["episodes_per_hour"], 1920)
        self.assertEqual(result["comparisons"]["candidate"]["speedup"], 2)
        self.assertEqual(result["comparisons"]["candidate"]["environment_steps_per_second_speedup"], 1)
        incomplete = benchmark.summarize(rows[:2] + [batch("candidate", 8, 15, 25)],
                                         ("baseline", "candidate"))
        self.assertEqual(incomplete["comparisons"], {})
        legacy = benchmark.summarize([batch("threads", 8, 30, 50), batch("processes", 8, 15, 50)])
        self.assertEqual(legacy["speedup"], 2)
        self.assertEqual(legacy["environment_steps_per_second_speedup"], 2)


if __name__ == "__main__":
    unittest.main()
