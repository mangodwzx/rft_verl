"""ALFWorld official split integrity and task preparation without GPU work."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from math_rft import alfworld_data as data
from math_rft.io_utils import read_jsonl, sha256_file


def write_task(root: Path, split: str, name: str, task_type: str = "pick_and_place_simple", **game_fields) -> Path:
    folder = root / split / name
    folder.mkdir(parents=True)
    (folder / "traj_data.json").write_text(json.dumps({"task_type": task_type, "private_expert": "hidden_solution"}))
    (folder / "game.tw-pddl").write_text(json.dumps({"solvable": True, "pddl_problem": name, **game_fields}))
    return folder


class FakeEnvironment:
    def __init__(self, task, **kwargs):
        self.task = task

    def reset(self):
        return f"You are in a room. Your task is {self.task['source_id']}."

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class AlfWorldDiscoveryTests(unittest.TestCase):
    def test_text_extraction_excludes_visual_assets_and_private_test_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "release.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("json_2.1.1/train/task/trial/traj_data.json", "{}")
                zipped.writestr("json_2.1.1/train/task/trial/game.tw-pddl", "{}")
                zipped.writestr("detectors/model.pth", "weights")
                zipped.writestr("json_2.1.1/test_unseen/task/trial/traj_data.json", "{}")
            destination = root / "extract"
            result = data._extract_text_assets(archive, destination)
            self.assertEqual(result["files"], 2)
            self.assertFalse((destination / "detectors").exists())
            self.assertFalse((destination / "json_2.1.1/test_unseen").exists())

    def test_download_archive_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "release.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("../../escaped.json", "{}")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                data._extract_text_assets(archive, root / "extract")
            self.assertFalse((root.parent / "escaped.json").exists())

    def test_all_loader_exclusions_are_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_task(root, "train", "ok")
            write_task(root, "train", "AppleSliced")
            write_task(root, "train", "movable")
            write_task(root, "train", "bad_type", task_type="unsupported")
            write_task(root, "train", "unsolvable", solvable=False)
            missing = write_task(root, "train", "missing_game")
            (missing / "game.tw-pddl").unlink()
            missing_flag = write_task(root, "train", "missing_flag")
            (missing_flag / "game.tw-pddl").write_text("{}")
            tasks, excluded = data.discover_tasks(root, "train")
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["source_id"], "train/ok")
            self.assertEqual(len(excluded), 6)
            self.assertEqual({row["reason"] for row in excluded}, {
                "unsupported_movable_or_sliced", "unsupported_task_type", "unsolvable", "missing_game", "missing_solvable_flag"})

    def test_duplicate_games_between_splits_fail_instead_of_silently_dropping(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            one = write_task(root, "train", "one")
            two = write_task(root, "valid_unseen", "two")
            (two / "game.tw-pddl").write_bytes((one / "game.tw-pddl").read_bytes())
            splits = {split: data.discover_tasks(root, split)[0] for split in ("train", "valid_unseen")}
            with self.assertRaisesRegex(ValueError, "overlapping"):
                data.validate_disjoint_tasks(splits)

    def test_actual_root_can_be_download_container_or_split_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "json_2.1.1"
            for split in data.SOURCE_SPLITS:
                (source / split).mkdir(parents=True)
            self.assertEqual(data.resolve_data_root(root), source)
            self.assertEqual(data.resolve_data_root(source), source)
            for split in data.SOURCE_SPLITS:
                (root / split).mkdir()
            with self.assertRaises(ValueError):
                data.resolve_data_root(root)

    def test_pilot_is_deterministic_task_stratified_and_nested(self):
        rows = [{"id": str(i), "subject": str(i % 2)} for i in range(20)]
        left = data.select_pilot(rows, 8, 42)
        right = data.select_pilot(list(reversed(rows)), 8, 42)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 8)
        self.assertEqual(sum(row["subject"] == "0" for row in left), 4)
        self.assertLessEqual({row["id"] for row in left}, {row["id"] for row in rows})
        with self.assertRaises(ValueError):
            data.select_pilot(rows, 21, 42)

    def test_held_out_demonstrations_are_rejected_instead_of_removed(self):
        for source_id in ("valid_seen/example", "valid_unseen/example"):
            with self.subTest(source_id=source_id), \
                    patch.object(data, "DEMONSTRATION_SOURCE_IDS", frozenset({source_id})):
                splits = {split: [{"source_id": f"{split}/example"}] for split in data.SOURCE_SPLITS}
                excluded = {split: [] for split in data.SOURCE_SPLITS}
                with self.assertRaisesRegex(ValueError, "only from train"):
                    data._exclude_demonstration_tasks(splits, excluded)
                self.assertEqual([len(tasks) for tasks in splits.values()], [1, 1, 1])
                self.assertTrue(all(not entries for entries in excluded.values()))


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Requires project data stack")
class AlfWorldPreparationTests(unittest.TestCase):
    def test_demonstration_tasks_are_excluded_before_train_sampling_only(self):
        from math_rft.alfworld_episode import PROMPT_VERSION

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, names in {"train": ("example", "a", "b", "c"),
                                 "valid_seen": ("d", "e"), "valid_unseen": ("f", "g")}.items():
                for name in names:
                    write_task(root / "source", split, name)
            output = root / "prepared"
            args = data.parser().parse_args(["prepare", "--data-root", str(root / "source"),
                                             "--output-dir", str(output), "--pilot-size", "2", "--workers", "1"])
            with patch.object(data, "DEMONSTRATION_SOURCE_IDS", frozenset({"train/example"})), \
                    patch.object(data, "AlfWorldEnvironment", side_effect=FakeEnvironment) as environment, \
                    patch.object(data, "environment_identity", return_value={"test": "fake"}):
                data.prepare(args)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual({name: row["count"] for name, row in manifest["splits"].items()},
                             {"pilot": 2, "full": 3, "debug": 3, "dev": 2, "test": 2})
            for name in ("pilot", "full", "debug"):
                rows = read_jsonl(output / f"{name}.jsonl")
                self.assertNotIn("alfworld:train/example", {row["id"] for row in rows})
                self.assertTrue(all(row["source_split"] == "train" for row in rows))
            self.assertEqual({row["source_id"] for row in read_jsonl(output / "dev.jsonl")},
                             {"valid_seen/d", "valid_seen/e"})
            self.assertEqual({row["source_id"] for row in read_jsonl(output / "test.jsonl")},
                             {"valid_unseen/f", "valid_unseen/g"})
            self.assertNotIn("train/example", {call.args[0]["source_id"] for call in environment.call_args_list})
            self.assertEqual(manifest["excluded"], {
                "train": [{"source_id": "train/example", "reason": "prompt_demonstration"}],
                "valid_seen": [], "valid_unseen": [],
            })
            self.assertEqual(manifest["source_counts"]["train"], {"eligible": 3, "excluded": 1})
            self.assertEqual(manifest["prompt_version"], PROMPT_VERSION)
            self.assertEqual(manifest["demonstrations"], {
                "version": data.DEMONSTRATIONS_VERSION, "source_ids": ["train/example"],
                "provenance_artifact_sha256": data.PROVENANCE_ARTIFACT_SHA256,
            })

    def test_duplicate_demonstration_game_in_heldout_split_is_not_hidden_by_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            demonstration = write_task(root / "source", "train", "example")
            write_task(root / "source", "train", "other")
            heldout = write_task(root / "source", "valid_seen", "copied_example")
            (heldout / "game.tw-pddl").write_bytes((demonstration / "game.tw-pddl").read_bytes())
            write_task(root / "source", "valid_unseen", "test")
            output = root / "prepared"
            args = data.parser().parse_args(["prepare", "--data-root", str(root / "source"),
                                             "--output-dir", str(output), "--pilot-size", "1", "--workers", "1"])
            with patch.object(data, "DEMONSTRATION_SOURCE_IDS", frozenset({"train/example"})), \
                    patch.object(data, "_observe_task") as observe, \
                    self.assertRaisesRegex(ValueError, "overlapping"):
                data.prepare(args)
            observe.assert_not_called()
            self.assertFalse(output.exists())

    def test_preparation_preserves_official_splits_and_private_state_is_not_prompted(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, names in {"train": ("a", "b", "c"), "valid_seen": ("d",), "valid_unseen": ("e",)}.items():
                for name in names:
                    write_task(root / "source", split, name)
            output = root / "prepared"
            args = data.parser().parse_args(["prepare", "--data-root", str(root / "source"),
                                             "--output-dir", str(output), "--pilot-size", "2", "--workers", "1"])
            with patch.object(data, "AlfWorldEnvironment", FakeEnvironment), \
                    patch.object(data, "environment_identity", return_value={"test": "fake"}):
                data.prepare(args)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["benchmark"], "alfworld")
            self.assertEqual({name: row["count"] for name, row in manifest["splits"].items()},
                             {"pilot": 2, "full": 3, "dev": 1, "test": 1, "debug": 3})
            full = read_jsonl(output / "full.jsonl")
            self.assertTrue(all(row["source_split"] == "train" for row in full))
            self.assertTrue(all(row["source_split"] == "valid_unseen" for row in read_jsonl(output / "test.jsonl")))
            self.assertTrue(all("hidden_solution" not in row["prompt"] for row in full))
            parquet = pq.read_table(output / "pilot.parquet").to_pylist()
            self.assertEqual(parquet[0]["ability"], "agent")
            self.assertIn("initial_observation", parquet[0]["extra_info"]["alfworld_task"])
            self.assertEqual(manifest["splits"]["test"]["jsonl_sha256"], sha256_file(output / "test.jsonl"))
            with self.assertRaises(FileExistsError):
                data.prepare(args)

    def test_environment_failure_does_not_leave_partial_prepared_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in data.SOURCE_SPLITS:
                write_task(root / "source", split, split)
            output = root / "prepared"
            args = data.parser().parse_args(["prepare", "--data-root", str(root / "source"),
                                             "--output-dir", str(output), "--pilot-size", "1", "--workers", "1"])
            with patch.object(data, "environment_identity", return_value={}), \
                    patch.object(data, "AlfWorldEnvironment", side_effect=RuntimeError("backend unavailable")), \
                    self.assertRaisesRegex(RuntimeError, "backend unavailable"):
                data.prepare(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
