"""Partial evaluation imports preserve records and reject protocol drift."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from math_rft import evaluate, evaluation_migration


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "old" / "dev.jsonl"
        self.source.parent.mkdir()
        self.output = self.root / "new" / "dev.jsonl"
        self.rows = [{"id": f"task-{index}", "data_source": "alfworld", "prompt": f"Room {index}",
                      "subject": "pick", "level": None} for index in range(2)]
        self.config = {
            "benchmark": "alfworld", "questions": len(self.rows),
            "question_ids": [row["id"] for row in self.rows], "dataset_sha256": evaluate.canonical_hash(self.rows),
            "sampling": {"n": 3, "seed": 23, "temperature": 1.0},
            "model": {"revision": "a" * 40}, "role": "development",
            "episode_protocol": {"max_steps": 50, "max_action_tokens": 128},
            "grader": {"source_sha256": "b" * 64}, "versions": {"torch": "test"},
            "engine": {"batch_size": 8}, "preparation_manifest_sha256": "c" * 64,
            "evaluator_sha256": "d" * 64, "ledger_source_sha256": "e" * 64,
        }
        self.new_config = deepcopy(self.config)
        self.new_config.update(evaluator_sha256="f" * 64, ledger_source_sha256="0" * 64,
                               environment_execution={"workers": 4, "source_sha256": "1" * 64})
        metadata = {"config": self.config, "manifest_hash": evaluate.canonical_hash(self.config),
                    "created_at_utc": "2026-11-13T01:10:00+00:00"}
        self.source_paths = evaluate.artifact_paths(self.source)
        evaluate.atomic_json(self.source_paths["meta"], metadata)
        self.source_paths["prompts"].write_text("".join(
            json.dumps({**row, "prompt_token_ids": [10, 20 + index]}) + "\n"
            for index, row in enumerate(self.rows)))
        self.source.write_text("".join(json.dumps(self.record(index, sample)) + "\n"
                                       for index, sample in ((0, 0), (0, 1), (1, 0))))
        self.original = {path: path.read_bytes() for path in
                         (self.source, self.source_paths["meta"], self.source_paths["prompts"])}

    def record(self, index, sample):
        qid = self.rows[index]["id"]
        return {"id": qid, "sample_index": sample, "seed": evaluate.sample_seed(23, qid, sample),
                "score": 0, "response_token_ids": [1, 2], "response_mask": [1, 0],
                "turns": [{"observation": "Nothing happens."}], "termination": "max_steps",
                "finish_reason": "length", "grader_status": "max_steps"}

    def migrate(self, config=None):
        return evaluation_migration.migrate_evaluation(
            self.source, self.output, self.rows, self.new_config if config is None else config)

    def assert_source_unchanged(self):
        for path, before in self.original.items():
            self.assertEqual(path.read_bytes(), before)
        self.assertFalse(self.source_paths["lock"].exists())

    def test_preserves_source_bytes_and_records_complete_provenance(self):
        metadata = self.migrate()
        target = evaluate.artifact_paths(self.output)
        self.assertEqual(self.output.read_bytes(), self.original[self.source])
        self.assertEqual(target["prompts"].read_bytes(), self.original[self.source_paths["prompts"]])
        self.assertEqual(metadata["config"], self.new_config)
        provenance = metadata["resume_source"]
        self.assertEqual(provenance["source_metadata"], json.loads(self.original[self.source_paths["meta"]]))
        self.assertEqual(provenance["ledger_sha256"], evaluate.file_hash(self.source))
        self.assertEqual(provenance["records"], 3)
        self.assertEqual(len(evaluate.read_existing(self.output, self.rows, 3, 23)[0]), 3)
        self.assertFalse(target["summary"].exists())
        self.assertFalse(target["lock"].exists())
        self.assert_source_unchanged()

    def test_idempotent_after_new_samples_and_parent_owned_lock(self):
        first = self.migrate()
        with self.output.open("a") as stream:
            stream.write(json.dumps(self.record(1, 1)) + "\n")
        lock = evaluate.artifact_paths(self.output)["lock"]
        lock.write_text(f"pid={os.getpid()}\n")
        before = self.output.read_bytes()
        self.assertEqual(self.migrate(), first)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertTrue(lock.exists())
        self.assert_source_unchanged()

    def test_protocol_engine_seed_and_sharding_changes_rejected(self):
        for key, value in (("episode_protocol", {"max_steps": 49}), ("engine", {"batch_size": 16}),
                           ("sampling", {"n": 3, "seed": 99, "temperature": 1.0}),
                           ("sharding", {"index": 0, "count": 2}), ("model", {"revision": "z" * 40}),
                           ("extra_field", None)):
            with self.subTest(key=key):
                config = {**self.new_config, key: value}
                with self.assertRaisesRegex(ValueError, "configuration"):
                    self.migrate(config)
                self.assertFalse(self.output.exists())
        self.assert_source_unchanged()

    def test_active_source_and_destination_locks_rejected(self):
        self.source_paths["lock"].write_text("pid=123\n")
        with self.assertRaisesRegex(ValueError, "locked"):
            self.migrate()
        self.assertTrue(self.source_paths["lock"].exists())
        self.source_paths["lock"].unlink()
        destination_lock = evaluate.artifact_paths(self.output)["lock"]
        destination_lock.write_text("pid=123\n")
        with self.assertRaisesRegex(ValueError, "locked"):
            self.migrate()
        self.assertTrue(destination_lock.exists())
        self.assert_source_unchanged()

    def test_duplicate_sample_and_wrong_seed_rejected(self):
        with self.source.open("a") as stream:
            stream.write(json.dumps(self.record(0, 0)) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.migrate()
        record = self.record(0, 0)
        record["seed"] += 1
        self.source.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            self.migrate()
        self.assertFalse(self.output.exists())

    def test_source_manifest_and_prompt_tampering_rejected(self):
        metadata = json.loads(self.source_paths["meta"].read_text())
        metadata["config"]["sampling"]["n"] = 99
        evaluate.atomic_json(self.source_paths["meta"], metadata)
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            self.migrate()
        self.source_paths["meta"].write_bytes(self.original[self.source_paths["meta"]])
        prompts = list(evaluate.read_jsonl(self.source_paths["prompts"]))
        prompts[0]["prompt"] = "changed"
        self.source_paths["prompts"].write_text("".join(json.dumps(row) + "\n" for row in prompts))
        with self.assertRaisesRegex(ValueError, "prompt index"):
            self.migrate()

    def test_partial_line_rejected(self):
        self.source.write_bytes(self.original[self.source].rstrip(b"\n"))
        with self.assertRaisesRegex(ValueError, "committed newline"):
            self.migrate()

    def test_source_change_during_copy_rejected(self):
        real_copy = evaluation_migration._copy

        def mutating_copy(source, target):
            real_copy(source, target)
            if source == self.source:
                with self.source.open("a") as stream:
                    stream.write(json.dumps(self.record(1, 1)) + "\n")

        with patch.object(evaluation_migration, "_copy", side_effect=mutating_copy), \
                self.assertRaisesRegex(ValueError, "source changed"):
            self.migrate()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.source_paths["lock"].exists())

    def test_interrupted_publish_recovers_without_dropping_records(self):
        original_replace = Path.replace
        target = evaluate.artifact_paths(self.output)

        def failing_replace(path, destination):
            if Path(destination) == target["prompts"]:
                raise OSError("simulated interrupted publication")
            return original_replace(path, destination)

        with patch.object(Path, "replace", failing_replace), self.assertRaisesRegex(OSError, "interrupted"):
            self.migrate()
        self.assertTrue(self.output.exists())
        self.assertFalse(target["meta"].exists())
        self.assertTrue(Path(str(self.output) + ".migration.json").exists())
        metadata = self.migrate()
        self.assertEqual(metadata["resume_source"]["records"], 3)
        self.assertEqual(self.output.read_bytes(), self.original[self.source])
        self.assertFalse(Path(str(self.output) + ".migration.json").exists())
        self.assert_source_unchanged()

    def test_destination_modification_or_existing_unrelated_file_rejected(self):
        self.output.parent.mkdir()
        self.output.write_text("unrelated\n")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            self.migrate()
        self.output.unlink()
        self.migrate()
        self.output.write_bytes(self.output.read_bytes().replace(b"max_steps", b"bad_steps", 1))
        with self.assertRaisesRegex(ValueError, "modified"):
            self.migrate()
        self.assert_source_unchanged()

    def test_source_destination_alias_rejected(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            evaluation_migration.migrate_evaluation(self.source, self.source, self.rows, self.new_config)


if __name__ == "__main__":
    unittest.main()
