"""Question sharding preserves complete sample budgets and auditable merging."""
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from math_rft import evaluate, merge_evaluations, reward
from math_rft.analyze import assert_comparable, load_run


@contextmanager
def fake_runtime():
    class Engine:
        def __init__(self, **kwargs):
            pass

        def get_tokenizer(self):
            return SimpleNamespace(encode=lambda text, add_special_tokens: [1, 2])

        def generate(self, prompts, sampling_params, use_tqdm):
            return [SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], outputs=[
                SimpleNamespace(text=r"\boxed{2}", token_ids=[3, 4], finish_reason="stop", stop_reason=None),
            ]) for prompt in prompts]

    vllm = ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = Engine, SimpleNamespace
    identity = {"kind": "huggingface", "repo_id": "fake/base", "revision": "a" * 40}
    with patch.dict(sys.modules, {"vllm": vllm}), \
            patch.object(evaluate, "model_identity", return_value=("fake/base", "a" * 40, identity)), \
            patch.object(evaluate.importlib.metadata, "version", return_value="fake"), \
            patch.object(reward, "score_answer", return_value={"score": 1, "status": "correct", "extracted_answer": "2"}), \
            redirect_stdout(io.StringIO()):
        yield


class EvaluationShardsTests(unittest.TestCase):
    def data(self, root, count=5, formal=False):
        rows = [{"id": f"q{i}", "problem": "1+1", "prompt": "Solve 1+1", "answer": "2",
                 "subject": "algebra", "level": 4, "split": "math500" if formal else "dev",
                 "data_source": "math500" if formal else "math"} for i in range(count)]
        path = root / "data.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path, rows

    def args(self, data, output, *extra):
        return ["--model", "fake/base", "--base-model", "fake/base", "--method", "grpo",
                "--training-seed", "42", "--data", str(data), "--output", str(output),
                "--role", "debug", "--profile", "throughput", "--n", "8", *extra]

    def shards(self, root):
        data, rows = self.data(root)
        shards = [root / f"shard_{i}.jsonl" for i in range(2)]
        with fake_runtime():
            for index, output in enumerate(shards):
                evaluate.main(self.args(data, output, "--num-shards", "2", "--shard-index", str(index)))
        return data, rows, shards

    def rewrite_config(self, path, change):
        meta_path = evaluate.artifact_paths(path)["meta"]
        meta = json.loads(meta_path.read_text())
        change(meta["config"])
        meta["manifest_hash"] = evaluate.canonical_hash(meta["config"])
        meta_path.write_text(json.dumps(meta))

    def test_partition_preserves_ids_seeds_and_complete_full_order_merge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, rows, shards = self.shards(root)
            for index, shard in enumerate(shards):
                run = load_run(shard)
                self.assertEqual(run["rows"], rows[index::2])
                self.assertEqual(run["config"]["sharding"]["full_dataset_sha256"], evaluate.canonical_hash(rows))
            # A resumed ledger can append earlier missing samples at its end.
            original = shards[0].read_bytes().splitlines(keepends=True)
            shards[0].write_bytes(b"".join(reversed(original)))
            merged, single = root / "merged.jsonl", root / "single.jsonl"
            with redirect_stdout(io.StringIO()):
                merge_evaluations.merge(list(reversed(shards)), data, merged)
            with fake_runtime():
                evaluate.main(self.args(data, single))
            self.assertEqual(list(evaluate.read_jsonl(merged)), list(evaluate.read_jsonl(single)))
            run, reference = load_run(merged), load_run(single)
            self.assertEqual(run["rows"], rows)
            self.assertEqual(run["config"]["sharding"], {"count": 2, "scheme": "question_round_robin_v1"})
            self.assertEqual(len(run["meta"]["source_shards"]), 2)
            self.assertEqual(sum(v["n"] for v in run["counts"].values()), len(rows) * 8)
            assert_comparable(run, reference)
            paths = [merged, *[p for k, p in evaluate.artifact_paths(merged).items() if k != "lock"]]
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
            with redirect_stdout(io.StringIO()):
                merge_evaluations.merge(shards, data, merged)
            self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths})

    def test_formal_full_population_is_validated_before_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _ = self.data(root, count=500, formal=True)
            output = root / "formal.jsonl"
            with patch.object(evaluate, "run") as run:
                evaluate.main(self.args(data, output, "--role", "formal", "--n", "256",
                                        "--num-shards", "2", "--shard-index", "1"))
            self.assertEqual(len(run.call_args.args[1]), 250)
            self.assertEqual(len(run.call_args.args[0].sharding["full_question_ids"]), 500)
            data, _ = self.data(root, count=499, formal=True)
            with patch.object(evaluate, "run") as run, self.assertRaisesRegex(ValueError, "all 500"):
                evaluate.main(self.args(data, output, "--role", "formal", "--n", "256", "--num-shards", "2"))
            run.assert_not_called()

    def test_invalid_shard_assignment_and_changed_resume_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _, shards = self.shards(root)
            for extra in (("--num-shards", "0"), ("--num-shards", "2", "--shard-index", "2"),
                          ("--shard-index", "-1"), ("--num-shards", "6")):
                with self.subTest(extra=extra), patch.object(evaluate, "run") as run, self.assertRaises(ValueError):
                    evaluate.main(self.args(data, root / "invalid.jsonl", *extra))
                run.assert_not_called()
            with fake_runtime(), self.assertRaisesRegex(ValueError, "Resume rejected"):
                evaluate.main(self.args(data, shards[0], "--num-shards", "2", "--shard-index", "1"))

    def test_missing_duplicate_active_incomplete_and_incompatible_shards_are_rejected(self):
        for change in ("missing", "duplicate_path", "duplicate_index", "active", "incomplete", "engine", "allocation"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data, rows, shards = self.shards(root)
                if change == "missing":
                    shards = shards[:1]
                elif change == "duplicate_path":
                    shards = [shards[0], shards[0]]
                elif change == "duplicate_index":
                    self.rewrite_config(shards[1], lambda c: c["sharding"].update(index=0))
                elif change == "active":
                    evaluate.artifact_paths(shards[0])["lock"].touch()
                elif change == "incomplete":
                    shards[0].write_bytes(b"".join(shards[0].read_bytes().splitlines(keepends=True)[:-1]))
                elif change == "engine":
                    self.rewrite_config(shards[1], lambda c: c["engine"].update(batch_size=128))
                elif change == "allocation":
                    for i, shard in enumerate(shards):
                        self.rewrite_config(shard, lambda c, i=i: c["sharding"].update(index=1-i))
                with self.assertRaises(ValueError):
                    merge_evaluations.merge(shards, data, root / "merged.jsonl")
                self.assertFalse((root / "merged.jsonl").exists())

    def test_conflicting_outputs_and_changed_source_content_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _, shards = self.shards(root)
            output = root / "merged.jsonl"
            with self.assertRaisesRegex(ValueError, "overwrite"):
                merge_evaluations.merge(shards, data, shards[0])
            with redirect_stdout(io.StringIO()):
                merge_evaluations.merge(shards, data, output)
            records = list(evaluate.read_jsonl(shards[0]))
            records[0]["text"] = "changed while retaining same IDs and score"
            shards[0].write_text("".join(json.dumps(r) + "\n" for r in records))
            with self.assertRaisesRegex(ValueError, "different merge provenance"):
                merge_evaluations.merge(shards, data, output)

    def test_interrupted_merge_with_matching_provenance_can_be_rebuilt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _, shards = self.shards(root)
            output = root / "merged.jsonl"
            with redirect_stdout(io.StringIO()):
                merge_evaluations.merge(shards, data, output)
            original = output.read_bytes()
            output.unlink()
            evaluate.artifact_paths(output)["summary"].unlink()
            with redirect_stdout(io.StringIO()):
                merge_evaluations.merge(shards, data, output)
            self.assertEqual(output.read_bytes(), original)
            self.assertTrue(load_run(output)["summary"]["complete"])
            self.assertFalse(evaluate.artifact_paths(output)["lock"].exists())


if __name__ == "__main__":
    unittest.main()
