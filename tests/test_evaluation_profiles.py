"""Exercise the complete ledger loop with a CPU fake engine at n=256."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from math_rft import evaluate, reward


def fake_engine(calls, initializations, parameters):
    class FakeLLM:
        def __init__(self, **kwargs):
            initializations.append((kwargs, os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")))

        def get_tokenizer(self):
            return SimpleNamespace(encode=lambda text, add_special_tokens: [1, 2])

        def generate(self, prompts, sampling_params, use_tqdm):
            calls.append(len(prompts))
            parameters.extend(sampling_params)
            return [SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], outputs=[
                SimpleNamespace(text=r"\boxed{2}", token_ids=[3, 4], finish_reason="stop", stop_reason=None),
            ]) for prompt in prompts]

    return FakeLLM


class EvaluationProfileTests(unittest.TestCase):
    def options(self, *extra):
        return evaluate.resolve_execution_options(evaluate.parser().parse_args([
            "--model", "fake/base", "--data", "unused.jsonl", "--output", "unused-out.jsonl", *extra,
        ]))

    def test_lowmem_changes_only_execution_defaults(self):
        default = vars(self.options())
        lowmem = vars(self.options("--profile", "lowmem"))
        changed = {key for key in default if default[key] != lowmem[key]}
        self.assertEqual(changed, {"profile", "batch_size", "gpu_memory_utilization", "max_num_seqs",
                                   "max_num_batched_tokens", "enable_chunked_prefill", "enforce_eager"})
        self.assertEqual(lowmem["n"], 256)
        self.assertEqual((lowmem["max_tokens"], lowmem["max_prompt_tokens"], lowmem["max_model_len"]),
                         (4096, 4096, 8192))
        self.assertEqual(evaluate.engine_execution_options(self.options()), {})

    def test_throughput_changes_only_execution_defaults(self):
        default = vars(self.options())
        throughput = vars(self.options("--profile", "throughput"))
        changed = {key for key in default if default[key] != throughput[key]}
        self.assertEqual(changed, {"profile", "batch_size", "gpu_memory_utilization", "max_num_seqs",
                                   "max_num_batched_tokens", "enable_chunked_prefill", "enforce_eager"})
        self.assertEqual(throughput["n"], 256)
        self.assertEqual((throughput["max_tokens"], throughput["max_prompt_tokens"], throughput["max_model_len"]),
                         (4096, 4096, 8192))
        self.assertGreater(throughput["batch_size"], throughput["max_num_seqs"])

    def test_explicit_execution_options_override_profile_defaults(self):
        for profile in ("lowmem", "throughput"):
            with self.subTest(profile=profile):
                args = self.options("--profile", profile, "--batch-size", "32", "--gpu-memory-utilization", "0.4",
                                    "--max-num-seqs", "8", "--max-num-batched-tokens", "8192", "--no-enforce-eager")
                self.assertEqual((args.batch_size, args.gpu_memory_utilization), (32, 0.4))
                self.assertEqual(evaluate.engine_execution_options(args), {
                    "max_num_seqs": 8, "max_num_batched_tokens": 8192,
                    "enable_chunked_prefill": True, "enforce_eager": False,
                })
        args = self.options("--profile", "throughput", "--enforce-eager", "--no-enable-chunked-prefill")
        self.assertTrue(args.enforce_eager)
        self.assertFalse(args.enable_chunked_prefill)

    def test_invalid_engine_budgets_fail_before_reading_data(self):
        for extra in (("--max-num-seqs", "0"), ("--max-num-batched-tokens", "0"),
                      ("--no-enable-chunked-prefill",)):
            with self.subTest(extra=extra), patch.object(evaluate, "load_data") as load:
                with self.assertRaises(ValueError):
                    evaluate.main(["--model", "fake/base", "--data", "unused", "--output", "unused",
                                   "--profile", "lowmem", *extra])
                load.assert_not_called()

    def test_invalid_training_objective_fails_before_reading_data(self):
        for extra in (("--training-loss-normalization", "fixed"),
                      ("--training-loss-normalization", "fixed", "--training-loss-scale-factor", "0"),
                      ("--training-loss-normalization", "per_response", "--training-loss-scale-factor", "4096"),
                      ("--training-loss-scale-factor", "4096"),
                      ("--method", "base", "--training-loss-normalization", "per_response")):
            with self.subTest(extra=extra), patch.object(evaluate, "load_data") as load:
                with self.assertRaises(ValueError):
                    evaluate.main(["--model", "fake/base", "--data", "unused", "--output", "unused",
                                   "--method", "rft", *extra])
                load.assert_not_called()

    def test_training_objective_is_recorded_without_changing_sampling_or_relabeling_old_runs(self):
        rows = [{"id": "q0", "problem": "1+1", "prompt": "Solve 1+1", "answer": "2",
                 "subject": "algebra", "level": 4, "split": "dev", "data_source": "math"}]
        calls, initializations, parameters = [], [], []
        fake_vllm = ModuleType("vllm")
        fake_vllm.LLM = fake_engine(calls, initializations, parameters)
        fake_vllm.SamplingParams = SimpleNamespace
        identity = {"kind": "local", "path": "fake/trained", "sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(sys.modules, {"vllm": fake_vllm}), \
                patch.object(evaluate, "model_identity", return_value=("fake/trained", None, identity)), \
                patch.object(evaluate.importlib.metadata, "version", return_value="fake-version"), \
                patch.object(reward, "score_answer", return_value={
                    "score": 1, "status": "correct", "extracted_answer": "2",
                }), redirect_stdout(io.StringIO()):
            root = Path(temporary)
            data = root / "dev.jsonl"
            data.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            records = []
            for normalization, factor in ((None, None), ("per_response", None), ("fixed", 4096)):
                output = root / f"{normalization}.jsonl"
                arguments = ["--model", "fake/trained", "--data", str(data), "--output", str(output),
                             "--role", "development", "--method", "rft", "--training-seed", "42", "--n", "2"]
                if normalization is not None:
                    arguments += ["--training-loss-normalization", normalization]
                if factor is not None:
                    arguments += ["--training-loss-scale-factor", str(factor)]
                evaluate.main(arguments)
                meta = json.loads(evaluate.artifact_paths(output)["meta"].read_text())["config"]
                if normalization is None:
                    self.assertNotIn("training_objective", meta)
                else:
                    self.assertEqual(meta["training_objective"], {
                        "loss_normalization": normalization, "loss_scale_factor": factor,
                    })
                records.append(output.read_bytes())
                before = len(initializations)
                evaluate.main(arguments)  # An identical objective can resume the complete ledger.
                self.assertEqual(len(initializations), before)
                with self.assertRaisesRegex(ValueError, "Resume rejected"):
                    evaluate.main([*arguments, "--training-loss-normalization", "fixed",
                                   "--training-loss-scale-factor", "2048"])
                self.assertEqual(len(initializations), before)
                self.assertEqual(output.read_bytes(), records[-1])
            self.assertEqual(records, [records[0]] * 3)

    def test_profiles_keep_all_256_samples_seeds_and_sampling_parameters(self):
        rows = [{"id": f"q{i}", "problem": "1+1", "prompt": "Solve 1+1", "answer": "2",
                 "subject": "algebra", "level": 4, "split": "dev", "data_source": "math"}
                for i in range(2)]
        records, sampling_configs = {}, {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "dev.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            for profile, batch_size in (("default", 32), ("lowmem", 8), ("throughput", 256)):
                calls, initializations, parameters = [], [], []
                fake_vllm = ModuleType("vllm")
                fake_vllm.LLM = fake_engine(calls, initializations, parameters)
                fake_vllm.SamplingParams = SimpleNamespace
                output = root / profile / "samples.jsonl"
                identity = {"kind": "huggingface", "repo_id": "fake/base", "revision": "a" * 40}
                with patch.dict(sys.modules, {"vllm": fake_vllm}), \
                        patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "1"}), \
                        patch.object(evaluate, "model_identity", return_value=("fake/base", "a" * 40, identity)), \
                        patch.object(evaluate.importlib.metadata, "version", return_value="fake-version"), \
                        patch.object(reward, "score_answer", return_value={
                            "score": 1, "status": "correct", "extracted_answer": "2",
                        }), redirect_stdout(io.StringIO()):
                    evaluate.main(["--model", "fake/base", "--data", str(data), "--output", str(output),
                                   "--role", "development", "--profile", profile])
                records[profile] = list(evaluate.read_jsonl(output))
                self.assertEqual(len(records[profile]), 512)
                self.assertEqual(calls, [batch_size] * (512 // batch_size))
                self.assertEqual(len(initializations), 1)
                engine, sampler_env = initializations[0]
                self.assertEqual(engine["max_model_len"], 8192)
                if profile == "lowmem":
                    self.assertEqual(sampler_env, "0")
                    self.assertEqual(engine["gpu_memory_utilization"], 0.3)
                    self.assertEqual(engine["max_num_seqs"], 16)
                    self.assertEqual(engine["max_num_batched_tokens"], 2048)
                    self.assertTrue(engine["enable_chunked_prefill"])
                    self.assertTrue(engine["enforce_eager"])
                elif profile == "throughput":
                    self.assertEqual(sampler_env, "0")
                    self.assertEqual(engine["gpu_memory_utilization"], 0.6)
                    self.assertEqual(engine["max_num_seqs"], 64)
                    self.assertEqual(engine["max_num_batched_tokens"], 8192)
                    self.assertTrue(engine["enable_chunked_prefill"])
                    self.assertFalse(engine["enforce_eager"])
                else:
                    self.assertEqual(sampler_env, "1")
                    self.assertEqual(engine["gpu_memory_utilization"], 0.9)
                    self.assertNotIn("max_num_seqs", engine)
                for parameter in parameters:
                    self.assertEqual((parameter.n, parameter.max_tokens, parameter.temperature,
                                      parameter.top_p, parameter.top_k), (1, 4096, 1.0, 1.0, -1))
                    self.assertFalse(parameter.ignore_eos)
                    self.assertEqual(parameter.stop, [])
                paths = evaluate.artifact_paths(output)
                meta = json.loads(paths["meta"].read_text())["config"]
                sampling_configs[profile] = meta["sampling"]
                self.assertEqual(meta["execution_profile"], profile)
                self.assertEqual(meta["engine"]["batch_size"], batch_size)
                self.assertEqual(meta["engine"]["flashinfer_sampler_env"], sampler_env)
                complete, _ = evaluate.read_existing(output, rows, 256, 20261025)
                self.assertEqual(len(complete), 512)
                summary = json.loads(paths["summary"].read_text())
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["ks"][-1], 256)
                self.assertEqual(summary["groups"]["all"]["pass_at_k"]["256"], 1.0)
                self.assertFalse(paths["lock"].exists())
        for profile in ("lowmem", "throughput"):
            self.assertEqual(sampling_configs["default"], sampling_configs[profile])
            self.assertEqual(records["default"], records[profile])

    def test_throughput_resumes_missing_samples_but_rejects_changed_execution_config(self):
        rows = [{"id": f"q{i}", "problem": "1+1", "prompt": "Solve 1+1", "answer": "2",
                 "subject": "algebra", "level": 4, "split": "dev", "data_source": "math"}
                for i in range(2)]
        calls, initializations, parameters = [], [], []
        fake_vllm = ModuleType("vllm")
        fake_vllm.LLM = fake_engine(calls, initializations, parameters)
        fake_vllm.SamplingParams = SimpleNamespace
        identity = {"kind": "huggingface", "repo_id": "fake/base", "revision": "a" * 40}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(sys.modules, {"vllm": fake_vllm}), \
                patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "1"}), \
                patch.object(evaluate, "model_identity", return_value=("fake/base", "a" * 40, identity)), \
                patch.object(evaluate.importlib.metadata, "version", return_value="fake-version"), \
                patch.object(reward, "score_answer", return_value={
                    "score": 1, "status": "correct", "extracted_answer": "2",
                }), redirect_stdout(io.StringIO()):
            root = Path(temporary)
            data, output = root / "dev.jsonl", root / "samples.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            arguments = ["--model", "fake/base", "--data", str(data), "--output", str(output),
                         "--role", "development", "--profile", "throughput"]
            evaluate.main(arguments)
            expected = output.read_bytes()
            paths = evaluate.artifact_paths(output)
            # Recreate a clean interruption after the first committed batch.
            output.write_bytes(b"".join(expected.splitlines(keepends=True)[:256]))
            paths["summary"].unlink()
            calls.clear()
            evaluate.main(arguments)
            self.assertEqual(calls, [256])
            self.assertEqual(output.read_bytes(), expected)
            self.assertTrue(json.loads(paths["summary"].read_text())["complete"])

            snapshots = {path: path.read_bytes() for name, path in paths.items() if name != "lock"}
            before = len(initializations)
            for extra in (("--batch-size", "128"), ("--enforce-eager",),
                          ("--gpu-memory-utilization", "0.5"), ("--max-num-seqs", "32")):
                with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "Resume rejected"):
                    evaluate.main([*arguments, *extra])
                self.assertEqual(len(initializations), before)
                self.assertEqual(output.read_bytes(), expected)
                self.assertFalse(paths["lock"].exists())
                for path, content in snapshots.items():
                    self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
