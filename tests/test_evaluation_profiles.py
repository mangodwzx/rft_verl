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

    def test_explicit_execution_options_override_profile_defaults(self):
        args = self.options("--profile", "lowmem", "--batch-size", "32", "--gpu-memory-utilization", "0.4",
                            "--max-num-seqs", "8", "--max-num-batched-tokens", "8192", "--no-enforce-eager")
        self.assertEqual((args.batch_size, args.gpu_memory_utilization), (32, 0.4))
        self.assertEqual(evaluate.engine_execution_options(args), {
            "max_num_seqs": 8, "max_num_batched_tokens": 8192,
            "enable_chunked_prefill": True, "enforce_eager": False,
        })

    def test_invalid_engine_budgets_fail_before_reading_data(self):
        for extra in (("--max-num-seqs", "0"), ("--max-num-batched-tokens", "0"),
                      ("--no-enable-chunked-prefill",)):
            with self.subTest(extra=extra), patch.object(evaluate, "load_data") as load:
                with self.assertRaises(ValueError):
                    evaluate.main(["--model", "fake/base", "--data", "unused", "--output", "unused",
                                   "--profile", "lowmem", *extra])
                load.assert_not_called()

    def test_lower_concurrency_keeps_all_256_samples_seeds_and_sampling_parameters(self):
        rows = [{"id": f"q{i}", "problem": "1+1", "prompt": "Solve 1+1", "answer": "2",
                 "subject": "algebra", "level": 4, "split": "dev", "data_source": "math"}
                for i in range(2)]
        records, sampling_configs = {}, {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "dev.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            for profile, batch_size in (("default", 32), ("lowmem", 8)):
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
        self.assertEqual(sampling_configs["default"], sampling_configs["lowmem"])
        self.assertEqual(records["default"], records["lowmem"])


if __name__ == "__main__":
    unittest.main()
