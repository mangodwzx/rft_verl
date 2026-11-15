"""Full ALFWorld episodes using the same runner and action format as training.

The vLLM engine is driven by one batching queue. Environment instances and
histories belong to individual attempts; observations never enter actor loss.
"""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import time

from math_rft import evaluate


def load_data(path: Path, role: str, n: int) -> list[dict]:
    from math_rft.alfworld_env import validate_task
    from math_rft.alfworld_episode import render_prompt

    rows = list(evaluate.read_jsonl(path))
    if not rows:
        raise ValueError("ALFWorld dataset is empty")
    ids = set()
    for row in rows:
        required = {"id", "problem", "prompt", "subject", "level", "split", "data_source", "extra_info"}
        if not required <= row.keys():
            raise ValueError(f"ALFWorld row lacks fields: {sorted(required - row.keys())}")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ValueError("ALFWorld IDs must be distinct nonempty strings")
        ids.add(row["id"])
        if row["data_source"] != "alfworld" or row["level"] is not None:
            raise ValueError("ALFWorld rows must identify their benchmark and have level=null")
        task = row["extra_info"].get("alfworld_task")
        if not isinstance(task, dict):
            raise ValueError("Missing extra_info.alfworld_task")
        if not {"initial_observation", "source_split", "task_type"} <= task.keys():
            raise ValueError("ALFWorld task must record its initial observation, source split and task type")
        validate_task(task)
        if row["prompt"] != render_prompt(task["initial_observation"]):
            raise ValueError("ALFWorld prepared prompt differs from the episode protocol; prepare data again")
        if role == "development" and (row["split"] != "dev" or task["source_split"] != "valid_seen"):
            raise ValueError("ALFWorld development requires dev/valid_seen; never tune on valid_unseen")
        if role == "formal" and (row["split"] != "test" or task["source_split"] != "valid_unseen"):
            raise ValueError("ALFWorld formal evaluation requires test/valid_unseen")
    if role == "formal":
        if n < 256:
            raise ValueError("Formal ALFWorld evaluation requires n>=256; use --role debug for smoke tests")
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Formal ALFWorld evaluation requires the preparation manifest alongside the dataset")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        test = manifest.get("splits", {}).get("test", {})
        if (manifest.get("benchmark") != "alfworld" or test.get("ids") != [row["id"] for row in rows]
                or test.get("count") != len(rows) or test.get("jsonl_sha256") != evaluate.file_hash(path)):
            raise ValueError("Formal ALFWorld evaluation must contain the full prepared test split unchanged")
    return rows


class BatchedGenerator:
    """Coalesce pending turn requests without concurrent access to the engine."""

    def __init__(self, engine, sampling_params, args):
        self.engine = engine
        self.sampling_params = sampling_params
        self.args = args
        self.pending = []
        self.scheduled = False

    async def __call__(self, prompt_ids: list[int], max_tokens: int, turn_seed: int):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.pending.append((list(prompt_ids), max_tokens, turn_seed, future))
        if not self.scheduled:
            self.scheduled = True
            loop.call_soon(self.flush)
        return await future

    def flush(self):
        from math_rft.alfworld_episode import GeneratedAction

        pending, self.pending = self.pending, []
        self.scheduled = False
        pending = [item for item in pending if not item[3].cancelled()]
        if not pending:
            return
        try:
            params = [self.sampling_params(
                n=1, temperature=self.args.temperature, top_p=self.args.top_p, top_k=self.args.top_k,
                seed=seed, max_tokens=budget, repetition_penalty=1.0, presence_penalty=0.0,
                frequency_penalty=0.0, ignore_eos=False, stop=["\n"], include_stop_str_in_output=True,
                skip_special_tokens=False,
            ) for _, budget, seed, _ in pending]
            requests = self.engine.generate(
                [{"prompt_token_ids": tokens} for tokens, _, _, _ in pending],
                sampling_params=params, use_tqdm=False,
            )
            if len(requests) != len(pending):
                raise RuntimeError("vLLM returned the wrong number of ALFWorld action requests")
            results = []
            for (tokens, budget, _, _), request in zip(pending, requests, strict=True):
                if len(request.outputs) != 1 or list(request.prompt_token_ids) != tokens:
                    raise RuntimeError("ALFWorld vLLM request ordering or prompt mismatch")
                result = request.outputs[0]
                if result.finish_reason not in ("stop", "length"):
                    raise RuntimeError(f"ALFWorld action generation failed: {result.finish_reason}")
                if len(result.token_ids) > budget:
                    raise RuntimeError("ALFWorld action generation exceeded its token budget")
                results.append(GeneratedAction(token_ids=list(result.token_ids), finish_reason=result.finish_reason))
            for (_, _, _, future), result in zip(pending, results, strict=True):
                if not future.done():
                    future.set_result(result)
        except Exception as exc:
            for _, _, _, future in pending:
                if not future.done():
                    future.set_exception(exc)


async def generate_batch(batch, engine, tokenizer, sampling_params, args, *, env_factory=None):
    from math_rft.alfworld_episode import run_episode

    generate = BatchedGenerator(engine, sampling_params, args)

    async def episode(row, index):
        seed = evaluate.sample_seed(args.seed, row["id"], index)
        result = await run_episode(
            row["extra_info"]["alfworld_task"], tokenizer, generate, seed=seed,
            max_steps=args.max_steps, max_action_tokens=args.max_action_tokens,
            max_response_tokens=args.max_tokens, max_prompt_tokens=args.max_prompt_tokens,
            env_factory=env_factory,
        )
        if result.prompt_ids != row["prompt_token_ids"]:
            raise ValueError("Environment reset differs from the prepared/tokenized initial observation")
        if result.score not in (0, 1):
            raise RuntimeError("ALFWorld returned a nonbinary success reward")
        truncated = result.termination in {"max_steps", "response_token_limit", "action_token_limit"}
        return {"id": row["id"], "sample_index": index, "seed": seed, "text": result.text,
                "score": int(result.score), "grader_status": result.termination,
                "response_token_ids": result.response_ids, "response_mask": result.response_mask,
                "finish_reason": "length" if truncated else "stop", "stop_reason": result.termination,
                "termination": result.termination, "turns": result.turns, "metrics": result.metrics}

    tasks = [asyncio.create_task(episode(row, index)) for row, index in batch]
    try:
        # Let peers finish cleanup after an ordinary inference/environment
        # error. Cancelling a peer already in its finally block can interrupt
        # close(). Nothing from a failed batch is committed to the ledger.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def run(args, rows: list[dict], paths: dict[str, Path]) -> None:
    from math_rft import alfworld_env, alfworld_episode

    # Check local assets even when a complete ledger is resumed. Model loading
    # and every environment reset also preserve the same immutable task inputs.
    for row in rows:
        alfworld_env.validate_task(row["extra_info"]["alfworld_task"])
    preparation_path = args.data.parent / "manifest.json"
    if not preparation_path.is_file():
        raise ValueError("ALFWorld evaluation requires the dataset preparation manifest")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    environment = alfworld_env.environment_identity()
    if preparation.get("benchmark") != "alfworld" or preparation.get("environment") != environment:
        raise ValueError("ALFWorld runtime differs from the prepared dataset environment")
    if args.profile in {"lowmem", "throughput"}:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm import LLM, SamplingParams

    model, revision, identity = evaluate.model_identity(args.model, args.revision)
    versions = {name: importlib.metadata.version(name) for name in
                ("vllm", "torch", "transformers", "huggingface-hub")}
    sampling = {"n": args.n, "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k, "max_tokens": args.max_tokens, "stop": ["\n"], "ignore_eos": False,
                "include_stop_str_in_output": True, "repetition_penalty": 1.0, "presence_penalty": 0.0,
                "frequency_penalty": 0.0, "add_special_tokens": False,
                "seed_scheme": "sha256_seed_question_id_sample_index_v1_then_episode_turn"}
    execution = evaluate.engine_execution_options(args)
    config = {"schema_version": evaluate.SCHEMA_VERSION, "benchmark": "alfworld", "model": identity,
              "execution_profile": args.profile,
              "base_model": args.base_model or (args.model if identity["kind"] == "huggingface" else None),
              "method": args.method, "training_seed": args.training_seed, "role": args.role,
              "dataset_sha256": evaluate.canonical_hash(rows), "questions": len(rows),
              "question_ids": [row["id"] for row in rows], "sampling": sampling,
              "episode_protocol": {"max_steps": args.max_steps, "max_action_tokens": args.max_action_tokens,
                                   "response_budget_includes_observations": True,
                                   "episode_source_sha256": evaluate.file_hash(Path(alfworld_episode.__file__)),
                                   "prompt_sha256": alfworld_episode.prompt_fingerprint()},
              "engine": {"dtype": args.dtype, "tensor_parallel_size": args.tensor_parallel_size,
                         "max_model_len": args.max_model_len, "max_prompt_tokens": args.max_prompt_tokens,
                         "batch_size": args.batch_size, "gpu_memory_utilization": args.gpu_memory_utilization,
                         "generation_config": "vllm", "enable_prefix_caching": True,
                         "flashinfer_sampler_env": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"), **execution},
              "grader": {"kind": "alfworld_episode_won", "environment": environment,
                         "source_sha256": evaluate.file_hash(Path(alfworld_env.__file__))},
              "preparation_manifest_sha256": evaluate.file_hash(preparation_path),
              "evaluator_sha256": evaluate.file_hash(Path(__file__)),
              "episode_source_sha256": evaluate.file_hash(Path(alfworld_episode.__file__)),
              "ledger_source_sha256": evaluate.file_hash(Path(evaluate.__file__)), "versions": versions}
    if args.training_objective is not None:
        config["training_objective"] = args.training_objective
    if getattr(args, "sharding", None) is not None:
        config["sharding"] = args.sharding
    env_workers = getattr(args, "env_workers", 0)
    if env_workers:
        from math_rft import alfworld_pool
        config["environment_execution"] = {
            "kind": "spawn_process_pool", "workers": env_workers,
            "source_sha256": evaluate.file_hash(Path(alfworld_pool.__file__)),
        }
    manifest_hash = evaluate.canonical_hash(config)
    if getattr(args, "resume_from", None) is not None:
        from math_rft.evaluation_migration import migrate_evaluation
        migrated = migrate_evaluation(args.resume_from, args.output, rows, config)
        print(f"Verified migrated prefix: {migrated['resume_source']['records']} episodes "
              f"from {migrated['resume_source']['path']}", flush=True)
    if paths["meta"].exists():
        old = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if old.get("manifest_hash") != manifest_hash or old.get("config") != config:
            raise ValueError("Resume rejected: ALFWorld model, tasks, episode protocol, environment or code changed")
    elif args.output.exists() or paths["prompts"].exists():
        raise ValueError("Existing ALFWorld samples/prompt index without metadata")
    else:
        evaluate.atomic_json(paths["meta"], {"manifest_hash": manifest_hash, "config": config,
                             "created_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                             "reproducibility_note": "Each attempt resets its own environment. Runtime/hardware changes may affect seeded model generation."})
    completed, counts = evaluate.read_existing(args.output, rows, args.n, args.seed)
    if len(completed) == len(rows) * args.n:
        if not paths["prompts"].exists():
            raise ValueError("Complete ALFWorld ledger is missing its prompt token index")
        evaluate.atomic_json(paths["summary"], evaluate.build_summary(rows, counts, manifest_hash, args.n))
        print(f"Already complete: {args.output}")
        return
    llm = LLM(model=model, revision=revision, tokenizer_revision=revision, dtype=args.dtype,
              tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed, trust_remote_code=False,
              generation_config="vllm", enable_prefix_caching=True, **execution)
    tokenizer = llm.get_tokenizer()
    token_rows = []
    for row in rows:
        tokens = tokenizer.encode(row["prompt"], add_special_tokens=False)
        if not tokens or len(tokens) > args.max_prompt_tokens:
            raise ValueError(f"{row['id']}: ALFWorld initial prompt exceeds budget; no task is dropped")
        token_rows.append({**row, "prompt_token_ids": tokens})
    if paths["prompts"].exists():
        if list(evaluate.read_jsonl(paths["prompts"])) != token_rows:
            raise ValueError("Resume rejected: ALFWorld prompt tokenization changed")
    else:
        temporary = paths["prompts"].with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in token_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(paths["prompts"])
    tasks = ((row, index) for row in token_rows for index in range(args.n) if (row["id"], index) not in completed)
    pool_context = alfworld_pool.EnvironmentPool(env_workers) if env_workers else nullcontext()
    with pool_context as pool, args.output.open("a", encoding="utf-8", newline="\n") as stream:
        while batch := list(itertools.islice(tasks, args.batch_size)):
            started = time.monotonic()
            records = asyncio.run(generate_batch(batch, llm, tokenizer, SamplingParams, args,
                                                 env_factory=pool.factory if pool is not None else None))
            # An infrastructure exception leaves this whole batch uncommitted.
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                evaluate.add_count(counts[record["id"]], record)
                completed.add((record["id"], record["sample_index"]))
            stream.flush()
            os.fsync(stream.fileno())
            elapsed = time.monotonic() - started
            print(f"{len(completed)}/{len(rows) * args.n} ALFWorld episodes committed "
                  f"(batch={len(records)}, seconds={elapsed:.2f}, "
                  f"episodes/hour={len(records) * 3600 / elapsed:.1f}, env_workers={env_workers})", flush=True)
    evaluate.atomic_json(paths["summary"], evaluate.build_summary(rows, counts, manifest_hash, args.n))
