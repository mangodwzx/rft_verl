"""Compare thread and process environments with one shared vLLM engine.

This writes a separate benchmark report and trajectories, never a formal
evaluation ledger. The same tasks, sample indices and seeds run in both modes.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    cli.add_argument("--revision", default="ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
    cli.add_argument("--data", type=Path, default=Path("data_alfworld/dev.jsonl"))
    cli.add_argument("--task-indices", type=int, nargs="+", default=[80, 100, 139])
    cli.add_argument("--samples-per-task", type=int, default=8)
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument("--env-workers", type=int, default=4)
    cli.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    cli.add_argument("--output", type=Path, default=Path("runs_alfworld_pool_benchmark/report.json"))
    return cli


async def warm_environments(factory, task, count):
    envs = [factory(task, max_steps=50, seed=0) for _ in range(count)]
    try:
        await asyncio.gather(*(asyncio.to_thread(env.reset) for env in envs))
    finally:
        await asyncio.gather(*(asyncio.to_thread(env.close) for env in envs))


def summarize(batches):
    by_mode = {}
    for mode in ("threads", "processes"):
        selected = [batch for batch in batches if batch["mode"] == mode]
        seconds = sum(batch["wall_seconds"] for batch in selected)
        episodes = sum(batch["episodes"] for batch in selected)
        steps = sum(batch["mean_episode_metrics"]["environment_steps"] * batch["episodes"]
                    for batch in selected)
        by_mode[mode] = {
            "batches": len(selected), "episodes": episodes, "wall_seconds": seconds,
            "episodes_per_hour": episodes * 3600 / seconds if seconds else None,
            "environment_steps": steps,
            "environment_steps_per_second": steps / seconds if seconds else None,
            "successes": sum(batch["successes"] for batch in selected),
        }
    baseline, parallel = by_mode["threads"], by_mode["processes"]
    if baseline["episodes"] == parallel["episodes"] and parallel["wall_seconds"]:
        by_mode["speedup"] = baseline["wall_seconds"] / parallel["wall_seconds"]
        if baseline["environment_steps_per_second"]:
            by_mode["environment_steps_per_second_speedup"] = (
                parallel["environment_steps_per_second"] / baseline["environment_steps_per_second"])
    return by_mode


def main():
    options = parser().parse_args()
    if min(options.samples_per_task, options.batch_size, options.env_workers) < 1:
        raise ValueError("Counts must be positive")
    if len(set(options.task_indices)) != len(options.task_indices):
        raise ValueError("Task indices must be distinct")
    if not 0 < options.gpu_memory_utilization < 1:
        raise ValueError("GPU memory utilization must be in (0, 1)")
    output = options.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Benchmark report exists; choose a new path: {output}")
    from math_rft import alfworld_env, alfworld_evaluate, evaluate
    from math_rft.alfworld_pool import EnvironmentPool

    args = evaluate.resolve_execution_options(evaluate.parser().parse_args([
        "--model", options.model, "--revision", options.revision,
        "--benchmark", "alfworld", "--profile", "lowmem", "--role", "development",
        "--data", str(options.data.resolve()), "--output", str(output), "--n", "256",
        "--batch-size", str(options.batch_size),
        "--gpu-memory-utilization", str(options.gpu_memory_utilization),
    ]))
    rows = alfworld_evaluate.load_data(args.data, args.role, args.n)
    if any(index < 0 or index >= len(rows) for index in options.task_indices):
        raise ValueError("Task index is outside the dataset")
    selected = [rows[index] for index in options.task_indices]
    report = {
        "status": "initializing", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Performance sample only; trajectories are excluded from formal evaluation.",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "configuration": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        "dataset_sha256": evaluate.file_hash(args.data),
        "task_indices": options.task_indices, "task_ids": [row["id"] for row in selected],
        "samples_per_task": options.samples_per_task, "env_workers": options.env_workers,
        "source_sha256": {name: evaluate.file_hash(Path(module.__file__)) for name, module in (
            ("evaluate", evaluate), ("alfworld_evaluate", alfworld_evaluate), ("alfworld_env", alfworld_env))},
        "warmup_seconds": {}, "batches": [],
    }
    evaluate.atomic_json(output, report)
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm import LLM, SamplingParams

    try:
        started = time.monotonic()
        model, revision, identity = evaluate.model_identity(args.model, args.revision)
        report["model_identity"] = identity
        engine = LLM(model=model, revision=revision, tokenizer_revision=revision, dtype=args.dtype,
                     tensor_parallel_size=1, max_model_len=args.max_model_len,
                     gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
                     trust_remote_code=False, generation_config="vllm", enable_prefix_caching=True,
                     **evaluate.engine_execution_options(args))
        tokenizer = engine.get_tokenizer()
        selected = [{**row, "prompt_token_ids": tokenizer.encode(row["prompt"], add_special_tokens=False)}
                    for row in selected]
        report["model_initialization_seconds"] = time.monotonic() - started
        generated = {"threads": {}, "processes": {}}
        with EnvironmentPool(options.env_workers) as pool:
            factories = {"threads": alfworld_env.AlfWorldEnvironment, "processes": pool.factory}
            # Preload native mappings and spawn workers outside measured batches.
            for mode in ("threads", "processes"):
                started = time.monotonic()
                asyncio.run(warm_environments(factories[mode], selected[0]["extra_info"]["alfworld_task"],
                                              options.batch_size))
                report["warmup_seconds"][mode] = time.monotonic() - started
            report["status"] = "running"
            evaluate.atomic_json(output, report)
            for task_position, row in enumerate(selected):
                modes = ("threads", "processes") if task_position % 2 == 0 else ("processes", "threads")
                for offset in range(0, options.samples_per_task, options.batch_size):
                    batch = [(row, index) for index in range(
                        offset, min(offset + options.batch_size, options.samples_per_task))]
                    for mode in modes:
                        started = time.monotonic()
                        records = asyncio.run(alfworld_evaluate.generate_batch(
                            batch, engine, tokenizer, SamplingParams, args,
                            env_factory=factories[mode]))
                        duration = time.monotonic() - started
                        record_path = output.with_name(f"{output.stem}.{mode}.jsonl")
                        with record_path.open("a", encoding="utf-8") as stream:
                            for record in records:
                                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                                fingerprint = evaluate.canonical_hash({
                                    key: value for key, value in record.items() if key != "metrics"})
                                generated[mode][(record["id"], record["sample_index"])] = fingerprint
                            stream.flush()
                            os.fsync(stream.fileno())
                        item = {
                            "mode": mode, "task_id": row["id"], "sample_indices": [index for _, index in batch],
                            "seeds": [record["seed"] for record in records],
                            "wall_seconds": duration, "episodes": len(records),
                            "episodes_per_hour": len(records) * 3600 / duration,
                            "successes": sum(record["score"] for record in records),
                            "terminations": dict(Counter(record["termination"] for record in records)),
                            "mean_episode_metrics": {key: sum(record["metrics"].get(key, 0)
                                                               for record in records) / len(records)
                                                     for key in ("wall_seconds", "tool_calls", "generate_sequences",
                                                                 "steps", "environment_steps")},
                        }
                        report["batches"].append(item)
                        report["summary"] = summarize(report["batches"])
                        evaluate.atomic_json(output, report)
                        print(json.dumps(item, ensure_ascii=False), flush=True)
        paired = sorted(set(generated["threads"]) & set(generated["processes"]))
        changed = [key for key in paired if generated["threads"][key] != generated["processes"][key]]
        report["trajectory_comparison"] = {
            "paired_samples": len(paired), "identical_samples": len(paired) - len(changed),
            "different_samples": len(changed), "different_keys": changed,
            "note": "Scheduling changes can change floating-point generation; semantic environment tests are separate.",
        }
        report["status"] = "complete"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        evaluate.atomic_json(output, report)
        print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        evaluate.atomic_json(output, report)
        raise


if __name__ == "__main__":
    main()
