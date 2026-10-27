"""Independent raw-completion sampling with an auditable, resumable ledger.

Nothing executes at import time. The CLI is intended for a Linux vLLM runtime.
vLLM API references: https://docs.vllm.ai/en/latest/api/vllm/entrypoints/llm/
and https://docs.vllm.ai/en/latest/api/vllm/sampling_params/ .
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from math_rft.metrics import DEFAULT_KS, pass_curve

SCHEMA_VERSION = 1


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank record in {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}; a crashed write may leave a partial "
                    "final line. Back up the file and remove only that incomplete line before resuming."
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object in {path}:{line_number}")
            yield record


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def artifact_paths(output: Path) -> dict[str, Path]:
    return {name: output.with_name(output.name + suffix) for name, suffix in (
        ("meta", ".meta.json"), ("prompts", ".prompts.jsonl"),
        ("summary", ".summary.json"), ("lock", ".lock"),
    )}


def sample_seed(seed: int, question_id: str, index: int) -> int:
    # Different samples have different RNG streams. The same experiment seed
    # aligns streams across methods without sharing responses or filtering draws.
    key = canonical_hash([seed, question_id, index])
    return int(key[:15], 16)


def model_identity(model: str, revision: str | None) -> tuple[str, str | None, dict]:
    local = Path(model).expanduser()
    if local.is_dir():
        if revision is not None:
            raise ValueError("--revision applies only to Hugging Face repositories")
        local = local.resolve()
        if not (local / "config.json").is_file():
            raise ValueError("Local evaluation requires an EXPORTED Hugging Face model with config.json")
        files = sorted(p for p in local.rglob("*") if p.is_file() and p.suffix in {
            ".json", ".safetensors", ".bin", ".model", ".txt", ".py", ".tiktoken"
        })
        if not any(p.suffix in {".safetensors", ".bin"} for p in files):
            raise ValueError("Local model contains no Hugging Face weight files; export the verl checkpoint first")
        hashes = {p.relative_to(local).as_posix(): file_hash(p) for p in files}
        return str(local), None, {"kind": "local", "path": str(local),
                                  "files": hashes, "sha256": canonical_hash(hashes)}
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo_id=model, revision=revision)
    if not info.sha:
        raise RuntimeError("Could not resolve immutable model revision")
    return model, info.sha, {"kind": "huggingface", "repo_id": model, "revision": info.sha}


def load_data(path: Path, role: str, n: int) -> list[dict]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError("Dataset is empty")
    seen: set[str] = set()
    required = ("id", "problem", "prompt", "answer", "subject", "level", "split", "data_source")
    for row in rows:
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Data row lacks fields: {missing}")
        if not isinstance(row["id"], str) or row["id"] in seen:
            raise ValueError("Dataset IDs must be unique strings")
        seen.add(row["id"])
        for field in ("problem", "prompt", "answer"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"{row['id']}: {field} must be nonempty text")
        if role in {"development", "calibration", "probe"} and (
            row["split"] in {"test", "math500"} or "math500" in str(row["data_source"]).lower().replace("-", "")
        ):
            raise ValueError("Development/calibration/probe samples must not use the formal test set")
    if role == "formal":
        if len(rows) != 500 or n < 256:
            raise ValueError("Formal evaluation requires all 500 MATH-500 questions and n>=256 (default1024)")
        if not all("math500" in (str(r["data_source"]) + str(r["split"])).lower().replace("-", "") for r in rows):
            raise ValueError("Formal evaluation requires the prepared MATH-500 split")
    return rows


def read_existing(output: Path, rows: list[dict], n: int, seed: int) -> tuple[set[tuple[str, int]], dict]:
    indexed = {row["id"]: row for row in rows}
    completed: set[tuple[str, int]] = set()
    counts = {key: {"n": 0, "correct": 0, "generated_tokens": 0,
                    "truncated": 0, "grader_statuses": Counter()} for key in indexed}
    if not output.exists():
        return completed, counts
    for record in read_jsonl(output):
        question_id, index = record.get("id"), record.get("sample_index")
        if question_id not in indexed or type(index) is not int or not 0 <= index < n:
            raise ValueError("Existing sample has an unknown question or invalid sample index")
        key = (question_id, index)
        if key in completed:
            raise ValueError(f"Duplicate sample in ledger: {key}")
        if record.get("seed") != sample_seed(seed, question_id, index):
            raise ValueError(f"Sample RNG seed mismatch: {key}")
        if record.get("score") not in (0, 1):
            raise ValueError(f"Non-binary reward: {key}")
        if not isinstance(record.get("response_token_ids"), list):
            raise ValueError(f"Missing response token IDs: {key}")
        completed.add(key)
        add_count(counts[question_id], record)
    return completed, counts


def add_count(count: dict, sample: dict) -> None:
    count["n"] += 1
    count["correct"] += int(sample["score"])
    count["generated_tokens"] += len(sample["response_token_ids"])
    count["truncated"] += int(sample["finish_reason"] == "length")
    count["grader_statuses"][sample["grader_status"]] += 1


def build_summary(rows: list[dict], counts: dict, manifest_hash: str, n: int) -> dict:
    ks = [k for k in DEFAULT_KS if k <= n]
    per_question = []
    for row in rows:
        count = counts[row["id"]]
        if count["n"] != n:
            raise ValueError(f"Incomplete question {row['id']}: {count['n']}/{n}")
        per_question.append({"id": row["id"], "subject": row["subject"], "level": row["level"],
                             **count, "pass_at_k": pass_curve(n, count["correct"], ks)})
    groups = {}
    for name, selected in (
        ("all", per_question),
        ("level_4_5", [r for r in per_question if r["level"] in (4, 5)]),
    ):
        groups[name] = {
            "questions": len(selected),
            "pass_at_k": {str(k): sum(r["pass_at_k"][str(k)] for r in selected) / len(selected)
                          for k in ks} if selected else {},
            "observed_coverage_at_n": sum(r["correct"] > 0 for r in selected) / len(selected) if selected else None,
            "generated_tokens": sum(r["generated_tokens"] for r in selected),
            "truncation_rate": sum(r["truncated"] for r in selected) / (len(selected) * n) if selected else None,
        }
    return {"schema_version": SCHEMA_VERSION, "manifest_hash": manifest_hash,
            "complete": True, "n": n, "ks": ks, "groups": groups, "questions": per_question,
            "primary_endpoint": "pass@128", "key_secondary_endpoint": "pass@256",
            "note": "Finite samples estimate success probability; zero observed successes do not establish impossibility."}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True)
    result.add_argument("--revision")
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--role", choices=("formal", "development", "calibration", "probe", "debug"), default="formal")
    result.add_argument("--n", type=int, default=1024)
    result.add_argument("--seed", type=int, default=20261025)
    result.add_argument("--training-seed", type=int)
    result.add_argument("--method", default="base")
    result.add_argument("--base-model", help="Original Base model ID; required when comparing trained checkpoints")
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--top-p", type=float, default=1.0)
    result.add_argument("--top-k", type=int, default=-1)
    result.add_argument("--max-tokens", type=int, default=4096)
    result.add_argument("--max-model-len", type=int, default=8192)
    result.add_argument("--max-prompt-tokens", type=int, default=4096)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--tensor-parallel-size", type=int, default=1)
    result.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    result.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    result.add_argument("--grader-timeout", type=int, default=5)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if min(args.n, args.max_tokens, args.max_prompt_tokens, args.batch_size,
           args.tensor_parallel_size, args.grader_timeout) < 1:
        raise ValueError("Counts and token budgets must be positive")
    if args.seed < 0 or (args.training_seed is not None and args.training_seed < 0):
        raise ValueError("Evaluation and training seeds must be nonnegative")
    if not math.isfinite(args.temperature) or not args.temperature > 0 or not 0 < args.top_p <= 1 or args.top_k == 0 or args.top_k < -1:
        raise ValueError("IID stochastic evaluation requires temperature>0, 0<top_p<=1, top_k=-1 or positive")
    if args.max_model_len < args.max_prompt_tokens + args.max_tokens:
        raise ValueError("Context must accommodate maximum prompt plus response; silent truncation is forbidden")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be in (0,1)")
    rows = load_data(args.data, args.role, args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(args.output)
    try:
        lock = paths["lock"].open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Evaluation lock exists: {paths['lock']}. If its process has ended, remove only this lock and resume.") from exc
    try:
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        run(args, rows, paths)
    finally:
        lock.close()
        paths["lock"].unlink()


def run(args: argparse.Namespace, rows: list[dict], paths: dict[str, Path]) -> None:
    # Deliberately lazy: importing or statically checking this package never
    # imports CUDA libraries, downloads models, or starts an inference process.
    from vllm import LLM, SamplingParams
    from math_rft import reward

    model, revision, identity = model_identity(args.model, args.revision)
    versions = {name: importlib.metadata.version(name) for name in
                ("vllm", "torch", "transformers", "math-verify", "huggingface-hub")}
    sampling = {"n": args.n, "seed": args.seed, "temperature": args.temperature,
                "top_p": args.top_p, "top_k": args.top_k, "max_tokens": args.max_tokens,
                "stop": [], "ignore_eos": False, "repetition_penalty": 1.0,
                "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "seed_scheme": "sha256_seed_question_id_sample_index_v1", "add_special_tokens": False}
    config = {"schema_version": SCHEMA_VERSION, "model": identity,
              "base_model": args.base_model or (args.model if identity["kind"] == "huggingface" else None),
              "method": args.method, "training_seed": args.training_seed, "role": args.role,
              "dataset_sha256": canonical_hash(rows), "questions": len(rows),
              "question_ids": [r["id"] for r in rows], "sampling": sampling,
              "engine": {"dtype": args.dtype, "tensor_parallel_size": args.tensor_parallel_size,
                         "max_model_len": args.max_model_len, "max_prompt_tokens": args.max_prompt_tokens,
                         "batch_size": args.batch_size, "gpu_memory_utilization": args.gpu_memory_utilization,
                         "generation_config": "vllm", "enable_prefix_caching": True},
              "grader": {"source_sha256": file_hash(Path(reward.__file__)), "timeout_seconds": args.grader_timeout},
              "evaluator_sha256": file_hash(Path(__file__)), "versions": versions}
    manifest_hash = canonical_hash(config)
    if paths["meta"].exists():
        old = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if old.get("manifest_hash") != manifest_hash or old.get("config") != config:
            raise ValueError("Resume rejected: checkpoint, dataset, decoding, package versions or code changed. Use a new output path.")
    elif args.output.exists() or paths["prompts"].exists():
        raise ValueError("Existing samples/prompt index without metadata; refuse to guess provenance")
    else:
        atomic_json(paths["meta"], {"manifest_hash": manifest_hash, "config": config,
                                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                   "platform": platform.platform(),
                                   "reproducibility_note": "Seeded sampling is reproducible only within compatible hardware/kernel/runtime scheduling; bitwise equality across systems is not promised."})
    completed, counts = read_existing(args.output, rows, args.n, args.seed)
    if len(completed) == len(rows) * args.n:
        if not paths["prompts"].exists():
            raise ValueError("Complete sample ledger is missing its prompt token index")
        atomic_json(paths["summary"], build_summary(rows, counts, manifest_hash, args.n))
        print(f"Already complete: {args.output}")
        return
    llm = LLM(model=model, revision=revision, tokenizer_revision=revision,
              dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              seed=args.seed, trust_remote_code=False, generation_config="vllm",
              enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    token_rows = []
    for row in rows:
        tokens = tokenizer.encode(row["prompt"], add_special_tokens=False)
        if not tokens or len(tokens) > args.max_prompt_tokens:
            raise ValueError(f"{row['id']}: prompt length {len(tokens)} exceeds configured budget; no question is dropped")
        token_rows.append({**row, "prompt_token_ids": tokens})
    if paths["prompts"].exists():
        if list(read_jsonl(paths["prompts"])) != token_rows:
            raise ValueError("Resume rejected: prompt tokenization differs from recorded token IDs")
    else:
        temporary = paths["prompts"].with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in token_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(paths["prompts"])
    tasks = ((row, index) for row in token_rows for index in range(args.n)
             if (row["id"], index) not in completed)
    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        while batch := list(itertools.islice(tasks, args.batch_size)):
            seeds = [sample_seed(args.seed, row["id"], index) for row, index in batch]
            params = [SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                                     top_k=args.top_k, seed=seed, max_tokens=args.max_tokens,
                                     repetition_penalty=1.0, presence_penalty=0.0,
                                     frequency_penalty=0.0, ignore_eos=False, stop=[],
                                     skip_special_tokens=True) for seed in seeds]
            generated = llm.generate([{"prompt_token_ids": row["prompt_token_ids"]} for row, _ in batch],
                                     sampling_params=params, use_tqdm=False)
            if len(generated) != len(batch):
                raise RuntimeError("vLLM returned an unexpected number of requests")
            records = []
            for (row, index), request, seed in zip(batch, generated, seeds, strict=True):
                if len(request.outputs) != 1 or list(request.prompt_token_ids) != row["prompt_token_ids"]:
                    raise RuntimeError("vLLM request ordering or output count mismatch")
                completion = request.outputs[0]
                if completion.finish_reason not in ("stop", "length"):
                    raise RuntimeError(f"Generation did not complete normally: {completion.finish_reason}")
                score = reward.score_answer(completion.text, row["answer"], timeout_seconds=args.grader_timeout)
                if score["score"] not in (0, 1):
                    raise RuntimeError("Grader returned a nonbinary reward")
                record = {"id": row["id"], "sample_index": index, "seed": seed,
                          "text": completion.text, "score": int(score["score"]),
                          "grader_status": score["status"], "extracted_answer": score["extracted_answer"],
                          # Exact generated IDs are authoritative for gradient
                          # diagnostics; do not retokenize decoded text (which
                          # omits special tokens such as a generated EOS).
                          "response_token_ids": list(completion.token_ids),
                          "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason}
                records.append(record)
            # A grading/inference infrastructure exception aborts the batch; it
            # is never silently written as an incorrect answer.
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                add_count(counts[record["id"]], record)
                completed.add((record["id"], record["sample_index"]))
            stream.flush()
            os.fsync(stream.fileno())
            print(f"{len(completed)}/{len(rows) * args.n} samples committed", flush=True)
    atomic_json(paths["summary"], build_summary(rows, counts, manifest_hash, args.n))


if __name__ == "__main__":
    main()
