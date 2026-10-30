"""Prepare reproducible MATH/DeepMath splits and verl-compatible parquet.

This command is deliberately explicit: invoking it downloads data, but importing
the package never accesses the network. Source revisions are resolved and saved.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, sha256_file, sha256_text, write_json, write_jsonl
from .prompts import DEMONSTRATIONS, PROMPT_VERSION, RAW_CHAT_TEMPLATE, prompt_fingerprint, render_prompt
from .reward import extract_boxed

MATH_REPO = "EleutherAI/hendrycks_math"
MATH500_REPO = "HuggingFaceH4/MATH-500"
GSM8K_REPO = "openai/gsm8k"
DEEPMATH_REPO = "zwhe99/DeepMath-103K"
MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")
REFERENCE_CORRECTIONS = Path(__file__).resolve().parents[2] / "configs/math_reference_corrections.json"


def problem_key(problem: str) -> str:
    """Conservative exact deduplication; does not claim semantic contamination removal."""
    normalized = unicodedata.normalize("NFKC", problem).replace("\r\n", "\n")
    return sha256_text(re.sub(r"\s+", " ", normalized).strip())


def _text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing/empty field {key!r} in row {row.get('id', row.get('unique_id', '?'))}")
    return value.strip()


def _level(value: Any, *, allow_unknown: bool = False) -> int | None:
    if allow_unknown and str(value).strip() == "Level ?":
        return None
    match = re.fullmatch(r"(?:Level\s*)?([1-5])", str(value).strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid MATH difficulty: {value!r}")
    return int(match.group(1))


@lru_cache(maxsize=1)
def _reference_corrections() -> dict[str, dict[str, str]]:
    entries = json.loads(REFERENCE_CORRECTIONS.read_text(encoding="utf-8"))
    return {entry["problem_sha256"]: entry for entry in entries}


def math_reference_answer(problem: str, solution: str) -> tuple[str | None, dict | None]:
    """Read historical MATH reference formatting without relaxing model grading.

    Only a single unbraced digit followed by a math delimiter is supported;
    arbitrary text and malformed final boxes are never guessed. Empty-box
    source defects require an explicitly audited problem AND solution match.
    """
    answer = extract_boxed(solution)
    if answer:
        return answer, None
    boxes = list(re.finditer(r"\\(?:boxed|fbox)\b", solution))
    if boxes:
        unbraced = re.match(r"\s+([0-9])\s*\$", solution[boxes[-1].end():])
        if unbraced:
            return unbraced[1], {"kind": "unbraced_digit_box", "solution_sha256": sha256_text(solution)}
    correction = _reference_corrections().get(problem_key(problem))
    if correction and sha256_text(solution) == correction["solution_sha256"]:
        return correction["answer"], {"kind": "audited_empty_box", **correction}
    return None, None


def normalize_row(row: dict[str, Any], source: str, source_split: str) -> dict[str, Any]:
    reference_adjustment = None
    if source == "deepmath":
        problem = _text(row, "question")
        answer = _text(row, "final_answer")
        solution = str(row.get("r1_solution_1") or "")
        subject = _text(row, "topic")
        difficulty = float(row["difficulty"])
        if not math.isfinite(difficulty):
            raise ValueError("DeepMath difficulty must be finite")
        level: int | None = None
    elif source == "gsm8k":
        problem = _text(row, "question")
        solution = _text(row, "answer")
        if "####" not in solution:
            raise ValueError("GSM8K reference answer has no #### delimiter")
        answer = solution.rsplit("####", 1)[1].strip().replace(",", "")
        subject, level, difficulty = "arithmetic", None, None
    else:
        problem = _text(row, "problem")
        solution = _text(row, "solution")
        answer = row.get("answer")
        if not answer:
            if source == "math" and source_split == "train":
                answer, reference_adjustment = math_reference_answer(problem, solution)
            else:
                answer = extract_boxed(solution)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"No reference answer extracted: {problem[:100]!r}")
        subject = _text(row, "subject" if "subject" in row else "type")
        level = _level(row["level"], allow_unknown=(source == "math" and source_split == "train"))
        difficulty = None
    key = problem_key(problem)
    original_id = str(row.get("unique_id") or row.get("id") or "")
    if source == "math500" and not original_id.startswith("test/"):
        raise ValueError("MATH-500 must preserve official unique_id beginning with test/")
    return {
        "id": f"{source}:{original_id or key}", "problem": problem,
        "prompt": render_prompt(problem), "answer": answer.strip(), "solution": solution,
        "subject": subject, "level": level, "difficulty": difficulty,
        "split": source_split, "source_split": source_split, "data_source": source,
        "source_id": original_id, "problem_sha256": key,
        "reference_answer_adjustment": reference_adjustment,
    }


def _deduplicate(rows: list[dict[str, Any]], excluded: set[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    seen: dict[str, dict[str, Any]] = {}
    id_to_problem: dict[str, str] = {}
    excluded = excluded or set()
    removed = []
    for row in sorted(rows, key=lambda item: item["id"]):
        key = row["problem_sha256"]
        if row["id"] in id_to_problem and id_to_problem[row["id"]] != key:
            raise ValueError(f"One source ID refers to different problems: {row['id']}")
        id_to_problem[row["id"]] = key
        if key in excluded:
            removed.append({"id": row["id"], "reason": "overlap_with_protected_split"})
        elif key in seen:
            if row["answer"] != seen[key]["answer"]:
                raise ValueError(f"Duplicate problem has conflicting reference answers: {row['id']}")
            removed.append({"id": row["id"], "reason": "duplicate_problem"})
        else:
            seen[key] = row
    return list(seen.values()), removed


def stratified_take(rows: list[dict[str, Any]], count: int, seed: int, salt: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exact-size proportional subject x level allocation, with stable hash ranks."""
    if not 0 < count <= len(rows):
        raise ValueError(f"Cannot select {count} rows from {len(rows)}")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["subject"], str(row["level"]))].append(row)
    keys = sorted(groups)
    targets = {key: count * len(groups[key]) / len(rows) for key in keys}
    allocated = {key: math.floor(targets[key]) for key in keys}
    remainder = count - sum(allocated.values())
    for key in sorted(keys, key=lambda item: (-(targets[item] - allocated[item]), item))[:remainder]:
        allocated[key] += 1
    selected, remaining = [], []
    for key in keys:
        ranked = sorted(groups[key], key=lambda row: sha256_text(f"{seed}:{salt}:{row['id']}"))
        selected.extend(ranked[:allocated[key]])
        remaining.extend(ranked[allocated[key]:])
    order = lambda row: sha256_text(f"{seed}:{salt}:order:{row['id']}")
    return sorted(selected, key=order), sorted(remaining, key=order)


def _load_source(repo: str, config: str | None, split: str, revision: str | None, local_path: str | None, provenance: dict[str, Any], label: str) -> list[dict[str, Any]]:
    if local_path:
        path = Path(local_path).resolve()
        provenance[label] = {"kind": "local_jsonl", "path": str(path), "sha256": sha256_file(path), "asserted_original_split": split}
        return read_jsonl(path)
    from datasets import load_dataset
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(repo, revision=revision or "main").sha
    if not sha:
        raise RuntimeError(f"Cannot resolve an immutable dataset revision: {repo}")
    dataset = load_dataset(repo, config, split=split, revision=sha)
    provenance[label] = {"kind": "huggingface", "repo": repo, "config": config, "split": split, "revision": sha, "fingerprint": dataset._fingerprint}
    return [dict(row) for row in dataset]


def _write_split(output: Path, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from datasets import Dataset

    rows = [{**row, "split": name} for row in rows]
    jsonl = output / f"{name}.jsonl"
    parquet = output / f"{name}.parquet"
    write_jsonl(jsonl, rows)
    training_rows = []
    for index, row in enumerate(rows):
        training_rows.append({
            "data_source": row["data_source"], "prompt": [{"role": "user", "content": row["prompt"]}],
            "ability": "math", "reward_model": {"style": "rule", "ground_truth": row["answer"]},
            "extra_info": {"index": index, "id": row["id"], "split": name, "subject": row["subject"], "level": row["level"], "problem_sha256": row["problem_sha256"]},
        })
    Dataset.from_list(training_rows).to_parquet(str(parquet))
    strata = collections.Counter(f"{row['subject']}|{row['level']}" for row in rows)
    return {"count": len(rows), "ids": [row["id"] for row in rows], "jsonl_sha256": sha256_file(jsonl), "parquet_sha256": sha256_file(parquet), "strata": dict(sorted(strata.items()))}


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty data directory: {output}")
    provenance: dict[str, Any] = {}
    raw_test = _load_source(MATH500_REPO, None, "test", args.math500_revision, args.math500_jsonl, provenance, "math500")
    if len(raw_test) != 500:
        raise ValueError(f"Expected official MATH-500's 500 rows, got {len(raw_test)}")
    test = [normalize_row(row, "math500", "test") for row in raw_test]
    test, removed_test = _deduplicate(test)
    if removed_test or len({row["id"] for row in test}) != 500:
        raise ValueError("MATH-500 contains duplicate problems or IDs; refusing to alter the official evaluation set")
    protected = {row["problem_sha256"] for row in test}
    protected.update(problem_key(question) for question, _ in DEMONSTRATIONS)
    if args.math_train_jsonl:
        raw_train = _load_source(MATH_REPO, None, "train", None, args.math_train_jsonl, provenance, "math_train")
    else:
        raw_train = []
        math_revision = args.math_revision
        for subject in MATH_SUBJECTS:
            label = f"math_train/{subject}"
            raw_train.extend(_load_source(MATH_REPO, subject, "train", math_revision, None, provenance, label))
            math_revision = provenance[label]["revision"]
    if len(raw_train) != 7500:
        raise ValueError(f"Expected original MATH train's 7500 rows, got {len(raw_train)}; do not substitute a merged train/test mirror")
    train = [normalize_row(row, "math", "train") for row in raw_train]
    reference_adjustments = [
        {"id": row["id"], "problem_sha256": row["problem_sha256"], "answer": row["answer"],
         **row["reference_answer_adjustment"]}
        for row in train if row["reference_answer_adjustment"] is not None
    ]
    unknown_training_levels = [row["id"] for row in train if row["level"] is None]
    train, removed_train = _deduplicate(train, protected)
    dev, full = stratified_take(train, args.dev_size, args.seed, "development")
    pilot, _ = stratified_take(full, args.pilot_size, args.seed, "pilot")
    if not ({(row["subject"], row["level"]) for row in full} <= {(row["subject"], row["level"]) for row in pilot}):
        raise ValueError("Pilot selection does not cover every available subject/difficulty stratum")
    raw_gsm = _load_source(GSM8K_REPO, "main", "train", args.gsm8k_revision, args.gsm8k_jsonl, provenance, "gsm8k")
    gsm = [normalize_row(row, "gsm8k", "train") for row in raw_gsm]
    math_keys = {row["problem_sha256"] for row in train} | protected
    gsm, removed_gsm = _deduplicate(gsm, math_keys)
    gsm = sorted(gsm, key=lambda row: sha256_text(f"{args.seed}:gsm8k:{row['id']}"))[:args.debug_size]
    if len(gsm) != args.debug_size:
        raise ValueError("Too few unique GSM8K examples for the requested debug set")
    splits = {"pilot": pilot, "dev": dev, "full": full, "math500": test, "gsm8k_debug": gsm}
    removed = {"math_train": removed_train, "gsm8k": removed_gsm}
    if args.deepmath_size:
        raw_deep = _load_source(DEEPMATH_REPO, None, "train", args.deepmath_revision, args.deepmath_jsonl, provenance, "deepmath")
        deep = [normalize_row(row, "deepmath", "train") for row in raw_deep]
        deep, removed_deep = _deduplicate(deep, math_keys | {row["problem_sha256"] for row in gsm})
        deep = sorted(deep, key=lambda row: sha256_text(f"{args.seed}:deepmath:{row['id']}"))[:args.deepmath_size]
        if len(deep) != args.deepmath_size:
            raise ValueError("Too few non-overlapping DeepMath examples")
        splits[f"deepmath_{args.deepmath_size}"] = deep
        removed["deepmath"] = removed_deep
    # Pilot is a fixed subset of full; both exclude dev and test.
    nested_training = {"pilot", "full"}
    for first, left in splits.items():
        for second, right in splits.items():
            if first >= second or {first, second} <= nested_training:
                continue
            if {row["problem_sha256"] for row in left} & {row["problem_sha256"] for row in right}:
                raise ValueError(f"Data leakage detected: {first} overlaps {second}")
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1, "split_seed": args.seed, "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_fingerprint(), "raw_chat_template": RAW_CHAT_TEMPLATE,
        "source_revisions": provenance, "removed_rows": removed,
        "reference_answer_adjustments": reference_adjustments,
        "reference_corrections_sha256": sha256_file(REFERENCE_CORRECTIONS),
        "unknown_training_levels": unknown_training_levels,
        "deduplication": "NFKC + whitespace-normalized exact problem hash; no semantic-overlap guarantee",
        "full_train_includes_pilot": True, "full_train_excludes_dev": True,
        "sources": {"math": f"https://huggingface.co/datasets/{MATH_REPO}", "math500": f"https://huggingface.co/datasets/{MATH500_REPO}", "deepmath": f"https://huggingface.co/datasets/{DEEPMATH_REPO}"},
        "splits": {},
    }
    for name, rows in splits.items():
        manifest["splits"][name] = _write_split(output, name, rows)
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {', '.join(f'{name}={len(rows)}' for name, rows in splits.items())} in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("prepare")
    command.add_argument("--output-dir", default="data")
    command.add_argument("--seed", type=int, default=20261025)
    command.add_argument("--pilot-size", type=int, default=2048)
    command.add_argument("--dev-size", type=int, default=256)
    command.add_argument("--debug-size", type=int, default=32)
    command.add_argument("--deepmath-size", type=int, choices=(0, 8192, 16384), default=0)
    command.add_argument("--math-train-jsonl", help="Local JSONL containing only the original 7500 MATH train rows")
    for name in ("math500", "gsm8k", "deepmath"):
        command.add_argument(f"--{name}-jsonl")
    for name in ("math", "math500", "gsm8k", "deepmath"):
        command.add_argument(f"--{name}-revision", help="HF commit/tag; resolved SHA is saved in manifest")
    args = parser.parse_args()
    if min(args.pilot_size, args.dev_size, args.debug_size) < 1:
        parser.error("split sizes must be positive")
    if args.deepmath_jsonl and not args.deepmath_size:
        parser.error("--deepmath-jsonl requires --deepmath-size")
    prepare(args)


if __name__ == "__main__":
    main()
