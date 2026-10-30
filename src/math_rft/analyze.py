"""Summarize complete ledgers and compare methods with paired uncertainty.

All intervals use questions (and, when supplied, paired training seeds) as the
resampling unit. Multiple generated responses to one question are NOT separate
experimental replicates. No model or GPU dependencies are imported here.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from math_rft.evaluate import (
    artifact_paths, atomic_json, build_summary, canonical_hash, read_existing, read_jsonl,
)
from math_rft.metrics import (
    DEFAULT_KS, describe_gap_curve, hierarchical_delta_ci, mean_ci, paired_delta_ci, pass_at_k,
)


def load_run(path: Path) -> dict[str, Any]:
    paths = artifact_paths(path)
    if not path.is_file() or not paths["meta"].is_file() or not paths["prompts"].is_file():
        raise ValueError(f"A run requires samples, .meta.json and .prompts.jsonl: {path}")
    if paths["lock"].exists():
        raise ValueError(f"Run is still locked; do not analyze a changing ledger: {paths['lock']}")
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    config = meta["config"]
    if canonical_hash(config) != meta.get("manifest_hash"):
        raise ValueError(f"Manifest hash mismatch: {path}")
    prompts = list(read_jsonl(paths["prompts"]))
    rows = [{key: value for key, value in row.items() if key != "prompt_token_ids"} for row in prompts]
    if canonical_hash(rows) != config["dataset_sha256"]:
        raise ValueError(f"Dataset/prompt index hash mismatch: {path}")
    if [row["id"] for row in rows] != config["question_ids"]:
        raise ValueError(f"Prompt index ordering/IDs mismatch: {path}")
    n = config["sampling"]["n"]
    complete, counts = read_existing(path, rows, n, config["sampling"]["seed"])
    if len(complete) != len(rows) * n:
        raise ValueError(f"Incomplete evaluation: {path}; {len(complete)}/{len(rows)*n} samples")
    summary = build_summary(rows, counts, meta["manifest_hash"], n)
    return {"path": str(path.resolve()), "meta": meta, "config": config,
            "rows": rows, "counts": counts, "summary": summary}


def groups(rows: list[dict]) -> dict[str, list[str]]:
    selected = {"all": [row["id"] for row in rows],
                "level_4_5": [row["id"] for row in rows if row["level"] in (4, 5)]}
    for subject in sorted({str(row["subject"]) for row in rows}):
        selected[f"subject/{subject}"] = [row["id"] for row in rows if str(row["subject"]) == subject]
    return selected


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize(args: argparse.Namespace) -> None:
    run = load_run(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(run["summary"])
    result["source"] = run["path"]
    result["config"] = run["config"]
    intervals, csv_rows = {}, []
    n = run["config"]["sampling"]["n"]
    for group_name, ids in groups(run["rows"]).items():
        intervals[group_name] = {}
        if not ids:
            continue
        for k in DEFAULT_KS:
            if k > n:
                continue
            values = [pass_at_k(n, run["counts"][qid]["correct"], k) for qid in ids]
            estimate = mean_ci(values, repeats=args.bootstrap, seed=args.seed)
            intervals[group_name][str(k)] = estimate
            csv_rows.append({"group": group_name, "k": k, "n": n, "questions": len(ids),
                             "pass_at_k": estimate["mean"], "ci95_low": estimate["ci95"][0],
                             "ci95_high": estimate["ci95"][1]})
    result["question_bootstrap"] = intervals
    atomic_json(args.output_dir / "summary.json", result)
    write_csv(args.output_dir / "pass_at_k.csv", csv_rows)
    write_csv(args.output_dir / "per_question.csv", [
        {"id": row["id"], "subject": row["subject"], "level": row["level"],
         "n": n, "correct": run["counts"][row["id"]]["correct"],
         "estimated_p": run["counts"][row["id"]]["correct"] / n,
         **{f"pass@{k}": pass_at_k(n, run["counts"][row["id"]]["correct"], k)
            for k in DEFAULT_KS if k <= n}} for row in run["rows"]
    ])
    print(f"Wrote {args.output_dir / 'summary.json'}")


def assert_comparable(left: dict, right: dict) -> None:
    lc, rc = left["config"], right["config"]
    for key in ("dataset_sha256", "question_ids", "sampling", "grader", "role", "versions"):
        if lc[key] != rc[key]:
            raise ValueError(f"Cannot compare runs with different {key}: {left['path']} vs {right['path']}")
    for key in ("dtype", "max_model_len", "max_prompt_tokens", "generation_config"):
        if lc["engine"][key] != rc["engine"][key]:
            raise ValueError(f"Cannot compare runs with different engine {key}")
    if not lc.get("base_model") or lc["base_model"] != rc.get("base_model"):
        raise ValueError("Comparisons must use the same --base-model; cross-generation differences are not treatment effects")
    ls, rs = lc.get("training_seed"), rc.get("training_seed")
    if ls is not None and rs is not None and ls != rs:
        raise ValueError("Pair the same training seed across methods")
    if (ls is None and lc["method"] != "base") or (rs is None and rc["method"] != "base"):
        raise ValueError("Trained methods require --training-seed provenance")


def pair_seed(left: dict, right: dict) -> int | None:
    ls, rs = left["config"].get("training_seed"), right["config"].get("training_seed")
    return ls if ls is not None else rs


def gap_trends(pairs: list[tuple[dict, dict]], group_ids: dict[str, list[str]], n: int,
               repeats: int, seed: int) -> dict:
    """Describe means and estimate predeclared high-k changes using paired data."""
    ks = [k for k in DEFAULT_KS if k <= n]
    result = {"per_seed": [], "aggregate": {}, "fixed_k_changes": {},
              "interpretation": [
                  "Signed delta is left minus right; absolute gap is the magnitude of that mean delta.",
                  "Peak locations and curve shapes are descriptive, restricted to the reported k grid.",
                  "Fixed-k changes compare deltas on the same questions and training seeds; intervals are pointwise.",
                  "A negative signed change does not necessarily mean the absolute gap shrank.",
              ]}
    for left, right in pairs:
        result["per_seed"].append({"training_seed": pair_seed(left, right), "groups": {}})
    for group, ids in group_ids.items():
        if not ids:
            continue
        differences = {
            k: [[pass_at_k(n, left["counts"][qid]["correct"], k)
                 - pass_at_k(n, right["counts"][qid]["correct"], k) for qid in ids]
                for left, right in pairs] for k in ks
        }
        seed_means = [[statistics.fmean(differences[k][index]) for k in ks] for index in range(len(pairs))]
        for entry, values in zip(result["per_seed"], seed_means, strict=True):
            entry["groups"][group] = describe_gap_curve(ks, values)
        means = [statistics.fmean(values[index] for values in seed_means) for index in range(len(ks))]
        result["aggregate"][group] = describe_gap_curve(ks, means)
        result["fixed_k_changes"][group] = {}
        for low, high in ((64, 128), (128, 256), (64, 256)):
            if low not in differences or high not in differences:
                continue
            changes = [[b - a for a, b in zip(low_values, high_values, strict=True)]
                       for low_values, high_values in zip(differences[low], differences[high], strict=True)]
            estimate = hierarchical_delta_ci(changes, repeats=repeats, seed=seed)
            estimate.update({"lower_k": low, "higher_k": high, "direction": "delta_at_higher_k_minus_delta_at_lower_k"})
            result["fixed_k_changes"][group][f"{high}_minus_{low}"] = estimate
    return result


def compare(args: argparse.Namespace) -> None:
    if len(args.left) != len(args.right) or not args.left:
        raise ValueError("Supply equally many --left and --right paths, ordered by paired training seed")
    pairs = [(load_run(left), load_run(right)) for left, right in zip(args.left, args.right, strict=True)]
    reference_left, reference_right = pairs[0]
    seeds = []
    for left, right in pairs:
        assert_comparable(left, right)
        for candidate, reference in ((left, reference_left), (right, reference_right)):
            for key in ("base_model", "method", "dataset_sha256", "question_ids", "sampling", "grader", "role", "versions", "engine"):
                if candidate["config"][key] != reference["config"][key]:
                    raise ValueError(f"Cross-seed series has inconsistent {key}")
        seeds.append(pair_seed(left, right))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate training seeds would create pseudoreplication")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n = reference_left["config"]["sampling"]["n"]
    group_ids = groups(reference_left["rows"])
    csv_rows, per_seed, aggregate = [], [], {}
    for left, right in pairs:
        seed = pair_seed(left, right)
        entry: dict[str, Any] = {"training_seed": seed, "left": left["path"], "right": right["path"], "groups": {}}
        for group_name, ids in group_ids.items():
            if not ids:
                entry["groups"][group_name] = {"questions": 0}
                continue
            left_success = {qid for qid in ids if left["counts"][qid]["correct"] > 0}
            right_success = {qid for qid in ids if right["counts"][qid]["correct"] > 0}
            group_result = {
                "questions": len(ids), "pass_at_k_delta": {},
                "observed_coverage_at_n": {
                    "n": n, "both": len(left_success & right_success),
                    "left_only": sorted(left_success - right_success),
                    "right_only": sorted(right_success - left_success),
                    "neither": len(ids) - len(left_success | right_success),
                    "note": "Finite-budget observations, not latent capability/support labels.",
                },
            }
            for k in DEFAULT_KS:
                if k > n:
                    continue
                lv = [pass_at_k(n, left["counts"][qid]["correct"], k) for qid in ids]
                rv = [pass_at_k(n, right["counts"][qid]["correct"], k) for qid in ids]
                estimate = paired_delta_ci(lv, rv, repeats=args.bootstrap, seed=args.seed)
                group_result["pass_at_k_delta"][str(k)] = estimate
                csv_rows.append({"training_seed": seed, "group": group_name, "k": k,
                                 "questions": len(ids), "left_mean": estimate["left_mean"],
                                 "right_mean": estimate["right_mean"], "delta_left_minus_right": estimate["mean"],
                                 "ci95_low": estimate["ci95"][0], "ci95_high": estimate["ci95"][1]})
            entry["groups"][group_name] = group_result
        per_seed.append(entry)
    aggregate_rows = []
    for group_name, ids in group_ids.items():
        aggregate[group_name] = {}
        if not ids:
            continue
        for k in DEFAULT_KS:
            if k > n:
                continue
            differences = [[pass_at_k(n, left["counts"][qid]["correct"], k)
                            - pass_at_k(n, right["counts"][qid]["correct"], k) for qid in ids]
                           for left, right in pairs]
            estimate = hierarchical_delta_ci(differences, repeats=args.bootstrap, seed=args.seed)
            estimate["left_mean"] = statistics.fmean(
                pass_at_k(n, left["counts"][qid]["correct"], k) for left, _ in pairs for qid in ids
            )
            estimate["right_mean"] = statistics.fmean(
                pass_at_k(n, right["counts"][qid]["correct"], k) for _, right in pairs for qid in ids
            )
            aggregate[group_name][str(k)] = estimate
            aggregate_rows.append({"group": group_name, "k": k, "training_seeds": len(pairs),
                                   "questions": len(ids), "left_mean": estimate["left_mean"],
                                   "right_mean": estimate["right_mean"], "delta_left_minus_right": estimate["mean"],
                                   "seed_std": estimate["seed_std"],
                                   "ci95_low": estimate["ci95"][0], "ci95_high": estimate["ci95"][1]})
    output = {"direction": "left_minus_right", "base_model": reference_left["config"]["base_model"],
              "dataset_sha256": reference_left["config"]["dataset_sha256"],
              "question_ids": reference_left["config"]["question_ids"],
              "left_method": reference_left["config"]["method"], "right_method": reference_right["config"]["method"],
              "n": n, "paired_training_seeds": seeds, "primary_endpoint": "pass@128",
              "key_secondary_endpoint": "pass@256", "per_seed": per_seed, "aggregate": aggregate,
              "interpretation": [
                  "CI resamples questions and paired training seeds, not individual completions.",
                  "The intervals are percentile bootstrap intervals, not exact fixed-benchmark binomial intervals.",
                  "Subject/level curves and non-primary k values are exploratory; no multiple-testing adjustment is implied.",
                  "At n=k, pass@k is observed coverage; larger n gives a finer per-question estimate at the same k.",
                  "With only a few training seeds, seed uncertainty remains poorly estimated; report each seed.",
              ]}
    trends = gap_trends(pairs, group_ids, n, args.bootstrap, args.seed)
    output["trends"] = trends
    atomic_json(args.output_dir / "comparison.json", output)
    atomic_json(args.output_dir / "trend_summary.json", trends)
    trend_rows = []
    for group, descriptions in trends["aggregate"].items():
        for quantity in ("signed_delta", "absolute_gap"):
            trend_rows.append({"scope": "aggregate", "training_seed": "", "group": group,
                               "quantity": quantity, **descriptions[quantity]})
    for entry in trends["per_seed"]:
        for group, descriptions in entry["groups"].items():
            for quantity in ("signed_delta", "absolute_gap"):
                trend_rows.append({"scope": "per_seed", "training_seed": entry["training_seed"], "group": group,
                                   "quantity": quantity, **descriptions[quantity]})
    write_csv(args.output_dir / "trend_summary.csv", trend_rows)
    write_csv(args.output_dir / "high_k_changes.csv", [
        {"group": group, "lower_k": value["lower_k"], "higher_k": value["higher_k"],
         "delta_change": value["mean"], "ci95_low": value["ci95"][0], "ci95_high": value["ci95"][1]}
        for group, changes in trends["fixed_k_changes"].items() for value in changes.values()
    ])
    write_csv(args.output_dir / "per_seed_deltas.csv", csv_rows)
    write_csv(args.output_dir / "aggregate_deltas.csv", aggregate_rows)
    print(f"Wrote {args.output_dir / 'comparison.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("summarize", help="Summarize a complete sample ledger")
    single.add_argument("--input", type=Path, required=True)
    paired = subparsers.add_parser("compare", help="Compare paired methods; optional multiple paired training seeds")
    paired.add_argument("--left", type=Path, nargs="+", required=True)
    paired.add_argument("--right", type=Path, nargs="+", required=True)
    for command in (single, paired):
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--bootstrap", type=int, default=2000)
        command.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates (default2000)")
    (summarize if args.command == "summarize" else compare)(args)


if __name__ == "__main__":
    main()
