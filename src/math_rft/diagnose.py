"""Inspect saved training/evaluation behavior without loading models or using GPUs.

Character lengths come from JSONL text. Token lengths and gradient norms, when
available, come from evaluation token IDs and optional TensorBoard scalars.
This describes behavior; it does not establish why a training run changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

from math_rft.metrics import DEFAULT_KS, pass_at_k


def records(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("score") not in (0, 1):
                raise ValueError(f"Expected a record with binary score: {path}:{line_number}")
            yield row


def mean(values):
    return statistics.mean(values) if values else None


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rollout_metrics(path: Path, method: str, split: str) -> dict:
    rows = list(records(path))
    if not rows:
        raise ValueError(f"Empty rollout file: {path}")
    lengths = [len(row["output"]) for row in rows]
    correct = [len(row["output"]) for row in rows if row["score"] == 1]
    incorrect = [len(row["output"]) for row in rows if row["score"] == 0]
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["input"]].append(row["score"])
    return {
        "method": method, "split": split, "step": int(path.stem), "samples": len(rows),
        "questions": len(by_question), "samples_per_question_min": min(map(len, by_question.values())),
        "samples_per_question_max": max(map(len, by_question.values())),
        "mean_correctness": mean([row["score"] for row in rows]),
        "observed_question_coverage": mean([int(any(scores)) for scores in by_question.values()]),
        "mean_response_chars": mean(lengths), "median_response_chars": statistics.median(lengths),
        "correct_mean_response_chars": mean(correct), "incorrect_mean_response_chars": mean(incorrect),
        "response_chars_le32_fraction": mean([int(length <= 32) for length in lengths]),
        "source": str(path.resolve()),
    }


def tensorboard_metrics(directory: Path) -> dict[int, dict]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = accumulator.Tags()["scalars"]
    result = defaultdict(dict)
    tags = {
        "response_length/mean": "mean_response_tokens",
        "actor/grad_norm": "pre_clip_grad_norm",
        "actor/pg_clipfrac": "policy_clip_fraction",
        "actor/lr": "learning_rate",
        "actor/pg_loss": "policy_loss",
    }
    for tag, field in tags.items():
        if tag in available:
            for event in accumulator.Scalars(tag):
                result[event.step][field] = event.value
    return result


def evaluation_metrics(path: Path, method: str) -> dict:
    meta_path = path.with_name(path.name + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    config = meta.get("config", {})
    expected_n = config.get("sampling", {}).get("n")
    questions = defaultdict(lambda: {"n": 0, "correct": 0, "texts": Counter(), "tokens": 0, "indices": set()})
    lengths, grades, finishes = [], Counter(), Counter()
    for row in records(path):
        question = questions[row["id"]]
        index = row["sample_index"]
        if type(index) is not int or index < 0 or index in question["indices"]:
            raise ValueError(f"Invalid or duplicate sample index in {path}: {row['id']}, {index}")
        question["indices"].add(index)
        question["n"] += 1
        question["correct"] += int(row["score"])
        question["texts"][text_hash(row["text"])] += 1
        length = len(row["response_token_ids"])
        question["tokens"] += length
        lengths.append(length)
        grades[row.get("grader_status", "unknown")] += 1
        finishes[row.get("finish_reason", "unknown")] += 1
    if not questions:
        raise ValueError(f"Empty evaluation file: {path}")
    sizes = {q["n"] for q in questions.values()}
    if len(sizes) != 1 or (expected_n is not None and sizes != {expected_n}):
        raise ValueError(f"Incomplete or unequal evaluation sample counts: {path}")
    n = next(iter(sizes))
    if any(q["indices"] != set(range(n)) for q in questions.values()):
        raise ValueError(f"Missing evaluation sample indices: {path}")
    if "question_ids" in config and set(config["question_ids"]) != set(questions):
        raise ValueError(f"Evaluation question set differs from metadata: {path}")
    ks = [k for k in DEFAULT_KS if k <= n]
    per_question = [{
        "id": key, "samples": q["n"], "correct": q["correct"],
        "unique_texts": len(q["texts"]), "mean_response_tokens": q["tokens"] / q["n"],
        "most_frequent_text_fraction": max(q["texts"].values()) / q["n"],
    } for key, q in sorted(questions.items())]
    return {
        "method": method, "source": str(path.resolve()), "samples": len(lengths),
        "questions": len(questions), "n": n,
        "pass_at_k": {str(k): mean([pass_at_k(n, q["correct"], k) for q in questions.values()]) for k in ks},
        "covered_questions": sum(q["correct"] > 0 for q in questions.values()),
        "mean_response_tokens": mean(lengths), "median_response_tokens": statistics.median(lengths),
        "response_tokens_le32_fraction": mean([int(length <= 32) for length in lengths]),
        "mean_unique_texts_per_question": mean([q["unique_texts"] for q in per_question]),
        "mean_most_frequent_text_fraction": mean([q["most_frequent_text_fraction"] for q in per_question]),
        "grader_status_counts": dict(grades), "finish_reason_counts": dict(finishes),
        "protocol": {key: config.get(key) for key in
                     ("dataset_sha256", "sampling", "execution_profile", "evaluator_sha256", "grader")},
        "per_question": per_question,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_diagnostics(training: list[dict], evaluations: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir()
    colors = {"base": "#777777", "rft": "#d55e00", "grpo": "#0072b2"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})

    def save(figure, name):
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7), layout="constrained")
    for method in sorted({row["method"] for row in training}):
        train = [r for r in training if r["method"] == method and r["split"] == "train"]
        dev = [r for r in training if r["method"] == method and r["split"] == "development"]
        token = [r for r in train if "mean_response_tokens" in r]
        source = token or train
        field = "mean_response_tokens" if token else "mean_response_chars"
        unit = "tokens" if token else "characters"
        axes[0, 0].plot([r["step"] for r in source], [r[field] for r in source],
                        color=colors.get(method), label=f"{method.upper()} ({unit})")
        axes[0, 1].plot([r["step"] for r in dev], [100 * r["mean_correctness"] for r in dev],
                        marker="o", color=colors.get(method), label=method.upper())
        axes[1, 0].plot([r["step"] for r in train], [100 * r["response_chars_le32_fraction"] for r in train],
                        color=colors.get(method), label=method.upper())
        axes[1, 1].plot([r["step"] for r in train], [r["correct_mean_response_chars"] for r in train],
                        color=colors.get(method), label=f"{method.upper()} correct")
        axes[1, 1].plot([r["step"] for r in train], [r["incorrect_mean_response_chars"] for r in train],
                        linestyle="--", color=colors.get(method), label=f"{method.upper()} incorrect")
    titles = [("Training response length", "Mean length (unit in legend)"),
              ("Development mean correctness (not pass@16)", "Correct samples (%)"),
              ("Training short responses", "Responses with <=32 characters (%)"),
              ("Correct / incorrect response length", "Mean characters")]
    for axis, (title, ylabel) in zip(axes.flat, titles, strict=True):
        axis.set(title=title, xlabel="Training step", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    save(figure, "training_behavior")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for result in evaluations:
        ks = sorted(map(int, result["pass_at_k"]))
        axes[0].plot(ks, [100 * result["pass_at_k"][str(k)] for k in ks], marker="o",
                     color=colors.get(result["method"]), label=result["method"].upper())
    axes[0].set_xscale("log", base=2)
    axes[0].set(xlabel="Attempts k", ylabel="Estimated pass@k (%)", title="Saved evaluation curves")
    axes[0].legend()
    labels = [r["method"].upper() for r in evaluations]
    bar_colors = [colors.get(r["method"], "#777777") for r in evaluations]
    for axis, field, title in [(axes[1], "mean_response_tokens", "Mean response tokens"),
                                (axes[2], "mean_unique_texts_per_question", "Mean unique texts / question")]:
        axis.bar(labels, [r[field] for r in evaluations], color=bar_colors)
        axis.set_title(title)
    profiles = ", ".join(f"{r['method']}: {r['protocol']['execution_profile'] or 'unspecified'}" for r in evaluations)
    figure.suptitle(f"Descriptive comparison of saved results; execution profiles: {profiles}", fontsize=10)
    save(figure, "evaluation_behavior")


def diagnose(args: argparse.Namespace) -> dict:
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite a non-empty output path: {output}")
    training, evaluations = [], []
    for method in args.methods:
        run = args.run_root / args.model / method / args.stage / f"seed_{args.seed}"
        tb = tensorboard_metrics(args.tensorboard_root / f"{args.model}_{method}_{args.stage}_s{args.seed}") \
            if args.tensorboard_root else {}
        for directory, split in (("rollouts", "train"), ("dev_rollouts", "development")):
            for path in sorted((run / directory).glob("*.jsonl"), key=lambda p: int(p.stem)):
                row = rollout_metrics(path, method, split)
                if split == "train":
                    row.update(tb.get(row["step"], {}))
                training.append(row)
        path = args.eval_root / args.model / method / args.stage / f"seed_{args.seed}" / "dev.jsonl"
        evaluations.append(evaluation_metrics(path, method))
    if args.baseline:
        evaluations.insert(0, evaluation_metrics(args.baseline, "base"))
    reference = evaluations[0]
    ids = [r["id"] for r in reference["per_question"]]
    if any(r["n"] != reference["n"] or [q["id"] for q in r["per_question"]] != ids for r in evaluations):
        raise ValueError("Evaluation comparison requires identical question IDs and sample budgets")
    coverage = []
    for result in evaluations[1:]:
        pairs = list(zip(reference["per_question"], result["per_question"], strict=True))
        coverage.append({"left": reference["method"], "right": result["method"],
                         "both": sum(a["correct"] > 0 and b["correct"] > 0 for a, b in pairs),
                         "left_only": sum(a["correct"] > 0 and b["correct"] == 0 for a, b in pairs),
                         "right_only": sum(a["correct"] == 0 and b["correct"] > 0 for a, b in pairs),
                         "neither": sum(a["correct"] == 0 and b["correct"] == 0 for a, b in pairs)})
    training_totals = []
    for method in args.methods:
        rows = [r for r in training if r["method"] == method and r["split"] == "train"]
        tokens_available = bool(rows) and all("mean_response_tokens" in r for r in rows)
        training_totals.append({
            "method": method, "steps": len(rows), "samples": sum(r["samples"] for r in rows),
            "generated_response_tokens": round(sum(r["mean_response_tokens"] * r["samples"] for r in rows))
                if tokens_available else None,
            "policy_clip_fraction_min": min((r["policy_clip_fraction"] for r in rows
                                              if "policy_clip_fraction" in r), default=None),
            "policy_clip_fraction_max": max((r["policy_clip_fraction"] for r in rows
                                              if "policy_clip_fraction" in r), default=None),
        })
    report = {"schema_version": 1, "model": args.model, "stage": args.stage, "training_seed": args.seed,
              "notes": ["Read-only analysis of saved samples; no generation or rescoring.",
                        "Character lengths and token lengths are distinct units.",
                        "Development mean correctness averages samples; it is not pass@16.",
                        "Length and reward associations are descriptive, not causal evidence.",
                        "Execution profiles and evaluator versions may differ; see each evaluation protocol.",
                        "Exact text uniqueness is not semantic reasoning diversity."],
              "training": training, "training_totals": training_totals,
              "evaluations": evaluations, "paired_observed_coverage": coverage}
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output / "training_metrics.csv", training)
    flat = []
    for result in evaluations:
        row = {k: v for k, v in result.items() if not isinstance(v, (dict, list))}
        row.update({f"pass_at_{k}": value for k, value in result["pass_at_k"].items()})
        row["execution_profile"] = result["protocol"]["execution_profile"]
        flat.append(row)
    write_csv(output / "evaluation_metrics.csv", flat)
    write_csv(output / "paired_observed_coverage.csv", coverage)
    if not args.no_plots:
        plot_diagnostics(training, evaluations, output / "figures")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", default="pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=["rft", "grpo"])
    parser.add_argument("--tensorboard-root", type=Path, help="Optional directory containing per-run event directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    diagnose(args)
    print(f"Wrote offline diagnostics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
