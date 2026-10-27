"""Render saved evaluation summaries to standalone PNG/PDF figures.

This command never loads a model or generates samples. Install the optional
``plots`` dependency group, then explicitly invoke the CLI after evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("complete") or not isinstance(data.get("groups"), dict):
        raise ValueError(f"Expected a completed evaluation .summary.json: {path}")
    for name in ("all", "level_4_5"):
        group = data["groups"].get(name)
        if group is None:
            raise ValueError(f"Summary lacks {name}: {path}")
        for k, score in group["pass_at_k"].items():
            if int(k) < 1 or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid pass@k score in {path}")
    return data


def save_figure(figure, output: Path, name: str) -> None:
    figure.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight")
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")


def setup_axis(axis, ks: list[int], title: str, ylabel: str) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(ks, labels=[str(k) for k in ks])
    axis.set_xlabel("Number of independent attempts k")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="major", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.summaries) != len(args.labels) or len(set(args.labels)) != len(args.labels):
        parser.error("Provide one distinct --labels entry per --summaries file")
    summaries = [read_summary(path) for path in args.summaries]
    # In particular, do not overlay a complete MATH-500 run and a pilot subset.
    reference_questions = [(row["id"], row["subject"], row["level"]) for row in summaries[0]["questions"]]
    for summary in summaries[1:]:
        questions = [(row["id"], row["subject"], row["level"]) for row in summary["questions"]]
        if questions != reference_questions:
            raise ValueError("Overlaid curves must use the identical ordered question set")
        if summary["n"] != summaries[0]["n"] or summary["ks"] != summaries[0]["ks"]:
            raise ValueError("Overlaid curves must share n and the k grid")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "pdf.fonttype": 42, "ps.fonttype": 42})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group_name, group_label in (("all", "All questions"), ("level_4_5", "MATH level 4–5")):
        if not summaries[0]["groups"][group_name]["questions"]:
            continue
        figure, axis = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
        ks = sorted(int(k) for k in summaries[0]["groups"][group_name]["pass_at_k"])
        for label, summary in zip(args.labels, summaries, strict=True):
            values = summary["groups"][group_name]["pass_at_k"]
            axis.plot(ks, [100 * values[str(k)] for k in ks], marker="o", markersize=4,
                      linewidth=1.7, label=label)
        count = summaries[0]["groups"][group_name]["questions"]
        setup_axis(axis, ks, f"{group_label} · {count} questions · n={summaries[0]['n']}", "Estimated pass@k (%)")
        axis.set_ylim(0, 100)
        axis.legend(frameon=False, loc="best")
        save_figure(figure, args.output_dir, f"pass_at_k_{group_name}")
        plt.close(figure)
    if args.comparison:
        comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
        if comparison.get("direction") != "left_minus_right":
            raise ValueError("Comparison must explicitly define left_minus_right")
        if comparison["n"] != summaries[0]["n"]:
            raise ValueError("Comparison n differs from plotted summaries")
        if comparison.get("question_ids") != [row["id"] for row in summaries[0]["questions"]]:
            raise ValueError("Comparison question IDs differ from plotted summaries")
        for group_name, group_label in (("all", "All questions"), ("level_4_5", "MATH level 4–5")):
            estimates = comparison["aggregate"].get(group_name, {})
            if not estimates:
                continue
            ks = sorted(int(k) for k in estimates)
            means, lows, highs = [], [], []
            for k in ks:
                estimate = estimates[str(k)]
                mean, (low, high) = estimate["mean"], estimate["ci95"]
                if not all(math.isfinite(value) for value in (mean, low, high)) or low > high:
                    raise ValueError("Comparison contains invalid confidence intervals")
                if estimate["questions_per_seed"] != summaries[0]["groups"][group_name]["questions"]:
                    raise ValueError("Comparison question count differs from plotted summaries")
                means.append(100 * mean)
                lows.append(100 * low)
                highs.append(100 * high)
            figure, axis = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
            label = f"{comparison['left_method']} − {comparison['right_method']}"
            axis.plot(ks, means, marker="o", linewidth=1.7, color="#2563eb", label=label)
            axis.fill_between(ks, lows, highs, alpha=0.18, color="#2563eb", label="95% paired bootstrap CI")
            axis.axhline(0, color="#4b5563", linewidth=0.9, linestyle="--")
            bound = max(1.0, max(abs(value) for value in [*lows, *highs]) * 1.15)
            axis.set_ylim(-bound, bound)
            setup_axis(axis, ks, f"{group_label} · {len(comparison['paired_training_seeds'])} paired seed(s)",
                       "Difference in pass@k (percentage points)")
            axis.legend(frameon=False, loc="best")
            save_figure(figure, args.output_dir, f"pass_at_k_delta_{group_name}")
            plt.close(figure)
    print(f"PNG and PDF figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
