"""Finite-sample pass@k estimators and question-level uncertainty.

These functions have no model/runtime dependencies. At n=k, pass@k is observed
coverage, not a proof that the model has (or lacks) the underlying capability.
"""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

DEFAULT_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def pass_at_k(n: int, correct: int, k: int) -> float:
    """Unbiased estimator 1 - choose(n-c,k)/choose(n,k), for IID samples."""
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (n, correct, k)):
        raise TypeError("n, correct and k must be integers")
    if not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError(f"Invalid pass@k counts: n={n}, correct={correct}, k={k}")
    if correct == 0:
        return 0.0
    if n - correct < k:
        return 1.0
    # log1p/expm1 avoid cancellation when success probability is small.
    log_failure = math.fsum(math.log1p(-correct / (n - j)) for j in range(k))
    return -math.expm1(log_failure)


def pass_curve(n: int, correct: int, ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    return {str(k): pass_at_k(n, correct, k) for k in ks if k <= n}


def quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("Quantiles require nonempty values and probability in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def mean_ci(values: Sequence[float], *, repeats: int = 2000, seed: int = 0) -> dict:
    """Percentile bootstrap over QUESTIONS, never over individual completions."""
    if not values or repeats < 100:
        raise ValueError("Bootstrap requires nonempty questions and >=100 replicates")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Bootstrap values must be finite")
    rng = random.Random(seed)
    n = len(values)
    means = [math.fsum(rng.choices(values, k=n)) / n for _ in range(repeats)]
    return {
        "mean": statistics.fmean(values),
        "ci95": [quantile(means, 0.025), quantile(means, 0.975)],
        "questions": n,
        "bootstrap_repeats": repeats,
        "bootstrap_unit": "question",
    }


def paired_delta_ci(
    left: Sequence[float], right: Sequence[float], *, repeats: int = 2000, seed: int = 0
) -> dict:
    if len(left) != len(right):
        raise ValueError("Paired comparison requires equal question counts")
    result = mean_ci([a - b for a, b in zip(left, right, strict=True)], repeats=repeats, seed=seed)
    result["left_mean"] = statistics.fmean(left)
    result["right_mean"] = statistics.fmean(right)
    result["direction"] = "left_minus_right"
    return result


def hierarchical_delta_ci(
    paired_seed_differences: Sequence[Sequence[float]], *, repeats: int = 2000, seed: int = 0
) -> dict:
    """Resample paired training seeds and their common question set.

    A common question resample is used across seeds, preserving the crossed
    seed-by-question design. Completion samples are never treated as replicates.
    """
    if not paired_seed_differences or repeats < 100:
        raise ValueError("Need paired seeds and >=100 bootstrap replicates")
    lengths = {len(values) for values in paired_seed_differences}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("All seeds must share an equal, nonempty question set")
    if any(not math.isfinite(value) for values in paired_seed_differences for value in values):
        raise ValueError("Bootstrap values must be finite")
    seed_count = len(paired_seed_differences)
    question_count = len(paired_seed_differences[0])
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        selected_seeds = rng.choices(range(seed_count), k=seed_count)
        selected_questions = rng.choices(range(question_count), k=question_count)
        draws.append(math.fsum(
            paired_seed_differences[s][q] for s in selected_seeds for q in selected_questions
        ) / (seed_count * question_count))
    seed_means = [statistics.fmean(values) for values in paired_seed_differences]
    return {
        "mean": statistics.fmean(seed_means),
        "ci95": [quantile(draws, 0.025), quantile(draws, 0.975)],
        "training_seeds": seed_count,
        "questions_per_seed": question_count,
        "seed_means": seed_means,
        "seed_std": statistics.stdev(seed_means) if seed_count > 1 else None,
        "bootstrap_repeats": repeats,
        "bootstrap_unit": "paired_training_seed_and_common_question",
        "note": "With one training seed, the interval reflects question uncertainty only.",
    }
