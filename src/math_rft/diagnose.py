"""Common-checkpoint gradient probes using independent calibration and holdout draws.

This is a local mechanism diagnostic, not a training loop. Exact generated token
IDs are replayed; no retokenization, optimizer step, or test-set selection occurs.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
from pathlib import Path

from .evaluate import artifact_paths, canonical_hash, model_identity, read_existing, read_jsonl
from .io_utils import sha256_file, write_json

METHODS = ("rft_fixed", "grpo_fixed", "grpo_weighted", "grpo_positive", "grpo_negative")


def coefficients(m: int, p: float, epsilon: float = 1e-6) -> tuple[float, float]:
    """Finite-m coefficients for sequence-score gradients, sample std (ddof=1).

    E[g_RFT] = a_m(p) grad p and E[g_GRPO] = b_m(p) grad p.
    A shared constant length denominator multiplies both sides equally.
    """
    if m < 2 or not 0 < p < 1 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Need m>=2, p strictly inside (0,1), and positive epsilon")
    a = -math.expm1(m * math.log1p(-p)) / p
    expectation = 0.0
    for c in range(1, m):
        q = c / m
        # The log form avoids integer-to-float overflow for large group sizes.
        log_probability = (math.lgamma(m + 1) - math.lgamma(c + 1) - math.lgamma(m - c + 1)
                           + c * math.log(p) + (m - c) * math.log1p(-p))
        probability = math.exp(log_probability)
        std = math.sqrt(m / (m - 1) * q * (1 - q))
        expectation += probability * q * (1 - q) / (std + epsilon)
    b = expectation / (p * (1 - p))
    return a, b


def load_ledger(path: Path, role: str) -> tuple[dict, list[dict], dict[str, list[dict]]]:
    paths = artifact_paths(path)
    if paths["lock"].exists():
        raise ValueError(f"Cannot probe a locked, potentially changing ledger: {paths['lock']}")
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    config = meta["config"]
    if meta["manifest_hash"] != canonical_hash(config) or config["role"] != role:
        raise ValueError(f"Expected an intact {role} manifest: {path}")
    prompts = list(read_jsonl(paths["prompts"]))
    if any(row["split"] in {"test", "math500"}
           or "math500" in str(row["data_source"]).lower().replace("-", "") for row in prompts):
        raise ValueError("Gradient probes must never consume MATH-500")
    if len(prompts) != config["questions"] or [row["id"] for row in prompts] != config["question_ids"]:
        raise ValueError("Prompt index does not match manifest")
    original_rows = [{key: value for key, value in row.items() if key != "prompt_token_ids"} for row in prompts]
    if canonical_hash(original_rows) != config["dataset_sha256"]:
        raise ValueError("Prompt content changed after sampling")
    n, seed = config["sampling"]["n"], config["sampling"]["seed"]
    complete, _ = read_existing(path, prompts, n, seed)
    if len(complete) != n * len(prompts):
        raise ValueError("Gradient analysis requires a complete sample ledger")
    grouped: dict[str, list[dict]] = {row["id"]: [] for row in prompts}
    for sample in read_jsonl(path):
        grouped[sample["id"]].append(sample)
    for samples in grouped.values():
        samples.sort(key=lambda sample: sample["sample_index"])
        if any(not sample["response_token_ids"] for sample in samples):
            raise ValueError("Empty trajectories cannot supply score gradients")
    return config, prompts, grouped


def load_model(model_path: str, revision: str | None, dtype: str, device: str):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, Qwen3_5ForConditionalGeneration

    config = AutoConfig.from_pretrained(model_path, revision=revision, trust_remote_code=False)
    cls = Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else AutoModelForCausalLM
    model = cls.from_pretrained(model_path, revision=revision, config=config,
                               torch_dtype=getattr(torch, dtype), trust_remote_code=False,
                               attn_implementation="eager")
    model.to(device)
    model.eval()  # no dropout noise; gradients remain enabled
    return model


def score_gradient(model, selected: list[tuple[str, object]], prompt: list[int], response: list[int]):
    """Gradient of SUM of generated-token log probabilities, including sampled EOS."""
    import torch

    if not prompt or not response:
        raise ValueError("Prompt and response token lists must be nonempty")
    token_ids = torch.tensor([prompt + response], device=next(model.parameters()).device)
    logits = model(input_ids=token_ids, attention_mask=torch.ones_like(token_ids), use_cache=False).logits
    response_logits = logits[0, len(prompt) - 1:-1].float()
    labels = token_ids[0, len(prompt):]
    log_probability = -torch.nn.functional.cross_entropy(response_logits, labels, reduction="sum")
    gradients = torch.autograd.grad(log_probability, [parameter for _, parameter in selected], allow_unused=True)
    return torch.cat([
        torch.zeros(parameter.numel(), dtype=torch.float32) if gradient is None
        else gradient.detach().reshape(-1).float().cpu()
        for (_, parameter), gradient in zip(selected, gradients, strict=True)
    ])


def compare_vectors(left, right) -> dict:
    import torch

    left_norm, right_norm = float(torch.linalg.vector_norm(left)), float(torch.linalg.vector_norm(right))
    return {"left_norm": left_norm, "right_norm": right_norm,
            "difference_norm": float(torch.linalg.vector_norm(left - right)),
            "cosine": float(torch.dot(left, right)) / (left_norm * right_norm) if left_norm and right_norm else None}


def run(args: argparse.Namespace) -> None:
    import torch

    calibration, calibration_prompts, calibration_samples = load_ledger(args.calibration, "calibration")
    probe, prompts, samples = load_ledger(args.rollouts, "probe")
    if calibration_prompts != prompts or calibration["model"] != probe["model"]:
        raise ValueError("Calibration and probe must use the same checkpoint and identical prompts/token IDs")
    if calibration["sampling"]["seed"] == probe["sampling"]["seed"]:
        raise ValueError("Calibration must use an independent seed from gradient probe samples")
    for config in (calibration, probe):
        sampling = config["sampling"]
        if (sampling["temperature"], sampling["top_p"], sampling["top_k"], sampling["repetition_penalty"]) != (1.0, 1.0, -1, 1.0):
            raise ValueError("Score-gradient mechanism requires untruncated model sampling at temperature 1")
        if sampling["stop"] or sampling["ignore_eos"]:
            raise ValueError("Use EOS-only sampling with no string stopping")
        if sampling["presence_penalty"] != 0.0 or sampling["frequency_penalty"] != 0.0:
            raise ValueError("Raw model score gradients require zero presence/frequency penalties")
    if calibration["grader"] != probe["grader"] or calibration["versions"] != probe["versions"]:
        raise ValueError("Calibration/probe grader and software versions must match")
    for key in ("max_tokens", "presence_penalty", "frequency_penalty"):
        if calibration["sampling"][key] != probe["sampling"][key]:
            raise ValueError(f"Calibration and probe differ in {key}")
    for key in ("dtype", "max_model_len", "max_prompt_tokens", "generation_config"):
        if calibration["engine"][key] != probe["engine"][key]:
            raise ValueError(f"Calibration and probe differ in engine {key}")
    if args.dtype != probe["engine"]["dtype"]:
        raise ValueError("Replay dtype must equal the explicit sampling dtype; use bfloat16 or float32 for both")
    replay_versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers")}
    if any(version != probe["versions"][name] for name, version in replay_versions.items()):
        raise ValueError("Gradient replay must use the sampled run's torch and transformers versions")
    m = args.group_size
    n = probe["sampling"]["n"]
    if n % (2 * m) or n < 4 * m:
        raise ValueError("Probe n must be divisible by 2*m and contain >=2 direction and >=2 holdout groups")
    group_count = n // (2 * m)
    model_path, revision, identity = model_identity(args.model, args.revision)
    if identity != probe["model"]:
        raise ValueError("Requested checkpoint differs from the checkpoint that produced the trajectories")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory for each gradient probe")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_path, revision, args.dtype, args.device)
    pattern = re.compile(args.parameter_regex)
    selected = []
    for name, parameter in model.named_parameters():
        enabled = pattern.search(name) is not None and ".visual." not in name and not name.startswith("visual.")
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append((name, parameter))
    if not selected:
        raise ValueError("Parameter regex matches no language-model parameters")
    dimension = sum(parameter.numel() for _, parameter in selected)
    directions = {method: torch.zeros(dimension, dtype=torch.float32) for method in METHODS}
    question_count = len(prompts)
    probabilities, statistics = {}, []
    for prompt in prompts:
        identifier = prompt["id"]
        c_cal = sum(sample["score"] for sample in calibration_samples[identifier])
        n_cal = len(calibration_samples[identifier])
        # Prespecified Jeffreys smoothing prevents false certainty at c=0 or c=n.
        p = (c_cal + 0.5) / (n_cal + 1)
        probabilities[identifier] = p
        a, b = coefficients(m, p)
        if not math.isfinite(a) or not math.isfinite(b) or b <= 0:
            raise ValueError(f"Invalid finite-sample gradient coefficients for {identifier}")
        weight = a / b
        stats = {"id": identifier, "calibration_n": n_cal, "calibration_correct": c_cal,
                 "p_smoothed": p, "a_m": a, "b_m": b, "a_over_b": weight,
                 "empty_groups": 0, "all_success_groups": 0, "mixed_groups": 0,
                 "direction_groups": group_count, "holdout_groups": group_count}
        for start in range(0, n // 2, m):
            group = samples[identifier][start:start + m]
            c = sum(sample["score"] for sample in group)
            stats["empty_groups" if c == 0 else "all_success_groups" if c == m else "mixed_groups"] += 1
            q = c / m
            std = math.sqrt(m / (m - 1) * q * (1 - q))
            for sample in group:
                r = sample["score"]
                rft = r / c if c else 0.0
                grpo = (r - q) / (std + 1e-6) / m
                if not rft and not grpo:
                    continue
                vector = score_gradient(model, selected, prompt["prompt_token_ids"], sample["response_token_ids"])
                scale = 1 / (question_count * group_count * args.fixed_denominator)
                weights = {"rft_fixed": rft, "grpo_fixed": grpo,
                           "grpo_weighted": grpo * weight, "grpo_positive": max(grpo, 0),
                           "grpo_negative": min(grpo, 0)}
                for method, coefficient in weights.items():
                    directions[method].add_(vector, alpha=coefficient * scale)
        statistics.append(stats)
        print(f"Directions accumulated for {len(statistics)}/{question_count} questions", flush=True)
    slopes = {method: {str(k): 0.0 for k in (1, 128, 256)} for method in METHODS}
    projections = []
    for prompt in prompts:
        identifier = prompt["id"]
        p = probabilities[identifier]
        gradient_p = torch.zeros(dimension, dtype=torch.float32)
        # Independent half of probe draws estimates grad p with a frozen,
        # independently calibrated baseline. It was not used to form updates.
        for sample in samples[identifier][n // 2:]:
            vector = score_gradient(model, selected, prompt["prompt_token_ids"], sample["response_token_ids"])
            gradient_p.add_(vector, alpha=(sample["score"] - p) / (n // 2))
        delta = {method: float(torch.dot(gradient_p, direction)) for method, direction in directions.items()}
        projections.append({"id": identifier, "p_smoothed": p, "grad_p_norm": float(torch.linalg.vector_norm(gradient_p)),
                            "delta": delta, "delta_rft_minus_grpo": delta["rft_fixed"] - delta["grpo_fixed"]})
        for method in METHODS:
            for k in (1, 128, 256):
                slopes[method][str(k)] += k * (1 - p) ** (k - 1) * delta[method] / question_count
    report = {
        "schema_version": 1, "model": identity, "calibration_sha256": sha256_file(args.calibration),
        "replay_dtype": args.dtype, "replay_versions": replay_versions,
        "replay_attention_implementation": "eager",
        "probe_sha256": sha256_file(args.rollouts), "group_size": m, "fixed_denominator": args.fixed_denominator,
        "selected_parameters": [{"name": name, "numel": parameter.numel()} for name, parameter in selected],
        "gradient_dimension": dimension, "parameter_regex": args.parameter_regex,
        "direction_convention": "ascent; optimizer-free; common checkpoint; no norm matching",
        "unweighted_vs_rft": compare_vectors(directions["grpo_fixed"], directions["rft_fixed"]),
        "weighted_vs_rft": compare_vectors(directions["grpo_weighted"], directions["rft_fixed"]),
        "positive_vs_rft": compare_vectors(directions["grpo_positive"], directions["rft_fixed"]),
        "negative_norm": float(torch.linalg.vector_norm(directions["grpo_negative"])),
        "question_statistics": statistics, "holdout_projections": projections,
        "plugin_pass_at_k_directional_derivatives": slopes,
        "limitations": ["Finite noisy calibration probabilities, with Jeffreys smoothing; not exact true-p weights.",
                        "Plug-in pass@k slopes are especially uncertain in the low-probability tail.",
                        "Parameter regex restricts the claim to that subspace if it is not .*.",
                        "Replay uses Transformers eager kernels; small numerical differences from vLLM sampling remain even with equal checkpoint, dtype and software versions.",
                        "These are local raw gradients, not an AdamW or PPO multi-step guarantee."]}
    write_json(args.output_dir / "gradient_report.json", report)
    if args.save_vectors:
        torch.save({key: value for key, value in directions.items()}, args.output_dir / "gradient_vectors.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--fixed-denominator", type=int, default=4096)
    parser.add_argument("--parameter-regex", default=".*")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--save-vectors", action="store_true")
    args = parser.parse_args()
    if args.group_size < 2 or args.fixed_denominator < 1:
        parser.error("group-size must be >=2 and fixed-denominator positive")
    run(args)


if __name__ == "__main__":
    main()
