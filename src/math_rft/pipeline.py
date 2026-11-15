"""Plan or explicitly execute the four-model, two-method, three-seed matrix."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from .io_utils import sha256_file, write_json
from .train import BENCHMARKS, LOSS_NORMALIZATIONS, default_output_dir, loss_reduction

ROOT = Path(__file__).resolve().parents[2]


def invoke(arguments: list[str], execute: bool, *, profile: str = "default") -> None:
    command = [sys.executable, "-m", *arguments]
    print(shlex.join(command), flush=True)
    if execute:
        if profile == "lowmem":
            subprocess.run(command, check=True, env={**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
        else:
            subprocess.run(command, check=True)


def code_identity() -> dict[str, str]:
    paths = sorted((ROOT / "src/math_rft").glob("*.py")) + sorted((ROOT / "configs").glob("*"))
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths if path.is_file()}


def frozen_revisions(output: Path, models: dict, selected: list[str], revision: str | None = None) -> dict[str, str]:
    """Resolve HF revisions once for the whole matrix, before any trial starts."""
    from .model_cache import hub_revision

    path = output / "model_revisions.json"
    revisions = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if revision is not None and (len(selected) != 1 or not re.fullmatch(r"[a-fA-F0-9]{40}", revision)):
        raise ValueError("An explicit --model-revision requires exactly one model and an immutable 40-character SHA")
    for model in selected:
        repo = models[model]["repo_id"]
        if model not in revisions:
            sha = hub_revision(repo, revision)
            revisions[model] = {"repo_id": repo, "revision": sha}
        if revisions[model]["repo_id"] != repo or not re.fullmatch(r"[a-fA-F0-9]{40}", str(revisions[model]["revision"])):
            raise ValueError(f"Invalid frozen model revision for {model}")
        if revision is not None and revisions[model]["revision"].lower() != revision.lower():
            raise ValueError(f"Requested revision differs from the already frozen revision for {model}")
    write_json(path, revisions)
    return {model: revisions[model]["revision"] for model in selected}


def evaluation(model: str, data: Path, output: Path, repo: str, n: int, method: str,
               evaluation_seed: int, training_seed: int | None, revision: str | None, split: str,
               profile: str = "default", loss_normalization: str | None = None,
               loss_scale_factor: int | None = None, benchmark: str = "math",
               max_steps: int = 50, max_action_tokens: int = 128,
               env_workers: int = 0, resume_from: Path | None = None) -> list[str]:
    role = "formal" if split in ("math500", "test") else "development"
    args = ["math_rft.evaluate", "--model", model, "--data", str(data / f"{split}.jsonl"),
            "--output", str(output), "--role", role, "--n", str(n),
            "--seed", str(evaluation_seed), "--method", method, "--base-model", repo]
    if benchmark == "alfworld":
        args += ["--benchmark", benchmark, "--max-steps", str(max_steps),
                 "--max-action-tokens", str(max_action_tokens)]
        if env_workers:
            args += ["--env-workers", str(env_workers)]
        if resume_from is not None:
            args += ["--resume-from", str(resume_from)]
    if training_seed is not None:
        args += ["--training-seed", str(training_seed)]
    if revision:
        args += ["--revision", revision]
    if profile == "lowmem":
        args += ["--profile", profile]
    if loss_normalization is not None:
        args += ["--training-loss-normalization", loss_normalization]
        if loss_scale_factor is not None:
            args += ["--training-loss-scale-factor", str(loss_scale_factor)]
    return args


def main() -> None:
    matrix = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    models = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="math")
    parser.add_argument("--max-steps", type=int, default=50, help="ALFWorld actions per episode, train and eval")
    parser.add_argument("--max-action-tokens", type=int, default=128, help="ALFWorld tokens per action, train and eval")
    parser.add_argument("--eval-env-workers", type=int, default=0,
                        help="ALFWorld evaluation environment processes; does not change training workers")
    parser.add_argument("--baseline-resume-from", type=Path,
                        help="Import an inactive baseline ledger into a new pipeline output (one ALFWorld model)")
    parser.add_argument("--profile", choices=("default", "lowmem"), default="default",
                        help="lowmem changes execution memory settings, preserving training and evaluation budgets")
    parser.add_argument("--loss-normalization", choices=LOSS_NORMALIZATIONS, default="fixed",
                        help="Shared RFT/GRPO normalization: fixed constant (default) or legacy per_response")
    parser.add_argument("--loss-scale-factor", type=int, help="Shared fixed loss scale, default 4096")
    parser.add_argument("--models", nargs="+", choices=list(models), default=matrix["models"])
    parser.add_argument("--model-revision", help="Pin one selected model to an immutable SHA, e.g. when repeating a pilot")
    parser.add_argument("--methods", nargs="+", choices=("rft", "grpo"), default=matrix["main_methods"])
    parser.add_argument("--seeds", type=int, nargs="+", default=matrix["training_seeds"])
    parser.add_argument("--stage", choices=("pilot", "full", "deepmath_8192", "deepmath_16384"), default="pilot")
    parser.add_argument("--data-dir", type=Path, help="Default: data or data_alfworld according to benchmark")
    parser.add_argument("--output-dir", type=Path, help="Default: runs_fixed; lowmem: runs_lowmem_fixed_pipeline; per_response retains legacy defaults")
    parser.add_argument("--evaluation-split", choices=("dev", "math500", "test"),
                        help="Default: dev for pilot; math500 (math) or test (ALFWorld) for full")
    parser.add_argument("--n", type=int, help="Samples per question, default 256 for both dev and MATH-500")
    parser.add_argument("--evaluation-seed", type=int, default=matrix["evaluation_seed"])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--execute", action="store_true", help="Without this flag, print the plan and start no jobs")
    args = parser.parse_args()
    if args.eval_env_workers < 0 or (args.benchmark != "alfworld" and
                                   (args.eval_env_workers or args.baseline_resume_from)):
        parser.error("Evaluation environment workers and baseline migration require ALFWorld and nonnegative workers")
    if args.baseline_resume_from is not None and len(args.models) != 1:
        parser.error("--baseline-resume-from requires exactly one model")
    try:
        _, loss_scale = loss_reduction(args.loss_normalization, args.loss_scale_factor)
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir = args.output_dir or Path(default_output_dir(
        args.profile, args.loss_normalization, pipeline=True, benchmark=args.benchmark))
    args.data_dir = args.data_dir or Path("data_alfworld" if args.benchmark == "alfworld" else "data")
    formal_split = "test" if args.benchmark == "alfworld" else "math500"
    args.evaluation_split = args.evaluation_split or ("dev" if args.stage == "pilot" else formal_split)
    if args.evaluation_split not in ("dev", formal_split):
        parser.error(f"Benchmark {args.benchmark} requires dev or {formal_split} evaluation")
    if args.benchmark == "alfworld":
        if args.stage not in ("pilot", "full") or min(args.max_steps, args.max_action_tokens) <= 0:
            parser.error("ALFWorld requires pilot/full and positive episode/action budgets")
        # These overrides would silently change training while evaluation kept
        # a different trajectory budget. Standalone train/evaluate allow explicit budgets.
        for override in args.overrides:
            if override.split("=", 1)[0] in {
                "data.max_prompt_length", "data.max_response_length", "actor_rollout_ref.rollout.max_model_len",
                "alfworld.max_steps", "alfworld.max_action_tokens",
            }:
                parser.error("Use --max-steps/--max-action-tokens for shared ALFWorld limits; "
                             "custom context budgets require matched standalone train/evaluate commands")
    args.n = args.n if args.n is not None else matrix[
        "formal_samples_per_question" if args.evaluation_split == formal_split else "development_samples_per_question"
    ]
    minimum_n = matrix["max_k"]
    if args.n < minimum_n or not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("This experiment requires n>=256 and distinct nonnegative training seeds; use evaluate --role debug for a smoke test")
    if len(set(args.models)) != len(args.models) or len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate models or methods are not allowed")
    if args.model_revision is not None and (len(args.models) != 1 or not re.fullmatch(r"[a-fA-F0-9]{40}", args.model_revision)):
        parser.error("--model-revision requires exactly one --models entry and a 40-character commit SHA")
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    revisions = {args.models[0]: args.model_revision} if args.model_revision else {}
    if args.execute:
        for filename in (f"{args.stage}.parquet", "dev.parquet", f"{args.evaluation_split}.jsonl", "manifest.json"):
            if not (args.data_dir / filename).is_file():
                raise FileNotFoundError(f"Prepare the data before running the matrix: {filename}")
        prepared = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
        if prepared.get("benchmark", "math") != args.benchmark:
            raise ValueError("Prepared data benchmark does not match --benchmark")
        revisions = (frozen_revisions(args.output_dir, models, args.models, revision=args.model_revision)
                     if args.model_revision else frozen_revisions(args.output_dir, models, args.models))
    else:
        print("PLAN ONLY. --execute freezes model revisions, then launches these jobs. No data/models are loaded.")
    source_hashes = code_identity()
    for model in args.models:
        repo, revision = models[model]["repo_id"], revisions.get(model)
        # Keep stage/sample budgets separate so later stages cannot accidentally
        # resume a baseline generated with a different evaluation protocol.
        baseline_output = args.output_dir / "baselines" / args.stage / model / f"{args.evaluation_split}.jsonl"
        invoke(evaluation(repo, args.data_dir, baseline_output, repo, args.n, "base", args.evaluation_seed,
                          None, revision, args.evaluation_split, args.profile, benchmark=args.benchmark,
                          max_steps=args.max_steps, max_action_tokens=args.max_action_tokens,
                          env_workers=args.eval_env_workers, resume_from=args.baseline_resume_from),
               args.execute, profile=args.profile)
        evaluated: dict[str, list[Path]] = {}
        for method in args.methods:
            evaluated[method] = []
            for seed in args.seeds:
                run_dir = args.output_dir / model / method / args.stage / f"seed_{seed}"
                complete_path = run_dir / "training_complete.json"
                request_path = run_dir / "matrix_request.json"
                train_args = ["math_rft.train", "--model", model, "--method", method,
                              "--stage", args.stage, "--seed", str(seed), "--data-dir", str(args.data_dir),
                              "--output-dir", str(args.output_dir),
                              "--loss-normalization", args.loss_normalization]
                if args.benchmark == "alfworld":
                    train_args += ["--benchmark", args.benchmark, "--max-steps", str(args.max_steps),
                                   "--max-action-tokens", str(args.max_action_tokens)]
                if loss_scale is not None:
                    train_args += ["--loss-scale-factor", str(loss_scale)]
                if args.profile == "lowmem":
                    train_args += ["--profile", args.profile]
                if revision:
                    train_args += ["--model-revision", revision]
                for override in args.overrides:
                    train_args += ["--set", override]
                if args.execute:
                    request = {"model": model, "method": method, "stage": args.stage, "seed": seed,
                               "revision": revision, "overrides": args.overrides, "source_hashes": source_hashes,
                               "loss_normalization": args.loss_normalization, "loss_scale_factor": loss_scale,
                               "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json")}
                    if args.benchmark == "alfworld":
                        request.update(benchmark=args.benchmark, max_steps=args.max_steps,
                                       max_action_tokens=args.max_action_tokens)
                    if args.profile == "lowmem":
                        request["profile"] = args.profile
                    if request_path.exists():
                        if json.loads(request_path.read_text(encoding="utf-8")) != request:
                            raise ValueError(f"Existing trial has a different matrix request: {run_dir}")
                    else:
                        if (run_dir / "run_manifest.json").exists():
                            raise ValueError(f"Existing manual trial cannot be adopted implicitly: {run_dir}")
                        write_json(request_path, request)
                    if not complete_path.exists():
                        # train.py deliberately rejects an incomplete existing run;
                        # finish it explicitly with --resume-from, then rerun matrix.
                        invoke(train_args, True, profile=args.profile)
                    complete = json.loads(complete_path.read_text(encoding="utf-8"))
                    if complete.get("benchmark", "math") != args.benchmark:
                        raise ValueError(f"Completed trial benchmark mismatch in {run_dir}")
                    for key in ("model", "method", "stage", "seed"):
                        if complete[key] != request[key]:
                            raise ValueError(f"Completed trial identity mismatch: {key} in {run_dir}")
                    if complete.get("profile", "default") != args.profile:
                        raise ValueError(f"Completed trial profile mismatch in {run_dir}")
                    if (complete.get("loss_normalization", "per_response") != args.loss_normalization
                            or complete.get("loss_scale_factor") != loss_scale):
                        raise ValueError(f"Completed trial training objective mismatch in {run_dir}; use a separate output directory")
                    if complete["resolved_config_sha256"] != sha256_file(run_dir / "resolved_config.yaml"):
                        raise ValueError("Completed trial's resolved configuration was modified")
                    prior = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
                    if prior.get("benchmark", "math") != args.benchmark:
                        raise ValueError(f"Training manifest benchmark mismatch in {run_dir}")
                    if prior.get("profile", "default") != args.profile:
                        raise ValueError(f"Training manifest profile mismatch in {run_dir}")
                    if (prior.get("loss_normalization", "per_response") != args.loss_normalization
                            or prior.get("loss_scale_factor") != loss_scale):
                        raise ValueError(f"Training manifest objective mismatch in {run_dir}")
                    if prior["source_sha256"] != complete["source_sha256"]:
                        raise ValueError("Training completion does not match the original source manifest")
                    for dataset_path, digest in prior["data_sha256"].items():
                        if sha256_file(dataset_path) != digest:
                            raise ValueError(f"Training data changed after the completed trial: {dataset_path}")
                    final_step = complete["global_step"]
                    expected_export = (run_dir / "checkpoints" / f"global_step_{final_step}" / "actor/huggingface").resolve()
                    exported = Path(complete["final_hf_model"]).resolve()
                    if final_step != prior["expected_steps"] or exported != expected_export:
                        raise ValueError("Completion marker does not identify this trial's planned final checkpoint")
                    if not (exported / "config.json").is_file() or not any(exported.glob("*.safetensors")):
                        raise FileNotFoundError(f"Successful training marker has no exported model: {exported}")
                else:
                    invoke(train_args, False, profile=args.profile)
                    # Printed symbolic path is informational; execution resolves
                    # it only from a successful training completion marker.
                    exported = run_dir / "checkpoints" / "FINAL_GLOBAL_STEP" / "actor" / "huggingface"
                eval_output = run_dir / f"{args.evaluation_split}.jsonl"
                invoke(evaluation(str(exported), args.data_dir, eval_output, repo, args.n, method,
                                  args.evaluation_seed, seed, None, args.evaluation_split, args.profile,
                                  args.loss_normalization, loss_scale, benchmark=args.benchmark,
                                  max_steps=args.max_steps, max_action_tokens=args.max_action_tokens,
                                  env_workers=args.eval_env_workers),
                       args.execute, profile=args.profile)
                evaluated[method].append(eval_output)
        if "rft" in evaluated and "grpo" in evaluated:
            comparison_dir = args.output_dir / "comparisons" / args.stage / model / "rft_vs_grpo"
            invoke(["math_rft.analyze", "compare", "--left", *map(str, evaluated["rft"]),
                    "--right", *map(str, evaluated["grpo"]), "--output-dir", str(comparison_dir),
                    "--bootstrap", str(matrix["bootstrap_replicates"])], args.execute, profile=args.profile)
            invoke(["math_rft.plot", "--comparison", str(comparison_dir / "comparison.json"),
                    "--baseline", str(baseline_output) + ".summary.json",
                    "--output-dir", str(comparison_dir / "figures")], args.execute, profile=args.profile)


if __name__ == "__main__":
    main()
