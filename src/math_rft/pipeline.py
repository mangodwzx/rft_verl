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


def frozen_revisions(output: Path, models: dict, selected: list[str]) -> dict[str, str]:
    """Resolve HF revisions once for the whole matrix, before any trial starts."""
    from huggingface_hub import HfApi

    path = output / "model_revisions.json"
    revisions = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for model in selected:
        repo = models[model]["repo_id"]
        if model not in revisions:
            sha = HfApi().model_info(repo).sha
            revisions[model] = {"repo_id": repo, "revision": sha}
        if revisions[model]["repo_id"] != repo or not re.fullmatch(r"[a-fA-F0-9]{40}", str(revisions[model]["revision"])):
            raise ValueError(f"Invalid frozen model revision for {model}")
    write_json(path, revisions)
    return {model: revisions[model]["revision"] for model in selected}


def evaluation(model: str, data: Path, output: Path, repo: str, n: int, method: str,
               evaluation_seed: int, training_seed: int | None, revision: str | None, split: str,
               profile: str = "default") -> list[str]:
    role = "formal" if split == "math500" else "development"
    args = ["math_rft.evaluate", "--model", model, "--data", str(data / f"{split}.jsonl"),
            "--output", str(output), "--role", role, "--n", str(n),
            "--seed", str(evaluation_seed), "--method", method, "--base-model", repo]
    if training_seed is not None:
        args += ["--training-seed", str(training_seed)]
    if revision:
        args += ["--revision", revision]
    if profile == "lowmem":
        args += ["--profile", profile]
    return args


def main() -> None:
    matrix = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    models = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("default", "lowmem"), default="default",
                        help="lowmem changes execution memory settings, preserving training and evaluation budgets")
    parser.add_argument("--models", nargs="+", choices=list(models), default=matrix["models"])
    parser.add_argument("--methods", nargs="+", choices=("rft", "grpo"), default=matrix["main_methods"])
    parser.add_argument("--seeds", type=int, nargs="+", default=matrix["training_seeds"])
    parser.add_argument("--stage", choices=("pilot", "full", "deepmath_8192", "deepmath_16384"), default="pilot")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, help="Default: runs; lowmem: runs_lowmem_pipeline")
    parser.add_argument("--evaluation-split", choices=("dev", "math500"), help="Default: dev for pilot; math500 for full/DeepMath")
    parser.add_argument("--n", type=int, help="Samples per question, default 256 for both dev and MATH-500")
    parser.add_argument("--evaluation-seed", type=int, default=matrix["evaluation_seed"])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--execute", action="store_true", help="Without this flag, print the plan and start no jobs")
    args = parser.parse_args()
    args.output_dir = args.output_dir or Path("runs_lowmem_pipeline" if args.profile == "lowmem" else "runs")
    args.evaluation_split = args.evaluation_split or ("dev" if args.stage == "pilot" else "math500")
    args.n = args.n if args.n is not None else matrix[
        "formal_samples_per_question" if args.evaluation_split == "math500" else "development_samples_per_question"
    ]
    minimum_n = matrix["max_k"]
    if args.n < minimum_n or not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("This experiment requires n>=256 and distinct nonnegative training seeds; use evaluate --role debug for a smoke test")
    if len(set(args.models)) != len(args.models) or len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate models or methods are not allowed")
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    revisions = {}
    if args.execute:
        for filename in (f"{args.stage}.parquet", "dev.parquet", f"{args.evaluation_split}.jsonl", "manifest.json"):
            if not (args.data_dir / filename).is_file():
                raise FileNotFoundError(f"Prepare the data before running the matrix: {filename}")
        revisions = frozen_revisions(args.output_dir, models, args.models)
    else:
        print("PLAN ONLY. --execute freezes model revisions, then launches these jobs. No data/models are loaded.")
    source_hashes = code_identity()
    for model in args.models:
        repo, revision = models[model]["repo_id"], revisions.get(model)
        # Keep stage/sample budgets separate so later stages cannot accidentally
        # resume a baseline generated with a different evaluation protocol.
        baseline_output = args.output_dir / "baselines" / args.stage / model / f"{args.evaluation_split}.jsonl"
        invoke(evaluation(repo, args.data_dir, baseline_output, repo, args.n, "base", args.evaluation_seed,
                          None, revision, args.evaluation_split, args.profile), args.execute, profile=args.profile)
        evaluated: dict[str, list[Path]] = {}
        for method in args.methods:
            evaluated[method] = []
            for seed in args.seeds:
                run_dir = args.output_dir / model / method / args.stage / f"seed_{seed}"
                complete_path = run_dir / "training_complete.json"
                request_path = run_dir / "matrix_request.json"
                train_args = ["math_rft.train", "--model", model, "--method", method,
                              "--stage", args.stage, "--seed", str(seed), "--data-dir", str(args.data_dir),
                              "--output-dir", str(args.output_dir)]
                if args.profile == "lowmem":
                    train_args += ["--profile", args.profile]
                if revision:
                    train_args += ["--model-revision", revision]
                for override in args.overrides:
                    train_args += ["--set", override]
                if args.execute:
                    request = {"model": model, "method": method, "stage": args.stage, "seed": seed,
                               "revision": revision, "overrides": args.overrides, "source_hashes": source_hashes,
                               "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json")}
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
                    for key in ("model", "method", "stage", "seed"):
                        if complete[key] != request[key]:
                            raise ValueError(f"Completed trial identity mismatch: {key} in {run_dir}")
                    if complete.get("profile", "default") != args.profile:
                        raise ValueError(f"Completed trial profile mismatch in {run_dir}")
                    if complete["resolved_config_sha256"] != sha256_file(run_dir / "resolved_config.yaml"):
                        raise ValueError("Completed trial's resolved configuration was modified")
                    prior = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
                    if prior.get("profile", "default") != args.profile:
                        raise ValueError(f"Training manifest profile mismatch in {run_dir}")
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
                                  args.evaluation_seed, seed, None, args.evaluation_split, args.profile),
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
