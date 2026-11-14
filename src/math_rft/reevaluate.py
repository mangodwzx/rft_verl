"""Re-evaluate completed training checkpoints in a separate output tree.

The default prints a validated plan. --execute evaluates the original final
checkpoints with one shared execution configuration, then compares and plots
RFT/GRPO when both are selected. This command never starts training.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from .io_utils import sha256_file
from .evaluate import training_objective

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("pilot", "full", "deepmath_8192", "deepmath_16384")


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-root", type=Path, default=Path("runs_lowmem_pipeline"))
    cli.add_argument("--output-dir", type=Path, default=Path("runs_eval_throughput"))
    cli.add_argument("--model", default="qwen3_1p7b")
    cli.add_argument("--benchmark", choices=("math", "alfworld"), help="Default: infer from the completed training manifest")
    cli.add_argument("--stage", choices=STAGES, default="pilot")
    cli.add_argument("--seed", type=int, default=42, help="Training seed of the completed trials")
    cli.add_argument("--methods", choices=("rft", "grpo"), nargs="+", default=["rft", "grpo"])
    cli.add_argument("--data-dir", type=Path, help="Default: data or data_alfworld according to the recorded benchmark")
    cli.add_argument("--split", choices=("dev", "math500", "test"), help="Default: dev for pilot, benchmark test otherwise")
    cli.add_argument("--max-steps", type=int, help="ALFWorld episode limit; default: recorded training setting")
    cli.add_argument("--max-action-tokens", type=int, help="ALFWorld action token limit; default: recorded training setting")
    cli.add_argument("--profile", choices=("default", "lowmem", "throughput"), default="throughput")
    cli.add_argument("--n", type=int, default=256)
    cli.add_argument("--evaluation-seed", type=int, default=20261025)
    cli.add_argument("--batch-size", type=int)
    cli.add_argument("--max-num-seqs", type=int)
    cli.add_argument("--max-num-batched-tokens", type=int)
    cli.add_argument("--gpu-memory-utilization", type=float)
    cli.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=None)
    cli.add_argument("--gpus", nargs="+", help="Physical GPU IDs for question shards, e.g. --gpus 2 6; each uses TP=1")
    cli.add_argument("--execute", action="store_true", help="Run the printed commands; otherwise only validate the plan")
    return cli


def final_model(run_dir: Path, identity: dict, repo_id: str) -> tuple[Path, dict]:
    """Resolve a successful planned final export without requiring old source code."""
    complete = json.loads((run_dir / "training_complete.json").read_text(encoding="utf-8"))
    prior = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    for key, expected in identity.items():
        if complete.get(key) != expected or prior.get(key) != expected:
            raise ValueError(f"Completed trial identity mismatch for {key}: {run_dir}")
    if complete.get("profile", "default") != prior.get("profile", "default"):
        raise ValueError(f"Training completion profile differs from its manifest: {run_dir}")
    benchmark = prior.get("benchmark", "math")
    if benchmark not in {"math", "alfworld"} or complete.get("benchmark", "math") != benchmark:
        raise ValueError(f"Training completion benchmark differs from its manifest: {run_dir}")
    if benchmark == "alfworld" and (
        not isinstance(prior.get("alfworld"), dict) or prior["alfworld"] != complete.get("alfworld")
    ):
        raise ValueError(f"ALFWorld episode protocol differs between completion and manifest: {run_dir}")
    objective_keys = {"loss_normalization", "loss_scale_factor"}
    objective = None
    if objective_keys.intersection(prior) or objective_keys.intersection(complete):
        if (not objective_keys <= prior.keys() or not objective_keys <= complete.keys()
                or any(prior[key] != complete[key] for key in objective_keys)):
            raise ValueError(f"Training objective provenance differs between completion and manifest: {run_dir}")
        objective = training_objective(identity["method"], prior["loss_normalization"], prior["loss_scale_factor"])
        if objective is None:
            raise ValueError(f"Recorded training objective cannot be null: {run_dir}")
    config_hash = sha256_file(run_dir / "resolved_config.yaml")
    if complete.get("resolved_config_sha256") != config_hash or prior.get("resolved_config_sha256") != config_hash:
        raise ValueError(f"Completed trial's resolved configuration was modified: {run_dir}")
    if not prior.get("source_sha256") or prior["source_sha256"] != complete.get("source_sha256"):
        raise ValueError(f"Training completion does not match its original source manifest: {run_dir}")
    if prior.get("model_provenance", {}).get("repo_id") != repo_id:
        raise ValueError(f"Completed trial has a different base model: {run_dir}")
    revision = prior["model_provenance"].get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError(f"Completed trial must record an immutable 40-character base model revision: {run_dir}")
    if not prior.get("data_sha256"):
        raise ValueError(f"Training manifest is missing dataset hashes: {run_dir}")
    for filename, digest in prior["data_sha256"].items():
        if sha256_file(filename) != digest:
            raise ValueError(f"Training data changed after the completed trial: {filename}")
    step = complete.get("global_step")
    if type(step) is not int or step < 1 or step != prior.get("expected_steps"):
        raise ValueError(f"Completion marker does not identify the planned final step: {run_dir}")
    expected = (run_dir / "checkpoints" / f"global_step_{step}" / "actor/huggingface").resolve()
    exported = Path(complete["final_hf_model"]).resolve()
    if exported != expected or not exported.is_relative_to(run_dir.resolve()):
        raise ValueError(f"Completion marker does not identify this trial's final export: {run_dir}")
    if not (exported / "config.json").is_file() or not any(exported.glob("*.safetensors")):
        raise FileNotFoundError(f"Successful training marker has no exported model: {exported}")
    for index in exported.glob("*.index.json"):
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        for shard in set(weight_map.values()):
            if not (exported / shard).is_file():
                raise FileNotFoundError(f"Missing model shard {shard} referenced by {index}")
    return exported, {"base_revision": revision.lower(), "data_sha256": prior["data_sha256"],
                      "training_objective": objective, "benchmark": benchmark, "alfworld": prior.get("alfworld")}


def execution_options(args: argparse.Namespace) -> list[str]:
    options = []
    for name in ("batch_size", "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization"):
        value = getattr(args, name)
        if value is not None:
            options += ["--" + name.replace("_", "-"), str(value)]
    if args.enforce_eager is not None:
        options.append("--enforce-eager" if args.enforce_eager else "--no-enforce-eager")
    return options


def plan(args: argparse.Namespace) -> list[list[str]]:
    models = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    if args.model not in models:
        raise ValueError(f"Unknown model {args.model}; choose from {', '.join(models)}")
    if args.n < matrix["max_k"] or min(args.seed, args.evaluation_seed) < 0:
        raise ValueError("Re-evaluation requires n>=256 and nonnegative training/evaluation seeds")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Duplicate methods are not allowed")
    if args.gpus is not None and (not args.gpus or len(set(args.gpus)) != len(args.gpus)
            or any(not re.fullmatch(r"0|[1-9][0-9]*", gpu) for gpu in args.gpus)):
        raise ValueError("--gpus requires distinct nonnegative canonical GPU IDs, e.g. --gpus 2 6")
    for name in ("batch_size", "max_num_seqs", "max_num_batched_tokens", "max_steps", "max_action_tokens"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    memory = args.gpu_memory_utilization
    if memory is not None and (not math.isfinite(memory) or not 0 < memory < 1):
        raise ValueError("--gpu-memory-utilization must be finite and in (0,1)")
    source, output = args.run_root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("--output-dir and --run-root must be separate, non-overlapping directory trees")
    first_manifest = source / args.model / args.methods[0] / args.stage / f"seed_{args.seed}" / "run_manifest.json"
    recorded_benchmark = json.loads(first_manifest.read_text(encoding="utf-8")).get("benchmark", "math")
    benchmark = args.benchmark or recorded_benchmark
    if benchmark != recorded_benchmark:
        raise ValueError("Requested benchmark differs from the completed training benchmark")
    if benchmark == "alfworld" and args.stage not in {"pilot", "full"}:
        raise ValueError("ALFWorld supports pilot and full stages")
    split = args.split or ("dev" if args.stage == "pilot" else "test" if benchmark == "alfworld" else "math500")
    if split not in ({"dev", "test"} if benchmark == "alfworld" else {"dev", "math500"}):
        raise ValueError("Requested evaluation split does not belong to the benchmark")
    data_dir = args.data_dir or Path("data_alfworld" if benchmark == "alfworld" else "data")
    data = data_dir.expanduser().resolve() / f"{split}.jsonl"
    if not data.is_file():
        raise FileNotFoundError(f"Prepare the evaluation data first: {data}")
    role = "formal" if split in {"math500", "test"} else "development"
    repo_id = models[args.model]["repo_id"]
    commands, results = [], {}
    reference_protocol = None
    # Validate all selected checkpoints before launching even the first evaluator.
    for method in args.methods:
        relative = Path(args.model) / method / args.stage / f"seed_{args.seed}"
        identity = {"model": args.model, "method": method, "stage": args.stage, "seed": args.seed}
        exported, training_protocol = final_model(source / relative, identity, repo_id)
        if training_protocol["benchmark"] != benchmark:
            raise ValueError("Paired trials have different training benchmarks")
        if reference_protocol is None:
            reference_protocol = training_protocol
        elif training_protocol["base_revision"] != reference_protocol["base_revision"]:
            raise ValueError("Paired trials have different frozen base model revisions")
        elif training_protocol["data_sha256"] != reference_protocol["data_sha256"]:
            raise ValueError("Paired trials have different original training dataset hashes")
        elif training_protocol["training_objective"] != reference_protocol["training_objective"]:
            raise ValueError("Paired trials have different or unknown training objective provenance")
        elif training_protocol["alfworld"] != reference_protocol["alfworld"]:
            raise ValueError("Paired trials have different ALFWorld episode protocols")
        result = (output / relative / f"{split}.jsonl").resolve()
        if not result.is_relative_to(output):
            raise ValueError(f"Evaluation output escapes --output-dir through a symlink: {result}")
        results[method] = result
        evaluation = [
            "math_rft.evaluate", "--profile", args.profile, "--model", str(exported),
            "--data", str(data), "--output", str(result), "--role", role, "--n", str(args.n),
            "--seed", str(args.evaluation_seed), "--method", method, "--base-model", repo_id,
            "--training-seed", str(args.seed), *execution_options(args),
        ]
        objective = training_protocol["training_objective"]
        if benchmark == "alfworld":
            episode = training_protocol["alfworld"]
            evaluation += ["--benchmark", "alfworld",
                           "--max-steps", str(args.max_steps if args.max_steps is not None else episode["max_steps"]),
                           "--max-action-tokens", str(args.max_action_tokens if args.max_action_tokens is not None else episode["max_action_tokens"])]
        if objective is not None:
            evaluation += ["--training-loss-normalization", objective["loss_normalization"]]
            if objective["loss_scale_factor"] is not None:
                evaluation += ["--training-loss-scale-factor", str(objective["loss_scale_factor"])]
        if args.gpus is None or len(args.gpus) == 1:
            if args.gpus is not None and not Path(str(result) + ".log").resolve().is_relative_to(output):
                raise ValueError(f"Evaluation log escapes --output-dir through a symlink: {result}")
            commands.append(evaluation)
        else:
            shards = []
            for index in range(len(args.gpus)):
                directory = f"shard_{index:05d}_of_{len(args.gpus):05d}"
                shard = (result.parent / "shards" / directory / result.name).resolve()
                log = Path(str(shard) + ".log").resolve()
                if not shard.is_relative_to(output) or not log.is_relative_to(output):
                    raise ValueError(f"Shard output or log escapes --output-dir through a symlink: {shard}")
                shards.append(shard)
                command = evaluation.copy()
                command[command.index("--output") + 1] = str(shard)
                commands.append([*command, "--num-shards", str(len(args.gpus)), "--shard-index", str(index)])
            commands.append(["math_rft.merge_evaluations", "--inputs", *map(str, shards),
                             "--data", str(data), "--output", str(result)])
    if {"rft", "grpo"} <= results.keys():
        comparison = (output / "comparisons" / args.stage / args.model / f"seed_{args.seed}" / "rft_vs_grpo").resolve()
        if not comparison.is_relative_to(output):
            raise ValueError(f"Comparison output escapes --output-dir through a symlink: {comparison}")
        figures = (comparison / "figures").resolve()
        if not figures.is_relative_to(output):
            raise ValueError(f"Figure output escapes --output-dir through a symlink: {figures}")
        commands += [
            ["math_rft.analyze", "compare", "--left", str(results["rft"]), "--right", str(results["grpo"]),
             "--output-dir", str(comparison), "--bootstrap", str(matrix["bootstrap_replicates"])],
            ["math_rft.plot", "--comparison", str(comparison / "comparison.json"),
             "--output-dir", str(figures)],
        ]
    return commands


def stop_children(children: list[subprocess.Popen]) -> None:
    """Interrupt owned process groups and bound cleanup even if vLLM is stuck."""
    # Child sessions do not receive terminal Ctrl+C, so explicitly signal their
    # complete groups. A second Ctrl+C must not interrupt cleanup halfway through.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        for child in children:
            try:
                child.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        # The evaluator can exit before its engine, so signal remaining groups
        # even if the immediate child was already reaped.
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for child in children:
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    finally:
        signal.signal(signal.SIGINT, previous)


def remove_interrupted_locks(owners: list[tuple[subprocess.Popen, Path, bool]]) -> None:
    """Remove only newly created locks belonging to our confirmed-exited children."""
    for child, lock, existed_before_launch in owners:
        if existed_before_launch or child.poll() is None:
            continue
        try:
            before = lock.lstat()
            if not stat.S_ISREG(before.st_mode) or lock.read_text(encoding="utf-8") != f"pid={child.pid}\n":
                continue
            after = lock.lstat()
            if (before.st_dev, before.st_ino, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_mtime_ns):
                continue
            lock.unlink()
            print(f"Removed interrupted evaluator lock: {lock}", flush=True)
        except (OSError, UnicodeError):
            # A removed/replaced/unreadable lock must not mask the original
            # evaluator failure or cause another owner's lock to be removed.
            continue


def parallel_evaluations(commands: list[list[str]], gpus: list[str]) -> None:
    """Run all shards concurrently; propagate a failure before any merge starts."""
    children = []
    lock_owners = []
    with ExitStack() as stack:
        try:
            for arguments, gpu in zip(commands, gpus, strict=True):
                output = Path(arguments[arguments.index("--output") + 1])
                log = Path(str(output) + ".log")
                lock = Path(str(output) + ".lock")
                existed_before_launch = os.path.lexists(lock)
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = stack.enter_context(log.open("a", encoding="utf-8"))
                command = [sys.executable, "-u", "-m", *arguments]
                stream.write(f"\nCUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}\n")
                stream.flush()
                child = subprocess.Popen(
                    command, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                children.append(child)
                lock_owners.append((child, lock, existed_before_launch))
                method = arguments[arguments.index("--method") + 1]
                print(f"Started {method} GPU {gpu}, PID {child.pid}; log: {log}", flush=True)
            pending = set(range(len(children)))
            while pending:
                for index in sorted(pending):
                    child = children[index]
                    code = child.poll()
                    if code is None:
                        continue
                    child.wait()
                    if code:
                        raise subprocess.CalledProcessError(code, child.args)
                    pending.remove(index)
                    print(f"Finished shard {index + 1}/{len(children)} on GPU {gpus[index]}", flush=True)
                if pending:
                    time.sleep(0.5)
        except BaseException:
            stop_children(children)
            remove_interrupted_locks(lock_owners)
            raise


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    commands = plan(args)
    if not args.execute:
        print("PLAN ONLY. Completed checkpoints validated; no files written or jobs started. Add --execute to run.")
    index = 0
    while index < len(commands):
        arguments = commands[index]
        if args.gpus is not None and arguments[0] == "math_rft.evaluate":
            shards = commands[index:index + len(args.gpus)]
            for shard, gpu in zip(shards, args.gpus, strict=True):
                command = [sys.executable, "-u", "-m", *shard]
                print(f"CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}", flush=True)
            if args.execute:
                parallel_evaluations(shards, args.gpus)
            index += len(args.gpus)
            continue
        command = [sys.executable, "-m", *arguments]
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)
        index += 1


if __name__ == "__main__":
    main()
