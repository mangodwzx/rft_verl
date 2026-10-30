"""AST/config checks only: never import experiment modules or load model weights."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--ruff", action="store_true", help="Also invoke installed ruff as a static analyzer")
    args = parser.parse_args()
    root = args.root.resolve()
    errors, checked, trees = [], [], {}
    python_files = [path for directory in ("src", "scripts", "tests")
                    for path in sorted((root / directory).rglob("*.py"))]
    for path in python_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            trees[path.relative_to(root).as_posix()] = tree
            compile(tree, str(path), "exec")  # compile only, never exec/import
            # Detect duplicate top-level definitions (often accidental merge damage).
            definitions = set()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name in definitions:
                        errors.append(f"{path.name}: duplicate top-level definition {node.name}")
                    definitions.add(node.name)
            checked.append({"file": path.relative_to(root).as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        except (SyntaxError, ValueError, UnicodeError) as exc:
            errors.append(f"{path}: {exc}")
    for path in sorted((root / "configs").glob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
            checked.append({"file": path.relative_to(root).as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        except (ValueError, UnicodeError) as exc:
            errors.append(f"{path}: {exc}")
    try:
        import yaml

        for path in sorted((root / "configs").glob("*.yaml")):
            yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            checked.append({"file": path.relative_to(root).as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        base = yaml.safe_load((root / "configs/verl_base.yaml").read_text(encoding="utf-8"))
        actor = base["actor_rollout_ref"]["actor"]
        rollout = base["actor_rollout_ref"]["rollout"]
        if actor["ppo_mini_batch_size"] != base["data"]["train_batch_size"] or actor["ppo_epochs"] != 1:
            errors.append("Training configuration does not use one fresh-rollout update per prompt batch")
        if rollout["n"] != 8 or rollout["max_model_len"] < base["data"]["max_prompt_length"] + base["data"]["max_response_length"]:
            errors.append("Default group size or context budgets are inconsistent")
        if base["data"]["filter_overlong_prompts"] or base["algorithm"]["filter_groups"]["enable"]:
            errors.append("Default configuration silently filters the fixed question population")
        raw_agents = yaml.safe_load((root / "configs/verl_raw_agent.yaml").read_text(encoding="utf-8"))
        if raw_agents != [{"name": "raw_math", "_target_": "math_rft.raw_loop.RawMathAgentLoop"}]:
            errors.append("Raw agent config points to an unexpected implementation")
    except (ImportError, OSError, ValueError, KeyError) as exc:
        errors.append(f"YAML static validation unavailable/failed: {exc}")
    try:
        tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        experiment = json.loads((root / "configs/experiment.json").read_text(encoding="utf-8"))
        models = json.loads((root / "configs/models.json").read_text(encoding="utf-8"))
        if set(models) != set(experiment["models"]) or len(models) != 4:
            errors.append("Model registry and experiment matrix do not describe the same four models")
        if experiment["max_k"] != 256 or experiment["formal_samples_per_question"] < 256:
            errors.append("Formal sampling cannot support required pass@256")
        if experiment["development_samples_per_question"] < 256:
            errors.append("Pilot development sampling cannot support required pass@256")
        if experiment["report_ks"][-1] != 256 or len(set(experiment["training_seeds"])) < 3:
            errors.append("Missing k=256 endpoint or three distinct training seeds")
        if len(experiment["verl_commit"]) != 40:
            errors.append("verl must be pinned to a full commit SHA")
        # Extract only literal constants from syntax trees; never import modules.
        constants = {}
        for relative in ("src/math_rft/compatibility.py", "src/math_rft/train.py"):
            for node in trees[relative].body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id in {"VERL_COMMIT", "METHODS"}:
                        constants[node.targets[0].id] = ast.literal_eval(node.value)
        if constants["VERL_COMMIT"] != experiment["verl_commit"]:
            errors.append("Training compatibility guard and experiment use different verl commits")
        if set(experiment["main_methods"]) != set(constants["METHODS"]) or set(constants["METHODS"]) != {"rft", "grpo"}:
            errors.append("The focused experiment must expose only RFT and GRPO")
        registrations = {"register_adv_est": {"grpo"}, "register_policy_loss": set()}
        for node in trees["src/math_rft/algorithms.py"].body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id in registrations:
                        registrations[decorator.func.id].add(ast.literal_eval(decorator.args[0]))
        for method, (advantage, loss, _) in constants["METHODS"].items():
            if advantage not in registrations["register_adv_est"] or loss not in registrations["register_policy_loss"]:
                errors.append(f"Method {method} refers to an unregistered algorithm/loss")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(str(exc))
    ruff_output = None
    if args.ruff:
        process = subprocess.run([sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"],
                                 cwd=root, text=True, capture_output=True, check=False)
        ruff_output = process.stdout + process.stderr
        if process.returncode:
            errors.append("ruff failed; see ruff_output")
    report = {"check_type": "static_only", "status": "passed" if not errors else "failed",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "files": checked,
              "errors": errors, "ruff_output": ruff_output,
              "not_executed": ["training", "model generation", "dataset preparation", "GPU/CUDA tests", "dynamic unit tests"]}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(f"Static check: {report['status']}; {len(python_files)} Python files; {len(errors)} errors")
    for error in errors:
        print(error)
    if ruff_output:
        print(ruff_output)
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
