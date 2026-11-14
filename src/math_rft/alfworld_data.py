"""Prepare frozen ALFWorld tasks for online training and independent episodes.

Reads official local ALFWorld JSON/TextWorld assets. ``train`` supplies pilot
and full, ``valid_seen`` supplies dev, and ``valid_unseen`` is locked as test.
The published counts are not assumed: every official-loader exclusion is
reported, and every included task is reset once before writing the manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any
import urllib.request
import zipfile

from .alfworld_env import AlfWorldEnvironment, environment_identity
from .io_utils import canonical_json, read_jsonl, sha256_file, sha256_text, write_json, write_jsonl

TASK_TYPES = (
    "pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
)
SOURCE_SPLITS = ("train", "valid_seen", "valid_unseen")
DATA_VERSION = "alfworld_tasks_v1"
FILTER_SOURCE = "https://github.com/alfworld/alfworld/blob/0.4.2/alfworld/agents/environment/alfred_tw_env.py"
TEXT_ARCHIVES = (
    ("json_2.1.1_json.zip", "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip",
     72018818, "25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc"),
    ("json_2.1.3_tw-pddl.zip", "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip",
     36507267, "5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf"),
)


def _extract_text_assets(archive: Path, destination: Path) -> dict[str, int]:
    """Extract only task metadata and executable text games from official zips."""
    files, size = 0, 0
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe ALFWorld archive path: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"ALFWorld archive contains symlink: {member.filename}")
            if member.is_dir() or path.name not in {"traj_data.json", "game.tw-pddl"}:
                continue
            if len(path.parts) < 4 or path.parts[0] != "json_2.1.1" or path.parts[1] not in SOURCE_SPLITS:
                continue
            target = destination / path
            if target.exists():
                raise ValueError(f"Duplicate archive member: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            files += 1
            size += member.file_size
    if files == 0:
        raise ValueError(f"No official train/valid_seen/valid_unseen text assets found in {archive}")
    return {"files": files, "uncompressed_bytes": size}


def download(args: argparse.Namespace) -> None:
    """Fetch only fixed official text-data releases, committing after both succeed."""
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty ALFWorld data directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.download-", dir=output.parent))
    sources = []
    try:
        for filename, url, expected_size, expected_sha256 in TEXT_ARCHIVES:
            archive = staging / filename
            request = urllib.request.Request(url, headers={"User-Agent": "rft-verl-alfworld-data/1"})
            print(f"Downloading {filename} ({expected_size / 1024**2:.1f} MiB)", flush=True)
            with urllib.request.urlopen(request, timeout=60) as response, archive.open("xb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
            if archive.stat().st_size != expected_size:
                raise ValueError(f"Official archive size changed for {filename}; expected {expected_size} bytes")
            digest = sha256_file(archive)
            if digest != expected_sha256:
                raise ValueError(f"Official archive checksum changed for {filename}; refusing unverified data")
            source = {"url": url, "bytes": expected_size, "sha256": digest}
            source["extracted"] = _extract_text_assets(archive, staging)
            sources.append(source)
            archive.unlink()
            print(f"Extracted {source['extracted']['files']} text assets from {filename}", flush=True)
        resolve_data_root(staging)
        write_json(staging / "download_manifest.json", {
            "benchmark": "alfworld", "text_only": True,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "archives": sources,
        })
        if output.exists():
            if any(output.iterdir()):
                raise FileExistsError(f"Download destination became nonempty: {output}")
            output.rmdir()
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Downloaded ALFWorld text assets to {output}")


def resolve_data_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    candidates = [path for path in (root, root / "json_2.1.1")
                  if all((path / split).is_dir() for split in SOURCE_SPLITS)]
    if len(candidates) != 1:
        raise ValueError("--data-root must contain train/valid_seen/valid_unseen, directly or under json_2.1.1")
    return candidates[0]


def discover_tasks(root: Path, source_split: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Mirror the official loader's eligibility rules, making exclusions visible."""
    if source_split not in SOURCE_SPLITS:
        raise ValueError(f"Unknown ALFWorld source split: {source_split}")
    split_root = root / source_split
    trajectories = sorted(split_root.rglob("traj_data.json"))
    if not trajectories:
        raise ValueError(f"No traj_data.json files found in {split_root}")
    tasks, excluded = [], []
    for trajectory in trajectories:
        relative = trajectory.parent.relative_to(split_root).as_posix()
        source_id = f"{source_split}/{relative}"
        game = trajectory.with_name("game.tw-pddl")
        reason = None
        if "movable" in relative or "Sliced" in relative:
            reason = "unsupported_movable_or_sliced"
        else:
            data = json.loads(trajectory.read_text(encoding="utf-8"))
            task_type = data.get("task_type")
            if task_type not in TASK_TYPES:
                reason = "unsupported_task_type"
            elif not game.is_file():
                reason = "missing_game"
            else:
                game_data = json.loads(game.read_text(encoding="utf-8"))
                if "solvable" not in game_data:
                    reason = "missing_solvable_flag"
                elif not game_data["solvable"]:
                    reason = "unsolvable"
        if reason:
            excluded.append({"source_id": source_id, "reason": reason})
            continue
        tasks.append({
            "source_id": source_id, "source_split": source_split, "task_type": task_type,
            "game_file": str(game.resolve()), "game_sha256": sha256_file(game),
            "trajectory_file": str(trajectory.resolve()), "trajectory_sha256": sha256_file(trajectory),
        })
    if not tasks:
        raise ValueError(f"No eligible solvable ALFWorld tasks in {split_root}")
    return tasks, excluded


def validate_disjoint_tasks(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Reject duplicate games or copied task directories instead of dropping them."""
    seen_games, seen_paths = {}, {}
    for split, tasks in splits.items():
        for task in tasks:
            source_id = task["source_id"]
            for key, registry in ((task["game_sha256"], seen_games),
                                  (source_id.split("/", 1)[1], seen_paths)):
                if key in registry:
                    raise ValueError(f"Duplicate/overlapping ALFWorld tasks: {registry[key]} and {source_id} ({split})")
                registry[key] = source_id


def select_pilot(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if type(count) is not int or count <= 0 or count > len(rows):
        raise ValueError(f"pilot_size must be between 1 and the available {len(rows)} training tasks")
    groups = defaultdict(list)
    for row in rows:
        groups[row["subject"]].append(row)
    keys = sorted(groups)
    targets = {key: count * len(groups[key]) / len(rows) for key in keys}
    allocated = {key: math.floor(targets[key]) for key in keys}
    for key in sorted(keys, key=lambda key: (-(targets[key] - allocated[key]), key))[:count - sum(allocated.values())]:
        allocated[key] += 1
    result = []
    for key in keys:
        ordered = sorted(groups[key], key=lambda row: sha256_text(f"{seed}:alfworld:pilot:{row['id']}"))
        result.extend(ordered[:allocated[key]])
    return sorted(result, key=lambda row: row["id"])


def task_row(task: dict[str, Any], observation: str) -> dict[str, Any]:
    from .alfworld_episode import render_prompt

    task = {**task, "initial_observation": observation}
    identity = sha256_text(canonical_json({"source_id": task["source_id"], "game_sha256": task["game_sha256"]}))
    return {
        "id": f"alfworld:{task['source_id']}", "problem": observation, "prompt": render_prompt(observation),
        "subject": task["task_type"], "level": None, "difficulty": None,
        "data_source": "alfworld", "source_split": task["source_split"], "source_id": task["source_id"],
        "problem_sha256": identity, "extra_info": {"alfworld_task": task},
    }


def _observe_task(task: dict[str, Any]) -> dict[str, Any]:
    """Spawn-safe CPU worker; each process owns its planner pool/parser state."""
    with AlfWorldEnvironment(task, seed=0) as env:
        observation = env.reset()
    return task_row(task, observation)


def _write_split(output: Path, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{**row, "split": name} for row in rows]
    jsonl, parquet = output / f"{name}.jsonl", output / f"{name}.parquet"
    write_jsonl(jsonl, rows)
    training_rows = [{
        "data_source": "alfworld", "prompt": [{"role": "user", "content": row["prompt"]}],
        "ability": "agent", "reward_model": {"style": "rule", "ground_truth": row["id"]},
        "extra_info": {**row["extra_info"], "index": index, "id": row["id"], "split": name,
                       "subject": row["subject"], "level": None, "problem_sha256": row["problem_sha256"]},
    } for index, row in enumerate(rows)]
    pq.write_table(pa.Table.from_pylist(training_rows), parquet)
    return {"count": len(rows), "ids": [row["id"] for row in rows],
            "jsonl_sha256": sha256_file(jsonl), "parquet_sha256": sha256_file(parquet),
            "strata": dict(sorted(Counter(row["subject"] for row in rows).items()))}


def prepare(args: argparse.Namespace) -> None:
    from .alfworld_episode import prompt_fingerprint

    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    workers = getattr(args, "workers", 1)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty dataset directory: {output}")
    root = resolve_data_root(args.data_root)
    source_tasks, excluded = {}, {}
    for split in SOURCE_SPLITS:
        source_tasks[split], excluded[split] = discover_tasks(root, split)
    validate_disjoint_tasks(source_tasks)
    if not 0 < args.pilot_size <= len(source_tasks["train"]):
        raise ValueError(f"pilot_size exceeds eligible training tasks ({len(source_tasks['train'])})")
    identity = environment_identity()
    rows = {}
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) if workers > 1 else None
    try:
        for split, tasks in source_tasks.items():
            rows[split] = []
            observed = pool.map(_observe_task, tasks, chunksize=4) if pool is not None else map(_observe_task, tasks)
            for index, row in enumerate(observed):
                rows[split].append(row)
                if (index + 1) % 100 == 0 or index + 1 == len(tasks):
                    print(f"ALFWorld {split}: verified {index + 1}/{len(tasks)} initial states", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    full, dev, test = (rows[split] for split in SOURCE_SPLITS)
    splits = {"pilot": select_pilot(full, args.pilot_size, args.seed), "full": full, "dev": dev, "test": test}
    # Debug is a fixed training subset, never a subset selected from final tests.
    splits["debug"] = sorted(full, key=lambda row: sha256_text(f"{args.seed}:debug:{row['id']}"))[:32]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
    try:
        manifest = {
            "schema_version": 1, "benchmark": "alfworld", "dataset_version": DATA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "data_root": str(root),
            "seed": args.seed, "pilot_size": args.pilot_size, "prompt_sha256": prompt_fingerprint(),
            "preparation_workers": workers,
            "environment": identity, "eligibility_reference": FILTER_SOURCE,
            "split_policy": {"pilot": "stratified subset of train", "full": "all eligible train",
                             "dev": "valid_seen", "test": "valid_unseen", "debug": "up to 32 train tasks"},
            "excluded": excluded,
            "source_counts": {split: {"eligible": len(source_tasks[split]), "excluded": len(excluded[split])}
                              for split in SOURCE_SPLITS},
            "splits": {name: _write_split(staging, name, values) for name, values in splits.items()},
        }
        download_record = root.parent / "download_manifest.json"
        if download_record.is_file():
            manifest["download_provenance"] = json.loads(download_record.read_text(encoding="utf-8"))
        write_json(staging / "manifest.json", manifest)
        # Replace only an empty destination checked at commit time as well.
        if output.exists():
            if any(output.iterdir()):
                raise FileExistsError(f"Dataset destination became nonempty: {output}")
            output.rmdir()
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Prepared ALFWorld dataset at {output}: " + ", ".join(f"{name}={len(values)}" for name, values in splits.items()))


def smoke(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError("No ALFWorld tasks in --data")
    row = next((row for row in rows if row["id"] == args.task_id), None) if args.task_id else rows[0]
    if row is None:
        raise ValueError(f"Task ID not found: {args.task_id}")
    task = row["extra_info"]["alfworld_task"]
    with AlfWorldEnvironment(task, max_steps=args.max_steps, seed=args.seed) as env:
        print(json.dumps({"id": row["id"], "observation": env.reset()}, ensure_ascii=False))
        for action in args.actions:
            observation, success, done, info = env.step(action)
            print(json.dumps({"action": action, "observation": observation,
                              "success": success, "done": done, "info": info}, ensure_ascii=False))
            if done:
                break


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Download official text-only assets without visual models")
    fetch.add_argument("--output-dir", type=Path, default=Path("vendor/alfworld_data"))
    fetch.set_defaults(function=download)
    prep = commands.add_parser("prepare", help="Validate local official text-game assets and prepare all task splits")
    prep.add_argument("--data-root", required=True, type=Path)
    prep.add_argument("--output-dir", type=Path, default=Path("data_alfworld"))
    prep.add_argument("--pilot-size", type=int, default=2048)
    prep.add_argument("--seed", type=int, default=20261025)
    prep.add_argument("--workers", type=int, default=4,
                      help="CPU processes for initial-state validation; use 1 for serial execution")
    prep.set_defaults(function=prepare)
    check = commands.add_parser("smoke", help="Reset and execute explicit commands without loading a language model")
    check.add_argument("--data", type=Path, required=True)
    check.add_argument("--task-id")
    check.add_argument("--actions", nargs="+", default=["look"])
    check.add_argument("--max-steps", type=int, default=50)
    check.add_argument("--seed", type=int, default=0)
    check.set_defaults(function=smoke)
    return cli


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
