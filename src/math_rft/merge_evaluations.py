"""Validate and combine complete question shards without generating new samples."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .analyze import load_run
from .evaluate import artifact_paths, build_summary, canonical_hash, file_hash, load_data, read_jsonl

SCHEME = "question_round_robin_v1"


def common_config(config: dict) -> dict:
    return {key: value for key, value in config.items()
            if key not in {"sharding", "dataset_sha256", "questions", "question_ids"}}


def signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def merge(inputs: list[Path], data: Path, output: Path) -> None:
    inputs = [path.expanduser().resolve() for path in inputs]
    output = output.expanduser().resolve()
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("Provide distinct shard inputs")
    paths = artifact_paths(output)
    targets = {output, *paths.values()}
    sources = {path for source in inputs for path in (source, *artifact_paths(source).values())}
    if targets & sources or data.expanduser().resolve() in targets:
        raise ValueError("Merged output must not overwrite an input or its artifacts")
    snapshots = {path: signature(path) for source in inputs
                 for path in (source, artifact_paths(source)["meta"], artifact_paths(source)["prompts"])}
    runs = [load_run(path) for path in inputs]
    reference = runs[0]["config"]
    n = reference["sampling"]["n"]
    if type(n) is not int or n < 1:
        raise ValueError("Invalid sample budget in shard metadata")
    rows = load_data(data, reference["role"], n, reference.get("benchmark", "math"))
    full_ids = [row["id"] for row in rows]
    full_hash = canonical_hash(rows)
    count = len(inputs)
    indexed, token_rows, counts, provenance = {}, {}, {}, []
    for run, source in zip(runs, inputs, strict=True):
        config = run["config"]
        shard = config.get("sharding", {})
        index = shard.get("index")
        if type(index) is not int or not 0 <= index < count or index in indexed:
            raise ValueError("Missing or duplicate shard index")
        expected_sharding = {"index": index, "count": count, "full_dataset_sha256": full_hash,
                             "full_question_ids": full_ids}
        if shard != expected_sharding:
            raise ValueError("Shard count or full dataset identity does not match the merge")
        if common_config(config) != common_config(reference):
            raise ValueError("Shard model, execution or scientific configuration differs")
        expected_rows = rows[index::count]
        if not expected_rows or run["rows"] != expected_rows or config.get("questions") != len(expected_rows):
            raise ValueError("Shard does not contain its expected round-robin question allocation")
        indexed[index] = (run, source)
        counts.update(run["counts"])
        for row in read_jsonl(artifact_paths(source)["prompts"]):
            if not isinstance(row.get("prompt_token_ids"), list) or not row["prompt_token_ids"]:
                raise ValueError("Shard prompt index has invalid token IDs")
            token_rows[row["id"]] = row
        provenance.append({"index": index, "path": str(source),
                           "manifest_hash": run["meta"]["manifest_hash"],
                           "ledger_sha256": file_hash(source),
                           "prompts_sha256": file_hash(artifact_paths(source)["prompts"])})
    if set(indexed) != set(range(count)) or set(counts) != set(full_ids):
        raise ValueError("Shards do not cover the complete question set")
    provenance.sort(key=lambda item: item["index"])
    config = {**common_config(reference), "dataset_sha256": full_hash,
              "questions": len(rows), "question_ids": full_ids,
              "sharding": {"count": count, "scheme": SCHEME}}
    manifest_hash = canonical_hash(config)
    summary = build_summary(rows, counts, manifest_hash, n)

    def unchanged_sources() -> None:
        if any(artifact_paths(source)["lock"].exists() for source in inputs):
            raise ValueError("A source shard became active during merging")
        if any(signature(path) != value for path, value in snapshots.items()):
            raise ValueError("A source shard changed during merging")

    unchanged_sources()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock = paths["lock"].open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Merge output is locked: {paths['lock']}") from exc
    try:
        lock.write(f"pid={os.getpid()} operation=merge\n")
        lock.flush()
        final_artifacts = (output, paths["meta"], paths["prompts"], paths["summary"])
        existing = [path.exists() for path in final_artifacts]
        if any(existing):
            if not paths["meta"].exists():
                raise ValueError("Existing merge artifacts lack provenance; use a new output path")
            old = json.loads(paths["meta"].read_text(encoding="utf-8"))
            if (old.get("config") != config or old.get("manifest_hash") != manifest_hash
                    or old.get("source_shards") != provenance):
                raise ValueError("Existing output has different merge provenance; use a new output path")
            if all(existing):
                if (file_hash(output) != old.get("ledger_sha256")
                        or file_hash(paths["prompts"]) != old.get("prompts_sha256")
                        or json.loads(paths["summary"].read_text(encoding="utf-8")) != summary):
                    raise ValueError("Existing complete merge artifacts were modified")
                unchanged_sources()
                print(f"Already merged: {output}")
                return
            # Metadata is published first and identifies an interrupted merge.
            # Matching incomplete transactions can be rebuilt from immutable shards.

        offsets = {}
        for index, (_, source) in indexed.items():
            with source.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    record = json.loads(line)
                    key = (record["id"], record["sample_index"])
                    if key in offsets:
                        raise ValueError("Duplicate sample while indexing shards")
                    offsets[key] = (index, offset)
        if len(offsets) != len(rows) * n:
            raise ValueError("Incomplete samples while indexing shards")
        with tempfile.TemporaryDirectory(prefix=".merge-", dir=output.parent) as temporary:
            stage = Path(temporary)
            ledger_hash = hashlib.sha256()
            with ExitStack() as stack:
                readers = {index: stack.enter_context(source.open("rb")) for index, (_, source) in indexed.items()}
                merged = stack.enter_context((stage / "samples.jsonl").open("wb"))
                for row in rows:
                    for sample_index in range(n):
                        index, offset = offsets[(row["id"], sample_index)]
                        readers[index].seek(offset)
                        line = readers[index].readline()
                        if not line.endswith(b"\n"):
                            line += b"\n"
                        merged.write(line)
                        ledger_hash.update(line)
                merged.flush()
                os.fsync(merged.fileno())
            with (stage / "prompts.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(token_rows[row["id"]], ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            metadata = {"manifest_hash": manifest_hash, "config": config, "source_shards": provenance,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "merger_sha256": file_hash(Path(__file__)), "ledger_sha256": ledger_hash.hexdigest(),
                        "prompts_sha256": file_hash(stage / "prompts.jsonl"),
                        "reproducibility_note": "Question-shard ledgers merged without regenerating or rescoring samples; source execution manifests and content hashes are retained."}
            write_json(stage / "meta.json", metadata)
            write_json(stage / "summary.json", summary)
            unchanged_sources()
            # Publish provenance before data so an interrupted multi-file commit
            # can be distinguished from unrelated output on the next invocation.
            for source_name, target in (("meta.json", paths["meta"]), ("prompts.jsonl", paths["prompts"]),
                                        ("samples.jsonl", output), ("summary.json", paths["summary"])):
                (stage / source_name).replace(target)
        print(f"Merged {count} shards: {len(rows) * n} samples -> {output}")
    finally:
        lock.close()
        paths["lock"].unlink()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    merge(args.inputs, args.data, args.output)


if __name__ == "__main__":
    main()
