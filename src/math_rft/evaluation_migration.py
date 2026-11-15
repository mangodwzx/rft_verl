"""Copy an immutable ALFWorld partial ledger into a new execution lineage.

The experiment protocol and every existing sample stay unchanged. Execution
code changes are explicit provenance, never edits to the source metadata.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

from .evaluate import artifact_paths, atomic_json, canonical_hash, file_hash, read_existing, read_jsonl

EXECUTION_FIELDS = frozenset({"evaluator_sha256", "ledger_source_sha256", "environment_execution"})


@contextmanager
def _lock(path: Path, *, allow_current_process: bool = False):
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        if allow_current_process and path.read_text(encoding="utf-8").splitlines() == [f"pid={os.getpid()}"]:
            yield
            return
        raise ValueError(f"Migration source or destination is locked: {path}") from exc
    try:
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        yield
    finally:
        stream.close()
        path.unlink()


def _signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy(source: Path, target: Path) -> None:
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _prefix_hash(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while size:
            block = stream.read(min(size, 8 * 1024 * 1024))
            if not block:
                raise ValueError("Migrated ledger lost records from its source")
            digest.update(block)
            size -= len(block)
    return digest.hexdigest()


def migrate_evaluation(source: Path, output: Path, rows: list[dict], config: dict) -> dict:
    """Import source records without regeneration, returning destination metadata.

    The destination may be locked by this process's normal evaluator. A source
    lock always rejects migration, even when stale: the owner must stop first.
    An interrupted publication can be retried with the same arguments. The
    transaction marker is removed only after all three artifacts are durable.
    """
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    source_paths, paths = artifact_paths(source), artifact_paths(output)
    source_files = {path.resolve() for path in (source, *source_paths.values())}
    target_files = {path.resolve() for path in (output, *paths.values())}
    if source_files & target_files:
        raise ValueError("Migration requires separate source and destination artifacts")
    if config.get("benchmark") != "alfworld":
        raise ValueError("Execution migration is supported only for ALFWorld")
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = output.with_name(output.name + ".migration.json")
    with _lock(paths["lock"], allow_current_process=True), _lock(source_paths["lock"]):
        watched = (source, source_paths["meta"], source_paths["prompts"])
        snapshots = {path: _signature(path) for path in watched}
        source_metadata = json.loads(source_paths["meta"].read_text(encoding="utf-8"))
        old = source_metadata.get("config")
        if not isinstance(old, dict) or canonical_hash(old) != source_metadata.get("manifest_hash"):
            raise ValueError("Migration source manifest hash mismatch")
        missing = object()
        changed = {key for key in old.keys() | config.keys()
                   if old.get(key, missing) != config.get(key, missing)}
        forbidden = changed - EXECUTION_FIELDS
        if forbidden:
            raise ValueError(f"Migration changes experiment configuration: {sorted(forbidden)}")
        ids = [row["id"] for row in rows]
        if (len(set(ids)) != len(ids) or old.get("question_ids") != ids
                or old.get("questions") != len(rows) or old.get("dataset_sha256") != canonical_hash(rows)):
            raise ValueError("Migration source dataset identity mismatch")
        prompts = list(read_jsonl(source_paths["prompts"]))
        unindexed = [{key: value for key, value in row.items() if key != "prompt_token_ids"} for row in prompts]
        if unindexed != rows or any(
                not isinstance(row.get("prompt_token_ids"), list) or not row["prompt_token_ids"]
                or any(type(token) is not int or token < 0 for token in row["prompt_token_ids"])
                for row in prompts):
            raise ValueError("Migration source prompt index mismatch")
        sampling = old["sampling"]
        if type(sampling.get("n")) is not int or sampling["n"] < 1 or type(sampling.get("seed")) is not int:
            raise ValueError("Migration source has invalid sampling configuration")
        completed, _ = read_existing(source, rows, sampling["n"], sampling["seed"])
        source_size = source.stat().st_size
        if source_size:
            with source.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    raise ValueError("Migration source must end at a committed newline")
        provenance = {
            "path": str(source), "manifest_hash": source_metadata["manifest_hash"],
            "ledger_sha256": file_hash(source), "ledger_bytes": source_size,
            "prompts_sha256": file_hash(source_paths["prompts"]),
            "metadata_sha256": file_hash(source_paths["meta"]),
            "records": len(completed), "source_metadata": source_metadata,
            "changed_execution_fields": sorted(changed),
        }
        metadata = {
            "manifest_hash": canonical_hash(config), "config": config,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "resume_source": provenance,
            "reproducibility_note": (
                "Source records are retained byte-for-byte. Only missing samples use the new execution "
                "implementation; scheduling changes may alter seeded generation. Source metadata and hashes "
                "identify the imported prefix separately from newly generated records."
            ),
        }

        def unchanged_source() -> None:
            if any(_signature(path) != signature for path, signature in snapshots.items()):
                raise ValueError("Migration source changed while being copied")

        unchanged_source()
        # Completed migrations are idempotent, including after new samples were
        # appended. Verify the immutable imported prefix before accepting them.
        if paths["meta"].exists():
            existing = json.loads(paths["meta"].read_text(encoding="utf-8"))
            if (existing.get("config") != config or existing.get("manifest_hash") != canonical_hash(config)
                    or existing.get("resume_source") != provenance):
                raise ValueError("Destination has different migration provenance or configuration")
            if (_prefix_hash(output, source_size) != provenance["ledger_sha256"]
                    or file_hash(paths["prompts"]) != provenance["prompts_sha256"]):
                raise ValueError("Destination imported records or prompts were modified")
            read_existing(output, rows, sampling["n"], sampling["seed"])
            unchanged_source()
            if marker.exists():
                transaction = json.loads(marker.read_text(encoding="utf-8"))
                if transaction.get("metadata") != existing:
                    raise ValueError("Destination migration transaction does not match metadata")
                marker.unlink()
                _sync_directory(output.parent)
            return existing

        if paths["summary"].exists():
            raise ValueError("Migration destination already has an unrelated summary")
        transaction = {"schema_version": 1, "metadata": metadata}
        if marker.exists():
            transaction = json.loads(marker.read_text(encoding="utf-8"))
            previous = transaction.get("metadata", {})
            if (previous.get("config") != config or previous.get("manifest_hash") != canonical_hash(config)
                    or previous.get("resume_source") != provenance):
                raise ValueError("Unfinished migration has different provenance or configuration")
            metadata = previous
            for path, expected in ((output, provenance["ledger_sha256"]),
                                   (paths["prompts"], provenance["prompts_sha256"])):
                if path.exists() and file_hash(path) != expected:
                    raise ValueError("Unfinished migration artifacts were modified")
        elif output.exists() or paths["prompts"].exists():
            raise ValueError("Migration destination must be empty or have a matching transaction")

        with tempfile.TemporaryDirectory(prefix=".evaluation-migration-", dir=output.parent) as temporary:
            staged = Path(temporary)
            _copy(source, staged / "ledger")
            _copy(source_paths["prompts"], staged / "prompts")
            if (file_hash(staged / "ledger") != provenance["ledger_sha256"]
                    or file_hash(staged / "prompts") != provenance["prompts_sha256"]):
                raise ValueError("Migration source changed while being copied")
            unchanged_source()
            # Marker first, metadata last: neither a crash nor partial publish
            # may turn an incomplete import into an ordinary resumable ledger.
            atomic_json(marker, transaction)
            _sync_directory(output.parent)
            (staged / "ledger").replace(output)
            (staged / "prompts").replace(paths["prompts"])
            atomic_json(paths["meta"], metadata)
            _sync_directory(output.parent)
            marker.unlink()
            _sync_directory(output.parent)
        return metadata
