"""Resolve immutable Hub models locally before making any network request."""
from __future__ import annotations

import json
from pathlib import Path
import re


def hub_revision(repo_id: str, revision: str | None = None) -> str:
    from huggingface_hub import HfApi, constants

    if revision is not None and re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        return revision.lower()
    if constants.HF_HUB_OFFLINE:
        raise ValueError("Offline loading requires an immutable 40-character --model-revision/--revision; "
                         "do not resolve a mutable branch such as main offline")
    resolved = HfApi().model_info(repo_id, revision=revision or "main").sha
    if not isinstance(resolved, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", resolved):
        raise RuntimeError(f"Could not resolve immutable model revision for {repo_id}")
    return resolved.lower()


def require_snapshot_files(path: Path) -> None:
    """Reject partial caches before starting training or a vLLM engine."""
    required = [path / "config.json", path / "tokenizer_config.json"]
    for file in required:
        if not file.is_file() or file.stat().st_size == 0:
            raise FileNotFoundError(f"Incomplete model cache: missing or empty {file}")
    if not any((path / name).is_file() and (path / name).stat().st_size > 0
               for name in ("tokenizer.json", "tokenizer.model", "spiece.model")):
        raise FileNotFoundError(f"Incomplete model cache: no tokenizer in {path}")
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if not weights or any(not file.is_file() or file.stat().st_size == 0 for file in weights):
        raise FileNotFoundError(f"Incomplete model cache: missing or empty model weights in {path}")
    for index in path.glob("*.index.json"):
        for shard in set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values()):
            file = path / shard
            if not file.is_file() or file.stat().st_size == 0:
                raise FileNotFoundError(f"Incomplete model cache: missing shard {file}")


def hub_snapshot(repo_id: str, revision: str | None = None) -> tuple[str, str]:
    """Keep Hub revision provenance even though workers receive a local path."""
    from huggingface_hub import constants, snapshot_download

    resolved = hub_revision(repo_id, revision)
    try:
        path = Path(snapshot_download(repo_id, revision=resolved, local_files_only=True))
        require_snapshot_files(path)
    except FileNotFoundError as exc:
        if constants.HF_HUB_OFFLINE:
            raise FileNotFoundError(
                f"Offline model cache is incomplete for {repo_id}@{resolved}: {exc}. "
                "Use scripts/activate.sh to select the project cache, or download this revision while online."
            ) from exc
        path = Path(snapshot_download(repo_id, revision=resolved))
        require_snapshot_files(path)
    return str(path.resolve()), resolved
