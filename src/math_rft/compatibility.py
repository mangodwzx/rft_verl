"""Fail closed when training against an unaudited verl revision."""
from __future__ import annotations

import importlib.metadata
import json
import subprocess
from pathlib import Path

VERL_COMMIT = "3c6d9b43b67c74ec343d77874cd8fa6e841523da"


def check_verl_revision() -> dict[str, str]:
    import verl

    source_root = Path(verl.__file__).resolve().parent.parent
    installed_commit = None
    if (source_root / ".git").exists():
        installed_commit = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        # Local source edits could invalidate the audited interfaces even at HEAD.
        diff = subprocess.check_output(
            ["git", "-C", str(source_root), "diff", "HEAD", "--", "verl"], text=True
        )
        if diff.strip():
            raise RuntimeError("Pinned verl has local modifications under verl/; use a clean checkout")
    else:
        direct_url = importlib.metadata.distribution("verl").read_text("direct_url.json")
        if direct_url:
            installed_commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    if installed_commit != VERL_COMMIT:
        raise RuntimeError(f"Expected verl {VERL_COMMIT}, found {installed_commit}; use scripts/setup_linux.sh")
    return {"commit": installed_commit, "source_root": str(source_root)}


def environment_versions() -> dict[str, str]:
    packages = ("verl", "torch", "vllm", "transformers", "ray", "datasets", "math-verify", "flash-attn")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions
