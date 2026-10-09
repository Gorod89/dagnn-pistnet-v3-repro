from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import V3ExperimentConfig


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _cached_sha256(path: str, size: int, mtime_ns: int) -> str:
    # Size and nanosecond mtime are cache keys only; the manifest still stores
    # the content digest. A long publication study hashes each immutable raw file once.
    del size, mtime_ns
    return sha256_file(path)


def canonical_config_hash(config: V3ExperimentConfig) -> str:
    payload = json.dumps(
        asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", *command],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_reproducibility_manifest(
    config: V3ExperimentConfig,
    workspace: str | Path,
    command: list[str] | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    paths = {
        "data": config.data.data_path,
        "adjacency": config.data.adjacency_path,
        "edge_attributes": config.data.edge_attributes_path,
        "clusters": config.data.cluster_index_path,
        "events": config.data.event_features_path,
        "segment_volume": config.data.segment_volume_path,
        "boundary_flow": config.data.boundary_flow_path,
        "pretrained_encoder": config.training.pretrained_encoder_path,
    }
    fingerprints: dict[str, Any] = {}
    for name, raw_path in paths.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = workspace / path
        if path.exists() and path.is_file():
            stat = path.stat()
            fingerprints[name] = {
                "path": str(path.resolve()),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _cached_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns),
            }
        else:
            fingerprints[name] = {"path": str(path), "missing": True}
    dirty = _git(["status", "--porcelain"], workspace)
    if config.study.fail_on_dirty_worktree and dirty:
        raise RuntimeError("publication run refused: git worktree is dirty")
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command or sys.argv,
        "working_directory": str(workspace),
        "config": asdict(config),
        "config_sha256": canonical_config_hash(config),
        "data_fingerprints": fingerprints,
        "git": {
            "commit": _git(["rev-parse", "HEAD"], workspace),
            "branch": _git(["branch", "--show-current"], workspace),
            "dirty": bool(dirty),
            "status": dirty.splitlines() if dirty else [],
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def validate_resume_manifest(
    path: str | Path,
    config: V3ExperimentConfig,
    workspace: str | Path,
    allowed_mismatches: dict[str, tuple[Any, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Refuse to mix completed runs from incompatible code, data, or environments."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise RuntimeError(f"completed run has no reproducibility manifest: {path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        previous = json.load(stream)
    current = build_reproducibility_manifest(config, workspace)
    comparisons = {
        "schema_version": (previous.get("schema_version"), current["schema_version"]),
        "config_sha256": (previous.get("config_sha256"), current["config_sha256"]),
        "data_fingerprints": (
            previous.get("data_fingerprints"),
            current["data_fingerprints"],
        ),
        "git.commit": (
            previous.get("git", {}).get("commit"),
            current["git"]["commit"],
        ),
    }
    environment_keys = ("python", "numpy", "torch", "cuda_runtime", "cudnn", "gpu")
    for key in environment_keys:
        comparisons[f"environment.{key}"] = (
            previous.get("environment", {}).get(key),
            current["environment"][key],
        )
    mismatches = {
        name: {"previous": values[0], "current": values[1]}
        for name, values in comparisons.items()
        if values[0] != values[1]
    }
    allowed_mismatches = allowed_mismatches or {}
    rejected = []
    for name, values in mismatches.items():
        expected = allowed_mismatches.get(name)
        if expected != (values["previous"], values["current"]):
            rejected.append(name)
    if rejected:
        joined = ", ".join(rejected)
        raise RuntimeError(
            "refusing to resume an incompatible completed run; "
            f"mismatched provenance: {joined}. Use a new output directory or force rerun."
        )
    return mismatches
