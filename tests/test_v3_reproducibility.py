from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from dagnn_pistnet_v3.config import V3ExperimentConfig
from dagnn_pistnet_v3.reproducibility import (
    build_reproducibility_manifest,
    validate_resume_manifest,
    write_manifest,
)


def _config() -> V3ExperimentConfig:
    return V3ExperimentConfig.from_dict(
        {
            "data": {
                "dataset": "synthetic",
                "data_path": "missing.npy",
                "feature_names": ["flow"],
                "units": ["vehicles_per_step"],
            },
            "training": {"epochs": 1, "output_dir": "unused"},
        }
    )


def test_resume_manifest_accepts_identical_provenance(tmp_path: Path) -> None:
    config = _config()
    path = tmp_path / "manifest.json"
    write_manifest(path, build_reproducibility_manifest(config, tmp_path))
    validate_resume_manifest(path, config, tmp_path)


def test_resume_manifest_rejects_changed_config(tmp_path: Path) -> None:
    config = _config()
    path = tmp_path / "manifest.json"
    write_manifest(path, build_reproducibility_manifest(config, tmp_path))
    raw = asdict(config)
    raw["training"]["learning_rate"] = 0.123
    changed = V3ExperimentConfig.from_dict(raw)
    with pytest.raises(RuntimeError, match="config_sha256"):
        validate_resume_manifest(path, changed, tmp_path)


def test_resume_manifest_allows_only_exact_declared_transition(tmp_path: Path) -> None:
    config = _config()
    path = tmp_path / "manifest.json"
    manifest = build_reproducibility_manifest(config, tmp_path)
    manifest["git"]["commit"] = "old-commit"
    manifest["environment"]["gpu"] = "old-gpu"
    write_manifest(path, manifest)
    current = build_reproducibility_manifest(config, tmp_path)
    allowed = {
        "git.commit": ("old-commit", current["git"]["commit"]),
        "environment.gpu": ("old-gpu", current["environment"]["gpu"]),
    }
    mismatches = validate_resume_manifest(
        path, config, tmp_path, allowed_mismatches=allowed
    )
    assert set(mismatches) == {"git.commit", "environment.gpu"}

    allowed["environment.gpu"] = ("different-gpu", current["environment"]["gpu"])
    with pytest.raises(RuntimeError, match="environment.gpu"):
        validate_resume_manifest(path, config, tmp_path, allowed_mismatches=allowed)


def test_manifest_fingerprints_pretrained_encoder(tmp_path: Path) -> None:
    checkpoint = tmp_path / "encoder.pt"
    checkpoint.write_bytes(b"frozen-encoder")
    raw = asdict(_config())
    raw["training"]["pretrained_encoder_path"] = str(checkpoint)
    manifest = build_reproducibility_manifest(
        V3ExperimentConfig.from_dict(raw), tmp_path
    )
    fingerprint = manifest["data_fingerprints"]["pretrained_encoder"]
    assert fingerprint["bytes"] == len(b"frozen-encoder")
    assert len(fingerprint["sha256"]) == 64
