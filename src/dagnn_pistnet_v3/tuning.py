from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import V3ExperimentConfig
from .experiment import V3Experiment
from .reproducibility import (
    build_reproducibility_manifest,
    sha256_file,
    validate_resume_manifest,
    write_manifest,
)


def _set_dotted(raw: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    target = raw
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


@dataclass(frozen=True)
class GPUTransitionSpec:
    enabled: bool = False
    previous_git_commit: str | None = None
    previous_gpu: str | None = None
    current_gpu: str | None = None
    resume_incomplete_trials: bool = False
    expected_remaining_trials: int | None = None


@dataclass(frozen=True)
class ValidationTuningSpec:
    study_name: str
    datasets: dict[str, str]
    search_space: dict[str, tuple[Any, ...]]
    seed: int = 42
    max_trials: int = 32
    output_root: str = "artifacts/v3/tuning"
    pretrained_encoder_path: str | None = None
    gpu_transition: GPUTransitionSpec = GPUTransitionSpec()

    @classmethod
    def from_yaml(cls, path: str | Path) -> ValidationTuningSpec:
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("tuning specification must be a mapping")
        raw = dict(raw)
        raw["search_space"] = {
            key: tuple(values) for key, values in raw["search_space"].items()
        }
        raw["gpu_transition"] = GPUTransitionSpec(**raw.get("gpu_transition", {}))
        return cls(**raw)


def _transition_mismatches(
    spec: ValidationTuningSpec,
    config: V3ExperimentConfig,
    workspace: str | Path,
) -> tuple[dict[str, tuple[Any, Any]], dict[str, Any]]:
    transition = spec.gpu_transition
    current = build_reproducibility_manifest(config, workspace)
    if not transition.enabled:
        return {}, current
    actual_gpu = current["environment"]["gpu"]
    if not all(
        (transition.previous_git_commit, transition.previous_gpu, transition.current_gpu)
    ):
        raise ValueError("enabled GPU transition requires previous commit/GPU and current GPU")
    if actual_gpu != transition.current_gpu:
        raise RuntimeError(
            f"GPU transition requires {transition.current_gpu!r}, found {actual_gpu!r}"
        )
    return (
        {
            "git.commit": (
                transition.previous_git_commit,
                current["git"]["commit"],
            ),
            "environment.gpu": (transition.previous_gpu, transition.current_gpu),
        },
        current,
    )


def _write_transition_ledger(path: Path, ledger: dict[str, Any]) -> None:
    write_manifest(path, ledger)


def run_validation_tuning(
    spec_path: str | Path,
    workspace: str | Path = ".",
    dry_run: bool = False,
) -> dict[str, Any]:
    spec = ValidationTuningSpec.from_yaml(spec_path)
    keys = sorted(spec.search_space)
    combinations = list(itertools.product(*(spec.search_space[key] for key in keys)))
    combinations = combinations[: spec.max_trials]
    plan = {
        "study": spec.study_name,
        "datasets": list(spec.datasets),
        "trials_per_dataset": len(combinations),
        "total_trials": len(combinations) * len(spec.datasets),
        "uses_test_split": False,
    }
    if dry_run:
        return plan
    root = Path(spec.output_root) / spec.study_name
    remaining = [
        (dataset, trial)
        for dataset in spec.datasets
        for trial in range(len(combinations))
        if not (root / dataset / f"trial_{trial:03d}" / "validation_result.json").is_file()
    ]
    transition = spec.gpu_transition
    if transition.enabled and transition.expected_remaining_trials is not None:
        if len(remaining) != transition.expected_remaining_trials:
            raise RuntimeError(
                "GPU continuation refused: expected "
                f"{transition.expected_remaining_trials} remaining trials, found {len(remaining)}"
            )
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "explicit_gpu_transition" if transition.enabled else "standard",
        "policy": asdict(transition),
        "remaining_trials_at_start": [f"{d}/trial_{t:03d}" for d, t in remaining],
        "reused_completed_trials": [],
        "resumed_trials": [],
        "fresh_trials": [],
    }
    ledger_path = root / "gpu_transition_provenance.json"
    rows: list[dict[str, Any]] = []
    for dataset, config_path in spec.datasets.items():
        base = asdict(V3ExperimentConfig.from_yaml(config_path))
        for trial, values in enumerate(combinations):
            # JSON round-trip is a deterministic deep copy of config primitives.
            raw = json.loads(json.dumps(base))
            parameters = dict(zip(keys, values, strict=True))
            for key, value in parameters.items():
                _set_dotted(raw, key, value)
            output = root / dataset / f"trial_{trial:03d}"
            raw["training"]["seed"] = spec.seed
            raw["training"]["output_dir"] = str(output)
            if spec.pretrained_encoder_path is not None:
                raw["training"]["pretrained_encoder_path"] = (
                    spec.pretrained_encoder_path
                )
            raw["study"]["fail_on_dirty_worktree"] = True
            config = V3ExperimentConfig.from_dict(raw)
            allowed_mismatches, current_manifest = _transition_mismatches(
                spec, config, workspace
            )
            ledger["current_git"] = current_manifest["git"]
            ledger["current_environment"] = current_manifest["environment"]
            completed = output / "validation_result.json"
            if completed.exists():
                mismatches = validate_resume_manifest(
                    output / "manifest.json",
                    config,
                    workspace,
                    allowed_mismatches=allowed_mismatches,
                )
                with completed.open("r", encoding="utf-8") as stream:
                    result_data = json.load(stream)
                checkpoint = Path(result_data["checkpoint_path"])
                if not checkpoint.exists():
                    raise FileNotFoundError(
                        f"completed tuning trial has no checkpoint: {checkpoint}"
                    )
                result_row = {
                    "dataset": dataset,
                    "trial": trial,
                    "best_epoch": result_data["best_epoch"],
                    "validation_loss": result_data["best_validation_loss"],
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "checkpoint": result_data["checkpoint_path"],
                }
                ledger["reused_completed_trials"].append(
                    {
                        "trial": f"{dataset}/trial_{trial:03d}",
                        "manifest_sha256": sha256_file(output / "manifest.json"),
                        "allowed_mismatches": mismatches,
                    }
                )
            else:
                resume_checkpoint: Path | None = None
                candidates = (
                    output / "last_validation_state_v3.pt",
                    output / "best_validation_model_v3.pt",
                )
                if transition.enabled and transition.resume_incomplete_trials:
                    resume_checkpoint = next((p for p in candidates if p.is_file()), None)
                if resume_checkpoint is not None:
                    source_manifest = output / "manifest.json"
                    mismatches = validate_resume_manifest(
                        source_manifest,
                        config,
                        workspace,
                        allowed_mismatches=allowed_mismatches,
                    )
                    source_record = {
                        "trial": f"{dataset}/trial_{trial:03d}",
                        "source_checkpoint": str(resume_checkpoint),
                        "source_checkpoint_sha256": sha256_file(resume_checkpoint),
                        "source_manifest_sha256": sha256_file(source_manifest),
                        "source_manifest": json.loads(
                            source_manifest.read_text(encoding="utf-8")
                        ),
                        "allowed_mismatches": mismatches,
                    }
                    write_manifest(output / "pre_transition_provenance.json", source_record)
                    ledger["resumed_trials"].append(source_record)
                else:
                    ledger["fresh_trials"].append(f"{dataset}/trial_{trial:03d}")
                _write_transition_ledger(ledger_path, ledger)
                result = V3Experiment(config, workspace).tune_validation_only(
                    resume_checkpoint=resume_checkpoint
                )
                result_row = {
                    "dataset": dataset,
                    "trial": trial,
                    "best_epoch": result.best_epoch,
                    "validation_loss": result.best_validation_loss,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "checkpoint": result.checkpoint_path,
                }
            rows.append(result_row)
            _write_trials(root / "validation_trials.partial.csv", rows)
            _write_transition_ledger(ledger_path, ledger)
    result_path = root / "validation_trials.csv"
    _write_trials(result_path, rows)
    best = {
        dataset: min(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: row["validation_loss"],
        )
        for dataset in spec.datasets
    }
    with (root / "best_validation_configs.json").open("w", encoding="utf-8") as stream:
        json.dump(best, stream, ensure_ascii=False, indent=2)
    return {**plan, "results": str(result_path)}


def _write_trials(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
