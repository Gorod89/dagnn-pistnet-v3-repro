from __future__ import annotations

from pathlib import Path

import yaml

from dagnn_pistnet_v3.tuning import ValidationTuningSpec, run_validation_tuning


def test_tuning_dry_run_declares_no_test_access(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "dataset": "PEMS07",
                    "data_path": "not-loaded-in-dry-run.npz",
                    "feature_names": ["flow"],
                    "units": ["vehicles_per_step"],
                }
            }
        ),
        encoding="utf-8",
    )
    spec_path = tmp_path / "tuning.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "study_name": "unit",
                "datasets": {"synthetic": str(base_path)},
                "search_space": {
                    "model.hidden_dim": [8, 16],
                    "training.learning_rate": [0.001, 0.0003],
                },
                "max_trials": 3,
            }
        ),
        encoding="utf-8",
    )
    plan = run_validation_tuning(spec_path, dry_run=True)
    assert plan["total_trials"] == 3
    assert plan["uses_test_split"] is False


def test_tuning_parses_explicit_gpu_transition(tmp_path: Path) -> None:
    path = tmp_path / "transition.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "study_name": "unit",
                "datasets": {"synthetic": "base.yaml"},
                "search_space": {"training.learning_rate": [0.001]},
                "gpu_transition": {
                    "enabled": True,
                    "previous_git_commit": "abc",
                    "previous_gpu": "GPU A",
                    "current_gpu": "GPU B",
                    "resume_incomplete_trials": True,
                    "expected_remaining_trials": 6,
                },
            }
        ),
        encoding="utf-8",
    )
    spec = ValidationTuningSpec.from_yaml(path)
    assert spec.gpu_transition.enabled is True
    assert spec.gpu_transition.expected_remaining_trials == 6
