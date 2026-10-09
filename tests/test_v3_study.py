from __future__ import annotations

from pathlib import Path

import yaml

from dagnn_pistnet_v3.study import (
    PublicationStudySpec,
    paired_bootstrap,
    resolve_study_runs,
    run_publication_study,
)


def test_study_dry_run_expands_dataset_variant_seed_product(tmp_path: Path) -> None:
    base = {
        "data": {
            "dataset": "synthetic",
            "data_path": "unused.npy",
            "feature_names": ["flow"],
            "units": ["vehicles_per_step"],
        },
        "training": {"epochs": 1, "batch_size": 1},
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    spec = {
        "study_name": "unit",
        "datasets": {"one": str(base_path)},
        "ablations": {"full": {}, "no_delta": {"model": {"use_delta_decoder": False}}},
        "baseline": "full",
        "seeds": [1, 2, 3],
        "output_root": str(tmp_path / "out"),
    }
    spec_path = tmp_path / "study.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    plan = run_publication_study(spec_path, dry_run=True)
    assert plan["run_count"] == 6


def test_publication_study_requires_three_seeds(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "study_name": "invalid",
                "datasets": {"x": "x.yaml"},
                "ablations": {"full": {}},
                "seeds": [1, 2],
            }
        ),
        encoding="utf-8",
    )
    try:
        PublicationStudySpec.from_yaml(path)
    except ValueError as error:
        assert "three" in str(error)
    else:
        raise AssertionError("two-seed publication study must be rejected")


def test_paired_bootstrap_detects_constant_improvement() -> None:
    mean, low, high = paired_bootstrap(
        differences=-0.5 * __import__("numpy").ones(100),
        samples=200,
        confidence=0.95,
        seed=1,
    )
    assert mean == low == high == -0.5


def test_study_applies_validation_selected_parameters(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "dataset": "synthetic",
                    "data_path": "unused.npy",
                    "feature_names": ["flow"],
                    "units": ["vehicles_per_step"],
                }
            }
        ),
        encoding="utf-8",
    )
    selection_path = tmp_path / "best.json"
    selection_path.write_text(
        '{"one":{"parameters":"{\\"model.num_regimes\\":5,'
        '\\"training.learning_rate\\":0.0007}"}}',
        encoding="utf-8",
    )
    spec = PublicationStudySpec(
        study_name="selected",
        datasets={"one": str(base_path)},
        ablations={"full": {}},
        seeds=(1, 2, 3),
        output_root=str(tmp_path / "out"),
        validation_selection_path=str(selection_path),
    )
    runs = resolve_study_runs(spec)
    assert runs[0][4]["model"]["num_regimes"] == 5
    assert runs[0][4]["training"]["learning_rate"] == 0.0007
