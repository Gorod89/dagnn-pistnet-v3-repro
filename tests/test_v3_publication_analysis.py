from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dagnn_pistnet_v3.publication_analysis import (
    efficiency_summary_rows,
    load_common_evaluator_runs,
    load_efficiency_records,
    paired_block_sign_flip_pvalue,
    paired_competitor_rows,
    paired_moving_block_bootstrap,
)


def _evaluated_run(
    root: Path, model: str, seed: int, errors: np.ndarray, protocol: str = "a" * 64
) -> None:
    output = root / model / f"seed_{seed}"
    output.mkdir(parents=True)
    archive = output / "window_errors.npz"
    np.savez_compressed(archive, h12=errors)
    (output / "result.json").write_text(
        json.dumps(
            {
                "model": model,
                "dataset": "METR-LA",
                "seed": seed,
                "protocol_sha256": protocol,
                "window_errors_path": str(archive),
            }
        ),
        encoding="utf-8",
    )


def test_paired_competitor_analysis_matches_protocol_seed_and_windows(tmp_path: Path) -> None:
    for seed in (13, 42, 97):
        _evaluated_run(tmp_path, "full", seed, np.full(30, 2.0))
        _evaluated_run(tmp_path, "competitor", seed, np.full(30, 1.5))
    rows = paired_competitor_rows(
        load_common_evaluator_runs(tmp_path), "full", bootstrap_samples=100
    )
    assert len(rows) == 1
    assert rows[0]["mean_mae_difference"] == -0.5
    assert rows[0]["ci_high"] == -0.5
    assert rows[0]["effect_direction"] == "candidate_better"
    assert rows[0]["significant_0_05"] is True


def test_efficiency_summary_marks_mismatched_profiles_noncomparable(tmp_path: Path) -> None:
    common = {
        "dataset": "METR-LA",
        "seed": 13,
        "protocol_sha256": "a" * 64,
        "parameter_count": 10,
        "peak_accelerator_memory_mb": 20,
        "inference_latency_ms": 3,
        "training_wall_seconds": 40,
        "precision": "fp32",
        "batch_size": 1,
        "warmup_iterations": 20,
        "measured_iterations": 100,
    }
    for model, hardware in (("full", "gpu-a"), ("competitor", "gpu-b")):
        output = tmp_path / model
        output.mkdir()
        (output / "efficiency.json").write_text(
            json.dumps({**common, "model": model, "hardware_id": hardware}),
            encoding="utf-8",
        )
    rows = efficiency_summary_rows(load_efficiency_records(tmp_path), "full")
    competitor = next(row for row in rows if row["model"] == "competitor")
    assert competitor["comparable_to_baseline"] is False


def test_moving_block_inference_is_deterministic_and_preserves_mean() -> None:
    differences = [np.linspace(-1.0, 0.5, 120), np.linspace(-0.8, 0.7, 120)]
    first = paired_moving_block_bootstrap(differences, 200, 0.95, 17, block_length=12)
    second = paired_moving_block_bootstrap(differences, 200, 0.95, 17, block_length=12)
    assert first == second
    assert first[0] == np.concatenate(differences).mean()
    first_p = paired_block_sign_flip_pvalue(differences, 500, 19, block_length=12)
    second_p = paired_block_sign_flip_pvalue(differences, 500, 19, block_length=12)
    assert first_p == second_p


def test_moving_block_inference_validates_block_length() -> None:
    with __import__("pytest").raises(ValueError, match="cannot exceed"):
        paired_moving_block_bootstrap([np.ones(11)], 100, 0.95, 1, block_length=12)


def test_competitor_rows_keep_iid_path_and_offer_block_path(tmp_path: Path) -> None:
    for seed in (13, 42, 97):
        _evaluated_run(tmp_path, "full", seed, np.full(36, 2.0))
        _evaluated_run(tmp_path, "competitor", seed, np.full(36, 1.5))
    runs = load_common_evaluator_runs(tmp_path)
    iid = paired_competitor_rows(runs, "full", bootstrap_samples=100)
    blocked = paired_competitor_rows(
        runs, "full", bootstrap_samples=100, block_length=12
    )
    assert iid[0]["inference_method"] == "iid_window"
    assert iid[0]["block_length"] == 1
    assert blocked[0]["inference_method"] == "moving_block"
    assert blocked[0]["block_length"] == 12
    assert blocked[0]["resampling_units"] == 9
