"""Predeclared block-aware paired analysis of PEMS08 sensor-outage runs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import yaml


def window_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.isfinite(target) & (np.abs(target) > 1e-5) & mask
    count = valid.sum(axis=1)
    error = np.abs(prediction.astype(np.float64) - target.astype(np.float64))
    return np.divide(
        np.where(valid, error, 0.0).sum(axis=1),
        count,
        out=np.full(len(prediction), np.nan, dtype=np.float64),
        where=count > 0,
    )


def block_analysis(
    differences: np.ndarray,
    rng: np.random.Generator,
    sensitivity_rng: np.random.Generator,
) -> dict:
    # Shape [training_seed, chronological_test_window].
    seeds, windows = differences.shape
    if not np.isfinite(differences).all():
        raise ValueError("Every paired window must have a finite masked-sensor error")
    observed = float(differences.mean())
    block = 12
    draws = 2000
    means = np.empty(draws, dtype=np.float64)
    for iteration in range(draws):
        per_seed = []
        for row in differences:
            starts = rng.integers(0, windows, size=(windows + block - 1) // block)
            indices = (starts[:, None] + np.arange(block)) % windows
            per_seed.append(float(row[indices.ravel()[:windows]].mean()))
        means[iteration] = np.mean(per_seed)
    ci = np.quantile(means, [0.025, 0.975]).tolist()

    # Sign flips are applied to contiguous nonoverlapping blocks, separately
    # within each training seed; this is exploratory under serial dependence.
    groups = []
    for row in differences:
        for start in range(0, windows, block):
            groups.append(row[start:start + block])
    group_sums = np.asarray([group.sum() for group in groups], dtype=np.float64)
    null = np.empty(draws, dtype=np.float64)
    for iteration in range(draws):
        signs = rng.choice((-1.0, 1.0), size=len(groups))
        null[iteration] = float(np.dot(signs, group_sums) / (seeds * windows))
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (draws + 1))
    # Post-hoc sensitivity: treating the three independently trained seeds as
    # the replication unit avoids claiming that overlapping test windows
    # substitute for independently trained models.
    seed_means = differences.mean(axis=1)
    seed_null = np.asarray([
        np.mean(seed_means * signs)
        for signs in itertools.product((-1.0, 1.0), repeat=seeds)
    ])
    seed_sign_flip_p = float(np.mean(np.abs(seed_null) >= abs(observed) - 1e-12))
    two_level = np.empty(draws, dtype=np.float64)
    for iteration in range(draws):
        sampled_seeds = sensitivity_rng.integers(0, seeds, size=seeds)
        sampled_means = []
        for seed_index in sampled_seeds:
            row = differences[seed_index]
            starts = sensitivity_rng.integers(0, windows, size=(windows + block - 1) // block)
            indices = (starts[:, None] + np.arange(block)) % windows
            sampled_means.append(float(row[indices.ravel()[:windows]].mean()))
        two_level[iteration] = np.mean(sampled_means)
    return {
        "paired_mean_difference_in_degradation": observed,
        "paired_95pct_12_window_circular_block_bootstrap_ci": ci,
        "block_sign_flip_p": p_value,
        "posthoc_seed_level_sensitivity": {
            "seed_contrasts": seed_means.tolist(),
            "exact_three_seed_sign_flip_p": seed_sign_flip_p,
            "two_level_seed_and_12_window_block_bootstrap_95pct_ci":
                np.quantile(two_level, [0.025, 0.975]).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = yaml.safe_load(protocol_bytes)
    run_root = args.run_root
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    if state["status"] != "completed" or state["completed_runs"] != 18:
        raise RuntimeError("All 18 fixed-protocol runs must complete before analysis")
    target = np.load(run_root / "target_h12_flow_original.npy")
    results = []
    for rate in (0.1, 0.3):
        mask = np.load(run_root / f"mask_rate_{int(rate * 100):02d}.npy")
        per_variant = {}
        for variant in protocol["variants"]:
            per_seed = []
            for seed in protocol["seeds"]:
                base = run_root / variant / f"seed_{seed}" / "rate_00"
                corrupted = run_root / variant / f"seed_{seed}" / f"rate_{int(rate * 100):02d}"
                base_prediction = np.load(base / "prediction_h12_flow_original.npy")
                baseline_on_same_sensors = window_mae(base_prediction, target, mask)
                with np.load(corrupted / "window_errors.npz") as archive:
                    masked_mae = archive["masked_sensor_mae"]
                if len(masked_mae) != len(baseline_on_same_sensors):
                    raise ValueError("Window alignment mismatch")
                degradation = masked_mae - baseline_on_same_sensors
                per_seed.append(degradation)
            per_variant[variant] = np.stack(per_seed)
        difference = per_variant["full"] - per_variant["no_hierarchy"]
        analysis = block_analysis(
            difference,
            np.random.default_rng(20261008 + round(rate * 100)),
            np.random.default_rng(20261009 + round(rate * 100)),
        )
        results.append({
            "mask_rate": rate,
            "masked_sensors_per_window": int(mask[0].sum()),
            "positive_contrast_means": "full model degrades more than no_hierarchy",
            "full_mean_degradation": float(per_variant["full"].mean()),
            "no_hierarchy_mean_degradation": float(per_variant["no_hierarchy"].mean()),
            "per_seed": [
                {
                    "seed": seed,
                    "full_degradation": float(per_variant["full"][index].mean()),
                    "no_hierarchy_degradation": float(per_variant["no_hierarchy"][index].mean()),
                }
                for index, seed in enumerate(protocol["seeds"])
            ],
            **analysis,
        })
    ordered = sorted(enumerate(results), key=lambda item: item[1]["block_sign_flip_p"])
    corrected = [0.0] * len(results)
    running = 0.0
    for rank, (index, result) in enumerate(ordered):
        running = max(running, min(1.0, (len(results) - rank) * result["block_sign_flip_p"]))
        corrected[index] = running
    for result, adjusted in zip(results, corrected, strict=True):
        result["holm_adjusted_p"] = adjusted
    payload = {
        "schema": "dagnn-pistnet-v3-missing-sensor-analysis-v1",
        "status": "completed",
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "scope": "PEMS08_inference_only_flow_60min",
        "test_windows_per_seed": len(target),
        "seed_alignment": protocol["seeds"],
        "bootstrap_draws": 2000,
        "block_length_windows": 12,
        "p_value_method": "paired_nonoverlapping_12_window_block_sign_flip_monte_carlo",
        "multiple_testing_correction": "Holm-Bonferroni across two frozen nonzero mask rates",
        "limitations": [
            "Inference-only sensor outages were not included during training.",
            "One dataset and three training seeds; overlapping windows retain residual dependence.",
            "Masking all three normalized input features to their training means is only one imputation policy.",
            "The model comparison changes architectural capacity as well as hierarchy.",
            "No conservation or event-response claim can be inferred from this flow-target stress test.",
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
