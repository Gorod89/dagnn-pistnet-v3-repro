"""Paired analysis of the prespecified capacity-matched operator-order control.

Run only after all nine reverse-order result/window-error artifacts exist.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import yaml

from dagnn_pistnet_v3.config import V3ExperimentConfig
from dagnn_pistnet_v3.reproducibility import sha256_file


DATASETS = ("metr_la", "pems_bay", "pems08")
SEEDS = (13, 42, 97)
HORIZON = "60min"
WINDOW_KEY = "h12"
BLOCK = 12
RESAMPLES = 2000
RNG_SEED = 20261009


def primary_windows(path: Path, *, feature_index: int, feature_count: int, batch_size: int) -> np.ndarray:
    """Undo the feature-within-batch concatenation in metrics.py."""
    with np.load(path, allow_pickle=False) as archive:
        raw = np.asarray(archive[WINDOW_KEY], dtype=np.float64)
    if raw.ndim != 1 or raw.size % feature_count:
        raise ValueError(f"unexpected window-error shape: {path}: {raw.shape}")
    remaining = raw.size // feature_count
    offset = 0
    chunks = []
    while remaining:
        length = min(batch_size, remaining)
        chunks.append(raw[offset + feature_index * length : offset + (feature_index + 1) * length])
        offset += length * feature_count
        remaining -= length
    result = np.concatenate(chunks)
    if offset != raw.size or not np.isfinite(result).all():
        raise ValueError(f"invalid primary window errors: {path}")
    return result


def circular_bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> float:
    starts = rng.integers(0, len(values), size=math.ceil(len(values) / BLOCK))
    indices = (starts[:, None] + np.arange(BLOCK)[None, :]) % len(values)
    return float(values[indices.reshape(-1)[: len(values)]].mean())


def block_sign_flip_p(seed_differences: list[np.ndarray], observed: float, rng: np.random.Generator) -> float:
    """Two-sided block sign-flip randomization; blocks never cross seeds."""
    exceed = 0
    for _ in range(RESAMPLES):
        permuted = []
        for values in seed_differences:
            signs = rng.choice((-1, 1), size=math.ceil(len(values) / BLOCK))
            permuted.append(float((values * np.repeat(signs, BLOCK)[: len(values)]).mean()))
        exceed += abs(float(np.mean(permuted))) >= abs(observed)
    return (exceed + 1) / (RESAMPLES + 1)


def exact_seed_sign_flip_p(seed_means: list[float]) -> float:
    observed = abs(float(np.mean(seed_means)))
    values = [abs(float(np.mean(np.asarray(seed_means) * signs))) for signs in itertools.product((-1, 1), repeat=len(seed_means))]
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.__getitem__)
    result = {}
    maximum = 0.0
    for position, name in enumerate(ordered):
        maximum = max(maximum, min(1.0, raw[name] * (len(raw) - position)))
        result[name] = maximum
    return result


def analyze(reference_root: Path, reverse_root: Path) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    datasets = {}
    raw_p = {}
    for dataset in DATASETS:
        differences = []
        records = []
        for seed in SEEDS:
            reference = reference_root / dataset / "full" / f"seed_{seed}"
            reverse = reverse_root / dataset / "reverse_order" / f"seed_{seed}"
            for directory in (reference, reverse):
                for filename in ("result.json", "resolved_config.yaml", "window_errors.npz", "manifest.json"):
                    if not (directory / filename).is_file():
                        raise FileNotFoundError(directory / filename)
            old_config = yaml.safe_load((reference / "resolved_config.yaml").read_text(encoding="utf-8"))
            new_config = yaml.safe_load((reverse / "resolved_config.yaml").read_text(encoding="utf-8"))
            old_normalized = V3ExperimentConfig.from_dict(old_config)
            new_normalized = V3ExperimentConfig.from_dict(new_config)
            if old_normalized.data != new_normalized.data or old_normalized.training.seed != seed or new_normalized.training.seed != seed:
                raise ValueError(f"split or seed mismatch: {dataset}/{seed}")
            features = old_config["data"]["feature_names"]
            feature_index = old_config["data"]["primary_feature_index"]
            feature = features[feature_index]
            old_batch = old_config["training"].get("evaluation_batch_size") or old_config["training"]["batch_size"]
            new_batch = new_config["training"].get("evaluation_batch_size") or new_config["training"]["batch_size"]
            old_windows = primary_windows(reference / "window_errors.npz", feature_index=feature_index, feature_count=len(features), batch_size=old_batch)
            new_windows = primary_windows(reverse / "window_errors.npz", feature_index=feature_index, feature_count=len(features), batch_size=new_batch)
            if old_windows.shape != new_windows.shape:
                raise ValueError(f"unaligned test-window count: {dataset}/{seed}")
            delta = new_windows - old_windows
            differences.append(delta)
            old_result = json.loads((reference / "result.json").read_text(encoding="utf-8"))
            new_result = json.loads((reverse / "result.json").read_text(encoding="utf-8"))
            old_mae = old_result["test_metrics"][HORIZON][feature]["mae"]
            new_mae = new_result["test_metrics"][HORIZON][feature]["mae"]
            if old_result["parameter_count"] != new_result["parameter_count"]:
                raise ValueError(f"parameter mismatch: {dataset}/{seed}")
            records.append({
                "seed": seed,
                "feature": feature,
                "test_windows": int(delta.size),
                "parameters_each": old_result["parameter_count"],
                "reference_sensor_weighted_mae": old_mae,
                "reverse_sensor_weighted_mae": new_mae,
                "sensor_weighted_delta_reverse_minus_reference": new_mae - old_mae,
                "window_mean_delta_reverse_minus_reference": float(delta.mean()),
                "reference_result_sha256": sha256_file(reference / "result.json"),
                "reverse_result_sha256": sha256_file(reverse / "result.json"),
                "reference_window_errors_sha256": sha256_file(reference / "window_errors.npz"),
                "reverse_window_errors_sha256": sha256_file(reverse / "window_errors.npz"),
            })
        seed_means = [float(values.mean()) for values in differences]
        observed = float(np.mean(seed_means))
        boot = np.asarray([
            np.mean([circular_bootstrap_mean(values, rng) for values in differences])
            for _ in range(RESAMPLES)
        ])
        p_value = block_sign_flip_p(differences, observed, rng)
        raw_p[dataset] = p_value
        datasets[dataset] = {
            "runs": records,
            "mean_sensor_weighted_delta": float(np.mean([r["sensor_weighted_delta_reverse_minus_reference"] for r in records])),
            "mean_window_delta": observed,
            "window_delta_95pct_block_bootstrap_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "block_sign_flip_two_sided_p": p_value,
            "exact_seed_sign_flip_two_sided_p": exact_seed_sign_flip_p(seed_means),
        }
    adjusted = holm_adjust(raw_p)
    for name, value in adjusted.items():
        datasets[name]["holm_adjusted_block_sign_flip_p"] = value
    return {
        "analysis": "prespecified capacity-matched operator order",
        "contrast": "reverse source/sink→diffusion→transport minus original transport→diffusion→source/sink; positive favors original",
        "primary_endpoint": "60-minute MAE of primary target in original units",
        "reference_root": str(reference_root),
        "reverse_root": str(reverse_root),
        "inference": {"block_windows": BLOCK, "resamples": RESAMPLES, "random_seed": RNG_SEED, "multiple_testing": "Holm across three dataset primary comparisons", "seed_sensitivity": "exact sign-flip over three seed-paired means"},
        "datasets": datasets,
        "limitations": ["Only three independent training seeds; block-window resampling does not create independent model replicates.", "Global test MAE is sensor-weighted, while paired-window inference averages per-window valid-node MAE.", "Reference and reverse runs share architecture and selection protocol but were executed at different times; retain environment manifests."],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, default=Path("artifacts/v3/studies/dagnn_pistnet_v3_ablation"))
    parser.add_argument("--reverse-root", type=Path, default=Path("artifacts/v3/studies/operator_order_capacity_matched"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v3/studies/operator_order_capacity_matched/paired_analysis.json"))
    args = parser.parse_args()
    report = analyze(args.reference_root, args.reverse_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: data["mean_window_delta"] for name, data in report["datasets"].items()}, indent=2))
