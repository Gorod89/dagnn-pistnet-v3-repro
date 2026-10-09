from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .study import holm_adjust, paired_bootstrap, paired_sign_flip_pvalue

EFFICIENCY_FIELDS = (
    "parameter_count",
    "peak_accelerator_memory_mb",
    "inference_latency_ms",
    "training_wall_seconds",
)

PROFILE_FIELDS = (
    "hardware_id",
    "precision",
    "batch_size",
    "warmup_iterations",
    "measured_iterations",
)

DEFAULT_TEMPORAL_BLOCK_LENGTH = 12


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _window_archive(result_path: Path, result: dict[str, Any]) -> Path:
    configured = result.get("window_errors_path")
    if configured:
        candidate = Path(str(configured))
        if candidate.is_file():
            return candidate
        relative = result_path.parent / candidate.name
        if relative.is_file():
            return relative
    candidate = result_path.parent / "window_errors.npz"
    if not candidate.is_file():
        raise ValueError(f"window_errors.npz is missing beside {result_path}")
    return candidate


def load_common_evaluator_runs(root: str | Path) -> list[dict[str, Any]]:
    """Load evaluated model runs without silently accepting ambiguous metadata."""

    runs: list[dict[str, Any]] = []
    for path in sorted(Path(root).rglob("result.json")):
        result = _read_json(path)
        required = ("model", "dataset", "seed", "protocol_sha256")
        missing = [field for field in required if field not in result]
        if missing:
            continue
        archive = _window_archive(path, result)
        runs.append({**result, "result_path": str(path), "archive_path": str(archive)})
    if not runs:
        raise ValueError("no common-evaluator result.json files with complete metadata found")
    return runs


def _validated_series(
    differences_by_series: Iterable[np.ndarray], block_length: int
) -> list[np.ndarray]:
    if block_length < 1:
        raise ValueError("block_length must be positive")
    series = [np.asarray(values, dtype=np.float64).reshape(-1) for values in differences_by_series]
    if not series:
        raise ValueError("at least one paired difference series is required")
    for values in series:
        if values.size < block_length:
            raise ValueError("block_length cannot exceed a paired time series length")
        if not np.isfinite(values).all():
            raise ValueError("paired differences must be finite")
    return series


def paired_moving_block_bootstrap(
    differences_by_series: Iterable[np.ndarray],
    samples: int,
    confidence: float,
    seed: int,
    block_length: int = DEFAULT_TEMPORAL_BLOCK_LENGTH,
) -> tuple[float, float, float]:
    """Circular moving-block bootstrap that never crosses a seed boundary."""

    if samples < 100:
        raise ValueError("samples must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    series = _validated_series(differences_by_series, block_length)
    total_count = sum(values.size for values in series)
    observed = sum(float(values.sum()) for values in series) / total_count
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    offsets = np.arange(block_length, dtype=np.int64)
    batch_size = 64
    for batch_start in range(0, samples, batch_size):
        batch_stop = min(samples, batch_start + batch_size)
        batch = batch_stop - batch_start
        sampled_sum = np.zeros(batch, dtype=np.float64)
        for values in series:
            block_count = math.ceil(values.size / block_length)
            starts = rng.integers(0, values.size, size=(batch, block_count))
            indices = (starts[..., None] + offsets) % values.size
            sampled = values[indices].reshape(batch, -1)[:, : values.size]
            sampled_sum += sampled.sum(axis=1)
        estimates[batch_start:batch_stop] = sampled_sum / total_count
    alpha = (1.0 - confidence) / 2.0
    return (
        observed,
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def paired_block_sign_flip_pvalue(
    differences_by_series: Iterable[np.ndarray],
    samples: int,
    seed: int,
    block_length: int = DEFAULT_TEMPORAL_BLOCK_LENGTH,
) -> float:
    """Monte Carlo sign-flip with one sign per contiguous non-overlapping block."""

    if samples < 1:
        raise ValueError("samples must be positive")
    series = _validated_series(differences_by_series, block_length)
    block_sums: list[float] = []
    for values in series:
        block_sums.extend(
            float(values[start : start + block_length].sum())
            for start in range(0, values.size, block_length)
        )
    sums = np.asarray(block_sums, dtype=np.float64)
    if sums.size < 2:
        return math.nan
    total_count = sum(values.size for values in series)
    observed = abs(sum(float(values.sum()) for values in series) / total_count)
    rng = np.random.default_rng(seed)
    extreme = 0
    batch_size = 256
    for batch_start in range(0, samples, batch_size):
        batch = min(batch_size, samples - batch_start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(batch, sums.size))
        permuted = np.abs(signs @ sums / total_count)
        extreme += int(np.count_nonzero(permuted >= observed))
    return (extreme + 1.0) / (samples + 1.0)


def paired_competitor_rows(
    runs: Iterable[dict[str, Any]],
    baseline_model: str,
    *,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    random_seed: int = 20260904,
    block_length: int | None = None,
) -> list[dict[str, Any]]:
    """Compare models on identical protocol, seed, horizon, and test windows.

    Differences are candidate minus baseline, so negative MAE differences favour
    the candidate. Holm correction is applied once over the returned comparison
    family. The caller should therefore create separate files for separately
    declared primary and secondary hypothesis families.
    """

    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    indexed: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for run in runs:
        key = (
            str(run["dataset"]),
            str(run["protocol_sha256"]),
            int(run["seed"]),
            str(run["model"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate evaluated run: {key}")
        indexed[key] = run

    paired: dict[tuple[str, str, str, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for (dataset, protocol, seed, model), run in indexed.items():
        if model == baseline_model:
            continue
        baseline = indexed.get((dataset, protocol, seed, baseline_model))
        if baseline is None:
            continue
        with np.load(run["archive_path"]) as candidate_archive, np.load(
            baseline["archive_path"]
        ) as baseline_archive:
            for horizon in sorted(set(candidate_archive.files) & set(baseline_archive.files)):
                candidate = np.asarray(candidate_archive[horizon], dtype=np.float64).reshape(-1)
                reference = np.asarray(baseline_archive[horizon], dtype=np.float64).reshape(-1)
                if candidate.shape != reference.shape:
                    raise ValueError(
                        f"paired window count differs for {dataset}/{model}/seed={seed}/{horizon}"
                    )
                if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
                    raise ValueError(f"non-finite window errors for {dataset}/{model}/seed={seed}")
                paired[(dataset, protocol, model, horizon)].append((seed, candidate - reference))

    rows: list[dict[str, Any]] = []
    for index, (key, seed_arrays) in enumerate(sorted(paired.items())):
        dataset, protocol, model, horizon = key
        seed_arrays.sort(key=lambda item: item[0])
        difference_series = [values for _, values in seed_arrays]
        differences = np.concatenate(difference_series)
        if block_length is None:
            mean, low, high = paired_bootstrap(
                differences, bootstrap_samples, confidence, random_seed + 2 * index
            )
            p_raw = paired_sign_flip_pvalue(
                differences, max(1000, bootstrap_samples), random_seed + 2 * index + 1
            )
            inference_method = "iid_window"
            effective_block_length = 1
            resampling_units = differences.size
        else:
            mean, low, high = paired_moving_block_bootstrap(
                difference_series,
                bootstrap_samples,
                confidence,
                random_seed + 2 * index,
                block_length,
            )
            p_raw = paired_block_sign_flip_pvalue(
                difference_series,
                max(1000, bootstrap_samples),
                random_seed + 2 * index + 1,
                block_length,
            )
            inference_method = "moving_block"
            effective_block_length = block_length
            resampling_units = sum(
                math.ceil(values.size / block_length) for values in difference_series
            )
        rows.append(
            {
                "dataset": dataset,
                "protocol_sha256": protocol,
                "candidate": model,
                "baseline": baseline_model,
                "horizon": horizon,
                "seeds": ";".join(str(seed) for seed, _ in seed_arrays),
                "windows": int(differences.size),
                "inference_method": inference_method,
                "block_length": effective_block_length,
                "resampling_units": int(resampling_units),
                "mean_mae_difference": mean,
                "ci_low": low,
                "ci_high": high,
                "p_raw": p_raw,
            }
        )
    adjusted = holm_adjust([float(row["p_raw"]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["p_holm"] = value
        row["significant_0_05"] = bool(
            math.isfinite(value)
            and value < 0.05
            and (float(row["ci_high"]) < 0.0 or float(row["ci_low"]) > 0.0)
        )
        row["effect_direction"] = (
            "candidate_better"
            if float(row["mean_mae_difference"]) < 0.0
            else "baseline_better"
        )
    return rows


def paired_competitor_block_rows(
    runs: Iterable[dict[str, Any]],
    baseline_model: str,
    *,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    random_seed: int = 20260904,
    block_length: int = DEFAULT_TEMPORAL_BLOCK_LENGTH,
) -> list[dict[str, Any]]:
    """Publication path for overlapping windows; defaults to one 12-step horizon."""

    return paired_competitor_rows(
        runs,
        baseline_model,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        random_seed=random_seed,
        block_length=block_length,
    )


def write_paired_comparisons(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    if not rows:
        raise ValueError("no protocol/seed-matched model pairs found")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target


def load_efficiency_records(root: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(Path(root).rglob("efficiency.json")):
        record = _read_json(path)
        required = ("model", "dataset", "seed", "protocol_sha256", *EFFICIENCY_FIELDS)
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(f"missing efficiency fields {missing}: {path}")
        for field in EFFICIENCY_FIELDS:
            value = float(record[field])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid {field} in {path}")
        missing_profile = [field for field in PROFILE_FIELDS if field not in record]
        if missing_profile:
            raise ValueError(f"missing profiling fields {missing_profile}: {path}")
        records.append({**record, "efficiency_path": str(path)})
    if not records:
        raise ValueError("no efficiency.json records found")
    return records


def efficiency_summary_rows(
    records: Iterable[dict[str, Any]], baseline_model: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (str(record["dataset"]), str(record["protocol_sha256"]), str(record["model"]))
        ].append(record)

    baseline_profiles: dict[tuple[str, str], set[tuple[Any, ...]]] = defaultdict(set)
    for (dataset, protocol, model), values in grouped.items():
        if model == baseline_model:
            baseline_profiles[(dataset, protocol)].update(
                tuple(value[field] for field in PROFILE_FIELDS) for value in values
            )

    rows: list[dict[str, Any]] = []
    for (dataset, protocol, model), values in sorted(grouped.items()):
        profiles = {tuple(value[field] for field in PROFILE_FIELDS) for value in values}
        reference_profiles = baseline_profiles.get((dataset, protocol), set())
        row: dict[str, Any] = {
            "dataset": dataset,
            "protocol_sha256": protocol,
            "model": model,
            "seeds": ";".join(
                str(value["seed"]) for value in sorted(values, key=lambda x: x["seed"])
            ),
            "runs": len(values),
            "profile_consistent": len(profiles) == 1,
            "comparable_to_baseline": bool(
                len(profiles) == 1 and profiles == reference_profiles
            ),
        }
        for field in EFFICIENCY_FIELDS:
            array = np.asarray([float(value[field]) for value in values], dtype=np.float64)
            row[f"{field}_mean"] = float(array.mean())
            row[f"{field}_std"] = float(array.std(ddof=1)) if array.size > 1 else math.nan
        rows.append(row)
    return rows


def write_efficiency_summary(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    if not rows:
        raise ValueError("no efficiency records to summarize")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target
