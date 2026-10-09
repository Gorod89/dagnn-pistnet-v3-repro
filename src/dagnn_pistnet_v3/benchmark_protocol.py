from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from .config import V3ExperimentConfig
from .data import prepare_v3_data
from .reproducibility import build_reproducibility_manifest


def _canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def export_benchmark_protocol(
    config: V3ExperimentConfig,
    output_path: str | Path,
    workspace: str | Path = ".",
) -> dict[str, Any]:
    data = prepare_v3_data(config)
    test = data.test_dataset
    manifest = build_reproducibility_manifest(config, workspace)
    protocol: dict[str, Any] = {
        "schema": "dagnn-pistnet-benchmark-v1",
        "dataset": config.data.dataset,
        "data_fingerprints": manifest["data_fingerprints"],
        "split": {
            "train_end": data.split.train_end,
            "val_end": data.split.val_end,
            "total": data.split.total,
            "test_first_target": test.first_target,
            "test_last_target": test.last_target,
        },
        "prediction_shape": [
            len(test),
            config.data.forecast_steps,
            data.series.num_nodes,
            data.series.num_features,
        ],
        "feature_names": list(data.series.feature_names),
        "units": list(data.series.units),
        "sampling_seconds": config.data.sampling_seconds,
        "split_policy": config.data.split_policy,
        "zero_is_missing": True,
        "null_threshold": 1e-5,
        "horizon_indices": [
            index for index in (0, 2, 5, 11) if index < config.data.forecast_steps
        ],
    }
    protocol["protocol_sha256"] = _canonical_hash(protocol)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)
    return protocol


def create_prediction_artifact(
    protocol_path: str | Path,
    output_path: str | Path,
    model_name: str,
    seed: int,
) -> Path:
    """Create a chunked HDF5 file that an external competitor can fill."""

    with Path(protocol_path).open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    shape = tuple(int(value) for value in protocol["prediction_shape"])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    chunks = (1, shape[1], min(shape[2], 512), shape[3])
    with h5py.File(target, "w") as handle:
        handle.attrs["schema"] = "dagnn-pistnet-predictions-v1"
        handle.attrs["protocol_sha256"] = protocol["protocol_sha256"]
        handle.attrs["model_name"] = model_name
        handle.attrs["seed"] = int(seed)
        handle.create_dataset(
            "prediction",
            shape=shape,
            dtype="float32",
            chunks=chunks,
            compression="gzip",
            shuffle=True,
            fillvalue=np.nan,
        )
    return target


def evaluate_prediction_artifact(
    config: V3ExperimentConfig,
    protocol_path: str | Path,
    prediction_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 16,
) -> dict[str, Any]:
    with Path(protocol_path).open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    expected_hash = protocol.pop("protocol_sha256")
    actual_hash = _canonical_hash(protocol)
    protocol["protocol_sha256"] = expected_hash
    if expected_hash != actual_hash:
        raise ValueError("benchmark protocol hash is invalid")
    data = prepare_v3_data(config)
    test = data.test_dataset
    shape = tuple(protocol["prediction_shape"])
    if shape[0] != len(test):
        raise ValueError("protocol does not match current test split")
    sums: dict[tuple[int, int], dict[str, float]] = {}
    window_errors: dict[int, list[np.ndarray]] = {
        index: [] for index in protocol["horizon_indices"]
    }
    with h5py.File(prediction_path, "r") as handle:
        if handle.attrs.get("protocol_sha256") != expected_hash:
            raise ValueError("prediction artifact was produced for another protocol")
        predictions = handle["prediction"]
        if tuple(predictions.shape) != shape:
            raise ValueError("prediction artifact has invalid shape")
        for batch_start in range(0, len(test), batch_size):
            batch_end = min(len(test), batch_start + batch_size)
            prediction = np.asarray(predictions[batch_start:batch_end], dtype=np.float32)
            if not np.isfinite(prediction).all():
                raise ValueError("prediction artifact contains NaN or infinity")
            targets = []
            for index in range(batch_start, batch_end):
                start = test.first_target + index
                normalized = test.values[start : start + test.forecast_steps]
                targets.append(data.scaler.inverse_transform(normalized))
            target = np.stack(targets)
            for horizon in protocol["horizon_indices"]:
                for feature in range(target.shape[-1]):
                    y = target[:, horizon, :, feature]
                    p = prediction[:, horizon, :, feature]
                    valid = np.isfinite(y) & (np.abs(y) > protocol["null_threshold"])
                    error = p - y
                    absolute = np.abs(error)
                    stats = sums.setdefault(
                        (horizon, feature),
                        {
                            "count": 0.0,
                            "absolute": 0.0,
                            "squared": 0.0,
                            "target_absolute": 0.0,
                            "smape": 0.0,
                        },
                    )
                    stats["count"] += valid.sum()
                    stats["absolute"] += absolute[valid].sum()
                    stats["squared"] += np.square(error[valid]).sum()
                    stats["target_absolute"] += np.abs(y[valid]).sum()
                    stats["smape"] += (
                        2.0
                        * absolute[valid]
                        / np.maximum(np.abs(p[valid]) + np.abs(y[valid]), 1e-5)
                    ).sum()
                    per_window = (absolute * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
                    window_errors[horizon].append(per_window.astype(np.float32))
        model_name = str(handle.attrs["model_name"])
        seed = int(handle.attrs["seed"])

    metrics: dict[str, Any] = {}
    step_minutes = config.data.sampling_seconds / 60
    for (horizon, feature), stats in sorted(sums.items()):
        count = max(stats["count"], 1)
        key = f"{(horizon + 1) * step_minutes:g}min"
        metrics.setdefault(key, {})[data.series.feature_names[feature]] = {
            "mae": float(stats["absolute"] / count),
            "rmse": float(math.sqrt(stats["squared"] / count)),
            "wape": float(stats["absolute"] / max(stats["target_absolute"], 1e-8)),
            "smape": float(stats["smape"] / count),
            "count": int(stats["count"]),
        }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "window_errors.npz"
    np.savez_compressed(
        archive_path,
        **{
            f"h{horizon + 1}": np.concatenate(chunks)
            for horizon, chunks in window_errors.items()
        },
    )
    result = {
        "model": model_name,
        "dataset": protocol["dataset"],
        "seed": seed,
        "protocol_sha256": expected_hash,
        "test_metrics": metrics,
        "window_errors_path": str(archive_path),
    }
    with (output / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def validate_competitor_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    competitors = raw.get("competitors", []) if isinstance(raw, dict) else []
    errors: list[str] = []
    for item in competitors:
        name = item.get("name", "<unnamed>")
        commit = str(item.get("commit", ""))
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            errors.append(f"{name}: commit must be a pinned 40-character lowercase SHA")
        if not item.get("repository") or not item.get("paper"):
            errors.append(f"{name}: repository and paper are required")
        lock_value = item.get("environment_lock")
        if not lock_value:
            errors.append(f"{name}: environment_lock is required")
        else:
            lock_path = Path(lock_value)
            if not lock_path.is_absolute():
                lock_path = registry_path.parent / lock_path
            if not lock_path.is_file():
                errors.append(f"{name}: environment lock does not exist: {lock_path}")
    if errors:
        raise ValueError("invalid competitor registry:\n" + "\n".join(errors))
    return {"competitors": len(competitors), "status": "valid"}
