"""Read and validate the model-neutral competitor HDF5 bundle.

This module intentionally depends only on h5py and NumPy so it can be copied or
placed on PYTHONPATH inside each isolated competitor environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class BundleMetadata:
    path: Path
    dataset: str
    protocol_sha256: str
    sampling_seconds: int
    history_steps: int
    forecast_steps: int
    train_end: int
    val_end: int
    total: int
    num_nodes: int
    num_features: int


def metadata(path: str | Path) -> BundleMetadata:
    source = Path(path).resolve()
    with h5py.File(source, "r") as handle:
        if handle.attrs.get("schema") != "dagnn-pistnet-competitor-data-v1":
            raise ValueError(f"unsupported canonical bundle schema in {source}")
        values = handle["values"]
        if values.ndim != 3:
            raise ValueError("values must have shape [time,node,feature]")
        total, num_nodes, num_features = map(int, values.shape)
        recorded_total = int(handle.attrs["total"])
        if total != recorded_total:
            raise ValueError("values length differs from recorded total")
        targets = np.asarray(handle["test_target_index"], dtype=np.int64)
        if targets.ndim != 1 or targets.size == 0 or np.any(np.diff(targets) != 1):
            raise ValueError("test_target_index must be a non-empty contiguous vector")
        val_end = int(handle.attrs["val_end"])
        forecast = int(handle.attrs["forecast_steps"])
        if targets[0] != val_end or targets[-1] + forecast != total:
            raise ValueError("test_target_index is inconsistent with split metadata")
        return BundleMetadata(
            path=source,
            dataset=str(handle.attrs["dataset"]),
            protocol_sha256=str(handle.attrs["protocol_sha256"]),
            sampling_seconds=int(handle.attrs["sampling_seconds"]),
            history_steps=int(handle.attrs["history_steps"]),
            forecast_steps=forecast,
            train_end=int(handle.attrs["train_end"]),
            val_end=val_end,
            total=total,
            num_nodes=num_nodes,
            num_features=num_features,
        )


def target_indices(meta: BundleMetadata, split_policy: str) -> dict[str, np.ndarray]:
    """Return first-future indices in the canonical evaluator order."""

    if split_policy not in {"nonoverlapping_timeline", "official_sliding_windows"}:
        raise ValueError(
            "split_policy must be nonoverlapping_timeline or official_sliding_windows"
        )
    history, forecast = meta.history_steps, meta.forecast_steps
    overlap = forecast - 1 if split_policy == "official_sliding_windows" else 0
    train_stop = meta.train_end + overlap - forecast + 1
    val_stop = meta.val_end + overlap - forecast + 1
    result = {
        "train": np.arange(history, train_stop, dtype=np.int64),
        "val": np.arange(meta.train_end, val_stop, dtype=np.int64),
        "test": np.arange(meta.val_end, meta.total - forecast + 1, dtype=np.int64),
    }
    if any(indices.size == 0 for indices in result.values()):
        raise ValueError("canonical split contains an empty partition")
    return result


def index_triplets(meta: BundleMetadata, split_policy: str) -> dict[str, np.ndarray]:
    return {
        name: np.column_stack(
            (indices - meta.history_steps, indices, indices + meta.forecast_steps)
        ).astype(np.int64, copy=False)
        for name, indices in target_indices(meta, split_policy).items()
    }


def calendar_features(handle: h5py.File, sampling_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    """Return time-of-day [0,1) and weekday [0,6] vectors."""

    total = int(handle["values"].shape[0])
    if "timestamps_ns" in handle:
        timestamps = np.asarray(handle["timestamps_ns"], dtype=np.int64).astype(
            "datetime64[ns]"
        )
        days = timestamps.astype("datetime64[D]")
        tod = ((timestamps - days) / np.timedelta64(1, "D")).astype(np.float32)
        # 1970-01-01 is Thursday; Monday is zero.
        dow = ((days.astype(np.int64) + 3) % 7).astype(np.int64)
        return tod, dow
    steps_per_day = 86400 // sampling_seconds
    if steps_per_day * sampling_seconds != 86400:
        raise ValueError("calendar synthesis requires sampling to divide one day")
    index = np.arange(total, dtype=np.int64)
    return (index % steps_per_day).astype(np.float32) / steps_per_day, (
        index // steps_per_day
    ) % 7
