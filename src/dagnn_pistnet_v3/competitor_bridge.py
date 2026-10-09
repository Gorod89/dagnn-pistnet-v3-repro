from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .benchmark_protocol import _canonical_hash, create_prediction_artifact
from .config import V3ExperimentConfig
from .data import prepare_v3_data


def _load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)
    expected_hash = protocol.get("protocol_sha256")
    unhashed = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if not expected_hash or expected_hash != _canonical_hash(unhashed):
        raise ValueError("benchmark protocol hash is invalid")
    return protocol


def export_competitor_data_bundle(
    config: V3ExperimentConfig,
    protocol_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Export model-neutral data without depending on a competitor repository.

    Values remain in physical units. The train-only scaler and exact target indices
    are included so an isolated competitor environment can reproduce the split and
    return predictions in the order required by the common evaluator.
    """

    protocol = _load_protocol(protocol_path)
    data = prepare_v3_data(config)
    expected_shape = tuple(int(value) for value in protocol["prediction_shape"])
    if expected_shape[0] != len(data.test_dataset):
        raise ValueError("protocol does not match current test split")
    if expected_shape[2:] != (data.series.num_nodes, data.series.num_features):
        raise ValueError("protocol does not match current node/feature dimensions")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(data.series.values, dtype=np.float32)
    chunks = (min(values.shape[0], 288), min(values.shape[1], 512), values.shape[2])
    with h5py.File(target, "w") as handle:
        handle.attrs["schema"] = "dagnn-pistnet-competitor-data-v1"
        handle.attrs["protocol_sha256"] = protocol["protocol_sha256"]
        handle.attrs["dataset"] = config.data.dataset
        handle.attrs["sampling_seconds"] = config.data.sampling_seconds
        handle.attrs["history_steps"] = config.data.history_steps
        handle.attrs["forecast_steps"] = config.data.forecast_steps
        handle.attrs["train_end"] = data.split.train_end
        handle.attrs["val_end"] = data.split.val_end
        handle.attrs["total"] = data.split.total
        handle.create_dataset(
            "values",
            data=values,
            chunks=chunks,
            compression="gzip",
            shuffle=True,
        )
        handle.create_dataset("scaler_mean", data=data.scaler.mean.reshape(-1))
        handle.create_dataset("scaler_std", data=data.scaler.std.reshape(-1))
        handle.create_dataset(
            "test_target_index",
            data=np.arange(
                data.test_dataset.first_target,
                data.test_dataset.last_target + 1,
                dtype=np.int64,
            ),
        )
        handle.create_dataset("edge_index", data=data.graph.edge_index.astype(np.int64))
        handle.create_dataset("edge_weight", data=data.graph.edge_weight.astype(np.float32))
        text_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "sensor_ids",
            data=np.asarray(data.series.sensor_ids, dtype=object),
            dtype=text_dtype,
        )
        handle.create_dataset(
            "feature_names",
            data=np.asarray(data.series.feature_names, dtype=object),
            dtype=text_dtype,
        )
        handle.create_dataset(
            "units", data=np.asarray(data.series.units, dtype=object), dtype=text_dtype
        )
        if data.series.timestamps is not None:
            timestamps = data.series.timestamps.astype("datetime64[ns]").astype(np.int64)
            handle.create_dataset("timestamps_ns", data=timestamps)
    return target


def _source_array(
    stack: ExitStack,
    path: Path,
    source_key: str,
) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    if suffix == ".npz":
        archive = stack.enter_context(np.load(path, allow_pickle=False))
        if source_key not in archive.files:
            raise ValueError(f"NPZ source has no dataset {source_key!r}")
        return archive[source_key]
    if suffix in {".h5", ".hdf5"}:
        handle = stack.enter_context(h5py.File(path, "r"))
        if source_key not in handle:
            raise ValueError(f"HDF5 source has no dataset {source_key!r}")
        return handle[source_key]
    raise ValueError("prediction source must be .npy, .npz, .h5, or .hdf5")


def _layout_shape(shape: tuple[int, ...], layout: str) -> dict[str, int]:
    axes = layout.upper()
    if len(axes) != len(shape) or len(set(axes)) != len(axes):
        raise ValueError("layout must name each source axis exactly once")
    if set(axes) not in ({"S", "H", "N", "F"}, {"S", "H", "N"}):
        raise ValueError("layout must contain S,H,N and optionally F")
    return dict(zip(axes, shape, strict=True))


def import_competitor_predictions(
    config: V3ExperimentConfig,
    protocol_path: str | Path,
    source_path: str | Path,
    output_path: str | Path,
    model_name: str,
    seed: int,
    *,
    layout: str = "SHNF",
    source_key: str = "prediction",
    value_space: str = "original",
    batch_size: int = 16,
) -> Path:
    """Convert official-repository output into the common prediction artifact.

    ``S,H,N,F`` mean sample/window, horizon, node and feature. A missing F axis is
    accepted only for one-feature protocols. Normalized values are inverted with
    the train-only scaler from the common configuration.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if value_space not in {"original", "normalized"}:
        raise ValueError("value_space must be 'original' or 'normalized'")
    protocol = _load_protocol(protocol_path)
    expected = tuple(int(value) for value in protocol["prediction_shape"])
    axes = layout.upper()

    with ExitStack() as stack:
        source = _source_array(stack, Path(source_path), source_key)
        source_shape = tuple(int(value) for value in source.shape)
        dimensions = _layout_shape(source_shape, axes)
        if "F" not in dimensions:
            dimensions["F"] = 1
        actual = tuple(dimensions[axis] for axis in "SHNF")
        if actual != expected:
            raise ValueError(
                f"source shape/layout resolves to {actual}, expected {expected}"
            )

        scaler = prepare_v3_data(config).scaler if value_space == "normalized" else None
        target = create_prediction_artifact(
            protocol_path, output_path, model_name=model_name, seed=seed
        )
        sample_axis = axes.index("S")
        output_axes = [axes.index(axis) for axis in "SHN" if axis in axes]
        if "F" in axes:
            output_axes.append(axes.index("F"))
        with h5py.File(target, "r+") as handle:
            destination = handle["prediction"]
            handle.attrs["source_layout"] = axes
            handle.attrs["source_value_space"] = value_space
            handle.attrs["source_file"] = str(Path(source_path).resolve())
            for start in range(0, expected[0], batch_size):
                end = min(expected[0], start + batch_size)
                selector = [slice(None)] * len(source_shape)
                selector[sample_axis] = slice(start, end)
                batch = np.asarray(source[tuple(selector)], dtype=np.float32)
                batch = np.transpose(batch, axes=output_axes)
                if "F" not in axes:
                    batch = batch[..., None]
                if not np.isfinite(batch).all():
                    raise ValueError("prediction source contains NaN or infinity")
                if scaler is not None:
                    batch = scaler.inverse_transform(batch)
                destination[start:end] = batch
    return Path(output_path)
