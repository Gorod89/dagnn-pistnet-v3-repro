from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from dagnn_pistnet_v3.benchmark_protocol import export_benchmark_protocol
from dagnn_pistnet_v3.competitor_bridge import (
    export_competitor_data_bundle,
    import_competitor_predictions,
)
from dagnn_pistnet_v3.config import V3ExperimentConfig


def _config(tmp_path: Path) -> V3ExperimentConfig:
    values = np.arange(60 * 3, dtype=np.float32).reshape(60, 3, 1) + 1
    data_path = tmp_path / "data.npz"
    np.savez(data_path, data=values)
    return V3ExperimentConfig.from_dict(
        {
            "data": {
                "dataset": "PEMS07",
                "data_path": str(data_path),
                "feature_names": ["flow"],
                "units": ["vehicles_per_step"],
                "history_steps": 4,
                "forecast_steps": 3,
                "train_ratio": 0.5,
                "val_ratio": 0.25,
            },
            "model": {"hidden_dim": 8, "temporal_layers": 1},
            "training": {"epochs": 1, "batch_size": 2},
        }
    )


def test_export_bundle_carries_exact_split_and_scaler(tmp_path: Path) -> None:
    config = _config(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = export_benchmark_protocol(config, protocol_path, tmp_path)
    bundle_path = export_competitor_data_bundle(
        config, protocol_path, tmp_path / "bundle.h5"
    )
    with h5py.File(bundle_path, "r") as handle:
        assert handle.attrs["protocol_sha256"] == protocol["protocol_sha256"]
        assert tuple(handle["values"].shape) == (60, 3, 1)
        assert len(handle["test_target_index"]) == protocol["prediction_shape"][0]
        assert handle["scaler_mean"].shape == (1,)


def test_import_transposes_single_feature_predictions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = export_benchmark_protocol(config, protocol_path, tmp_path)
    shape = tuple(protocol["prediction_shape"])
    canonical = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    source = np.transpose(canonical[..., 0], (1, 0, 2))
    source_path = tmp_path / "external.npy"
    np.save(source_path, source)
    output_path = import_competitor_predictions(
        config,
        protocol_path,
        source_path,
        tmp_path / "predictions.h5",
        "STID",
        13,
        layout="HSN",
        batch_size=2,
    )
    with h5py.File(output_path, "r") as handle:
        np.testing.assert_array_equal(handle["prediction"][:], canonical)
        assert handle.attrs["model_name"] == "STID"


def test_import_inverts_train_only_normalization(tmp_path: Path) -> None:
    config = _config(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = export_benchmark_protocol(config, protocol_path, tmp_path)
    shape = tuple(protocol["prediction_shape"])
    source_path = tmp_path / "normalized.npz"
    np.savez(source_path, prediction=np.zeros(shape, dtype=np.float32))
    output_path = import_competitor_predictions(
        config,
        protocol_path,
        source_path,
        tmp_path / "predictions.h5",
        "PDFormer",
        42,
        value_space="normalized",
    )
    with h5py.File(output_path, "r") as handle:
        expected_mean = (
            np.arange(60 * 3, dtype=np.float32).reshape(60, 3, 1)[:30] + 1
        ).mean()
        np.testing.assert_allclose(handle["prediction"][:], expected_mean)
        assert handle.attrs["source_value_space"] == "normalized"


def test_import_rejects_wrong_window_count(tmp_path: Path) -> None:
    config = _config(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = export_benchmark_protocol(config, protocol_path, tmp_path)
    shape = list(protocol["prediction_shape"])
    shape[0] -= 1
    source_path = tmp_path / "wrong.npy"
    np.save(source_path, np.zeros(shape, dtype=np.float32))
    try:
        import_competitor_predictions(
            config,
            protocol_path,
            source_path,
            tmp_path / "predictions.h5",
            "STAEformer",
            97,
        )
    except ValueError as error:
        assert "expected" in str(error)
    else:
        raise AssertionError("wrong window count must be rejected")


def test_import_rejects_tampered_protocol(tmp_path: Path) -> None:
    config = _config(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    export_benchmark_protocol(config, protocol_path, tmp_path)
    raw = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw["sampling_seconds"] = 900
    protocol_path.write_text(json.dumps(raw), encoding="utf-8")
    source_path = tmp_path / "external.npy"
    np.save(source_path, np.zeros((1, 3, 3, 1), dtype=np.float32))
    try:
        import_competitor_predictions(
            config,
            protocol_path,
            source_path,
            tmp_path / "predictions.h5",
            "STAEformer",
            13,
        )
    except ValueError as error:
        assert "hash" in str(error)
    else:
        raise AssertionError("tampered protocol must be rejected")
