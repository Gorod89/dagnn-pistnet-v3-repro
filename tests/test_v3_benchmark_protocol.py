from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from dagnn_pistnet_v3.benchmark_protocol import (
    create_prediction_artifact,
    evaluate_prediction_artifact,
    export_benchmark_protocol,
    validate_competitor_registry,
)
from dagnn_pistnet_v3.config import V3ExperimentConfig
from dagnn_pistnet_v3.data import prepare_v3_data


def _config(tmp_path: Path) -> V3ExperimentConfig:
    values = np.arange(48 * 3, dtype=np.float32).reshape(48, 3, 1) + 1
    data_path = tmp_path / "pems07.npz"
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


def test_common_prediction_protocol_scores_perfect_forecast(tmp_path: Path) -> None:
    config = _config(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = export_benchmark_protocol(config, protocol_path, tmp_path)
    prediction_path = tmp_path / "predictions.h5"
    create_prediction_artifact(protocol_path, prediction_path, "oracle-test", 7)
    data = prepare_v3_data(config)
    targets = []
    for index in range(len(data.test_dataset)):
        _, normalized, *_ = data.test_dataset[index]
        targets.append(data.scaler.inverse_transform(normalized.numpy()))
    with h5py.File(prediction_path, "r+") as handle:
        handle["prediction"][:] = np.stack(targets)
    result = evaluate_prediction_artifact(
        config, protocol_path, prediction_path, tmp_path / "result", batch_size=2
    )
    assert result["protocol_sha256"] == protocol["protocol_sha256"]
    assert result["test_metrics"]["5min"]["flow"]["mae"] == 0.0
    assert result["test_metrics"]["15min"]["flow"]["rmse"] == 0.0


def test_unfilled_prediction_artifact_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    export_benchmark_protocol(config, protocol_path, tmp_path)
    prediction_path = tmp_path / "predictions.h5"
    create_prediction_artifact(protocol_path, prediction_path, "unfilled", 1)
    with pytest.raises(ValueError, match="NaN"):
        evaluate_prediction_artifact(
            config, protocol_path, prediction_path, tmp_path / "result"
        )


def test_registry_rejects_unlocked_environment(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
competitors:
  - name: model
    repository: https://example.test/repo
    commit: 0123456789012345678901234567890123456789
    paper: https://example.test/paper
    environment_lock: null
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="environment_lock"):
        validate_competitor_registry(registry)


def test_protocol_file_contains_hash(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = tmp_path / "protocol.json"
    export_benchmark_protocol(config, path, tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert len(raw["protocol_sha256"]) == 64
