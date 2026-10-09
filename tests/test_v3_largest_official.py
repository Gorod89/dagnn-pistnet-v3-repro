from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dagnn_pistnet_v3.benchmark_protocol import export_benchmark_protocol
from dagnn_pistnet_v3.config import V3ExperimentConfig
from dagnn_pistnet_v3.data import prepare_v3_data
from dagnn_pistnet_v3.largest_official import aggregate_official_15min


def test_official_aggregation_matches_notebook_formula() -> None:
    index = pd.date_range("2019-01-01", periods=6, freq="5min")
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 4.0, np.nan, np.nan, np.nan], "b": [2.0, 3.0, 3.0, 1, 2, 3]},
        index=index,
    )
    result = aggregate_official_15min(frame)
    expected = frame.resample("15min").mean().round(0).fillna(0)
    pd.testing.assert_frame_equal(result, expected)
    assert result.iloc[0].tolist() == [2.0, 3.0]
    assert result.iloc[1].tolist() == [0.0, 2.0]


def test_official_split_uses_eligible_window_counts(tmp_path: Path) -> None:
    values = np.arange(40, dtype=np.float32).reshape(40, 1, 1) + 1
    data_path = tmp_path / "flow.npz"
    np.savez(data_path, data=values)
    config = V3ExperimentConfig.from_dict(
        {
            "data": {
                "dataset": "PEMS07",
                "data_path": str(data_path),
                "history_steps": 4,
                "forecast_steps": 3,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "split_policy": "official_sliding_windows",
            },
            "model": {"hidden_dim": 8, "temporal_layers": 1},
            "training": {"epochs": 1, "batch_size": 2},
        }
    )
    prepared = prepare_v3_data(config)
    # 40 - 4 - 3 + 1 = 34 eligible windows -> round(20.4), round(6.8), 7.
    assert [
        len(prepared.train_dataset),
        len(prepared.validation_dataset),
        len(prepared.test_dataset),
    ] == [20, 7, 7]
    assert prepared.test_dataset.first_target == 31
    assert prepared.test_dataset.last_target == 37
    # Official builder fits on data[:idx_val[0] - history_steps] == data[:19].
    assert prepared.scaler.mean.item() == np.mean(values[:19]).item()
    protocol = export_benchmark_protocol(config, tmp_path / "protocol.json", tmp_path)
    assert protocol["split_policy"] == "official_sliding_windows"


def test_existing_2017_config_keeps_nonoverlapping_timeline_policy() -> None:
    config = V3ExperimentConfig.from_yaml("configs/v3/large_st_ca.yaml")
    assert config.data.dataset == "LargeST-CA"
    assert config.data.sampling_seconds == 300
    assert config.data.split_policy == "nonoverlapping_timeline"


def test_default_policy_retains_nonoverlapping_timeline_windows(tmp_path: Path) -> None:
    values = np.arange(40, dtype=np.float32).reshape(40, 1, 1) + 1
    data_path = tmp_path / "flow.npz"
    np.savez(data_path, data=values)
    config = V3ExperimentConfig.from_dict(
        {
            "data": {
                "dataset": "PEMS07",
                "data_path": str(data_path),
                "history_steps": 4,
                "forecast_steps": 3,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
            },
            "model": {"hidden_dim": 8, "temporal_layers": 1},
            "training": {"epochs": 1, "batch_size": 2},
        }
    )
    prepared = prepare_v3_data(config)
    assert [
        len(prepared.train_dataset),
        len(prepared.validation_dataset),
        len(prepared.test_dataset),
    ] == [18, 6, 6]
    assert prepared.split.train_end == 24
    assert prepared.split.val_end == 32
