from pathlib import Path

import h5py
import numpy as np
import pytest

from dagnn_pistnet.config import DataConfig
from dagnn_pistnet.data import (
    ChronologicalSplit,
    MetrLaAdapter,
    PemsNpzAdapter,
    StandardScaler,
    TrafficSeries,
    TrafficWindowDataset,
    build_adapter,
    encode_calendar_features,
)


def test_series_rejects_graph_from_another_dataset() -> None:
    with pytest.raises(ValueError, match="does not match nodes"):
        TrafficSeries(
            values=np.zeros((20, 5, 1)),
            feature_names=("flow",),
            units=("vehicles",),
            adjacency=np.zeros((4, 4)),
        )


def test_scaler_is_fit_on_training_values_only() -> None:
    values = np.concatenate(
        [np.zeros((10, 2, 1), dtype=np.float32), np.full((10, 2, 1), 100.0)], axis=0
    )
    scaler = StandardScaler.fit(values[:10])
    assert scaler.mean.item() == 0.0
    assert scaler.std.item() == 1.0
    assert scaler.transform(values[10:]).mean() == 100.0


def test_chronological_windows_do_not_share_targets() -> None:
    values = np.arange(100, dtype=np.float32).reshape(50, 2, 1)
    split = ChronologicalSplit.from_ratios(50, train_ratio=0.6, val_ratio=0.2)
    train = TrafficWindowDataset(values, 4, 3, *split.target_bounds("train"))
    validation = TrafficWindowDataset(values, 4, 3, *split.target_bounds("val"))
    _, train_last_target, _, _ = train[-1]
    _, validation_first_target, _, _ = validation[0]
    assert train_last_target[-1, 0, 0].item() == values[split.train_end - 1, 0, 0]
    assert validation_first_target[0, 0, 0].item() == values[split.train_end, 0, 0]


def test_pems_adapter_requires_explicit_feature_metadata(tmp_path: Path) -> None:
    path = tmp_path / "pems.npz"
    np.savez(path, data=np.ones((30, 4, 1), dtype=np.float32))
    adapter = PemsNpzAdapter(
        DataConfig(
            dataset="PEMS07",
            data_path=str(path),
            feature_names=("flow",),
            units=("vehicles_per_5_minutes",),
            start_timestamp="2017-05-01T00:00:00",
        )
    )
    series = adapter.load()
    assert series.feature_names == ("flow",)
    assert series.values.shape == (30, 4, 1)
    assert series.timestamps is not None
    assert series.timestamps[1] - series.timestamps[0] == np.timedelta64(300, "s")


def test_hdf_adapter_accepts_pems_bay_group_name(tmp_path: Path) -> None:
    path = tmp_path / "bay.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("speed/block0_values", data=np.ones((20, 3)))
        h5.create_dataset(
            "speed/axis1",
            data=np.arange(20, dtype=np.int64) * 300_000_000_000,
        )
        h5.create_dataset("speed/axis0", data=np.arange(3, dtype=np.int64))
    adapter = MetrLaAdapter(
        DataConfig(
            dataset="PEMS-BAY",
            data_path=str(path),
            feature_names=("speed",),
            units=("mph",),
        )
    )
    series = adapter.load()
    assert series.values.shape == (20, 3, 1)
    assert series.timestamps is not None


def test_hdf_adapter_returns_writable_values_from_pandas_hdf(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")
    path = tmp_path / "frame.h5"
    frame = pd.DataFrame(
        [[1.0, np.nan], [3.0, 4.0]],
        index=pd.date_range("2026-01-01", periods=2, freq="15min"),
        columns=["a", "b"],
    )
    frame.to_hdf(path, key="data")
    series = MetrLaAdapter(
        DataConfig(
            dataset="LargeST-CA",
            data_path=str(path),
            feature_names=("flow",),
            units=("vehicles",),
            missing_value_policy="zero",
        )
    ).load()
    assert series.values.flags.writeable
    assert series.values[0, 1, 0] == 0.0


def test_calendar_features_encode_midnight_and_weekend() -> None:
    timestamps = np.array(
        ["2026-07-17T00:00:00", "2026-07-18T12:00:00"],
        dtype="datetime64[s]",
    )
    features = encode_calendar_features(timestamps)
    assert features.shape == (2, 5)
    assert features[0, 0] == pytest.approx(0.0, abs=1e-7)
    assert features[0, 1] == pytest.approx(1.0, abs=1e-7)
    assert features[0, 4] == 0.0
    assert features[1, 4] == 1.0


def test_largest_adapter_loads_npy_graph_and_zero_fills_missing_values(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "largest.h5"
    graph_path = tmp_path / "adj.npy"
    values = np.ones((20, 4), dtype=np.float64)
    values[3, 2] = np.nan
    with h5py.File(data_path, "w") as h5:
        h5.create_dataset("t/block0_values", data=values)
        h5.create_dataset("t/axis1", data=np.arange(20, dtype=np.int64) * 300_000_000_000)
        h5.create_dataset("t/axis0", data=np.arange(4, dtype=np.int64))
    np.save(graph_path, np.eye(4, dtype=np.float64))
    series = build_adapter(
        DataConfig(
            dataset="LargeST-CA",
            data_path=str(data_path),
            adjacency_path=str(graph_path),
            feature_names=("flow",),
            units=("vehicles",),
            missing_value_policy="zero",
        )
    ).load()
    assert series.values[3, 2, 0] == 0.0
    assert series.adjacency is not None
    assert series.adjacency.shape == (4, 4)
