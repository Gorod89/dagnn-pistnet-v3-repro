from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

RUNNER_DIR = Path(__file__).parents[1] / "scripts" / "v3" / "competitors"
sys.path.insert(0, str(RUNNER_DIR))


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bundle_api = _module("canonical_bundle")
prepare = _module("prepare_native")


def _bundle(path: Path, sampling_seconds: int) -> Path:
    total, nodes = 80, 3
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            schema="dagnn-pistnet-competitor-data-v1",
            dataset="fixture",
            protocol_sha256="a" * 64,
            sampling_seconds=sampling_seconds,
            history_steps=12,
            forecast_steps=12,
            train_end=46,
            val_end=58,
            total=total,
        )
        handle.create_dataset(
            "values", data=np.arange(total * nodes, dtype=np.float32).reshape(total, nodes, 1)
        )
        handle.create_dataset("test_target_index", data=np.arange(58, 69))
        handle.create_dataset("timestamps_ns", data=(
            np.datetime64("2019-01-07T00:00")
            + np.arange(total) * np.timedelta64(sampling_seconds, "s")
        ).astype("datetime64[ns]").astype(np.int64))
        dtype = h5py.string_dtype("utf-8")
        handle.create_dataset(
            "sensor_ids", data=np.array(["a", "b", "c"], dtype=object), dtype=dtype
        )
        handle.create_dataset("edge_index", data=np.array([[0, 1], [1, 2]], dtype=np.int64))
        handle.create_dataset("edge_weight", data=np.ones(2, dtype=np.float32))
    return path


@pytest.mark.parametrize(("sampling_seconds", "steps_per_day"), [(300, 288), (900, 96)])
def test_native_manifest_uses_canonical_sampling(tmp_path, sampling_seconds, steps_per_day):
    bundle = _bundle(tmp_path / "bundle.h5", sampling_seconds)
    output = tmp_path / "stid"
    prepare._write_basicts(bundle, output, "nonoverlapping_timeline")
    manifest = json.loads((output / "native_manifest.json").read_text())
    assert manifest["sampling_seconds"] == sampling_seconds
    assert manifest["steps_per_day"] == steps_per_day
    timestamps = np.load(output / "test_timestamps.npy")
    assert timestamps[1, 0] - timestamps[0, 0] == pytest.approx(
        1 / steps_per_day, abs=3e-8
    )


def test_split_policies_have_exact_expected_anchor_counts(tmp_path):
    meta = bundle_api.metadata(_bundle(tmp_path / "bundle.h5", 900))
    strict = bundle_api.target_indices(meta, "nonoverlapping_timeline")
    official = bundle_api.target_indices(meta, "official_sliding_windows")
    assert [len(strict[name]) for name in ("train", "val", "test")] == [23, 1, 11]
    assert [len(official[name]) for name in ("train", "val", "test")] == [34, 12, 11]
    assert np.array_equal(official["test"], np.arange(58, 69))


def test_staeformer_and_fast_native_layouts(tmp_path):
    bundle = _bundle(tmp_path / "bundle.h5", 900)
    stae = tmp_path / "stae"
    fast = tmp_path / "fast"
    prepare._write_staeformer(bundle, stae, "official_sliding_windows")
    prepare._write_fast(bundle, fast, "official_sliding_windows")
    with np.load(stae / "data.npz") as archive:
        assert archive["data"].shape == (80, 3, 3)
    with np.load(stae / "index.npz") as archive:
        assert archive["test"].shape == (11, 3)
    with np.load(fast / "his.npz") as archive:
        assert archive["data"].shape == (80, 3, 3)
    assert np.array_equal(np.load(fast / "12_12" / "idx_test.npy"), np.arange(69, 80))
