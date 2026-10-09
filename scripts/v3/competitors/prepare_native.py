"""Prepare exact native datasets for pinned official competitor runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from canonical_bundle import calendar_features, index_triplets, metadata, target_indices


def _write_manifest(output: Path, meta, model: str, split_policy: str) -> None:
    payload = {
        "schema": "dagnn-pistnet-native-competitor-v1",
        "model": model,
        "dataset": meta.dataset,
        "protocol_sha256": meta.protocol_sha256,
        "sampling_seconds": meta.sampling_seconds,
        "steps_per_day": 86400 // meta.sampling_seconds,
        "history_steps": meta.history_steps,
        "forecast_steps": meta.forecast_steps,
        "split_policy": split_policy,
        "num_nodes": meta.num_nodes,
        "num_features": meta.num_features,
    }
    (output / "native_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _write_staeformer(bundle: Path, output: Path, split_policy: str) -> None:
    meta = metadata(bundle)
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(bundle, "r") as handle:
        values = np.asarray(handle["values"][..., :1], dtype=np.float32)
        tod, dow = calendar_features(handle, meta.sampling_seconds)
    data = np.concatenate(
        (
            values,
            np.broadcast_to(tod[:, None, None], values.shape),
            np.broadcast_to(dow[:, None, None], values.shape).astype(np.float32),
        ),
        axis=-1,
    )
    np.savez_compressed(output / "data.npz", data=data)
    np.savez_compressed(output / "index.npz", **index_triplets(meta, split_policy))
    _write_manifest(output, meta, "STAEformer", split_policy)


def _split_bounds(meta, split_policy: str) -> dict[str, tuple[int, int]]:
    anchors = target_indices(meta, split_policy)
    return {
        name: (
            int(indices[0] - meta.history_steps),
            int(indices[-1] + meta.forecast_steps),
        )
        for name, indices in anchors.items()
    }


def _write_basicts(bundle: Path, output: Path, split_policy: str) -> None:
    meta = metadata(bundle)
    output.mkdir(parents=True, exist_ok=True)
    bounds = _split_bounds(meta, split_policy)
    with h5py.File(bundle, "r") as handle:
        values = handle["values"]
        tod, dow = calendar_features(handle, meta.sampling_seconds)
        timestamps = np.column_stack((tod, dow.astype(np.float32) / 7.0))
        for split, (start, stop) in bounds.items():
            # STID treats every sensor as one multivariate feature.
            np.save(output / f"{split}_data.npy", np.asarray(values[start:stop, :, 0]))
            np.save(output / f"{split}_timestamps.npy", timestamps[start:stop])
    _write_manifest(output, meta, "STID", split_policy)


def _write_fast(bundle: Path, output: Path, split_policy: str) -> None:
    meta = metadata(bundle)
    if split_policy != "official_sliding_windows":
        raise ValueError("FaST native runner is locked to official_sliding_windows")
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(bundle, "r") as handle:
        values = np.asarray(handle["values"][..., :1], dtype=np.float32)
        tod, dow = calendar_features(handle, meta.sampling_seconds)
    data = np.concatenate(
        (
            values,
            np.broadcast_to(tod[:, None, None], values.shape),
            np.broadcast_to((dow / 7.0)[:, None, None], values.shape).astype(np.float32),
        ),
        axis=-1,
    )
    np.savez_compressed(output / "his.npz", data=data)
    index_dir = output / f"{meta.history_steps}_{meta.forecast_steps}"
    index_dir.mkdir(parents=True, exist_ok=True)
    # FaST indices name the final output timestamp, not the first future timestamp.
    for name, indices in target_indices(meta, split_policy).items():
        np.save(index_dir / f"idx_{name}.npy", indices + meta.forecast_steps - 1)
    description = {
        "name": meta.dataset,
        "domain": "traffic flow",
        "shape": [meta.total, meta.num_nodes, 3],
        "num_time_steps": meta.total,
        "num_nodes": meta.num_nodes,
        "num_features": 3,
        "frequency (minutes)": meta.sampling_seconds // 60,
        "canonical_protocol_sha256": meta.protocol_sha256,
    }
    (output / "desc.json").write_text(json.dumps(description, indent=2), encoding="utf-8")
    _write_manifest(output, meta, "FaST", split_policy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("staeformer", "stid", "fast"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split-policy",
        choices=("nonoverlapping_timeline", "official_sliding_windows"),
        required=True,
    )
    args = parser.parse_args()
    writer = {
        "staeformer": _write_staeformer,
        "stid": _write_basicts,
        "fast": _write_fast,
    }[args.model]
    writer(args.bundle.resolve(), args.output.resolve(), args.split_policy)
    print(json.dumps(metadata(args.bundle).__dict__, default=str, indent=2))


if __name__ == "__main__":
    main()
