"""Evaluate flow-only SpecSTG with the shared point evaluator plus sample CRPS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from dagnn_pistnet_v3.benchmark_protocol import evaluate_prediction_artifact
from dagnn_pistnet_v3.config import V3ExperimentConfig


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checked_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    expected = int(np.prod(shape, dtype=np.int64)) * 4
    if path.stat().st_size != expected:
        raise ValueError(f"{path} has {path.stat().st_size} bytes, expected {expected}")
    return np.memmap(path, dtype="float32", mode="r", shape=shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[13, 42, 97])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    state = json.loads((run_dir / "inference_state.json").read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if state["status"] != "completed" or int(state["completed_windows"]) != 3561:
        raise RuntimeError("inference incomplete; refusing partial evaluation")
    if state["seed"] != args.seed:
        raise ValueError("seed mismatch")
    if tuple(protocol["prediction_shape"]) != (3561, 12, 170, 1):
        raise ValueError("not the canonical flow-only protocol")
    if protocol["split"]["test_first_target"] != 14284:
        raise ValueError("test split mismatch")
    mean_path = run_dir / "prediction.flow_mean.raw.f32"
    samples_path = run_dir / "samples.flow.raw.f32"
    q10_path = run_dir / "prediction.flow_q10.raw.f32"
    q90_path = run_dir / "prediction.flow_q90.raw.f32"
    point_shape = (3561, 12, 170)
    mean = checked_memmap(mean_path, point_shape)
    samples = checked_memmap(samples_path, (3561, 200, 12, 170))
    q10 = checked_memmap(q10_path, point_shape)
    q90 = checked_memmap(q90_path, point_shape)

    view_path = run_dir / "prediction.flow_mean.external.h5"
    if not view_path.exists():
        temporary = view_path.with_suffix(".h5.tmp")
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = "dagnn-pistnet-predictions-v1"
            handle.attrs["protocol_sha256"] = protocol["protocol_sha256"]
            handle.attrs["model_name"] = "SpecSTG"
            handle.attrs["seed"] = args.seed
            handle.attrs["zero_copy_external_storage"] = True
            handle.create_dataset(
                "prediction", shape=(3561, 12, 170, 1), dtype="float32",
                external=[(mean_path.as_posix(), 0, mean_path.stat().st_size)],
            )
        os.replace(temporary, view_path)

    config = V3ExperimentConfig.from_yaml(args.config)
    # Keep the shared evaluator's intermediate result separate so a stopped
    # CRPS pass cannot make the run look fully evaluated.
    result = evaluate_prediction_artifact(
        config, args.protocol, view_path, run_dir / "flow_common_point"
    )
    with h5py.File(args.bundle, "r") as handle:
        truth = handle["values"]
        anchors = np.asarray(handle["test_target_index"], dtype=np.int64)
        if len(anchors) != 3561 or int(anchors[0]) != 14284:
            raise ValueError("bundle test anchors mismatch")
        crps_windows = {h: np.empty(3561, dtype=np.float32) for h in (0, 2, 5, 11)}
        totals = {h: {"crps": 0.0, "coverage": 0.0, "width": 0.0, "count": 0} for h in crps_windows}
        rank_weight = (2 * np.arange(200, dtype=np.float64) - 200 + 1) / (200 * 200)
        for index, anchor in enumerate(anchors):
            target = np.asarray(truth[int(anchor) : int(anchor) + 12, :, 0], dtype=np.float32)
            for horizon in crps_windows:
                y = target[horizon]
                valid = np.isfinite(y) & (np.abs(y) > 1e-5)
                draw = np.asarray(samples[index, :, horizon, :], dtype=np.float64)
                term1 = np.abs(draw - y[None, :]).mean(axis=0)
                sorted_draw = np.sort(draw, axis=0)
                term2 = (rank_weight[:, None] * sorted_draw).sum(axis=0)
                values = np.maximum(term1 - term2, 0.0)
                count = int(valid.sum())
                crps_windows[horizon][index] = float(values[valid].mean()) if count else np.nan
                totals[horizon]["crps"] += float(values[valid].sum())
                totals[horizon]["coverage"] += float(((y >= q10[index, horizon]) & (y <= q90[index, horizon]) & valid).sum())
                totals[horizon]["width"] += float((q90[index, horizon] - q10[index, horizon])[valid].sum())
                totals[horizon]["count"] += count

    crps_path = run_dir / "window_crps.npz"
    np.savez_compressed(crps_path, **{f"h{h+1}": item for h, item in crps_windows.items()})
    for horizon, values in totals.items():
        time_key = f"{(horizon + 1) * 5}min"
        count = max(values["count"], 1)
        result["test_metrics"][time_key]["flow"].update(
            crps=values["crps"] / count,
            coverage80=values["coverage"] / count,
            mean_interval_width80=values["width"] / count,
        )
    result["evaluation_scope"] = "flow_only"
    result["point_evaluator"] = "dagnn_pistnet_v3.benchmark_protocol.evaluate_prediction_artifact"
    result["probabilistic_evaluator"] = "empirical_200_sample_CRPS_and_central_80_percent_interval"
    result["parent_three_feature_protocol_sha256"] = state["protocol_sha256"]
    result["window_crps_path"] = str(crps_path)
    result["prediction_mean_path"] = str(mean_path)
    result["samples_path"] = str(samples_path)
    write_json(run_dir / "result.json", result)
    print(json.dumps({"seed": args.seed, "status": "completed", "metrics": result["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
