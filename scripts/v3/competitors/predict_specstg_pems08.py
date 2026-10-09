"""Resumable 200-sample canonical PEMS08 flow inference for pinned SpecSTG."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from pts.feature import fourier_time_features_from_frequency

from probe_specstg_pems08 import install_upstream_compatibility_alias


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[13, 42, 97])
    parser.add_argument("--stop-after-windows", type=int, default=0)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.run_dir / "inference_state.json"
    checkpoint = args.run_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    install_upstream_compatibility_alias()
    sys.path.insert(0, str(args.source.resolve()))
    from algorithm.SpecSTG_network import SpecSTGPredictionNetwork  # noqa: E402
    from data.spectral_transform import SpecTransform  # noqa: E402

    with h5py.File(args.bundle, "r") as handle:
        if handle.attrs["schema"] != "dagnn-pistnet-competitor-data-v1":
            raise ValueError("wrong bundle schema")
        if handle.attrs["dataset"] != "PEMS08":
            raise ValueError("wrong dataset")
        values = np.asarray(handle["values"][:, :, 0], dtype=np.float32).T
        edge_index_np = np.asarray(handle["edge_index"], dtype=np.int64)
        edge_weight_np = np.asarray(handle["edge_weight"], dtype=np.float32)
        anchors = np.asarray(handle["test_target_index"], dtype=np.int64)
        mean = float(handle["scaler_mean"][0])
        std = float(handle["scaler_std"][0])
        train_end = int(handle.attrs["train_end"])
        val_end = int(handle.attrs["val_end"])
        protocol_sha = str(handle.attrs["protocol_sha256"])
    if anchors[0] != val_end or anchors[-1] + 12 != values.shape[1]:
        raise ValueError("test anchors disagree with canonical protocol")
    if abs(float(values[:, :train_end].mean()) - mean) > 1e-4:
        raise ValueError("train scaler mean mismatch")
    if abs(float(values[:, :train_end].std()) - std) > 1e-4:
        raise ValueError("train scaler std mismatch")

    transform = SpecTransform(dataset="PEMS08F")
    transform(
        X=(values - mean) / std,
        edge_weight=edge_weight_np,
        edge_index=edge_index_np,
        num_nodes=170,
    )
    spectral = np.ascontiguousarray(transform.X_f.T, dtype=np.float32)
    reconstruct = np.asarray(transform.reconstruct_matrix, dtype=np.float32)
    periods = pd.period_range("2018-01-01", periods=len(spectral), freq="5min")
    features = np.ascontiguousarray(
        np.concatenate([feature(periods) for feature in fourier_time_features_from_frequency("5min")], axis=0).T,
        dtype=np.float32,
    )
    device = torch.device("cuda")
    common = dict(
        input_size=7, num_nodes=170, num_cells=96, cell_type="GRU",
        history_length=12, context_length=11, prediction_length=12,
        dropout_rate=0.1, lags_seq=[1], conditioning_length=100,
        diff_steps=50, loss_type="l2", beta_end=0.2, beta_schedule="quad",
        residual_layers=8, residual_channels=8, dilation_cycle_length=2,
        K=2, normalization="sym", scaling=False,
    )
    network = SpecSTGPredictionNetwork(num_parallel_samples=200, **common).to(device)
    saved = torch.load(checkpoint, map_location=device)
    if saved["seed"] != args.seed or saved["protocol_sha256"] != protocol_sha:
        raise ValueError("best checkpoint provenance mismatch")
    network.load_state_dict(saved["network"])
    network.eval()
    edge_index = torch.as_tensor(edge_index_np[None], dtype=torch.long, device=device)
    edge_weight = torch.as_tensor(edge_weight_np[None], dtype=torch.float32, device=device)
    lambdas = torch.as_tensor(np.asarray(transform.Lambda)[None].copy(), dtype=torch.float32, device=device)

    count = len(anchors)
    shape_samples = (count, 200, 12, 170)
    shape_point = (count, 12, 170)
    files = {
        "samples": (args.run_dir / "samples.flow.raw.f32", shape_samples),
        "mean": (args.run_dir / "prediction.flow_mean.raw.f32", shape_point),
        "median": (args.run_dir / "prediction.flow_median.raw.f32", shape_point),
        "q10": (args.run_dir / "prediction.flow_q10.raw.f32", shape_point),
        "q90": (args.run_dir / "prediction.flow_q90.raw.f32", shape_point),
    }
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state["seed"] != args.seed or state["protocol_sha256"] != protocol_sha:
            raise ValueError("inference state provenance mismatch")
        start = int(state["next_index"])
        if state["status"] == "completed":
            print("inference already completed; skipping", flush=True)
            return
        state["status"] = "running"
        state.pop("error", None)
        mode = "r+"
    else:
        if any(path.exists() for path, _ in files.values()):
            raise RuntimeError("prediction files exist without state; refusing overwrite")
        start = 0
        mode = "w+"
        state = {
            "schema": "specstg-canonical-pems08-inference-v1",
            "status": "running",
            "seed": args.seed,
            "pid": os.getpid(),
            "protocol_sha256": protocol_sha,
            "checkpoint_epoch": int(saved["epoch"]),
            "checkpoint_val_loss": float(saved["val_loss"]),
            "sample_count": 200,
            "test_windows": count,
            "next_index": 0,
            "completed_windows": 0,
            "source_commit": "c434bcd27507851ad9f40aa92e7147d3e91e5b20",
            "value_space": "original_flow_vehicles_per_5_minutes",
            "sampling_seed_rule": "seed*1000000+test_window_index",
            "inference_seconds": 0.0,
            "peak_allocated_mib": 0.0,
        }
    maps = {
        name: np.memmap(path, dtype="float32", mode=mode, shape=shape)
        for name, (path, shape) in files.items()
    }
    save_json(progress_path, state)
    torch.cuda.reset_peak_memory_stats()
    try:
        for index in range(start, count):
            anchor = int(anchors[index])
            torch.manual_seed(args.seed * 1_000_000 + index)
            torch.cuda.manual_seed_all(args.seed * 1_000_000 + index)
            past = torch.as_tensor(spectral[anchor - 12 : anchor].copy(), dtype=torch.float32, device=device)[None]
            past_time = torch.as_tensor(features[anchor - 12 : anchor].copy(), dtype=torch.float32, device=device)[None]
            future_time = torch.as_tensor(features[anchor : anchor + 12].copy(), dtype=torch.float32, device=device)[None]
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                sample_spectral = network(
                    past_time, past, torch.ones_like(past),
                    torch.zeros((1, 12), dtype=torch.float32, device=device),
                    future_time, edge_index, edge_weight, lambdas,
                )
            torch.cuda.synchronize()
            state["inference_seconds"] += time.perf_counter() - started
            sample_np = sample_spectral[0].detach().cpu().numpy()
            sample_original = np.einsum("ij,shj->shi", reconstruct, sample_np, optimize=True)
            sample_original = np.maximum(sample_original * std + mean, 0).astype(np.float32)
            if not np.isfinite(sample_original).all():
                raise ValueError(f"non-finite prediction at test window {index}")
            maps["samples"][index] = sample_original
            maps["mean"][index] = sample_original.mean(axis=0)
            maps["median"][index] = np.median(sample_original, axis=0)
            maps["q10"][index] = np.quantile(sample_original, 0.1, axis=0)
            maps["q90"][index] = np.quantile(sample_original, 0.9, axis=0)
            if ((index + 1) % 25 == 0 or index + 1 == count
                    or (args.stop_after_windows and index + 1 >= args.stop_after_windows)):
                for view in maps.values():
                    view.flush()
                state["next_index"] = index + 1
                state["completed_windows"] = index + 1
                state["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
                save_json(progress_path, state)
                print(json.dumps({"completed_windows": index + 1, "test_windows": count,
                                  "inference_seconds": state["inference_seconds"]}), flush=True)
            if args.stop_after_windows and index + 1 >= args.stop_after_windows:
                break
        if state["next_index"] == count:
            state["status"] = "completed"
            save_json(progress_path, state)
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        save_json(progress_path, state)
        raise


if __name__ == "__main__":
    main()
