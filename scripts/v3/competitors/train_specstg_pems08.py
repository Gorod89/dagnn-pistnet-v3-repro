"""Resumable SpecSTG training on the canonical PEMS08 flow split.

This adapter keeps the pinned source architecture and published optimization
budget while disclosing the minimal 12-step-history and initialization fixes.
It does not compute test metrics; prediction/evaluation is a separate stage.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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


def make_batch(
    anchors: np.ndarray,
    spectral_time_first: np.ndarray,
    features: np.ndarray,
    graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    offsets = np.arange(-12, 12, dtype=np.int64)
    positions = anchors[:, None] + offsets
    sequences = torch.as_tensor(
        spectral_time_first[positions].copy(), dtype=torch.float32, device=device
    )
    times = torch.as_tensor(features[positions].copy(), dtype=torch.float32, device=device)
    batch = len(anchors)
    edge_index, edge_weight, lambdas = graph
    return (
        times[:, :12],
        sequences[:, :12],
        torch.ones_like(sequences[:, :12]),
        torch.zeros((batch, 12), dtype=torch.float32, device=device),
        times[:, 12:],
        sequences[:, 12:],
        torch.ones_like(sequences[:, 12:]),
        edge_index.repeat(batch, 1, 1),
        edge_weight.repeat(batch, 1),
        lambdas.repeat(batch, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[13, 42, 97])
    parser.add_argument("--stop-after-epochs", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "last.pt"
    best_path = args.output_dir / "best.pt"
    state_path = args.output_dir / "state.json"

    install_upstream_compatibility_alias()
    sys.path.insert(0, str(args.source.resolve()))
    from algorithm.SpecSTG_network import SpecSTGTrainingNetwork  # noqa: E402
    from data.spectral_transform import SpecTransform  # noqa: E402

    with h5py.File(args.bundle, "r") as handle:
        if handle.attrs["schema"] != "dagnn-pistnet-competitor-data-v1":
            raise ValueError("wrong bundle schema")
        if handle.attrs["dataset"] != "PEMS08":
            raise ValueError("wrong dataset")
        values = np.asarray(handle["values"][:, :, 0], dtype=np.float32).T
        edge_index_np = np.asarray(handle["edge_index"], dtype=np.int64)
        edge_weight_np = np.asarray(handle["edge_weight"], dtype=np.float32)
        mean = float(handle["scaler_mean"][0])
        std = float(handle["scaler_std"][0])
        train_end = int(handle.attrs["train_end"])
        val_end = int(handle.attrs["val_end"])
        protocol_sha = str(handle.attrs["protocol_sha256"])

    if abs(float(values[:, :train_end].mean()) - mean) > 1e-4:
        raise ValueError("train-only scaler mean mismatch")
    if abs(float(values[:, :train_end].std()) - std) > 1e-4:
        raise ValueError("train-only scaler std mismatch")
    transform = SpecTransform(dataset="PEMS08F")
    transform(
        X=(values - mean) / std,
        edge_weight=edge_weight_np,
        edge_index=edge_index_np,
        num_nodes=values.shape[0],
    )
    spectral = np.ascontiguousarray(transform.X_f.T, dtype=np.float32)
    periods = pd.period_range("2018-01-01", periods=len(spectral), freq="5min")
    features = np.ascontiguousarray(
        np.concatenate([feature(periods) for feature in fourier_time_features_from_frequency("5min")], axis=0).T,
        dtype=np.float32,
    )
    if features.shape != (len(spectral), 6):
        raise ValueError(f"unexpected Fourier feature shape {features.shape}")

    train_anchors = np.arange(12, train_end - 12 + 1, dtype=np.int64)
    val_anchors = np.arange(train_end, val_end - 12 + 1, dtype=np.int64)
    if train_anchors[-1] + 12 > train_end or val_anchors[-1] + 12 > val_end:
        raise ValueError("split leakage")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    common = dict(
        input_size=7, num_nodes=170, num_cells=96, cell_type="GRU",
        history_length=12, context_length=11, prediction_length=12,
        dropout_rate=0.1, lags_seq=[1], conditioning_length=100,
        diff_steps=50, loss_type="l2", beta_end=0.2, beta_schedule="quad",
        residual_layers=8, residual_channels=8, dilation_cycle_length=2,
        K=2, normalization="sym", scaling=False,
    )
    network = SpecSTGTrainingNetwork(**common).to(device)
    theta_names = []
    for name, module in network.named_modules():
        if module.__class__.__name__ == "SpecConv":
            torch.nn.init.ones_(module.theta)
            theta_names.append(name)
    graph = (
        torch.as_tensor(edge_index_np[None], dtype=torch.long, device=device),
        torch.as_tensor(edge_weight_np[None], dtype=torch.float32, device=device),
        torch.as_tensor(np.asarray(transform.Lambda)[None].copy(), dtype=torch.float32, device=device),
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=5e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, steps_per_epoch=200, epochs=300
    )
    rng = np.random.default_rng(args.seed)
    start_epoch = 0
    best_val = float("inf")
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device)
        if saved["seed"] != args.seed or saved["protocol_sha256"] != protocol_sha:
            raise ValueError("checkpoint provenance mismatch")
        network.load_state_dict(saved["network"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        start_epoch = int(saved["epoch"])
        best_val = float(saved["best_val"])
    elif state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("status") != "failed" or previous.get("last_completed_epoch") != 0:
            raise RuntimeError("state exists without checkpoint; refusing duplicate training")

    state = {
        "schema": "specstg-canonical-pems08-training-v1",
        "status": "running",
        "seed": args.seed,
        "pid": os.getpid(),
        "protocol_sha256": protocol_sha,
        "source_commit": "c434bcd27507851ad9f40aa92e7147d3e91e5b20",
        "source_adaptations": [
            "canonical 70/10/20 nonoverlapping timeline split",
            "flow-only train-fitted scaler and canonical graph",
            "12-step history via context=11, lags=[1], input_size=7",
            "explicit ones initialization of upstream uninitialized SpecConv.theta",
            "GluonTS 0.10 compatibility symbol alias for pytorchts 0.6",
        ],
        "theta_modules_initialized": theta_names,
        "max_epochs": 300,
        "batches_per_epoch": 200,
        "validation_batches": 50,
        "batch_size": 8,
        "last_completed_epoch": start_epoch,
        "best_val_loss": best_val,
        "parameter_count": sum(p.numel() for p in network.parameters()),
    }
    save_json(state_path, state)
    val_rng = np.random.default_rng(20240108)
    fixed_val = val_rng.choice(val_anchors, size=(50, 8), replace=True)
    try:
        for epoch in range(start_epoch, 300):
            if args.stop_after_epochs and epoch >= args.stop_after_epochs:
                break
            network.train()
            train_losses = []
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(200):
                anchors = rng.choice(train_anchors, size=8, replace=True)
                batch = make_batch(anchors, spectral, features, graph, device)
                optimizer.zero_grad(set_to_none=True)
                loss = network(*batch)[0]
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"non-finite train loss at epoch {epoch + 1}")
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_losses.append(float(loss.detach().cpu()))
            network.eval()
            val_losses = []
            with torch.no_grad():
                for anchors in fixed_val:
                    loss = network(*make_batch(anchors, spectral, features, graph, device))[0]
                    if not bool(torch.isfinite(loss)):
                        raise ValueError(f"non-finite validation loss at epoch {epoch + 1}")
                    val_losses.append(float(loss.detach().cpu()))
            torch.cuda.synchronize()
            val_loss = float(np.mean(val_losses))
            if val_loss < best_val:
                best_val = val_loss
                temporary_best = best_path.with_suffix(".pt.tmp")
                torch.save({"epoch": epoch + 1, "network": network.state_dict(), "seed": args.seed,
                            "protocol_sha256": protocol_sha, "val_loss": val_loss}, temporary_best)
                os.replace(temporary_best, best_path)
            temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            torch.save({
                "epoch": epoch + 1, "network": network.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(), "best_val": best_val,
                "seed": args.seed, "protocol_sha256": protocol_sha,
            }, temporary_checkpoint)
            os.replace(temporary_checkpoint, checkpoint_path)
            state.update(
                last_completed_epoch=epoch + 1,
                best_val_loss=best_val,
                last_train_loss=float(np.mean(train_losses)),
                last_val_loss=val_loss,
                last_epoch_seconds=time.perf_counter() - started,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            )
            save_json(state_path, state)
            print(json.dumps({"epoch": epoch + 1, "train_loss": state["last_train_loss"],
                              "val_loss": val_loss, "seconds": state["last_epoch_seconds"]}), flush=True)
        if state["last_completed_epoch"] == 300:
            state["status"] = "training_completed"
            save_json(state_path, state)
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        save_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
