"""Bounded SpecSTG PEMS08 GPU feasibility probe; not an accuracy benchmark."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import h5py
import numpy as np
import torch


def install_upstream_compatibility_alias() -> None:
    """Restore the GluonTS 0.10 symbol path expected by pytorchts 0.6.0."""
    from gluonts.torch.distributions import distribution_output
    from gluonts.torch.modules.lambda_layer import LambdaLayer

    module = types.ModuleType("gluonts.torch.modules.distribution_output")
    module.DistributionOutput = distribution_output.DistributionOutput
    module.PtArgProj = distribution_output.PtArgProj
    module.LambdaLayer = LambdaLayer
    sys.modules[module.__name__] = module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    if args.samples < 1 or args.samples > 200:
        raise ValueError("bounded probe allows 1–200 samples")

    install_upstream_compatibility_alias()
    sys.path.insert(0, str(args.source.resolve()))
    from algorithm.SpecSTG_network import (  # noqa: E402
        SpecSTGPredictionNetwork,
        SpecSTGTrainingNetwork,
    )
    from data.spectral_transform import SpecTransform  # noqa: E402

    with h5py.File(args.bundle, "r") as handle:
        if handle.attrs["schema"] != "dagnn-pistnet-competitor-data-v1":
            raise ValueError("wrong bundle schema")
        if handle.attrs["dataset"] != "PEMS08":
            raise ValueError("wrong dataset")
        values = np.asarray(handle["values"][:, :, 0], dtype=np.float32).T
        edge_index = np.asarray(handle["edge_index"], dtype=np.int64)
        edge_weight = np.asarray(handle["edge_weight"], dtype=np.float32)
        mean = float(handle["scaler_mean"][0])
        std = float(handle["scaler_std"][0])
        train_end = int(handle.attrs["train_end"])
        protocol_sha = str(handle.attrs["protocol_sha256"])
        anchor = int(handle["test_target_index"][0])

    # Verify the canonical scale is train-only before invoking upstream transform.
    train_mean = float(values[:, :train_end].mean())
    train_std = float(values[:, :train_end].std())
    if not math.isclose(mean, train_mean, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError("bundle mean is not train-only mean")
    if not math.isclose(std, train_std, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError("bundle std is not train-only std")
    x_norm = (values - mean) / std

    transform = SpecTransform(dataset="PEMS08F")
    started = time.perf_counter()
    transform(
        X=x_norm,
        edge_weight=edge_weight,
        edge_index=edge_index,
        num_nodes=values.shape[0],
    )
    spectral_seconds = time.perf_counter() - started
    spectral = np.asarray(transform.X_f)
    reconstruction_error = float(
        np.max(np.abs(transform.reconstruct_matrix @ spectral[:, anchor - 12 : anchor] - x_norm[:, anchor - 12 : anchor]))
    )
    if not np.isfinite(reconstruction_error) or reconstruction_error > 1e-3:
        raise ValueError(f"spectral transform reconstruction error: {reconstruction_error}")

    device = torch.device("cuda")
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    # Canonical 12-step history requires context 11 and one lag. The six
    # upstream Fourier time features plus this lag give input_size 7.
    common = dict(
        input_size=7,
        num_nodes=170,
        num_cells=96,
        cell_type="GRU",
        history_length=12,
        context_length=11,
        prediction_length=12,
        dropout_rate=0.1,
        lags_seq=[1],
        conditioning_length=100,
        diff_steps=50,
        loss_type="l2",
        beta_end=0.2,
        beta_schedule="quad",
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        K=2,
        normalization="sym",
        scaling=False,
    )
    network = SpecSTGTrainingNetwork(**common).to(device)
    # Both upstream SpecConv implementations allocate theta via torch.Tensor(K)
    # without initializing it; their reset_parameters methods are never called
    # and reference an undefined "ones" symbol. Make the intended ones
    # initialization explicit in this adapter, without altering source files.
    initialized_theta = []
    for name, module in network.named_modules():
        if module.__class__.__name__ == "SpecConv":
            torch.nn.init.ones_(module.theta)
            initialized_theta.append(name)
    prediction = SpecSTGPredictionNetwork(num_parallel_samples=args.samples, **common).to(device)
    prediction.load_state_dict(network.state_dict())
    torch.cuda.reset_peak_memory_stats()
    sample = torch.as_tensor(spectral[:, anchor - 12 : anchor].T.copy(), dtype=torch.float32, device=device)[None]
    future = torch.as_tensor(spectral[:, anchor : anchor + 12].T.copy(), dtype=torch.float32, device=device)[None]
    past_time = torch.zeros((1, 12, 6), dtype=torch.float32, device=device)
    future_time = torch.zeros((1, 12, 6), dtype=torch.float32, device=device)
    past_observed = torch.ones_like(sample)
    future_observed = torch.ones_like(future)
    is_pad = torch.zeros((1, 12), dtype=torch.float32, device=device)
    edge_index_t = torch.as_tensor(edge_index[None], dtype=torch.long, device=device)
    edge_weight_t = torch.as_tensor(edge_weight[None], dtype=torch.float32, device=device)
    lambda_t = torch.as_tensor(np.asarray(transform.Lambda)[None].copy(), dtype=torch.float32, device=device)

    torch.cuda.synchronize()
    started = time.perf_counter()
    loss, _, _ = network(
        past_time, sample, past_observed, is_pad, future_time, future,
        future_observed, edge_index_t, edge_weight_t, lambda_t,
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("non-finite loss despite explicit theta initialization")
    loss.backward()
    torch.cuda.synchronize()
    train_step_seconds = time.perf_counter() - started
    prediction.eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        samples = prediction(
            past_time, sample, past_observed, is_pad, future_time,
            edge_index_t, edge_weight_t, lambda_t,
        )
    torch.cuda.synchronize()
    sample_seconds = time.perf_counter() - started
    if tuple(samples.shape) != (1, args.samples, 12, 170):
        raise ValueError(f"unexpected sample shape: {tuple(samples.shape)}")
    if not bool(torch.isfinite(samples).all()):
        raise ValueError("non-finite samples despite explicit theta initialization")
    result = {
        "status": "completed",
        "kind": "bounded_untrained_feasibility_probe_not_accuracy",
        "protocol_sha256": protocol_sha,
        "anchor": anchor,
        "samples": args.samples,
        "source_adaptation": "context=11,lags=[1],input_size=7 for canonical 12-step history; upstream diffusion and hidden settings unchanged",
        "theta_initialization": "explicit ones for uninitialized upstream SpecConv.theta",
        "theta_modules_initialized": initialized_theta,
        "spectral_transform_seconds": spectral_seconds,
        "spectral_reconstruction_max_error": reconstruction_error,
        "parameter_count": sum(p.numel() for p in network.parameters()),
        "train_forward_backward_seconds": train_step_seconds,
        "sample_seconds": sample_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "loss": float(loss.detach().cpu()),
        "sample_shape": list(samples.shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
