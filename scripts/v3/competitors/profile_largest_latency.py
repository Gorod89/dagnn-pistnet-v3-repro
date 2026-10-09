"""Measure matched model-only inference latency for LargeST competitors.

The profiler intentionally avoids data-loader and disk timing.  Both models
receive the same device-resident batch shape (64, 12, 8,600) and are measured
with explicit CUDA synchronization after a warm-up period.  FaST uses the
FP16 autocast path enabled by its official configuration; STID uses FP32, as
in its pinned BasicTS runner.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def _state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    if any(key.startswith("module.") for key in state):
        state = {
            key.removeprefix("module."): value for key, value in state.items()
        }
    return state


def _build_stid(repo: Path, checkpoint: Path, batch_size: int, device: torch.device):
    sys.path.insert(0, str(repo / "src"))
    from basicts.models.STID import STID, STIDConfig

    model = STID(
        STIDConfig(
            input_len=12,
            output_len=12,
            num_features=8600,
            if_spatial=True,
            if_time_in_day=True,
            if_day_in_week=True,
            num_time_in_day=96,
        )
    )
    model.load_state_dict(_state_dict(checkpoint), strict=True)
    model = model.to(device).eval()
    inputs = torch.zeros((batch_size, 12, 8600), device=device)
    timestamps = torch.zeros((batch_size, 12, 2), device=device)

    def forward():
        return model(inputs, timestamps)

    return model, forward, False


def _build_fast(repo: Path, checkpoint: Path, batch_size: int, device: torch.device):
    project = repo / "main-master"
    os.chdir(project)
    sys.path.insert(0, str(project))
    from FaST.arch import FaST

    model = FaST(
        num_nodes=8600,
        input_len=12,
        output_len=12,
        layers=3,
        num_experts=8,
        daily_steps=96,
        weekly_days=7,
        hidden_dim=64,
        num_agent=32,
    )
    model.load_state_dict(_state_dict(checkpoint), strict=True)
    model = model.to(device).eval()
    # FaST creates node_idx through EasyTorch rather than registering a buffer.
    model.node_idx = model.node_idx.to(device)
    inputs = torch.zeros((batch_size, 12, 8600, 3), device=device)
    inputs[..., 1] = 0.25
    inputs[..., 2] = 2.0 / 7.0

    def forward():
        return model(history_data=inputs)

    return model, forward, True


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("stid", "fast"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--precision", choices=("native", "fp32", "fp16_autocast"), default="native")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for matched latency profiling")
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError("warmup must be >=1 and repeats must be >=2")

    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    builder = _build_stid if args.model == "stid" else _build_fast
    model, forward, native_amp = builder(
        repo, checkpoint, args.batch_size, device
    )
    use_amp = native_amp if args.precision == "native" else args.precision == "fp16_autocast"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    autocast = (
        (lambda: torch.cuda.amp.autocast(dtype=torch.float16))
        if use_amp
        else nullcontext
    )

    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            with autocast():
                forward()
        torch.cuda.synchronize(device)

        latency_ms: list[float] = []
        for _ in range(args.repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter_ns()
            with autocast():
                forward()
            torch.cuda.synchronize(device)
            latency_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    result = {
        "schema": "dagnn-pistnet-matched-latency-v1",
        "model": "STID" if args.model == "stid" else "FaST",
        "seed": args.seed,
        "measured_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "precision": "fp16_autocast" if use_amp else "fp32",
        "scope": "model_forward_device_resident_inputs",
        "input_shape": [args.batch_size, 12, 8600, 3 if args.model == "fast" else 1],
        "batch_size": args.batch_size,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.repeats,
        "parameter_count": parameter_count,
        "latency_ms": {
            "mean_per_batch": float(statistics.fmean(latency_ms)),
            "median_per_batch": float(statistics.median(latency_ms)),
            "std_per_batch": float(statistics.stdev(latency_ms)),
            "p05_per_batch": _percentile(latency_ms, 5),
            "p95_per_batch": _percentile(latency_ms, 95),
            "median_per_sample": float(statistics.median(latency_ms) / args.batch_size),
        },
        "peak_allocated_memory_mb": float(
            torch.cuda.max_memory_allocated(device) / (1024**2)
        ),
        "samples_per_second": float(
            args.batch_size / (statistics.median(latency_ms) / 1000.0)
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
