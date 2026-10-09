"""Time a complete LargeST forward assembled from the pinned v3 graph partitions.

Inputs and graph slices are prepared before timing. Each measured iteration
forwards every partition for the same device-resident test batch; no loader,
metric, or disk work is included in the timed region.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--compat-src", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError("warmup >= 1 and repeats >= 2 are required")

    # The current workspace adapter owns a writable pandas array before the
    # pinned v3 source applies its in-place missing-value normalization. Its
    # only local difference is that compatibility copy, not model logic.
    sys.path.insert(0, str(args.source_dir.resolve()))
    sys.path.insert(0, str(args.compat_src.resolve()))
    from dagnn_pistnet_v3.config import V3ExperimentConfig
    from dagnn_pistnet_v3.data import prepare_v3_data
    from dagnn_pistnet_v3.training import build_v3_runtime

    config = V3ExperimentConfig.from_yaml(args.config)
    data = prepare_v3_data(config)
    model, trainer, loaders = build_v3_runtime(config, data)
    if trainer.device.type != "cuda":
        raise RuntimeError("Configuration did not select CUDA")
    saved = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    if int(saved.get("architecture_version", -1)) != model.architecture_version:
        raise ValueError("Checkpoint architecture does not match model")
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    batch = next(iter(loaders[2]))
    selected_slices = [
        (trainer._select_batch(batch, graph_slice), graph_slice)
        for graph_slice in trainer.graph_slices
    ]
    batch_size = int(selected_slices[0][0][0].shape[0])
    node_counts = [int(graph_slice.nodes.numel()) for _, graph_slice in selected_slices]
    if sum(node_counts) != data.series.num_nodes:
        raise ValueError("Graph partitions do not cover every sensor exactly once")
    use_amp = trainer.grad_scaler.is_enabled()

    def forward_full_graph() -> None:
        with torch.autocast(device_type="cuda", enabled=use_amp):
            for selected, graph_slice in selected_slices:
                trainer._forecast(selected, graph_slice, config.data.forecast_steps)

    torch.cuda.reset_peak_memory_stats(trainer.device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            forward_full_graph()
        torch.cuda.synchronize(trainer.device)
        elapsed_ms = []
        for _ in range(args.repeats):
            torch.cuda.synchronize(trainer.device)
            start = time.perf_counter_ns()
            forward_full_graph()
            torch.cuda.synchronize(trainer.device)
            elapsed_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    result = {
        "schema": "dagnn-pistnet-partitioned-latency-v1",
        "model": "DAGNN-PISTNet-v3",
        "seed": config.training.seed,
        "measured_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.source_dir.resolve()),
        "compat_data_adapter_src": str(args.compat_src.resolve()),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(saved["epoch"]),
        "device": torch.cuda.get_device_name(trainer.device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "precision": "fp16_autocast" if use_amp else "fp32",
        "scope": "full_graph_partitioned_model_forward_device_resident_inputs",
        "batch_size": batch_size,
        "history_steps": config.data.history_steps,
        "forecast_steps": config.data.forecast_steps,
        "sensor_count": data.series.num_nodes,
        "partition_count": len(node_counts),
        "partition_node_counts": node_counts,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.repeats,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "latency_ms": {
            "mean_per_full_graph_batch": statistics.fmean(elapsed_ms),
            "median_per_full_graph_batch": statistics.median(elapsed_ms),
            "std_per_full_graph_batch": statistics.stdev(elapsed_ms),
            "p05_per_full_graph_batch": float(np.percentile(elapsed_ms, 5)),
            "p95_per_full_graph_batch": float(np.percentile(elapsed_ms, 95)),
            "median_per_full_graph_sample": statistics.median(elapsed_ms) / batch_size,
        },
        "peak_allocated_memory_mb": torch.cuda.max_memory_allocated(trainer.device)
        / 1024**2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
