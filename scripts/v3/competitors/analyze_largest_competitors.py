"""Build matched LargeST competitor metrics, paired inference, and an interim report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dagnn_pistnet.training import build_graph_partitions
from dagnn_pistnet_v3.config import V3ExperimentConfig
from dagnn_pistnet_v3.data import prepare_v3_data
from dagnn_pistnet_v3.publication_analysis import (
    load_common_evaluator_runs,
    paired_competitor_block_rows,
    paired_competitor_rows,
    write_paired_comparisons,
)

DATASET = "LargeST-CA-2019-15min"
OURS = "DAGNN-PISTNet-v3"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_our_temporal_window_errors(
    root: Path,
    study_root: Path,
    output: Path,
) -> dict[int, Path]:
    """Collapse historical node-partition errors into one value per time window.

    The original v3 evaluator appended one per-window MAE for each of the 17
    node partitions.  Pairing those 119,068 values directly with a competitor's
    7,004 temporal windows is invalid.  The original target series supplies the
    exact non-missing count for every partition, so the global temporal-window
    MAE can be reconstructed losslessly without rerunning model inference.
    """

    config = V3ExperimentConfig.from_yaml(
        root / "configs/v3/large_st_ca_2019_15min_official.yaml"
    )
    data = prepare_v3_data(config)
    edge_index = torch.as_tensor(data.graph.edge_index, dtype=torch.long)
    test = data.test_dataset
    batch_size = config.training.evaluation_batch_size
    if len(test) % batch_size:
        raise ValueError("reconstruction currently requires full evaluation batches")
    batch_count = len(test) // batch_size
    reconstructed: dict[int, Path] = {}
    horizon_by_key = {"h1": 0, "h3": 2, "h6": 5, "h12": 11}

    for seed in (13, 42, 97):
        partitions = build_graph_partitions(
            data.graph.num_nodes,
            edge_index,
            config.training.node_batch_size or data.graph.num_nodes,
            seed,
        )
        source = study_root / f"seed_{seed}/window_errors.npz"
        result = read_json(study_root / f"seed_{seed}/result.json")
        arrays: dict[str, np.ndarray] = {}
        with np.load(source) as archive:
            for key, horizon in horizon_by_key.items():
                partition_errors = np.asarray(archive[key], dtype=np.float64)
                expected = len(test) * len(partitions)
                if partition_errors.size != expected:
                    raise ValueError(
                        f"{source}:{key} has {partition_errors.size} values, expected {expected}"
                    )
                partition_errors = (
                    partition_errors.reshape(batch_count, len(partitions), batch_size)
                    .transpose(0, 2, 1)
                    .reshape(len(test), len(partitions))
                )
                targets = data.series.values[
                    test.first_target + horizon : test.first_target + horizon + len(test),
                    :,
                    0,
                ]
                numerator = np.zeros(len(test), dtype=np.float64)
                denominator = np.zeros(len(test), dtype=np.int64)
                for index, nodes in enumerate(partitions):
                    selected = targets[:, nodes.numpy()]
                    valid_count = np.count_nonzero(
                        np.isfinite(selected) & (np.abs(selected) > 1e-5), axis=1
                    )
                    numerator += partition_errors[:, index] * valid_count
                    denominator += valid_count
                temporal = numerator / np.maximum(denominator, 1)
                metric_key = f"{(horizon + 1) * 15}min"
                reconstructed_mae = float(numerator.sum() / denominator.sum())
                reported_mae = float(result["test_metrics"][metric_key]["flow"]["mae"])
                if not np.isclose(reconstructed_mae, reported_mae, rtol=0.0, atol=2e-5):
                    raise ValueError(
                        f"reconstructed MAE {reconstructed_mae} disagrees with "
                        f"reported {reported_mae} for seed={seed} {metric_key}"
                    )
                arrays[key] = temporal.astype(np.float32)
        target = output / "reconstructed_ours" / f"seed_{seed}" / "window_errors.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        reconstructed[seed] = target
    return reconstructed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=12)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = read_json(
        root / "artifacts/v3/protocols/large_st_ca_2019_15min_official.json"
    )
    competitor_root = (
        root / "artifacts/v3/competitor_runs/large_st_ca_2019_15min_official"
    )
    study_root = (
        root
        / "artifacts/v3/studies/large_st_ca_2019_15min_official"
        / "large_st_ca_2019_15min/full"
    )
    output = root / "artifacts/v3/competitor_analysis/large_st_ca_2019_15min_official"
    output.mkdir(parents=True, exist_ok=True)

    competitor_runs = load_common_evaluator_runs(competitor_root)
    reconstructed_archives = reconstruct_our_temporal_window_errors(
        root, study_root, output
    )
    our_runs: list[dict[str, Any]] = []
    for seed in (13, 42, 97):
        path = study_root / f"seed_{seed}/result.json"
        result = read_json(path)
        archive = study_root / f"seed_{seed}/window_errors.npz"
        our_runs.append(
            {
                **result,
                "model": OURS,
                "dataset": DATASET,
                "seed": seed,
                "protocol_sha256": protocol["protocol_sha256"],
                "result_path": str(path),
                "archive_path": str(reconstructed_archives[seed]),
            }
        )
    all_runs = [*our_runs, *competitor_runs]

    primary = [run for run in all_runs if run["model"] in {OURS, "FaST"}]
    primary_block = paired_competitor_block_rows(
        primary,
        "FaST",
        bootstrap_samples=args.bootstrap_samples,
        block_length=args.block_length,
    )
    write_paired_comparisons(primary_block, output / "paired_vs_fast_block_holm.csv")
    primary_iid = paired_competitor_rows(
        primary, "FaST", bootstrap_samples=args.bootstrap_samples
    )
    write_paired_comparisons(primary_iid, output / "paired_vs_fast_iid_sensitivity.csv")

    secondary = [run for run in all_runs if run["model"] in {OURS, "STID"}]
    secondary_block = paired_competitor_block_rows(
        secondary,
        "STID",
        bootstrap_samples=args.bootstrap_samples,
        block_length=args.block_length,
    )
    write_paired_comparisons(secondary_block, output / "paired_vs_stid_block_holm.csv")

    metric_rows: list[dict[str, Any]] = []
    for run in all_runs:
        for horizon, by_feature in sorted(run["test_metrics"].items()):
            metrics = by_feature["flow"]
            metric_rows.append(
                {
                    "model": run["model"],
                    "seed": run["seed"],
                    "horizon": horizon,
                    **{
                        field: metrics.get(field)
                        for field in ("mae", "rmse", "wape", "smape", "count")
                    },
                }
            )
    write_csv(output / "metrics_by_run.csv", metric_rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["model"]), str(row["horizon"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (model, horizon), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "model": model,
            "horizon": horizon,
            "seeds": ";".join(str(row["seed"]) for row in rows),
            "runs": len(rows),
        }
        for field in ("mae", "rmse", "wape", "smape"):
            values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            summary[f"{field}_mean"] = float(values.mean())
            summary[f"{field}_std"] = float(values.std(ddof=1))
        summary_rows.append(summary)
    write_csv(output / "metrics_summary.csv", summary_rows)

    efficiency_rows: list[dict[str, Any]] = []
    for run in our_runs:
        sensor_windows_per_second = float(
            run["inference_sensor_windows_per_second"]
        )
        efficiency_rows.append(
            {
                "model": OURS,
                "seed": run["seed"],
                "parameter_count": run["parameter_count"],
                "peak_accelerator_memory_mb": run["peak_accelerator_memory_mb"],
                "training_wall_seconds": run["training_wall_seconds"],
                "inference_latency_ms": run["inference_latency_ms"],
                "inference_latency_ms_per_sample": (
                    1000.0 * 8600 / sensor_windows_per_second
                ),
                "inference_sensor_windows_per_second": sensor_windows_per_second,
                "latency_batch_size": 2,
                "latency_precision": "fp16_autocast",
                "latency_scope": "partitioned device-resident model forward",
                "latency_status": "dedicated measurement; normalized to one 8600-node sample",
            }
        )
    parameter_counts = {"STID": 313484, "FaST": 1207564}
    for run in competitor_runs:
        run_file = Path(run["result_path"]).parent / "run.json"
        native = read_json(run_file)
        latency_file = Path(run["result_path"]).parent / "latency.json"
        if not latency_file.exists():
            raise FileNotFoundError(
                f"Missing dedicated latency measurement for {run['model']} "
                f"seed {run['seed']}: {latency_file}"
            )
        latency = read_json(latency_file)
        efficiency_rows.append(
            {
                "model": run["model"],
                "seed": run["seed"],
                "parameter_count": parameter_counts[str(run["model"])],
                "peak_accelerator_memory_mb": native.get(
                    "peak_accelerator_memory_mb"
                ),
                "training_wall_seconds": native.get("wall_seconds"),
                "inference_latency_ms": latency["latency_ms"]["median_per_batch"],
                "inference_latency_ms_per_sample": latency["latency_ms"][
                    "median_per_sample"
                ],
                "inference_sensor_windows_per_second": (
                    float(latency["samples_per_second"]) * 8600
                ),
                "latency_batch_size": latency["batch_size"],
                "latency_precision": latency["precision"],
                "latency_scope": latency["scope"],
                "latency_status": "dedicated matched-shape measurement",
            }
        )
    write_csv(output / "efficiency_by_run.csv", efficiency_rows)

    efficiency_summary: list[dict[str, Any]] = []
    for model in (OURS, "STID", "FaST"):
        rows = [row for row in efficiency_rows if row["model"] == model]
        walls = [
            float(row["training_wall_seconds"])
            for row in rows
            if row["training_wall_seconds"] is not None
        ]
        sample_latency = np.asarray(
            [float(row["inference_latency_ms_per_sample"]) for row in rows],
            dtype=np.float64,
        )
        throughput = np.asarray(
            [float(row["inference_sensor_windows_per_second"]) for row in rows],
            dtype=np.float64,
        )
        efficiency_summary.append(
            {
                "model": model,
                "parameter_count": int(rows[0]["parameter_count"]),
                "peak_accelerator_memory_mb_max": max(
                    float(row["peak_accelerator_memory_mb"])
                    for row in rows
                    if row["peak_accelerator_memory_mb"] is not None
                ),
                "training_wall_hours_mean": (
                    float(np.mean(walls)) / 3600.0 if walls else None
                ),
                "training_wall_runs": len(walls),
                "inference_latency_ms_per_sample_mean": float(sample_latency.mean()),
                "inference_latency_ms_per_sample_std": float(
                    sample_latency.std(ddof=1)
                ),
                "inference_sensor_windows_per_second_mean": float(throughput.mean()),
            }
        )
    write_csv(output / "efficiency_summary.csv", efficiency_summary)

    fast_summary = {
        row["horizon"]: row for row in summary_rows if row["model"] == "FaST"
    }
    our_summary = {
        row["horizon"]: row for row in summary_rows if row["model"] == OURS
    }
    lines = [
        "# LargeST-CA 2019 / 15-minute matched comparison",
        "",
        "All metrics use the same official sliding-window split, sensor order, "
        "test windows, and common evaluator. Values are mean ± sample SD over "
        "seeds 13, 42, and 97.",
        "",
        "| Horizon | DAGNN-PISTNet-v3 MAE | FaST MAE | Difference |",
        "|---:|---:|---:|---:|",
    ]
    for horizon in ("15min", "45min", "90min", "180min"):
        ours = our_summary[horizon]
        fast = fast_summary[horizon]
        difference = float(ours["mae_mean"]) - float(fast["mae_mean"])
        lines.append(
            f"| {horizon} | {ours['mae_mean']:.4f} ± {ours['mae_std']:.4f} | "
            f"{fast['mae_mean']:.4f} ± {fast['mae_std']:.4f} | {difference:+.4f} |"
        )
    horizon_labels = {"h1": "15min", "h3": "45min", "h6": "90min", "h12": "180min"}
    lines.extend(
        [
            "",
            "## Confirmatory paired comparison against FaST",
            "",
            "Positive ΔMAE means FaST is better.",
            "",
            "| Horizon | Paired ΔMAE | 95% block-bootstrap CI | Holm p |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(primary_block, key=lambda value: int(str(value["horizon"])[1:])):
        lines.append(
            f"| {horizon_labels[str(row['horizon'])]} | "
            f"{float(row['mean_mae_difference']):+.4f} | "
            f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] | "
            f"{float(row['p_holm']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Secondary paired comparison against STID",
            "",
            "Negative ΔMAE means DAGNN-PISTNet-v3 is better.",
            "",
            "| Horizon | Paired ΔMAE | 95% block-bootstrap CI | Holm p |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(secondary_block, key=lambda value: int(str(value["horizon"])[1:])):
        lines.append(
            f"| {horizon_labels[str(row['horizon'])]} | "
            f"{float(row['mean_mae_difference']):+.4f} | "
            f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] | "
            f"{float(row['p_holm']):.6f} |"
        )
    lines.extend(
        [
            "",
            "Confirmatory paired inference uses circular moving-block bootstrap "
            f"with block_length={args.block_length}, {args.bootstrap_samples} resamples, "
            "block sign-flip p-values, and Holm correction across the four "
            "predeclared horizons. IID-window results are retained only as a "
            "sensitivity analysis.",
            "",
            "STAEformer was infeasible on 8,600 nodes: eager attention exhausted "
            "48 GiB, while exact Flash Attention fit but implied more than 94 hours "
            "per epoch. PDFormer preprocessing was infeasible because the pinned "
            "official path requires cubic Floyd-Warshall plus all-pairs DTW.",
            "",
            "Physics/event limitation: LargeST is flow-only and does not provide "
            "independently measured density, segment/boundary flow, or event labels. "
            "Therefore these results do not validate physics/conservation correctness "
            "or causal effectiveness of event/jump modules.",
            "",
            "## Efficiency",
            "",
            "| Model | Parameters | Peak training VRAM (MiB) | Mean train wall time (h) | "
            "Inference latency (ms/sample) | Sensor-windows/s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in efficiency_summary:
        wall = row["training_wall_hours_mean"]
        wall_text = f"{float(wall):.2f}" if wall is not None else "n/a"
        lines.append(
            f"| {row['model']} | {int(row['parameter_count']):,} | "
            f"{float(row['peak_accelerator_memory_mb_max']):.1f} | {wall_text} "
            f"({int(row['training_wall_runs'])} runs) | "
            f"{float(row['inference_latency_ms_per_sample_mean']):.3f} ± "
            f"{float(row['inference_latency_ms_per_sample_std']):.3f} | "
            f"{float(row['inference_sensor_windows_per_second_mean']):,.0f} |"
        )
    lines.extend(
        [
            "",
            "Latency is synchronized GPU model-forward time with device-resident "
            "inputs. Competitors use the same batch shape (64 × 12 × 8,600); "
            "DAGNN-PISTNet-v3 uses its required 512-node graph partitions and "
            "evaluation batch size 2, then is normalized to one complete 8,600-node "
            "sample. It is not end-to-end data-loader latency. STID uses FP32; FaST "
            "and DAGNN-PISTNet-v3 use their configured FP16 autocast paths.",
            "",
            "Operational caveat: competitor latency was measured at explicit user "
            "direction while an unrelated long-running process (PID 8420) remained "
            "alive and intermittently used the same GPU. Re-run these short profiles "
            "on an exclusive GPU before using latency values in a final submission.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(all_runs)}, indent=2))


if __name__ == "__main__":
    main()
