"""Resumable, paired PEMS08 inference-only sensor outage stress test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def masks_for_rate(windows: int, nodes: int, rate: float, seed: int) -> np.ndarray:
    mask = np.zeros((windows, nodes), dtype=np.bool_)
    selected_count = round(rate * nodes)
    if selected_count:
        rng = np.random.default_rng(seed + round(rate * 1000))
        for row in mask:
            row[rng.choice(nodes, selected_count, replace=False)] = True
    return mask


def per_window_mae(error: np.ndarray, validity: np.ndarray) -> np.ndarray:
    counts = validity.sum(axis=1)
    return np.divide(
        np.where(validity, error, 0.0).sum(axis=1),
        counts,
        out=np.full(len(error), np.nan, dtype=np.float64),
        where=counts > 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-runs", type=int, default=0)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_evaluation":
        raise ValueError("Protocol must be frozen before evaluation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")

    source_dir = Path(protocol["source_checkout"]).resolve()
    sys.path.insert(0, str(source_dir))
    from dagnn_pistnet_v3.config import V3ExperimentConfig
    from dagnn_pistnet_v3.data import prepare_v3_data
    from dagnn_pistnet_v3.training import build_v3_runtime

    root = Path(protocol["checkpoints_root"])
    output_root = args.output_root.resolve()
    first = root / "full" / "seed_13" / "resolved_config.yaml"
    first_data = prepare_v3_data(V3ExperimentConfig.from_yaml(first))
    if first_data.series.num_nodes != 170:
        raise ValueError("Expected exactly 170 PEMS08 sensors")
    test_windows = len(first_data.test_dataset)
    target_indices = (
        np.arange(first_data.test_dataset.first_target,
                  first_data.test_dataset.last_target + 1)
        + 11
    )
    raw_targets = np.asarray(
        first_data.series.values[target_indices, :, 0], dtype=np.float32
    )
    target_path = output_root / "target_h12_flow_original.npy"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        if not np.allclose(np.load(target_path), raw_targets, atol=1e-5):
            raise ValueError("Existing test target differs")
    else:
        np.save(target_path, raw_targets)
    del first_data
    masks: dict[float, np.ndarray] = {}
    for rate in protocol["mask_rates"]:
        rate = float(rate)
        # The test split uses the same window anchors for all archived models.
        masks[rate] = masks_for_rate(test_windows, 170, rate, int(protocol["mask_seed"]))
        mask_path = output_root / f"mask_rate_{int(rate * 100):02d}.npy"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if mask_path.exists():
            if not np.array_equal(np.load(mask_path), masks[rate]):
                raise ValueError(f"Existing mask differs: {mask_path}")
        else:
            np.save(mask_path, masks[rate])

    state_path = output_root / "state.json"
    run_count = len(protocol["variants"]) * len(protocol["seeds"]) * len(masks)
    completed = sum(
        (output_root / variant / f"seed_{seed}" / f"rate_{int(rate * 100):02d}" / "result.json").is_file()
        for variant in protocol["variants"]
        for seed in protocol["seeds"]
        for rate in masks
    )
    new_runs = 0
    write_json(state_path, {
        "status": "running", "completed_runs": completed, "planned_runs": run_count,
        "protocol_sha256": sha256(protocol_path), "updated_utc": datetime.now(timezone.utc).isoformat(),
    })

    for variant in protocol["variants"]:
        for seed in protocol["seeds"]:
            run_dir = root / variant / f"seed_{seed}"
            config_path = run_dir / "resolved_config.yaml"
            checkpoint = run_dir / "best_model_v3.pt"
            config = V3ExperimentConfig.from_yaml(config_path)
            if config.training.seed != seed:
                raise ValueError(f"Seed mismatch: {run_dir}")
            data = prepare_v3_data(config)
            if len(data.test_dataset) != test_windows or data.series.num_nodes != 170:
                raise ValueError(f"Test alignment mismatch: {run_dir}")
            model, trainer, loaders = build_v3_runtime(config, data)
            saved = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
            if int(saved.get("architecture_version", -1)) != model.architecture_version:
                raise ValueError(f"Architecture mismatch: {checkpoint}")
            model.load_state_dict(saved["model"], strict=True)
            model.eval()
            if len(trainer.graph_slices) != 1:
                raise ValueError("PEMS08 must have one complete graph slice")
            graph_slice = trainer.graph_slices[0]
            if graph_slice.nodes.numel() != 170:
                raise ValueError("Graph slice does not cover all sensors")
            for rate, mask in masks.items():
                target_dir = output_root / variant / f"seed_{seed}" / f"rate_{int(rate * 100):02d}"
                result_path = target_dir / "result.json"
                if result_path.is_file():
                    print(f"SKIP {variant} seed={seed} rate={rate}", flush=True)
                    continue
                predictions = np.empty(mask.shape, dtype=np.float32)
                truths = np.empty(mask.shape, dtype=np.float32)
                offset = 0
                with torch.inference_mode():
                    for batch in loaders[2]:
                        selected = trainer._select_batch(batch, graph_slice)
                        batch_size = selected[0].shape[0]
                        selected_mask = torch.as_tensor(
                            mask[offset:offset + batch_size], device=trainer.device
                        )
                        x = selected[0].masked_fill(selected_mask[:, None, :, None], 0.0)
                        selected = (x, *selected[1:])
                        distribution = trainer._forecast(
                            selected, graph_slice, config.data.forecast_steps
                        )
                        predicted = data.scaler.inverse_tensor(distribution.mean)[..., 0]
                        truth = data.scaler.inverse_tensor(selected[1])[..., 0]
                        predictions[offset:offset + batch_size] = predicted[:, 11].float().cpu().numpy()
                        truths[offset:offset + batch_size] = truth[:, 11].float().cpu().numpy()
                        offset += batch_size
                if offset != len(mask):
                    raise ValueError(f"Test window count mismatch: {offset} != {len(mask)}")
                valid = np.isfinite(truths) & (np.abs(truths) > 1e-5)
                error = np.abs(predictions.astype(np.float64) - truths.astype(np.float64))
                all_mae = per_window_mae(error, valid)
                masked_mae = per_window_mae(error, valid & mask)
                unmasked_mae = per_window_mae(error, valid & ~mask)
                target_dir.mkdir(parents=True, exist_ok=True)
                np.save(target_dir / "prediction_h12_flow_original.npy", predictions)
                np.savez_compressed(
                    target_dir / "window_errors.npz",
                    all_sensor_mae=all_mae,
                    masked_sensor_mae=masked_mae,
                    unmasked_sensor_mae=unmasked_mae,
                )
                write_json(result_path, {
                    "schema": "dagnn-pistnet-v3-missing-sensor-result-v1",
                    "variant": variant, "seed": seed, "mask_rate": rate,
                    "protocol_sha256": sha256(protocol_path),
                    "source_commit": protocol["study_source_commit"],
                    "config_sha256": sha256(config_path),
                    "checkpoint_sha256": sha256(checkpoint),
                    "checkpoint_epoch": int(saved["epoch"]),
                    "mask_sha256": sha256(output_root / f"mask_rate_{int(rate * 100):02d}.npy"),
                    "test_windows": len(mask), "sensor_count": 170,
                    "masked_sensors_per_window": int(mask[0].sum()),
                    "mae_all_sensors": float(np.nanmean(all_mae)),
                    "mae_masked_sensors": None if rate == 0 else float(np.nanmean(masked_mae)),
                    "mae_unmasked_sensors": float(np.nanmean(unmasked_mae)),
                    "python": platform.python_version(), "torch": torch.__version__,
                    "precision": "fp32_as_original_test_evaluation",
                    "prediction_path": "prediction_h12_flow_original.npy",
                    "window_errors_path": "window_errors.npz",
                })
                completed += 1
                new_runs += 1
                write_json(state_path, {
                    "status": "running", "completed_runs": completed, "planned_runs": run_count,
                    "last_completed": f"{variant}/seed_{seed}/rate_{int(rate * 100):02d}",
                    "protocol_sha256": sha256(protocol_path),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                })
                print(f"DONE {completed}/{run_count} {variant} seed={seed} rate={rate}", flush=True)
                if args.max_new_runs and new_runs >= args.max_new_runs:
                    write_json(state_path, {
                        "status": "paused_after_bounded_run",
                        "completed_runs": completed, "planned_runs": run_count,
                        "protocol_sha256": sha256(protocol_path),
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    return
            del model, trainer, loaders, data
            gc.collect()
            torch.cuda.empty_cache()
    write_json(state_path, {
        "status": "completed", "completed_runs": completed, "planned_runs": run_count,
        "protocol_sha256": sha256(protocol_path), "updated_utc": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
