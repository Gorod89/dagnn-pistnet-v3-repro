"""Pinned BasicTS/STID runner for canonical native split files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PINNED_COMMIT = "c2bb6e31e591167e84459775a21a62e70a5893ce"


def _find_raw_prediction(output_dir: Path) -> Path:
    candidates = sorted(
        {
            *output_dir.glob("test_results/prediction.npy"),
            *output_dir.glob("*/test_results/prediction.npy"),
        }
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one BasicTS prediction memmap under {output_dir}, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _install_prediction_only_saver(runner_type: type) -> None:
    """Avoid BasicTS's multi-gigabyte duplicate input/target test dumps."""

    def _save_prediction_only(self, batch_idx, batch_data):
        prediction = batch_data["prediction"].detach().cpu().numpy()
        total_samples = len(self.test_data_loader.dataset)
        save_dir = Path(self.ckpt_save_dir) / "test_results"
        save_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = save_dir / "prediction.npy"

        if batch_idx == 0:
            self._prediction_memmap = np.memmap(
                prediction_path,
                dtype=prediction.dtype,
                mode="w+",
                shape=(total_samples, *prediction.shape[1:]),
            )

        start = batch_idx * prediction.shape[0]
        end = start + prediction.shape[0]
        self._prediction_memmap[start:end] = prediction
        if batch_idx == (total_samples // prediction.shape[0]):
            self._prediction_memmap.flush()

    runner_type._save_results = _save_prediction_only


def _finalize_prediction(
    output_dir: Path,
    test: np.ndarray,
    history: int,
    forecast: int,
    seed: int,
    wall_seconds: float | None,
    peak_accelerator_memory_mb: float | None,
) -> None:
    raw_path = _find_raw_prediction(output_dir)
    samples = len(test) - history - forecast + 1
    shape = (samples, forecast, int(test.shape[1]))
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Prediction size mismatch: got {actual_bytes} bytes, expected "
            f"{expected_bytes} for shape {shape}"
        )

    canonical_path = output_dir / "prediction.canonical.raw.f32"
    if canonical_path.exists() and canonical_path.resolve() != raw_path.resolve():
        raise FileExistsError(canonical_path)
    if raw_path.resolve() != canonical_path.resolve():
        raw_path.replace(canonical_path)

    metadata = {
        "path": canonical_path.name,
        "dtype": "float32",
        "shape": list(shape),
        "layout": "SHN",
        "value_space": "original",
        "format": "raw_memmap",
    }
    (output_dir / "prediction.canonical.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "model": "STID",
                "upstream_commit": PINNED_COMMIT,
                "seed": seed,
                "wall_seconds": wall_seconds,
                "peak_accelerator_memory_mb": peak_accelerator_memory_mb,
                "prediction": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--forecast", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != PINNED_COMMIT:
        raise RuntimeError(f"BasicTS checkout is {actual}, expected {PINNED_COMMIT}")
    sys.path.insert(0, str(repo / "src"))
    from basicts.configs import BasicTSForecastingConfig
    from basicts.data import BasicTSForecastingDataset
    from basicts.launcher import BasicTSLauncher
    from basicts.models.STID import STID, STIDConfig
    from basicts.runners.basicts_runner import BasicTSRunner

    test = np.load(args.data_dir / "test_data.npy", mmap_mode="r")
    manifest = json.loads(
        (args.data_dir / "native_manifest.json").read_text(encoding="utf-8")
    )
    if test.ndim != 2:
        raise ValueError("STID native data must have shape [time,node]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        _finalize_prediction(
            args.output_dir,
            test,
            args.history,
            args.forecast,
            args.seed,
            wall_seconds=None,
            peak_accelerator_memory_mb=None,
        )
        return

    _install_prediction_only_saver(BasicTSRunner)
    cfg = BasicTSForecastingConfig(
        model=STID,
        dataset_name="CANONICAL",
        model_config=STIDConfig(
            input_len=args.history,
            output_len=args.forecast,
            num_features=int(test.shape[1]),
            if_spatial=True,
            if_time_in_day=True,
            if_day_in_week=True,
            num_time_in_day=int(manifest["steps_per_day"]),
        ),
        dataset_type=BasicTSForecastingDataset,
        dataset_params={
            "input_len": args.history,
            "output_len": args.forecast,
            "use_timestamps": True,
            "data_file_path": str(args.data_dir.resolve()),
            "memmap": True,
        },
        gpus=args.gpu,
        seed=args.seed,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        input_len=args.history,
        output_len=args.forecast,
        use_timestamps=True,
        rescale=True,
        save_results=True,
        ckpt_save_dir=str(args.output_dir.resolve()),
    )
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    BasicTSLauncher.launch_training(cfg)
    _finalize_prediction(
        args.output_dir,
        test,
        args.history,
        args.forecast,
        args.seed,
        wall_seconds=time.perf_counter() - start,
        peak_accelerator_memory_mb=(
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available()
            else None
        ),
    )


if __name__ == "__main__":
    main()
