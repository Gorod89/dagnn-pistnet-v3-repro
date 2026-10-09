"""Pinned FaST runner for a canonical 12-to-12 LargeST native dataset."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PINNED_COMMIT = "c94a6273df801a0092f92d97a0c06a110d6bd90c"


def _install_prediction_stream(runner_type: type) -> None:
    """Persist test predictions without materializing inputs/targets in RAM."""

    def _test_iter(self, iter_index, data):
        forward_return = self.forward(
            data=data, epoch=None, iter_num=iter_index, train=False
        )
        loss = self.metric_forward(self.loss, forward_return)
        self.update_epoch_meter("test/loss", loss.item())
        for metric_name, metric_func in self.metrics.items():
            value_item, num_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(
                f"test/{metric_name}", value_item.item(), num_item.item()
            )

        prediction = (
            forward_return["prediction"].detach().cpu().numpy().astype(np.float32)
        )
        if not hasattr(self, "_canonical_prediction_memmap"):
            total_samples = len(self.test_data_loader.dataset)
            shape = (total_samples, *prediction.shape[1:])
            path = Path(self.ckpt_save_dir) / "prediction.raw.f32"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._canonical_prediction_path = path
            self._canonical_prediction_shape = shape
            self._canonical_prediction_offset = 0
            self._canonical_prediction_memmap = np.memmap(
                path, dtype=np.float32, mode="w+", shape=shape
            )

        start = self._canonical_prediction_offset
        end = start + prediction.shape[0]
        self._canonical_prediction_memmap[start:end] = prediction
        self._canonical_prediction_offset = end
        if end == self._canonical_prediction_shape[0]:
            self._canonical_prediction_memmap.flush()
            sidecar = self._canonical_prediction_path.with_suffix(".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "path": self._canonical_prediction_path.name,
                        "dtype": "float32",
                        "shape": list(self._canonical_prediction_shape),
                        "layout": "SHNF",
                        "value_space": "original",
                        "format": "raw_memmap",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    runner_type.test_iters_4OOM = _test_iter


def _finalize_prediction(
    output_dir: Path,
    seed: int,
    wall_seconds: float | None,
    peak_accelerator_memory_mb: float | None,
) -> None:
    sidecars = sorted(output_dir.glob("*/prediction.raw.json"))
    if len(sidecars) != 1:
        raise RuntimeError(
            f"Expected exactly one streamed FaST prediction under {output_dir}, "
            f"found {len(sidecars)}: {sidecars}"
        )
    metadata = json.loads(sidecars[0].read_text(encoding="utf-8"))
    raw_path = sidecars[0].with_name(metadata["path"])
    expected_bytes = (
        int(np.prod(metadata["shape"], dtype=np.int64))
        * np.dtype(metadata["dtype"]).itemsize
    )
    if raw_path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Incomplete FaST prediction: {raw_path.stat().st_size} bytes, "
            f"expected {expected_bytes}"
        )
    metadata["path"] = raw_path.relative_to(output_dir).as_posix()
    (output_dir / "prediction.canonical.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "model": "FaST",
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
    parser.add_argument("--dataset-name", default="CANONICAL_CA_2019")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != PINNED_COMMIT:
        raise RuntimeError(f"FaST checkout is {actual}, expected {PINNED_COMMIT}")
    project = repo / "main-master"
    expected_data = (project / "datasets" / args.dataset_name).resolve()
    if args.data_dir.resolve() != expected_data:
        raise ValueError(
            "FaST uses a fixed relative dataset root; prepare native data at "
            f"{expected_data}"
        )
    description = json.loads((expected_data / "desc.json").read_text(encoding="utf-8"))
    if description["num_nodes"] != description["shape"][1]:
        raise ValueError("FaST dataset description is inconsistent")
    # Resolve the durable artifact path before changing into the upstream
    # project directory (its configuration uses paths relative to that root).
    output_dir = args.output_dir.resolve()
    if args.finalize_only:
        _finalize_prediction(output_dir, args.seed, None, None)
        return

    # The official configuration resolves dataset metadata relative to the
    # FaST project root during module import.
    os.chdir(project)
    sys.path.insert(0, str(project))
    # The pinned FaST checkout imports GradScaler from torch.amp before its
    # existing torch.cuda.amp fallback.  PyTorch 2.2.1 (the upstream-pinned
    # version) does not yet expose GradScaler at torch.amp, so provide the
    # equivalent CUDA AMP symbol without changing the locked torch version.
    if not hasattr(torch.amp, "GradScaler"):
        torch.amp.GradScaler = torch.cuda.amp.GradScaler
    import basicts
    from FaST.CA_96_96 import CFG as upstream_cfg
    from basicts.runners import SimpleTimeSeriesForecastingRunner

    _install_prediction_stream(SimpleTimeSeriesForecastingRunner)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg = copy.deepcopy(upstream_cfg)
    cfg.SEED = args.seed
    cfg.DATASET.NAME = args.dataset_name
    cfg.DATASET.PARAM.dataset_name = args.dataset_name
    cfg.DATASET.PARAM.input_len = 12
    cfg.DATASET.PARAM.output_len = 12
    cfg.MODEL.PARAM.num_nodes = int(description["num_nodes"])
    cfg.MODEL.PARAM.input_len = 12
    cfg.MODEL.PARAM.output_len = 12
    frequency = int(description["frequency (minutes)"])
    if 1440 % frequency:
        raise ValueError("FaST daily embedding requires frequency to divide 1440 minutes")
    cfg.MODEL.PARAM.daily_steps = 1440 // frequency
    # EasyTorch's asynchronous CUDA prefetcher can strand the training loop if
    # its pin-memory thread runs out of accelerator memory on the 8,600-node
    # canonical dataset.  Keep the official model, batch size, optimizer, and
    # schedule unchanged, but load batches synchronously on the host so the
    # ordinary training loop can transfer them to CUDA itself.
    for phase in (cfg.TRAIN, cfg.VAL, cfg.TEST):
        phase.DATA.PREFETCH = False
        phase.DATA.PIN_MEMORY = False
        phase.DATA.NUM_WORKERS = 0
    cfg.TRAIN.CKPT_SAVE_DIR = str(output_dir)
    cfg.TEST.INTERVAL = 200
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    basicts.launch_training(cfg, args.gpu, node_rank=0)
    _finalize_prediction(
        output_dir,
        args.seed,
        time.perf_counter() - start,
        (
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available()
            else None
        ),
    )


if __name__ == "__main__":
    main()
