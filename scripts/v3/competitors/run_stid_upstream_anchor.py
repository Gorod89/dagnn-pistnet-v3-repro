"""Train the pinned upstream STID model on canonical LargeST split arrays.

This is an adapted, original-architecture anchor, not a reproduction of the
published PatchSTG row. The upstream CA data generator specifies global
train-flow scaling, original-unit rescaling and zero masking; this runner
retains the canonical LargeST sliding-window split and shared evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


UPSTREAM_COMMIT = "e8b313bc591bdd0101a1619962c9b503e75127c0"
PROTOCOL_HASH = "6e302d95e1fd2cb1125b1f2ad8fc6c360e864384f3d8cdf5e087d04393187251"
HISTORY = FORECAST = 12
NODES = 8600


class NativeWindows(Dataset):
    def __init__(self, native_dir: Path, split: str, mean: float, std: float):
        self.values = np.load(native_dir / f"{split}_data.npy", mmap_mode="r")
        self.times = np.load(native_dir / f"{split}_timestamps.npy", mmap_mode="r")
        if self.values.ndim != 2 or self.values.shape[1] != NODES:
            raise ValueError(f"Unexpected {split} flow shape: {self.values.shape}")
        if self.times.shape != (len(self.values), 2):
            raise ValueError(f"Unexpected {split} calendar shape: {self.times.shape}")
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.values) - HISTORY - FORECAST + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        flow = np.asarray(self.values[index : index + HISTORY], dtype=np.float32)
        times = np.asarray(self.times[index : index + HISTORY], dtype=np.float32)
        features = np.empty((HISTORY, NODES, 3), dtype=np.float32)
        features[..., 0] = (flow - self.mean) / self.std
        features[..., 1] = times[:, 0, None]
        features[..., 2] = times[:, 1, None]
        targets = np.array(
            self.values[index + HISTORY : index + HISTORY + FORECAST],
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(features), torch.from_numpy(targets[..., None])


def global_train_scaler(values: np.ndarray, train_end: int) -> tuple[float, float]:
    total = squared = 0.0
    count = 0
    for start in range(0, train_end, 256):
        block = np.asarray(values[start : min(start + 256, train_end)], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Train flow contains non-finite values")
        total += float(block.sum())
        squared += float(np.square(block).sum())
        count += block.size
    mean = total / count
    std = math.sqrt(max(squared / count - mean * mean, 0.0))
    if not math.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid train scaler standard deviation {std}")
    return mean, std


def original_unit_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(target) & (target.abs() > 1e-5)
    if not bool(valid.any()):
        raise ValueError("Batch has no valid targets")
    return (prediction - target).abs()[valid].mean()


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
) -> float:
    model.eval()
    absolute = 0.0
    count = 0
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = model(features, None, 0, 0, False) * std + mean
            valid = torch.isfinite(target) & (target.abs() > 1e-5)
            absolute += float((prediction - target).abs()[valid].sum().item())
            count += int(valid.sum().item())
    if count == 0:
        raise ValueError("Validation contains no valid targets")
    return absolute / count


def make_model(repo: Path) -> torch.nn.Module:
    sys.path.insert(0, str(repo))
    from stid.arch import STID  # pylint: disable=import-outside-toplevel

    model = STID(
        num_nodes=NODES,
        input_len=HISTORY,
        input_dim=3,
        embed_dim=32,
        output_len=FORECAST,
        num_layer=4,
        if_node=True,
        node_dim=128,
        if_T_i_D=True,
        if_D_i_W=True,
        temp_dim_tid=32,
        temp_dim_diw=32,
        time_of_day_size=96,
        day_of_week_size=7,
    )
    if sum(p.numel() for p in model.parameters()) != 1_511_180:
        raise ValueError("Unexpected upstream STID parameter count")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(13, 42, 97))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"Upstream commit {commit} != {UPSTREAM_COMMIT}")
    manifest = json.loads(
        (args.native_dir / "native_manifest.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        manifest["protocol_sha256"] != PROTOCOL_HASH
        or protocol["protocol_sha256"] != PROTOCOL_HASH
        or protocol["prediction_shape"] != [7004, 12, NODES, 1]
    ):
        raise ValueError("Canonical protocol mismatch")
    train_end = int(protocol["split"]["train_end"])
    train_values = np.load(args.native_dir / "train_data.npy", mmap_mode="r")
    if len(train_values) != train_end + FORECAST - 1:
        raise ValueError("Unexpected training native split length")
    # LargeST's published generator fits on data preceding the first
    # validation history anchor, not on every row in the train split.
    scaler_end = train_end - 1 - HISTORY
    mean, std = global_train_scaler(train_values, scaler_end)
    train = NativeWindows(args.native_dir, "train", mean, std)
    validation = NativeWindows(args.native_dir, "val", mean, std)
    test = NativeWindows(args.native_dir, "test", mean, std)
    if (len(train), len(validation), len(test)) != (21010, 7003, 7004):
        raise ValueError("Native window count mismatch")
    if abs(mean - 237.11562) > 0.01 or abs(std - 177.698) > 0.01:
        raise ValueError(f"Train scaler differs from canonical: {mean}, {std}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = make_model(repo)

    if args.preflight_only:
        features, target = train[0]
        prediction = model(features[None], None, 0, 0, True) * std + mean
        loss = original_unit_mae(prediction, target[None])
        loss.backward()
        if not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ValueError("CPU preflight found missing or non-finite gradients")
        print(
            json.dumps(
                {
                    "status": "CPU adapter preflight only; no training or test metric",
                    "upstream_commit": commit,
                    "protocol_sha256": PROTOCOL_HASH,
                    "mean": mean,
                    "std": std,
                    "train_windows": len(train),
                    "validation_windows": len(validation),
                    "test_windows": len(test),
                    "model_parameters": sum(p.numel() for p in model.parameters()),
                    "smoke_loss": float(loss),
                },
                indent=2,
            )
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; CPU preflight is available")
    device = torch.device("cuda:0")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Completed anchor run already exists; refusing duplicate")
    if (output / "curve.jsonl").exists() or (output / "best.pt").exists():
        raise FileExistsError("Partial anchor run exists; inspect before any retry")
    lock = output / "active.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()} seed={args.seed}\n")
    try:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[1, 30, 60, 80], gamma=0.5
        )
        train_loader = DataLoader(
            train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            validation,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        environment = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        (output / "environment.pip-freeze.txt").write_text(
            environment, encoding="utf-8"
        )
        save_json(
            output / "run_config.json",
            {
                "upstream_commit": commit,
                "runner_sha256": sha256_file(Path(__file__)),
                "upstream_model_sha256": sha256_file(repo / "stid/arch/stid_arch.py"),
                "upstream_ca_config_sha256": sha256_file(repo / "stid/CA.py"),
                "upstream_ca_generator_sha256": sha256_file(
                    repo / "scripts/data_preparation/CA/generate_training_data.py"
                ),
                "environment_sha256": sha256_file(output / "environment.pip-freeze.txt"),
                "protocol_sha256": PROTOCOL_HASH,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "optimizer": "Adam(lr=0.002, weight_decay=0.0001)",
                "scheduler": "MultiStepLR([1,30,60,80], gamma=0.5)",
                "gradient_clip_norm": 5.0,
                "train_scaler": "global scalar, including zero flow; official scaler_end prefix",
                "scaler_end": scaler_end,
                "train_mean": mean,
                "train_std": std,
                "loss": "original-unit masked MAE, |target|>1e-5",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "upstream_ca_settings": {
                    "norm_each_channel": False,
                    "rescale": True,
                    "null_val": 0.0,
                    "settings_source": "scripts/data_preparation/CA/generate_training_data.py",
                },
                "split_adaptation": "canonical LargeST sliding-window split, not upstream 60/20/20 timeline split",
                "published_score_replication": False,
            },
        )

        best = math.inf
        best_epoch = 0
        elapsed_start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_absolute = 0.0
            train_batches = 0
            for features, target in train_loader:
                features = features.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(features, None, 0, epoch, True) * std + mean
                loss = original_unit_mae(prediction, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_absolute += float(loss.item())
                train_batches += 1
            validation_mae = evaluate(model, val_loader, device, mean, std)
            scheduler.step()
            record = {
                "epoch": epoch,
                "train_batch_mean_mae": train_absolute / train_batches,
                "validation_global_mae": validation_mae,
                "lr_after_epoch": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - elapsed_start,
            }
            with (output / "curve.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if validation_mae < best:
                best = validation_mae
                best_epoch = epoch
                temporary = output / "best.pt.tmp"
                torch.save(model.state_dict(), temporary)
                temporary.replace(output / "best.pt")
            print(json.dumps(record), flush=True)

        model.load_state_dict(torch.load(output / "best.pt", map_location=device))
        model.eval()
        raw_temporary = output / "prediction.canonical.raw.f32.tmp"
        prediction_raw = np.memmap(
            raw_temporary,
            dtype="float32",
            mode="w+",
            shape=(len(test), FORECAST, NODES),
        )
        test_loader = DataLoader(
            test, batch_size=4, shuffle=False, num_workers=0, pin_memory=True
        )
        offset = 0
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(device, non_blocking=True)
                prediction = model(features, None, 0, best_epoch, False)
                prediction = (prediction * std + mean)[..., 0].cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise ValueError("Non-finite test prediction")
                prediction_raw[offset : offset + len(prediction)] = prediction
                offset += len(prediction)
        if offset != len(test):
            raise ValueError("Test prediction count mismatch")
        prediction_raw.flush()
        del prediction_raw
        raw_temporary.replace(output / "prediction.canonical.raw.f32")
        save_json(
            output / "prediction.canonical.json",
            {
                "path": "prediction.canonical.raw.f32",
                "dtype": "float32",
                "shape": [len(test), FORECAST, NODES],
                "layout": "SHN",
                "value_space": "original",
                "format": "raw_memmap",
            },
        )
        save_json(
            output / "result.json",
            {
                "status": "trained_prediction_exported; common evaluation pending",
                "seed": args.seed,
                "upstream_commit": commit,
                "protocol_sha256": PROTOCOL_HASH,
                "best_epoch": best_epoch,
                "best_validation_mae": best,
                "elapsed_seconds": time.perf_counter() - elapsed_start,
                "test_windows": len(test),
            },
        )
    except Exception as exc:
        save_json(
            output / "failure.json",
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
