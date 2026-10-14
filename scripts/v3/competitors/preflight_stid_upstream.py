"""CPU-only interface preflight for the pinned upstream STID California model.

This does not train or evaluate the model and does not certify PatchSTG fidelity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


EXPECTED_COMMIT = "e8b313bc591bdd0101a1619962c9b503e75127c0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Upstream STID commit {commit} != {EXPECTED_COMMIT}")

    manifest = json.loads(
        (args.native_dir / "native_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest["dataset"] != "LargeST-CA-2019-15min"
        or manifest["history_steps"] != 12
        or manifest["forecast_steps"] != 12
        or manifest["num_nodes"] != 8600
        or manifest["steps_per_day"] != 96
        or manifest["split_policy"] != "official_sliding_windows"
    ):
        raise ValueError(f"Unexpected native dataset protocol: {manifest}")

    for split in ("train", "val", "test"):
        values = np.load(args.native_dir / f"{split}_data.npy", mmap_mode="r")
        times = np.load(args.native_dir / f"{split}_timestamps.npy", mmap_mode="r")
        if values.ndim != 2 or values.shape[1] != 8600:
            raise ValueError(f"{split} values have shape {values.shape}")
        if times.shape != (values.shape[0], 2):
            raise ValueError(f"{split} timestamps have shape {times.shape}")
        if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(times >= 1):
            raise ValueError(f"{split} calendar features fall outside [0, 1)")

    sys.path.insert(0, str(repo))
    from stid.arch import STID  # pylint: disable=import-outside-toplevel

    model = STID(
        num_nodes=8600,
        input_len=12,
        input_dim=3,
        embed_dim=32,
        output_len=12,
        num_layer=4,
        if_node=True,
        node_dim=128,
        if_T_i_D=True,
        if_D_i_W=True,
        temp_dim_tid=32,
        temp_dim_diw=32,
        time_of_day_size=96,
        day_of_week_size=7,
    ).eval()
    if sum(p.numel() for p in model.parameters()) != 1_511_180:
        raise ValueError("Upstream model parameter count unexpectedly changed")

    values = np.load(args.native_dir / "train_data.npy", mmap_mode="r")
    times = np.load(args.native_dir / "train_timestamps.npy", mmap_mode="r")
    # Interface smoke test only: not the final train-fitted normalization.
    history = np.empty((1, 12, 8600, 3), dtype=np.float32)
    history[0, :, :, 0] = values[:12]
    history[0, :, :, 1] = times[:12, 0, None]
    history[0, :, :, 2] = times[:12, 1, None]
    with torch.no_grad():
        prediction = model(torch.from_numpy(history), None, 0, 0, False)
    if prediction.shape != (1, 12, 8600, 1) or not torch.isfinite(prediction).all():
        raise ValueError(f"Invalid upstream forward output: {prediction.shape}")
    print(
        json.dumps(
            {
                "upstream_commit": commit,
                "protocol_sha256": manifest["protocol_sha256"],
                "model_parameters": 1_511_180,
                "input_shape": list(history.shape),
                "output_shape": list(prediction.shape),
                "status": "CPU interface only; no training or metric",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
