"""Exploratory seed-stratified paired block analysis for PEMS08 flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HORIZONS = (("h1", "5min"), ("h3", "15min"), ("h6", "30min"), ("h12", "60min"))
SEEDS = (13, 42, 97)


def own_flow_window_errors(path: Path, batch_size: int = 16) -> dict[str, np.ndarray]:
    """Undo v3's batch-major, feature-major interleaving (flow is feature 0)."""
    out: dict[str, np.ndarray] = {}
    with np.load(path) as archive:
        for key, _ in HORIZONS:
            flat = np.asarray(archive[key], dtype=np.float64)
            if flat.shape != (3561 * 3,):
                raise ValueError(f"unexpected three-feature archive shape for {path}: {flat.shape}")
            chunks = []
            offset = 0
            for first in range(0, 3561, batch_size):
                size = min(batch_size, 3561 - first)
                chunks.append(flat[offset : offset + size])
                offset += 3 * size
            if offset != len(flat):
                raise ValueError("window-error extraction did not consume archive")
            out[key] = np.concatenate(chunks)
    return out


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.zeros(len(p_values), dtype=float)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--spec-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=12)
    args = parser.parse_args()
    if args.draws < 100 or args.block_length < 1:
        raise ValueError("draws >= 100 and block length >= 1 required")

    errors: dict[str, np.ndarray] = {}
    published: dict[str, list[dict[str, float]]] = {}
    for key, time_key in HORIZONS:
        per_seed = []
        published[key] = []
        for seed in SEEDS:
            own_dir = args.study_root / f"seed_{seed}"
            spec_dir = args.spec_root / f"specstg_pems08_seed{seed}"
            spec_result = json.loads((spec_dir / "result.json").read_text(encoding="utf-8"))
            own_result = json.loads((own_dir / "result.json").read_text(encoding="utf-8"))
            if spec_result.get("evaluation_scope") != "flow_only":
                raise ValueError(f"SpecSTG seed {seed} is not fully evaluated flow-only")
            if spec_result["protocol_sha256"] != "72bda3dc678ae76bef5f5a59d38b2c4161154630a7b0d5ae18bc79dec4671629":
                raise ValueError("SpecSTG flow-only protocol changed")
            own = own_flow_window_errors(own_dir / "window_errors.npz")[key]
            with np.load(spec_dir / "flow_common_point" / "window_errors.npz") as archive:
                spec = np.asarray(archive[key], dtype=np.float64)
            if own.shape != (3561,) or spec.shape != (3561,):
                raise ValueError("paired window arrays have different shapes")
            if abs(float(own.mean()) - own_result["test_metrics"][time_key]["flow"]["mae"]) > 0.15:
                raise ValueError("own extracted flow errors are inconsistent with reported MAE")
            if abs(float(spec.mean()) - spec_result["test_metrics"][time_key]["flow"]["mae"]) > 0.15:
                raise ValueError("SpecSTG window errors are inconsistent with reported MAE")
            per_seed.append(spec - own)
            published[key].append({
                "seed": seed,
                "dagnn_flow_mae": own_result["test_metrics"][time_key]["flow"]["mae"],
                "specstg_flow_mae": spec_result["test_metrics"][time_key]["flow"]["mae"],
                "specstg_flow_crps": spec_result["test_metrics"][time_key]["flow"]["crps"],
            })
        errors[key] = np.stack(per_seed)

    rng = np.random.default_rng(20261007)
    draw_count = args.draws
    block = args.block_length
    n = 3561
    blocks = (n + block - 1) // block
    offsets = np.arange(block, dtype=np.int64)
    rows: list[dict[str, object]] = []
    p_values = []
    for key, time_key in HORIZONS:
        d = errors[key]
        observed = float(d.mean())
        bootstrap = np.empty(draw_count, dtype=np.float64)
        sign_null = np.empty(draw_count, dtype=np.float64)
        for draw in range(draw_count):
            seed_means = []
            null_means = []
            for seed_index in range(3):
                starts = rng.integers(0, n, size=blocks)
                indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
                seed_means.append(float(d[seed_index, indices].mean()))
                # Approximate null: flip signs of *nonoverlapping* temporal
                # blocks, preserving local dependence inside each block.
                signs = rng.choice((-1.0, 1.0), size=blocks)
                padded = np.pad(d[seed_index], (0, blocks * block - n), mode="wrap")
                null_means.append(float((padded.reshape(blocks, block) * signs[:, None]).reshape(-1)[:n].mean()))
            bootstrap[draw] = float(np.mean(seed_means))
            sign_null[draw] = float(np.mean(null_means))
        p_value = float((1 + np.count_nonzero(np.abs(sign_null) >= abs(observed))) / (draw_count + 1))
        p_values.append(p_value)
        rows.append({
            "horizon": time_key,
            "paired_difference_specstg_minus_dagnn": observed,
            "paired_95pct_block_bootstrap_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
            "block_sign_flip_p": p_value,
            "seed_results": published[key],
        })
    adjusted = holm_adjust(p_values)
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
    report = {
        "status": "completed",
        "scope": "PEMS08_flow_only_specstg_vs_dagnn",
        "inference_label": "exploratory_posthoc",
        "seed_alignment": list(SEEDS),
        "test_windows_per_seed": n,
        "block_length_windows": block,
        "bootstrap_draws": draw_count,
        "p_value_method": "paired_nonoverlapping_12_window_block_sign_flip_monte_carlo",
        "multiple_testing_correction": "Holm-Bonferroni across four prelisted forecast horizons",
        "positive_difference_means": "SpecSTG has higher per-window flow MAE than DAGNN-PISTNet-v3",
        "limitations": [
            "Windows overlap in time; block length mitigates but does not remove all dependence.",
            "Only three training seeds and one dataset are included.",
            "SpecSTG required disclosed source compatibility and history adaptations.",
            "Paired analysis is post-hoc exploratory, not a preregistered claim.",
        ],
        "horizons": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
