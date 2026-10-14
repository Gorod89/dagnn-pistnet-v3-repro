"""Summarize the three adapted upstream STID LargeST anchor runs.

This analysis is deliberately separate from the earlier BasicTS STID results.
It never rewrites the original local competitor analysis or run artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dagnn_pistnet_v3.publication_analysis import paired_competitor_block_rows


SEEDS = (13, 42, 97)
HORIZONS = ("15min", "45min", "90min", "180min")
PROTOCOL = "6e302d95e1fd2cb1125b1f2ad8fc6c360e864384f3d8cdf5e087d04393187251"
UPSTREAM = "e8b313bc591bdd0101a1619962c9b503e75127c0"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "artifacts/v3/competitor_runs/large_st_ca_2019_15min_official/stid_upstream_anchor"
    study = root / "artifacts/v3/studies/large_st_ca_2019_15min_official/large_st_ca_2019_15min/full"
    reconstructed = root / "artifacts/v3/competitor_analysis/large_st_ca_2019_15min_official/reconstructed_ours"
    output = root / "artifacts/v3/competitor_analysis/large_st_ca_2019_15min_official/stid_upstream_anchor"
    runs: list[dict] = []
    anchor: list[dict] = []
    configs: list[dict] = []
    for seed in SEEDS:
        run_dir = base / f"seed_{seed}"
        training = read_json(run_dir / "result.json")
        config = read_json(run_dir / "run_config.json")
        evaluated_dir = run_dir / "common_evaluator"
        evaluated = read_json(evaluated_dir / "result.json")
        if (run_dir / "active.lock").exists():
            raise ValueError(f"seed {seed} remains locked")
        if training["seed"] != seed or training["upstream_commit"] != UPSTREAM:
            raise ValueError(f"seed {seed} training identity mismatch")
        if training["status"] != "trained_prediction_exported; common evaluation pending":
            raise ValueError(f"seed {seed} training did not finish")
        if evaluated["model"] != "STID-upstream-adapted" or evaluated["seed"] != seed:
            raise ValueError(f"seed {seed} evaluator identity mismatch")
        if evaluated["protocol_sha256"] != PROTOCOL or training["protocol_sha256"] != PROTOCOL:
            raise ValueError(f"seed {seed} protocol mismatch")
        if training["test_windows"] != 7004:
            raise ValueError(f"seed {seed} test-window count mismatch")
        archive = evaluated_dir / "window_errors.npz"
        with np.load(archive) as errors:
            if any(errors[key].shape != (7004,) for key in ("h1", "h3", "h6", "h12")):
                raise ValueError(f"seed {seed} window errors are incomplete")
        anchor.append(evaluated)
        configs.append(config)
        runs.append({**evaluated, "archive_path": str(archive)})

        ours_dir = study / f"seed_{seed}"
        ours = read_json(ours_dir / "result.json")
        ours_archive = reconstructed / f"seed_{seed}" / "window_errors.npz"
        if not ours_archive.is_file():
            raise FileNotFoundError(ours_archive)
        runs.append({
            **ours,
            "model": "DAGNN-PISTNet-v3",
            "dataset": "LargeST-CA-2019-15min",
            "seed": seed,
            "protocol_sha256": PROTOCOL,
            "archive_path": str(ours_archive),
        })

    for field in ("runner_sha256", "upstream_model_sha256", "environment_sha256", "protocol_sha256"):
        if len({config[field] for config in configs}) != 1:
            raise ValueError(f"anchor configurations differ on {field}")
    paired = paired_competitor_block_rows(
        runs, "STID-upstream-adapted", bootstrap_samples=2000, block_length=12
    )
    summary: dict[str, dict] = {}
    for horizon in HORIZONS:
        metrics: dict[str, dict] = {}
        for metric in ("mae", "rmse", "smape"):
            values = np.array([run["test_metrics"][horizon]["flow"][metric] for run in anchor])
            metrics[metric] = {
                "by_seed": {str(seed): float(value) for seed, value in zip(SEEDS, values, strict=True)},
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
            }
        counts = {run["test_metrics"][horizon]["flow"]["count"] for run in anchor}
        if len(counts) != 1:
            raise ValueError(f"{horizon} masks differ across seeds")
        summary[horizon] = {"metrics": metrics, "valid_target_count": counts.pop()}
    report = {
        "description": "Adapted pinned upstream-STID architecture on canonical LargeST-CA 2019/15-min sliding-window split",
        "not_exact_patchstg_replication": True,
        "upstream_commit": UPSTREAM,
        "protocol_sha256": PROTOCOL,
        "seeds": list(SEEDS),
        "config_identity": {field: configs[0][field] for field in ("runner_sha256", "upstream_model_sha256", "environment_sha256")},
        "published_patchstg_reference_mae": {"45min": 15.51, "90min": 18.53, "180min": 22.63},
        "horizons": summary,
        "paired_ours_minus_anchor": paired,
        "inference": "Three fixed seeds; 12-window circular moving-block bootstrap/sign-flip; Holm over four horizons. This is a within-pipeline comparison, not cross-paper replication.",
        "limitations": "Canonical sliding-window split/common evaluator differ from upstream CA timeline 60/20/20; PatchSTG runner not reconstructed; differences between old BasicTS baseline and this anchor have not been causally ablated.",
    }
    output.mkdir(parents=True, exist_ok=True)
    target = output / "analysis.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target)
    for horizon in HORIZONS:
        m = summary[horizon]["metrics"]["mae"]
        print(f"{horizon}: MAE {m['mean']:.4f} +/- {m['sample_sd']:.4f}")
    for row in paired:
        print(f"{row['horizon']}: delta MAE {row['mean_mae_difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}], Holm p={row['p_holm']:.6f}")


if __name__ == "__main__":
    main()
