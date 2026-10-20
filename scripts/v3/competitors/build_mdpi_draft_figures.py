"""Create publication-draft figures from completed, immutable result tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "artifacts/v3/competitor_analysis/large_st_ca_2019_15min_official"
READINESS = ROOT / "artifacts/v3/publication_readiness"
STUDY = ROOT / "artifacts/v3/studies/dagnn_pistnet_v3_ablation"
FIGURES = ROOT / "docs/manuscript_mdpi/figures"
COLORS = {"ours": "#24627A", "fast": "#C24C32", "stid": "#7E8B3A", "stid_old": "#888888"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 3.4))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    def box(x: float, y: float, w: float, title: str, subtitle: str, color: str) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, 0.9, boxstyle="round,pad=0.06,rounding_size=0.10", linewidth=1.2, edgecolor=color, facecolor="white"))
        ax.text(x + w / 2, y + 0.57, title, ha="center", va="center", fontsize=9.0, weight="bold", color=color)
        ax.text(x + w / 2, y + 0.27, subtitle, ha="center", va="center", fontsize=7.3, color="#333333")

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#57636B") -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=color))

    stages = [
        ("History", "12 steps", COLORS["ours"]),
        ("Encoder", "trend / residual", COLORS["ours"]),
        ("Fine graph", "directed messages", COLORS["ours"]),
        ("Coarse graph", "topology pooling", COLORS["ours"]),
        ("Transport", "edge conditioned", COLORS["fast"]),
        ("Diffusion", "bounded graph", COLORS["fast"]),
        ("Source / sink", "local forcing", COLORS["fast"]),
        ("Increment", "mixture modes", COLORS["ours"]),
    ]
    width, gap, left = 1.47, 0.18, 0.08
    for index, (title, subtitle, color) in enumerate(stages):
        x = left + index * (width + gap)
        box(x, 1.35, width, title, subtitle, color)
        if index:
            arrow(x - gap + 0.02, 1.80, x - 0.02, 1.80)
    ax.text(6.6, 0.57, "The learned operator block and increment update repeat at each autonomous forecast step", ha="center", fontsize=9.5)
    ax.text(6.6, 2.78, "Event jump and hard conservation are inactive in the evaluated flow-only configuration", ha="center", fontsize=9.2, color="#666666")
    save(fig, "figure1_architecture")


def ablations() -> None:
    data = rows(READINESS / "ablation_60min_effects.csv")
    datasets = ["metr_la", "pems_bay", "pems07", "pems08", "large_st_ca"]
    variants = ["no_operator_splitting", "no_hierarchy", "no_delta", "no_cross_dataset_pretraining", "no_decomposition", "no_probabilistic_head", "no_rollout_curriculum"]
    labels = ["No operator splitting", "No hierarchy", "No delta decoder", "No cross-data pretraining", "No decomposition", "No probabilistic head", "No rollout curriculum"]
    lookup = {(r["dataset"], r["variant"]): float(r["delta_pct"]) for r in data}
    matrix = np.array([[lookup[(dataset, variant)] for dataset in datasets] for variant in variants])
    fig, ax = plt.subplots(figsize=(10.7, 5.1))
    im = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=8), aspect="auto")
    ax.set_xticks(range(len(datasets)), ["METR-LA", "PEMS-BAY", "PEMS07", "PEMS08", "LargeST-CA"])
    ax.set_yticks(range(len(variants)), labels)
    ax.set_xlabel("Dataset (original 5-minute ablation protocol)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "0.0%" if abs(value) < 0.05 else f"{value:+.1f}%"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="white" if value > 5.5 or value < -2 else "#17202A")
    fig.colorbar(im, ax=ax, label="MAE change after removing component (%)")
    ax.set_title("60-minute ablation effects; positive values favor the full model", pad=13)
    save(fig, "figure2_ablation_heatmap")


def official_accuracy() -> None:
    data = rows(ANALYSIS / "metrics_summary.csv")
    anchor = json.loads((ANALYSIS / "stid_upstream_anchor/analysis.json").read_text(encoding="utf-8"))
    horizons = [15, 45, 90, 180]
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    for name, key, label in (("DAGNN-PISTNet-v3", "ours", "DAGNN-PISTNet-v3"), ("FaST", "fast", "FaST (12-step adapted)"), ("STID", "stid_old", "STID (prior BasicTS local)")):
        subset = {int(r["horizon"].replace("min", "")): r for r in data if r["model"] == name}
        means = [float(subset[h]["mae_mean"]) for h in horizons]
        sd = [float(subset[h]["mae_std"]) for h in horizons]
        ax.errorbar(horizons, means, yerr=sd, marker="o", linewidth=1.7, capsize=3, color=COLORS[key], label=label)
    means = [anchor["horizons"][f"{h}min"]["metrics"]["mae"]["mean"] for h in horizons]
    sd = [anchor["horizons"][f"{h}min"]["metrics"]["mae"]["sample_sd"] for h in horizons]
    ax.errorbar(horizons, means, yerr=sd, marker="s", linewidth=2.3, capsize=3, color=COLORS["stid"], label="STID (upstream-adapted anchor)")
    ax.set_xticks(horizons)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("MAE (original flow units; lower is better)")
    ax.set_title("Local runs on LargeST-CA 2019 / 15-minute targets")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    save(fig, "figure3_official_mae")


def paired_differences() -> None:
    horizons = {"h1": 15, "h3": 45, "h6": 90, "h12": 180}
    anchor = json.loads((ANALYSIS / "stid_upstream_anchor/analysis.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5), sharey=True)
    panels = (
        (axes[0], "FaST (12-step adapted)", "fast", rows(ANALYSIS / "paired_vs_fast_block_holm.csv")),
        (axes[1], "STID (upstream-adapted)", "stid", anchor["paired_ours_minus_anchor"]),
        (axes[2], "STID (prior BasicTS local)", "stid_old", rows(ANALYSIS / "paired_vs_stid_block_holm.csv")),
    )
    for ax, reference, key, source in panels:
        record = {horizons[r["horizon"]]: r for r in source}
        for i, horizon in enumerate((15, 45, 90, 180)):
            r = record[horizon]
            center, lo, hi = [float(r[col]) for col in ("mean_mae_difference", "ci_low", "ci_high")]
            ax.plot([lo, hi], [i, i], color=COLORS[key], linewidth=2.5)
            ax.plot(center, i, marker="o", color=COLORS[key], markersize=7)
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
        ax.set_title(f"Ours minus {reference}")
        ax.set_xlabel("Paired MAE difference (positive: reference better)")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(4), ["15 min", "45 min", "90 min", "180 min"])
    axes[0].invert_yaxis()
    fig.suptitle("Seed-stratified moving-block 95% confidence intervals", y=1.02)
    save(fig, "figure4_paired_ci")


def calibration() -> None:
    data = rows(STUDY / "summary.csv")
    datasets = ["metr_la", "pems_bay", "pems07", "pems08", "large_st_ca"]
    labels = {"metr_la": "METR-LA", "pems_bay": "PEMS-BAY", "pems07": "PEMS07", "pems08": "PEMS08", "large_st_ca": "LargeST-CA local"}
    horizons = [5, 15, 30, 60]
    metrics = {"coverage80": 0.8, "coverage95": 0.95}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), sharex=True)
    for ax, (metric, nominal) in zip(axes, metrics.items(), strict=True):
        for dataset in datasets:
            lookup = {(r["horizon"], r["feature"]): float(r["mean"]) for r in data if r["dataset"] == dataset and r["variant"] == "full" and r["metric"] == metric}
            feature = "speed" if dataset in ("metr_la", "pems_bay") else "flow"
            ax.plot(horizons, [lookup[(f"{h}min", feature)] for h in horizons], marker="o", label=labels[dataset])
        ax.axhline(nominal, color="#222222", linestyle="--", linewidth=1.1)
        ax.set_title(f"Nominal {int(nominal * 100)}% interval")
        ax.set_xlabel("Forecast horizon (minutes)")
        ax.set_ylabel("Empirical coverage (zoomed scale)")
        ax.set_ylim(0.70, 1.00)
        ax.set_xticks(horizons)
        ax.grid(alpha=0.2)
    axes[1].legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.suptitle("Coverage of the full model on 5-minute ablation datasets", y=1.02)
    save(fig, "figure5_calibration")


def efficiency() -> None:
    data = {r["model"]: r for r in rows(ANALYSIS / "efficiency_summary.csv")}
    names = ["DAGNN-PISTNet-v3", "FaST", "STID"]
    colors = [COLORS["ours"], COLORS["fast"], COLORS["stid"]]
    latency_dir = READINESS / "harmonized_latency"
    latency_model = {"DAGNN-PISTNet-v3": "dagnn", "FaST": "fast", "STID": "stid"}
    latency = {}
    for name, slug in latency_model.items():
        values = []
        for seed in (13, 42, 97):
            result = json.loads((latency_dir / f"{slug}_seed{seed}.json").read_text(encoding="utf-8"))
            key = "median_per_full_graph_sample" if slug == "dagnn" else "median_per_sample"
            values.append(float(result["latency_ms"][key]))
        latency[name] = float(np.mean(values))
    metrics = [
        ("parameter_count", "Parameters", 1 / 1e6, "million"),
        ("peak_accelerator_memory_mb_max", "Peak training VRAM", 1 / 1024, "GiB"),
        ("training_wall_hours_mean", "Mean training wall time", 1, "hours"),
        ("inference_latency_ms_per_sample_mean", "Model-forward latency", 1, "ms/sample"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.5))
    for ax, (field, title, scale, unit) in zip(axes.flat, metrics, strict=True):
        values = [(latency[name] if field == "inference_latency_ms_per_sample_mean" else float(data[name][field])) * scale for name in names]
        ax.bar(range(3), values, color=colors, width=0.65)
        ax.set_xticks(range(3), ["Ours", "FaST", "STID\nBasicTS"])
        ax.set_title(title)
        ax.set_ylabel(unit)
        if field == "inference_latency_ms_per_sample_mean":
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Previously profiled LargeST-CA 2019 implementations (batch-2 FP16 latency)")
    fig.tight_layout()
    save(fig, "figure6_efficiency")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    architecture()
    ablations()
    official_accuracy()
    paired_differences()
    calibration()
    efficiency()
    print(f"Wrote six PNG and six PDF figures to {FIGURES}")


if __name__ == "__main__":
    main()
