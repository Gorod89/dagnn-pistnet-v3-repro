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
COLORS = {"ours": "#24627A", "fast": "#C24C32", "stid": "#7E8B3A"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    def box(x: float, y: float, w: float, title: str, subtitle: str, color: str) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, 0.9, boxstyle="round,pad=0.1,rounding_size=0.13", linewidth=1.2, edgecolor=color, facecolor="white"))
        ax.text(x + w / 2, y + 0.57, title, ha="center", va="center", fontsize=10, weight="bold", color=color)
        ax.text(x + w / 2, y + 0.27, subtitle, ha="center", va="center", fontsize=8.3, color="#333333")

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#57636B") -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=color))

    box(0.2, 3.5, 2.0, "History + calendar", "12 observed steps", COLORS["ours"])
    box(2.7, 3.5, 2.1, "Encoder", "trend / residual", COLORS["ours"])
    box(5.3, 3.5, 2.1, "Fine graph", "directed messages", COLORS["ours"])
    box(7.9, 3.5, 2.1, "Coarse graph", "corridor pooling", COLORS["ours"])
    box(10.5, 3.5, 2.3, "Rollout decoder", "increment + regimes", COLORS["ours"])
    for x1, x2 in ((2.2, 2.7), (4.8, 5.3), (7.4, 7.9), (10.0, 10.5)):
        arrow(x1, 3.95, x2, 3.95)
    box(2.0, 1.45, 2.2, "Transport", "edge conditioned", COLORS["fast"])
    box(5.25, 1.45, 2.2, "Diffusion", "bounded graph term", COLORS["fast"])
    box(8.5, 1.45, 2.2, "Source / sink", "local forcing", COLORS["fast"])
    arrow(4.2, 1.9, 5.25, 1.9, COLORS["fast"])
    arrow(7.45, 1.9, 8.5, 1.9, COLORS["fast"])
    arrow(6.2, 3.5, 3.1, 2.45, COLORS["fast"])
    arrow(10.7, 1.9, 11.65, 3.5, COLORS["fast"])
    ax.text(6.4, 0.65, "Sequential operator block; probabilistic output yields mixture trajectories and predictive intervals", ha="center", fontsize=10)
    ax.text(6.4, 5.12, "Event jump and hard conservation are inactive in the evaluated flow-only configuration", ha="center", fontsize=9.4, color="#666666")
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
    horizons = [15, 45, 90, 180]
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    for name, key in (("DAGNN-PISTNet-v3", "ours"), ("FaST", "fast"), ("STID", "stid")):
        subset = {int(r["horizon"].replace("min", "")): r for r in data if r["model"] == name}
        means = [float(subset[h]["mae_mean"]) for h in horizons]
        sd = [float(subset[h]["mae_std"]) for h in horizons]
        ax.errorbar(horizons, means, yerr=sd, marker="o", linewidth=2, capsize=3, color=COLORS[key], label=name)
    ax.set_xticks(horizons)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("MAE (original flow units; lower is better)")
    ax.set_title("Official LargeST-CA 2019 / 15-minute protocol")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    save(fig, "figure3_official_mae")


def paired_differences() -> None:
    horizons = {"h1": 15, "h3": 45, "h6": 90, "h12": 180}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), sharey=True)
    for ax, reference, key in ((axes[0], "FaST", "fast"), (axes[1], "STID", "stid")):
        file = ANALYSIS / f"paired_vs_{reference.lower()}_block_holm.csv"
        record = {horizons[r["horizon"]]: r for r in rows(file)}
        for i, horizon in enumerate((15, 45, 90, 180)):
            r = record[horizon]
            center, lo, hi = [float(r[col]) for col in ("mean_mae_difference", "ci_low", "ci_high")]
            ax.plot([lo, hi], [i, i], color=COLORS[key], linewidth=2.5)
            ax.plot(center, i, marker="o", color=COLORS[key], markersize=7)
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
        ax.set_title(f"DAGNN-PISTNet-v3 minus {reference}")
        ax.set_xlabel("Paired MAE difference (positive: reference better)")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(4), ["15 min", "45 min", "90 min", "180 min"])
    axes[0].invert_yaxis()
    fig.suptitle("Seed-stratified moving-block 95% confidence intervals", y=1.02)
    save(fig, "figure4_paired_ci")


def calibration() -> None:
    data = rows(STUDY / "summary.csv")
    datasets = ["metr_la", "pems_bay", "pems07", "pems08", "large_st_ca"]
    horizons = [5, 15, 30, 60]
    metrics = {"coverage80": 0.8, "coverage95": 0.95}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), sharex=True)
    for ax, (metric, nominal) in zip(axes, metrics.items(), strict=True):
        for dataset in datasets:
            lookup = {(r["horizon"], r["feature"]): float(r["mean"]) for r in data if r["dataset"] == dataset and r["variant"] == "full" and r["metric"] == metric}
            feature = "speed" if dataset in ("metr_la", "pems_bay") else "flow"
            ax.plot(horizons, [lookup[(f"{h}min", feature)] for h in horizons], marker="o", label=dataset)
        ax.axhline(nominal, color="#222222", linestyle="--", linewidth=1.1)
        ax.set_title(f"Nominal {int(nominal * 100)}% interval")
        ax.set_xlabel("Forecast horizon (minutes)")
        ax.set_ylabel("Empirical coverage")
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
        ax.set_xticks(range(3), ["Ours", "FaST", "STID"])
        ax.set_title(title)
        ax.set_ylabel(unit)
        if field == "inference_latency_ms_per_sample_mean":
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Official LargeST-CA 2019 resource profile (batch-2 FP16 latency)")
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
