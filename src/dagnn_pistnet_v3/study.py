from __future__ import annotations

import csv
import hashlib
import json
import math
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import V3ExperimentConfig
from .experiment import V3Experiment
from .reproducibility import sha256_file, validate_resume_manifest, write_manifest


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class PublicationStudySpec:
    study_name: str
    datasets: dict[str, str]
    ablations: dict[str, dict[str, Any]]
    baseline: str = "full"
    seeds: tuple[int, ...] = (13, 42, 97)
    output_root: str = "artifacts/v3/studies"
    bootstrap_samples: int = 2000
    confidence_level: float = 0.95
    fail_fast: bool = True
    require_clean_worktree: bool = True
    validation_selection_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> PublicationStudySpec:
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("study specification must be a mapping")
        raw = dict(raw)
        raw["seeds"] = tuple(int(seed) for seed in raw.get("seeds", (13, 42, 97)))
        spec = cls(**raw)
        if len(spec.seeds) < 3:
            raise ValueError("publication study requires at least three independent seeds")
        if spec.baseline not in spec.ablations:
            raise ValueError("baseline must be present in ablations")
        if not spec.datasets or not spec.ablations:
            raise ValueError("study requires datasets and ablations")
        return spec

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.study_name


def _set_dotted(raw: dict[str, Any], key: str, value: Any) -> None:
    target = raw
    parts = key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _load_validation_selection(
    spec: PublicationStudySpec,
    required: bool,
) -> dict[str, dict[str, Any]]:
    if spec.validation_selection_path is None:
        return {}
    path = Path(spec.validation_selection_path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"validation selection is missing; run tuning first: {path}"
            )
        return {}
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    selection: dict[str, dict[str, Any]] = {}
    for dataset in spec.datasets:
        if dataset not in raw:
            raise ValueError(f"validation selection has no dataset: {dataset}")
        parameters = raw[dataset].get("parameters")
        if isinstance(parameters, str):
            parameters = json.loads(parameters)
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError(f"validation selection is invalid for dataset: {dataset}")
        selection[dataset] = parameters
    return selection


def resolve_study_runs(
    spec: PublicationStudySpec,
    require_validation_selection: bool = True,
) -> list[tuple[str, str, int, Path, dict[str, Any]]]:
    selection = _load_validation_selection(spec, require_validation_selection)
    runs: list[tuple[str, str, int, Path, dict[str, Any]]] = []
    for dataset_name, config_path in spec.datasets.items():
        base = V3ExperimentConfig.from_yaml(config_path)
        base_raw = asdict(base)
        for key, value in selection.get(dataset_name, {}).items():
            _set_dotted(base_raw, key, value)
        for variant, override in spec.ablations.items():
            for seed in spec.seeds:
                output = spec.output_dir / dataset_name / variant / f"seed_{seed}"
                resolved = _deep_merge(base_raw, override)
                resolved["training"]["seed"] = seed
                resolved["training"]["output_dir"] = str(output)
                resolved["study"]["seeds"] = list(spec.seeds)
                resolved["study"]["bootstrap_samples"] = spec.bootstrap_samples
                resolved["study"]["confidence_level"] = spec.confidence_level
                resolved["study"]["fail_on_dirty_worktree"] = spec.require_clean_worktree
                runs.append((dataset_name, variant, seed, output, resolved))
    return runs


def run_publication_study(
    spec_path: str | Path,
    workspace: str | Path = ".",
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    spec = PublicationStudySpec.from_yaml(spec_path)
    runs = resolve_study_runs(
        spec, require_validation_selection=not dry_run
    )
    plan = {
        "study": spec.study_name,
        "run_count": len(runs),
        "datasets": list(spec.datasets),
        "variants": list(spec.ablations),
        "seeds": list(spec.seeds),
        "output": str(spec.output_dir),
        "validation_selection": spec.validation_selection_path,
        "validation_selection_available": bool(
            spec.validation_selection_path
            and Path(spec.validation_selection_path).is_file()
        ),
    }
    if dry_run:
        return plan
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    with (spec.output_dir / "study_spec.resolved.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(asdict(spec), stream, sort_keys=False, allow_unicode=True)
    failures: list[dict[str, Any]] = []
    for index, (dataset, variant, seed, output, raw) in enumerate(runs, start=1):
        result_path = output / "result.json"
        if result_path.exists() and not force:
            config = V3ExperimentConfig.from_dict(raw)
            validate_resume_manifest(output / "manifest.json", config, workspace)
            print(f"study skip existing {dataset}/{variant}/seed_{seed}", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        with (output / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=True)
        print(
            f"study run={index}/{len(runs)} dataset={dataset} variant={variant} seed={seed}",
            flush=True,
        )
        try:
            config = V3ExperimentConfig.from_dict(raw)
            resume_checkpoint = output / "last_training_state_v3.pt"
            if resume_checkpoint.is_file():
                validate_resume_manifest(output / "manifest.json", config, workspace)
                write_manifest(
                    output / "pre_resume_provenance.json",
                    {
                        "source_checkpoint": str(resume_checkpoint),
                        "source_checkpoint_sha256": sha256_file(resume_checkpoint),
                        "source_manifest_sha256": sha256_file(output / "manifest.json"),
                    },
                )
            V3Experiment(config, workspace=workspace).train(
                resume_checkpoint=resume_checkpoint if resume_checkpoint.is_file() else None
            )
        except Exception as error:  # noqa: BLE001 - failures must be archived verbatim
            failure = {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            with (output / "failure.json").open("w", encoding="utf-8") as stream:
                json.dump(failure, stream, ensure_ascii=False, indent=2)
            if spec.fail_fast:
                raise
    with (spec.output_dir / "failures.json").open("w", encoding="utf-8") as stream:
        json.dump(failures, stream, ensure_ascii=False, indent=2)
    aggregate_publication_study(spec_path)
    return {**plan, "failures": len(failures)}


def _flatten_metrics(result: dict[str, Any]) -> list[tuple[str, str, str, float]]:
    rows: list[tuple[str, str, str, float]] = []
    for horizon, features in result["test_metrics"].items():
        for feature, metrics in features.items():
            for metric, value in metrics.items():
                if metric != "count":
                    rows.append((horizon, feature, metric, float(value)))
    return rows


def paired_bootstrap(
    differences: np.ndarray,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    differences = np.asarray(differences, dtype=np.float64)
    if differences.size < 2:
        return float(differences.mean()), math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    batch = 64
    for start in range(0, samples, batch):
        stop = min(samples, start + batch)
        indices = rng.integers(0, differences.size, size=(stop - start, differences.size))
        estimates[start:stop] = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(differences.mean()),
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def paired_sign_flip_pvalue(
    differences: np.ndarray,
    samples: int,
    seed: int,
) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    if differences.size < 2:
        return math.nan
    observed = abs(differences.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(samples):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=differences.size)
        extreme += abs((differences * signs).mean()) >= observed
    return (extreme + 1.0) / (samples + 1.0)


def holm_adjust(pvalues: list[float]) -> list[float]:
    adjusted = [math.nan] * len(pvalues)
    finite = [(index, value) for index, value in enumerate(pvalues) if math.isfinite(value)]
    finite.sort(key=lambda item: item[1])
    running = 0.0
    count = len(finite)
    for rank, (index, value) in enumerate(finite):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return adjusted


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def aggregate_publication_study(spec_path: str | Path) -> dict[str, str]:
    spec = PublicationStudySpec.from_yaml(spec_path)
    long_rows: list[dict[str, Any]] = []
    run_results: dict[tuple[str, str, int], tuple[dict[str, Any], Path]] = {}
    for dataset, variant, seed, output, _ in resolve_study_runs(spec):
        result_path = output / "result.json"
        if not result_path.exists():
            continue
        with result_path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        run_results[(dataset, variant, seed)] = (result, output)
        for horizon, feature, metric, value in _flatten_metrics(result):
            long_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "horizon": horizon,
                    "feature": feature,
                    "metric": metric,
                    "value": value,
                }
            )
    if not long_rows:
        raise ValueError("no completed study results found")
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    long_path = spec.output_dir / "metrics_long.csv"
    with long_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)

    grouped: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in long_rows:
        key = (
            row["dataset"],
            row["variant"],
            row["horizon"],
            row["feature"],
            row["metric"],
        )
        grouped.setdefault(key, []).append(row["value"])
    summary_rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        array = np.asarray(values)
        summary_rows.append(
            {
                "dataset": key[0],
                "variant": key[1],
                "horizon": key[2],
                "feature": key[3],
                "metric": key[4],
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if array.size > 1 else math.nan,
                "seeds": int(array.size),
            }
        )
    summary_path = spec.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    comparisons: list[dict[str, Any]] = []
    for dataset in spec.datasets:
        for variant in spec.ablations:
            if variant == spec.baseline:
                continue
            for horizon_index in (0, 2, 5, 11):
                paired: list[np.ndarray] = []
                used_seeds: list[int] = []
                key = f"h{horizon_index + 1}"
                for seed in spec.seeds:
                    base_item = run_results.get((dataset, spec.baseline, seed))
                    variant_item = run_results.get((dataset, variant, seed))
                    if base_item is None or variant_item is None:
                        continue
                    base_archive = np.load(base_item[1] / "window_errors.npz")
                    variant_archive = np.load(variant_item[1] / "window_errors.npz")
                    if key not in base_archive or key not in variant_archive:
                        continue
                    size = min(base_archive[key].size, variant_archive[key].size)
                    paired.append(variant_archive[key][:size] - base_archive[key][:size])
                    used_seeds.append(seed)
                if not paired:
                    continue
                differences = np.concatenate(paired)
                random_seed = _stable_seed(dataset, variant, key)
                mean, low, high = paired_bootstrap(
                    differences,
                    spec.bootstrap_samples,
                    spec.confidence_level,
                    random_seed,
                )
                pvalue = paired_sign_flip_pvalue(
                    differences, max(1000, spec.bootstrap_samples), random_seed + 1
                )
                comparisons.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "baseline": spec.baseline,
                        "horizon_steps": horizon_index + 1,
                        "seeds": ";".join(map(str, used_seeds)),
                        "windows": differences.size,
                        "mean_mae_difference": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "p_raw": pvalue,
                    }
                )
    adjusted = holm_adjust([row["p_raw"] for row in comparisons])
    for row, value in zip(comparisons, adjusted, strict=True):
        row["p_holm"] = value
        row["significant_0_05"] = bool(math.isfinite(value) and value < 0.05)
    comparison_path = spec.output_dir / "paired_comparisons.csv"
    if comparisons:
        with comparison_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)

    report_path = spec.output_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as stream:
        stream.write(f"# Publication study: {spec.study_name}\n\n")
        stream.write(
            f"Baseline: `{spec.baseline}`. Seeds: {', '.join(map(str, spec.seeds))}. "
            f"Paired bootstrap: {spec.bootstrap_samples} resamples, "
            f"confidence {spec.confidence_level:.1%}; sign-flip p-values use Holm correction.\n\n"
        )
        stream.write("## Completeness\n\n")
        stream.write(f"Completed runs: {len(run_results)} / {len(resolve_study_runs(spec))}.\n\n")
        stream.write("## Artifacts\n\n")
        stream.write("- `metrics_long.csv`: every seed-level metric\n")
        stream.write("- `summary.csv`: mean ± sample standard deviation\n")
        stream.write("- `paired_comparisons.csv`: paired window-level inference\n")
        stream.write("- each run directory: manifest, checkpoint, history, result, window errors\n")
    return {
        "metrics_long": str(long_path),
        "summary": str(summary_path),
        "comparisons": str(comparison_path),
        "report": str(report_path),
    }
