"""Check the new order control against the archived full-run configurations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from dagnn_pistnet_v3.config import V3ExperimentConfig
from dagnn_pistnet_v3.experiment import V3Experiment
from dagnn_pistnet_v3.model import DAGNNPISTNetV3
from dagnn_pistnet_v3.reproducibility import sha256_file
from dagnn_pistnet_v3.study import PublicationStudySpec, resolve_study_runs


def _comparable(config: V3ExperimentConfig) -> dict:
    raw = asdict(config)
    raw["model"].pop("operator_order")
    raw["training"].pop("output_dir")
    return raw


def _parameters(config: V3ExperimentConfig, *, nodes: int, calendar_dim: int, event_dim: int) -> int:
    features = len(config.data.feature_names)
    model = DAGNNPISTNetV3(
        input_dim=features,
        output_dim=features,
        num_nodes=nodes,
        forecast_steps=config.data.forecast_steps,
        calendar_dim=calendar_dim,
        event_dim=event_dim,
        config=config.model,
        physics=config.physics,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def run(spec_path: Path, baseline_root: Path, output: Path) -> dict:
    spec = PublicationStudySpec.from_yaml(spec_path)
    if set(spec.ablations) != {"reverse_order"}:
        raise ValueError("the prespecified study has one new reverse_order control")
    rows = []
    data_metadata = {}
    for dataset, variant, seed, target, resolved in resolve_study_runs(spec):
        if variant != "reverse_order":
            raise ValueError(f"unexpected variant: {variant}")
        baseline = baseline_root / dataset / "full" / f"seed_{seed}"
        required = [
            baseline / "resolved_config.yaml",
            baseline / "result.json",
            baseline / "manifest.json",
            baseline / "window_errors.npz",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"archived reference incomplete: {missing}")
        old_raw = yaml.safe_load(required[0].read_text(encoding="utf-8"))
        old = V3ExperimentConfig.from_dict(old_raw)
        new = V3ExperimentConfig.from_dict(resolved)
        if _comparable(old) != _comparable(new):
            raise ValueError(f"configuration mismatch beyond operator order/output: {dataset}/{seed}")
        if tuple(old.model.operator_order) != ("transport", "diffusion", "source_sink"):
            raise ValueError("archived reference is not the original operator order")
        if tuple(new.model.operator_order) != ("source_sink", "diffusion", "transport"):
            raise ValueError("new control is not the declared reverse order")
        if dataset not in data_metadata:
            data = V3Experiment(new).prepare_data()
            data_metadata[dataset] = {
                "nodes": data.series.num_nodes,
                "calendar_dim": data.calendar_dim,
                "event_dim": data.event_dim,
            }
        metadata = data_metadata[dataset]
        old_count = _parameters(old, **metadata)
        new_count = _parameters(new, **metadata)
        recorded_count = json.loads(required[1].read_text(encoding="utf-8"))["parameter_count"]
        if old_count != new_count or old_count != recorded_count:
            raise ValueError(
                f"parameter count mismatch {dataset}/{seed}: "
                f"archived={recorded_count}, original={old_count}, reverse={new_count}"
            )
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "reference": str(baseline),
                "reference_config_sha256": sha256_file(required[0]),
                "reference_result_sha256": sha256_file(required[1]),
                "reference_window_errors_sha256": sha256_file(required[3]),
                "new_output": str(target),
                "parameter_count": new_count,
                "matched_except": ["model.operator_order", "training.output_dir"],
            }
        )
    report = {
        "spec": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "comparison_count": len(rows),
        "records": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=Path("configs/v3/operator_order_capacity_matched.yaml"))
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("artifacts/v3/studies/dagnn_pistnet_v3_ablation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v3/studies/operator_order_capacity_matched/preflight.json"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.spec, args.baseline_root, args.output), indent=2))
