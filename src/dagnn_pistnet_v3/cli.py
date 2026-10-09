from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .benchmark_protocol import (
    create_prediction_artifact,
    evaluate_prediction_artifact,
    export_benchmark_protocol,
    validate_competitor_registry,
)
from .config import V3ExperimentConfig
from .experiment import V3Experiment
from .pretraining import CrossDatasetPretrainConfig, run_cross_dataset_pretraining
from .study import aggregate_publication_study, run_publication_study
from .tuning import run_validation_tuning


def _emit(value: Any) -> None:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dagnn_pistnet_v3",
        description="Independent DAGNN + PI-STNet v3 research pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect-data", "smoke-test", "train", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", required=True)
        if command == "evaluate":
            sub.add_argument("--checkpoint")
    pretrain = subparsers.add_parser("pretrain")
    pretrain.add_argument("--config", required=True)
    study = subparsers.add_parser("study")
    study.add_argument("--spec", required=True)
    study.add_argument("--dry-run", action="store_true")
    study.add_argument("--force", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--spec", required=True)
    export = subparsers.add_parser("export-protocol")
    export.add_argument("--config", required=True)
    export.add_argument("--output", required=True)
    create = subparsers.add_parser("create-prediction-artifact")
    create.add_argument("--protocol", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--model", required=True)
    create.add_argument("--seed", required=True, type=int)
    evaluate_predictions = subparsers.add_parser("evaluate-predictions")
    evaluate_predictions.add_argument("--config", required=True)
    evaluate_predictions.add_argument("--protocol", required=True)
    evaluate_predictions.add_argument("--predictions", required=True)
    evaluate_predictions.add_argument("--output-dir", required=True)
    registry = subparsers.add_parser("validate-registry")
    registry.add_argument("--registry", required=True)
    tune = subparsers.add_parser("tune")
    tune.add_argument("--spec", required=True)
    tune.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path.cwd()
    if args.command == "pretrain":
        checkpoint = run_cross_dataset_pretraining(
            CrossDatasetPretrainConfig.from_yaml(args.config)
        )
        _emit({"checkpoint": str(checkpoint)})
        return 0
    if args.command == "study":
        _emit(
            run_publication_study(
                args.spec,
                workspace=workspace,
                dry_run=args.dry_run,
                force=args.force,
            )
        )
        return 0
    if args.command == "aggregate":
        _emit(aggregate_publication_study(args.spec))
        return 0
    if args.command == "create-prediction-artifact":
        _emit(
            {
                "artifact": str(
                    create_prediction_artifact(
                        args.protocol, args.output, args.model, args.seed
                    )
                )
            }
        )
        return 0
    if args.command == "validate-registry":
        _emit(validate_competitor_registry(args.registry))
        return 0
    if args.command == "tune":
        _emit(run_validation_tuning(args.spec, workspace, dry_run=args.dry_run))
        return 0

    config = V3ExperimentConfig.from_yaml(args.config)
    experiment = V3Experiment(config, workspace=workspace)
    if args.command == "inspect-data":
        _emit(experiment.inspect_data())
    elif args.command == "smoke-test":
        _emit(experiment.smoke_test())
    elif args.command == "train":
        _emit(experiment.train())
    elif args.command == "evaluate":
        _emit(experiment.evaluate(args.checkpoint))
    elif args.command == "export-protocol":
        _emit(export_benchmark_protocol(config, args.output, workspace))
    elif args.command == "evaluate-predictions":
        _emit(
            evaluate_prediction_artifact(
                config,
                args.protocol,
                args.predictions,
                args.output_dir,
            )
        )
    return 0
