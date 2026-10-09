from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ExperimentConfig
from .experiment import Experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dagnn-pistnet")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect-data", "smoke-test", "baselines", "train", "evaluate"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", required=True, type=Path)
        if command == "evaluate":
            command_parser.add_argument("--checkpoint", type=Path)
    return parser


def main() -> None:
    try:
        args = _parser().parse_args()
        config = ExperimentConfig.from_yaml(args.config)
        experiment = Experiment(config)
        if args.command == "inspect-data":
            result = experiment.inspect_data()
        elif args.command == "smoke-test":
            result = experiment.smoke_test()
        elif args.command == "baselines":
            result = experiment.baselines()
        elif args.command == "train":
            result = asdict(experiment.train())
        elif args.command == "evaluate":
            result = experiment.evaluate(args.checkpoint)
        else:  # pragma: no cover - argparse guarantees command choices
            raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        # PyTorch CUDA on Windows can terminate with 0xC0000409 if the runtime
        # is unloaded while tensors are still pending destruction.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
