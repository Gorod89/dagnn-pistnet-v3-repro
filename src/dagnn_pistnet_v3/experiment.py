from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import V3ExperimentConfig
from .data import PreparedV3Data, prepare_v3_data
from .pretraining import load_pretrained_encoder
from .reproducibility import build_reproducibility_manifest, write_manifest
from .training import V3TrainingResult, V3ValidationResult, build_v3_runtime


class V3Experiment:
    def __init__(self, config: V3ExperimentConfig, workspace: str | Path = ".") -> None:
        self.config = config
        self.workspace = Path(workspace).resolve()

    def prepare_data(self) -> PreparedV3Data:
        return prepare_v3_data(self.config)

    def inspect_data(self) -> dict[str, Any]:
        data = self.prepare_data()
        return {
            "dataset": self.config.data.dataset,
            "shape": list(data.series.values.shape),
            "feature_names": list(data.series.feature_names),
            "units": list(data.series.units),
            "sampling_seconds": data.series.sampling_seconds,
            "graph_source": data.graph_source,
            "graph_directed": data.graph.directed,
            "graph_edges": int(data.graph.edge_weight.size),
            "clusters": data.hierarchy.num_clusters,
            "calendar_dim": data.calendar_dim,
            "event_dim": data.event_dim,
            "edge_attr_dim": self.config.model.edge_attr_dim,
            "physics_ready": data.segment_volume is not None,
            "split": asdict(data.split),
        }

    def _manifest(self) -> None:
        output = Path(self.config.training.output_dir)
        manifest = build_reproducibility_manifest(self.config, self.workspace)
        write_manifest(output / "manifest.json", manifest)

    def smoke_test(self) -> dict[str, Any]:
        self._manifest()
        data = self.prepare_data()
        model, trainer, loaders = build_v3_runtime(self.config, data)
        if self.config.training.pretrained_encoder_path:
            load_report = load_pretrained_encoder(
                model, self.config.training.pretrained_encoder_path
            )
        else:
            load_report = None
        batch = next(iter(loaders[0]))
        graph_slice = trainer.graph_slices[0]
        selected = trainer._select_batch(batch, graph_slice)
        steps = min(2, self.config.data.forecast_steps)
        trainer.optimizer.zero_grad(set_to_none=True)
        distribution = trainer._forecast(selected, graph_slice, steps)
        components = trainer._loss(distribution, selected, graph_slice, steps, train=True)
        components.total.backward()
        result = {
            "device": str(trainer.device),
            "input_shape": list(selected[0].shape),
            "mean_shape": list(distribution.mean.shape),
            "mixture_shape": list(distribution.means.shape),
            "loss": components.detached(),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "graph_partitions": len(trainer.graph_slices),
            "pretrained_encoder": load_report,
        }
        del model, trainer, loaders, batch, data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def train(self, resume_checkpoint: str | Path | None = None) -> V3TrainingResult:
        self._manifest()
        data = self.prepare_data()
        model, trainer, loaders = build_v3_runtime(self.config, data)
        if self.config.training.pretrained_encoder_path:
            report = load_pretrained_encoder(model, self.config.training.pretrained_encoder_path)
            target = Path(self.config.training.output_dir) / "pretrained_load_report.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
        result = trainer.fit(*loaders, resume_checkpoint=resume_checkpoint)
        del model, trainer, loaders, data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def tune_validation_only(
        self, resume_checkpoint: str | Path | None = None
    ) -> V3ValidationResult:
        self._manifest()
        data = self.prepare_data()
        model, trainer, loaders = build_v3_runtime(self.config, data)
        if self.config.training.pretrained_encoder_path:
            load_pretrained_encoder(model, self.config.training.pretrained_encoder_path)
        return trainer.fit_validation_only(
            loaders[0], loaders[1], resume_checkpoint=resume_checkpoint
        )

    def evaluate(self, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
        data = self.prepare_data()
        model, trainer, loaders = build_v3_runtime(self.config, data)
        path = Path(checkpoint_path or Path(self.config.training.output_dir) / "best_model_v3.pt")
        saved = torch.load(path, map_location=trainer.device, weights_only=False)
        if int(saved.get("architecture_version", -1)) != model.architecture_version:
            raise ValueError("checkpoint is not a DAGNN + PI-STNet v3 checkpoint")
        model.load_state_dict(saved["model"])
        loss, metrics, windows = trainer.evaluate(loaders[2])
        return {
            "checkpoint": str(path),
            "epoch": int(saved["epoch"]),
            "validation_loss": float(saved["validation_loss"]),
            "test_loss": loss,
            "test_metrics": metrics,
            "window_counts": {f"h{key + 1}": int(value.size) for key, value in windows.items()},
        }
