from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .baselines import evaluate_naive_baselines
from .config import ExperimentConfig
from .data import (
    ChronologicalSplit,
    GraphData,
    StandardScaler,
    TrafficSeries,
    TrafficWindowDataset,
    adjacency_to_graph,
    build_adapter,
    build_correlation_graph,
    encode_calendar_features,
)
from .models import DAGNNPISTNet
from .physics import DirectedConservationLoss, NoPhysicsConstraint, PhysicsConstraint
from .training import Trainer, TrainingResult, seed_everything


@dataclass(frozen=True)
class DataBundle:
    series: TrafficSeries
    split: ChronologicalSplit
    scaler: StandardScaler
    graph: GraphData
    graph_source: str
    calendar_dim: int
    train_dataset: TrafficWindowDataset
    validation_dataset: TrafficWindowDataset
    test_dataset: TrafficWindowDataset


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def inspect_data(self) -> dict[str, Any]:
        series = build_adapter(self.config.data).load()
        split = ChronologicalSplit.from_ratios(
            series.time_steps,
            self.config.data.train_ratio,
            self.config.data.val_ratio,
        )
        return {
            "dataset": self.config.data.dataset,
            "shape": list(series.values.shape),
            "feature_names": list(series.feature_names),
            "units": list(series.units),
            "sampling_seconds": series.sampling_seconds,
            "timestamp_start": str(series.timestamps[0]) if series.timestamps is not None else None,
            "timestamp_end": str(series.timestamps[-1]) if series.timestamps is not None else None,
            "calendar_features_enabled": self.config.data.use_calendar_features,
            "minimum": series.values.min(axis=(0, 1)).tolist(),
            "maximum": series.values.max(axis=(0, 1)).tolist(),
            "mean": series.values.mean(axis=(0, 1)).tolist(),
            "zero_fraction": (series.values == 0).mean(axis=(0, 1)).tolist(),
            "adjacency_shape": (
                list(series.adjacency.shape) if series.adjacency is not None else None
            ),
            "split": asdict(split),
        }

    def prepare_data(self) -> DataBundle:
        series = build_adapter(self.config.data).load()
        split = ChronologicalSplit.from_ratios(
            series.time_steps,
            self.config.data.train_ratio,
            self.config.data.val_ratio,
        )
        scaler = StandardScaler.fit(series.values[: split.train_end])
        normalized = scaler.transform(series.values)
        calendar_features = None
        if self.config.data.use_calendar_features:
            if series.timestamps is None:
                raise ValueError(
                    "calendar features require dataset timestamps or data.start_timestamp"
                )
            calendar_features = encode_calendar_features(series.timestamps)
        calendar_dim = 0 if calendar_features is None else calendar_features.shape[1]

        if series.adjacency is not None:
            adjacency = series.adjacency
            graph_source = "dataset"
        else:
            adjacency = build_correlation_graph(
                series.values[: split.train_end],
                top_k=self.config.data.graph_top_k,
            )
            graph_source = "train-only-correlation"
        if adjacency.shape != (series.num_nodes, series.num_nodes):
            raise ValueError(
                f"graph {adjacency.shape} is incompatible with {series.num_nodes} sensors"
            )
        graph = adjacency_to_graph(
            adjacency,
            threshold=self.config.data.graph_threshold,
            add_self_loops=False,
            remove_self_loops=True,
            directed=self.config.data.graph_directed,
        )

        datasets: dict[str, TrafficWindowDataset] = {}
        for split_name in ("train", "val", "test"):
            target_start, target_end = split.target_bounds(split_name)
            datasets[split_name] = TrafficWindowDataset(
                normalized,
                history_steps=self.config.data.history_steps,
                forecast_steps=self.config.data.forecast_steps,
                target_start=target_start,
                target_end=target_end,
                calendar_features=calendar_features,
            )
        return DataBundle(
            series=series,
            split=split,
            scaler=scaler,
            graph=graph,
            graph_source=graph_source,
            calendar_dim=calendar_dim,
            train_dataset=datasets["train"],
            validation_dataset=datasets["val"],
            test_dataset=datasets["test"],
        )

    def _device(self) -> torch.device:
        requested = self.config.training.device.lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device

    def _physics(self, bundle: DataBundle) -> PhysicsConstraint:
        config = self.config.physics
        if config.kind == "none" or config.weight == 0:
            return NoPhysicsConstraint()
        if config.kind == "directed_conservation":
            if bundle.graph_source != "dataset" or not bundle.graph.directed:
                raise ValueError(
                    "directed_conservation requires an explicitly directed dataset graph; "
                    "a correlation graph is not a physical road topology"
                )
            if config.density_channel is None or config.flow_channel is None:
                raise ValueError("directed_conservation requires density_channel and flow_channel")
            feature_names = self.config.data.feature_names
            if max(config.density_channel, config.flow_channel) >= len(feature_names):
                raise ValueError("physics channels exceed declared dataset features")
            density_name = feature_names[config.density_channel].lower()
            flow_name = feature_names[config.flow_channel].lower()
            if "density" not in density_name or "flow" not in flow_name:
                raise ValueError(
                    "physics channels must be explicitly named density and flow; "
                    "derived speed-to-density targets are not accepted"
                )
            return DirectedConservationLoss(
                density_channel=config.density_channel,
                flow_channel=config.flow_channel,
                dt_seconds=config.dt_seconds,
            )
        raise ValueError(f"unsupported physics constraint: {config.kind}")

    def build_runtime(
        self, bundle: DataBundle
    ) -> tuple[DAGNNPISTNet, Trainer, tuple[DataLoader, DataLoader, DataLoader]]:
        seed_everything(self.config.training.seed)
        device = self._device()
        model = DAGNNPISTNet(
            input_dim=bundle.series.num_features,
            output_dim=bundle.series.num_features,
            num_nodes=bundle.series.num_nodes,
            forecast_steps=self.config.data.forecast_steps,
            hidden_dim=self.config.model.hidden_dim,
            lstm_layers=self.config.model.lstm_layers,
            adaptive_embedding_dim=self.config.model.adaptive_embedding_dim,
            adaptive_top_k=self.config.model.adaptive_top_k,
            diffusion_steps=self.config.model.diffusion_steps,
            calendar_dim=bundle.calendar_dim,
            dropout=self.config.model.dropout,
            adaptive_candidate_graph=self.config.model.adaptive_candidate_graph,
        ).to(device)
        edge_index, edge_weight = bundle.graph.tensors(device)
        trainer = Trainer(
            model=model,
            config=self.config,
            scaler=bundle.scaler,
            edge_index=edge_index,
            edge_weight=edge_weight,
            physics_constraint=self._physics(bundle).to(device),
        )
        train_loader_options = {
            "batch_size": self.config.training.batch_size,
            "num_workers": self.config.training.num_workers,
            "pin_memory": device.type == "cuda",
        }
        loaders = (
            DataLoader(bundle.train_dataset, shuffle=True, **train_loader_options),
            DataLoader(bundle.validation_dataset, shuffle=False, **train_loader_options),
            DataLoader(
                bundle.test_dataset,
                shuffle=False,
                **{
                    **train_loader_options,
                    "batch_size": self.config.training.evaluation_batch_size
                    or self.config.training.batch_size,
                },
            ),
        )
        return model, trainer, loaders

    def smoke_test(self) -> dict[str, Any]:
        bundle = self.prepare_data()
        model, trainer, loaders = self.build_runtime(bundle)
        train_loader, _, _ = loaders
        x, target, history_calendar, future_calendar = next(iter(train_loader))
        x = x[:1].to(trainer.device)
        target = target[:1].to(trainer.device)
        history_calendar = history_calendar[:1].to(trainer.device)
        future_calendar = future_calendar[:1].to(trainer.device)
        trainer.optimizer.zero_grad(set_to_none=True)
        prediction = model(
            x,
            trainer.edge_index,
            trainer.edge_weight,
            history_calendar=history_calendar,
            future_calendar=future_calendar,
            teacher_forcing_target=target,
            teacher_forcing_ratio=0.5,
        )
        loss, components = trainer._loss(prediction, target)
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError("smoke-test produced a non-finite loss")
        result = {
            "device": str(trainer.device),
            "input_shape": list(x.shape),
            "prediction_shape": list(prediction.shape),
            "target_shape": list(target.shape),
            "loss": float(loss.detach()),
            "loss_components": components,
            "graph_source": bundle.graph_source,
            "graph_edges": int(bundle.graph.edge_weight.size),
            "calendar_dim": bundle.calendar_dim,
        }
        del loss, prediction, x, target, history_calendar, future_calendar
        del model, trainer, loaders, train_loader, bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return result

    def baselines(self) -> dict[str, Any]:
        bundle = self.prepare_data()
        return evaluate_naive_baselines(
            test_dataset=bundle.test_dataset,
            train_values=bundle.series.values[: bundle.split.train_end],
            scaler=bundle.scaler,
            feature_names=bundle.series.feature_names,
            forecast_steps=self.config.data.forecast_steps,
            batch_size=self.config.training.batch_size,
        )

    def train(self) -> TrainingResult:
        bundle = self.prepare_data()
        model, trainer, loaders = self.build_runtime(bundle)
        result = trainer.fit(*loaders)
        del model, trainer, loaders, bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return result

    def evaluate(self, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
        bundle = self.prepare_data()
        model, trainer, loaders = self.build_runtime(bundle)
        path = Path(checkpoint_path or self.config.training.output_dir) / (
            "best_model.pt" if checkpoint_path is None else ""
        )
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
        checkpoint_version = int(checkpoint.get("architecture_version", 1))
        if checkpoint_version != model.architecture_version:
            raise ValueError(
                f"checkpoint architecture v{checkpoint_version} is incompatible with "
                f"model architecture v{model.architecture_version}"
            )
        model.load_state_dict(checkpoint["model"])
        test_loss, test_metrics = trainer.evaluate(loaders[2])
        result = {
            "checkpoint_path": str(path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "validation_loss": float(checkpoint["validation_loss"]),
            "test_loss": test_loss,
            "test_metrics": test_metrics,
        }
        del checkpoint, model, trainer, loaders, bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return result
