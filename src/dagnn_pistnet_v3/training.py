from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dagnn_pistnet.training.trainer import build_graph_partitions, seed_everything

from .config import V3ExperimentConfig
from .data import PreparedV3Data
from .graph import induced_coarse_graph
from .losses import V3LossComponents, gaussian_mixture_nll
from .metrics import PublicationMetricAccumulator, rescale_distribution
from .model import DAGNNPISTNetV3, ForecastDistribution


@dataclass(frozen=True)
class GraphSlice:
    nodes_cpu: torch.Tensor
    nodes: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    edge_attributes: torch.Tensor | None
    cluster_index: torch.Tensor
    coarse_edge_index: torch.Tensor
    coarse_edge_weight: torch.Tensor
    segment_volume: torch.Tensor | None


@dataclass(frozen=True)
class V3TrainingResult:
    best_epoch: int
    best_validation_loss: float
    test_loss: float
    test_metrics: dict[str, Any]
    checkpoint_path: str
    window_errors_path: str
    parameter_count: int
    training_wall_seconds: float
    inference_latency_ms: float
    inference_sensor_windows_per_second: float
    peak_accelerator_memory_mb: float


@dataclass(frozen=True)
class V3ValidationResult:
    best_epoch: int
    best_validation_loss: float
    checkpoint_path: str


class V3Trainer:
    def __init__(
        self,
        model: DAGNNPISTNetV3,
        config: V3ExperimentConfig,
        data: PreparedV3Data,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.data = data
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.training.epochs
        )
        enabled = config.training.amp and device.type == "cuda"
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        self.graph_slices = self._build_graph_slices()

    def _build_graph_slices(self) -> list[GraphSlice]:
        graph = self.data.graph
        edge_index_cpu = torch.as_tensor(graph.edge_index, dtype=torch.long)
        limit = self.config.training.node_batch_size or graph.num_nodes
        partitions = build_graph_partitions(
            graph.num_nodes, edge_index_cpu, limit, self.config.training.seed
        )
        source = graph.edge_index[0]
        target = graph.edge_index[1]
        result: list[GraphSlice] = []
        for nodes_cpu in partitions:
            mapping = np.full(graph.num_nodes, -1, dtype=np.int64)
            mapping[nodes_cpu.numpy()] = np.arange(nodes_cpu.numel())
            mask = (mapping[source] >= 0) & (mapping[target] >= 0)
            local_edges = np.stack([mapping[source[mask]], mapping[target[mask]]], axis=0)
            local_edge_index = torch.as_tensor(local_edges, dtype=torch.long, device=self.device)
            local_edge_weight = torch.as_tensor(
                graph.edge_weight[mask], dtype=torch.float32, device=self.device
            )
            global_clusters = self.data.hierarchy.cluster_index.index_select(0, nodes_cpu)
            _, local_clusters = torch.unique(global_clusters, sorted=True, return_inverse=True)
            local_clusters = local_clusters.to(self.device)
            coarse_index, coarse_weight = induced_coarse_graph(
                local_clusters, local_edge_index, local_edge_weight
            )
            edge_attributes = (
                self.data.edge_attributes[torch.as_tensor(mask)].to(self.device)
                if self.data.edge_attributes is not None
                else None
            )
            volume = (
                self.data.segment_volume.index_select(0, nodes_cpu).to(self.device)
                if self.data.segment_volume is not None
                else None
            )
            result.append(
                GraphSlice(
                    nodes_cpu=nodes_cpu,
                    nodes=nodes_cpu.to(self.device),
                    edge_index=local_edge_index,
                    edge_weight=local_edge_weight,
                    edge_attributes=edge_attributes,
                    cluster_index=local_clusters,
                    coarse_edge_index=coarse_index,
                    coarse_edge_weight=coarse_weight,
                    segment_volume=volume,
                )
            )
        return result

    def _active_steps(self, epoch: int) -> int:
        start = self.config.training.curriculum_start_steps
        full_epoch = max(
            1,
            round(
                self.config.training.epochs
                * self.config.training.curriculum_full_fraction
            ),
        )
        progress = min(epoch / full_epoch, 1.0)
        span = self.config.data.forecast_steps - start
        return min(self.config.data.forecast_steps, start + round(span * progress))

    def _select_batch(
        self,
        batch: tuple[torch.Tensor, ...],
        graph_slice: GraphSlice,
    ) -> tuple[torch.Tensor, ...]:
        x, target, history_calendar, future_calendar, events, boundary = batch
        return (
            x.index_select(2, graph_slice.nodes_cpu).to(self.device, non_blocking=True),
            target.index_select(2, graph_slice.nodes_cpu).to(self.device, non_blocking=True),
            history_calendar.to(self.device, non_blocking=True),
            future_calendar.to(self.device, non_blocking=True),
            events.index_select(2, graph_slice.nodes_cpu).to(self.device, non_blocking=True),
            boundary.index_select(2, graph_slice.nodes_cpu).to(self.device, non_blocking=True),
        )

    def _forecast(
        self,
        selected: tuple[torch.Tensor, ...],
        graph_slice: GraphSlice,
        steps: int,
    ) -> ForecastDistribution:
        x, _, history_calendar, future_calendar, events, boundary = selected
        return self.model(
            x,
            graph_slice.edge_index,
            graph_slice.edge_weight,
            history_calendar=history_calendar,
            future_calendar=future_calendar,
            cluster_index=graph_slice.cluster_index,
            coarse_edge_index=graph_slice.coarse_edge_index,
            coarse_edge_weight=graph_slice.coarse_edge_weight,
            edge_attr=graph_slice.edge_attributes,
            future_events=events,
            segment_volume=graph_slice.segment_volume,
            boundary_flow=boundary,
            steps=steps,
        )

    def _semigroup_loss(
        self,
        distribution: ForecastDistribution,
        selected: tuple[torch.Tensor, ...],
        graph_slice: GraphSlice,
        steps: int,
    ) -> torch.Tensor:
        if steps < 2 or self.config.training.semigroup_weight == 0:
            return distribution.mean.sum() * 0.0
        x, _, history_calendar, future_calendar, events, boundary = selected
        half = steps // 2
        predicted = distribution.mean[:, :half]
        predicted_inputs = x.new_zeros(
            x.shape[0], half, x.shape[2], self.model.input_dim
        )
        predicted_inputs[..., : self.model.output_dim] = predicted
        rolled_x = torch.cat([x[:, half:], predicted_inputs], dim=1)
        rolled_history_calendar = torch.cat(
            [history_calendar[:, half:], future_calendar[:, :half]], dim=1
        )
        composed = self.model(
            rolled_x,
            graph_slice.edge_index,
            graph_slice.edge_weight,
            history_calendar=rolled_history_calendar,
            future_calendar=future_calendar[:, half:steps],
            cluster_index=graph_slice.cluster_index,
            coarse_edge_index=graph_slice.coarse_edge_index,
            coarse_edge_weight=graph_slice.coarse_edge_weight,
            edge_attr=graph_slice.edge_attributes,
            future_events=events[:, half:steps],
            segment_volume=graph_slice.segment_volume,
            boundary_flow=boundary[:, half:steps],
            steps=steps - half,
        )
        return (composed.mean - distribution.mean[:, half:steps]).abs().mean()

    def _loss(
        self,
        distribution: ForecastDistribution,
        selected: tuple[torch.Tensor, ...],
        graph_slice: GraphSlice,
        steps: int,
        train: bool,
    ) -> V3LossComponents:
        x, target, history_calendar, _, _, _ = selected
        target = target[:, :steps, :, : self.model.output_dim]
        original = self.data.scaler.inverse_tensor(target)
        mask = torch.isfinite(original) & (original.abs() > 1e-5)
        nll = gaussian_mixture_nll(distribution, target, mask)
        mae = (distribution.mean - target).abs()[mask].mean() if mask.any() else nll * 0.0
        semigroup = self._semigroup_loss(distribution, selected, graph_slice, steps)
        scale_regularization = distribution.scales.log().square().mean()

        reconstruction = nll * 0.0
        if train and self.config.training.masked_pretrain_weight:
            random_mask = torch.rand_like(x) < self.config.training.mask_ratio
            reconstructed = self.model.masked_reconstruction(x, history_calendar, random_mask)
            reconstruction = (
                (reconstructed - x).abs()[random_mask].mean()
                if random_mask.any()
                else reconstructed.sum() * 0.0
            )
        inverse_mean = self.data.scaler.inverse_tensor(distribution.mean)
        nonnegative = torch.relu(-inverse_mean).mean()
        total = (
            self.config.training.nll_weight * nll
            + self.config.training.mae_weight * mae
            + self.config.training.semigroup_weight * semigroup
            + self.config.training.scale_regularization_weight * scale_regularization
            + self.config.training.masked_pretrain_weight * reconstruction
            + self.config.physics.nonnegative_weight * nonnegative
        )
        return V3LossComponents(
            total=total,
            nll=nll,
            mae=mae,
            semigroup=semigroup,
            scale_regularization=scale_regularization,
            reconstruction=reconstruction,
            nonnegative=nonnegative,
        )

    def _run_epoch(self, loader: DataLoader, epoch: int, train: bool) -> float:
        self.model.train(train)
        active_steps = self._active_steps(epoch) if train else self.config.data.forecast_steps
        loss_sum = 0.0
        count = 0
        progress = tqdm(
            loader,
            desc="v3 train" if train else "v3 validation",
            leave=False,
            disable=not self.config.training.show_batch_progress,
        )
        offset = random.randrange(len(self.graph_slices)) if train else 0
        for batch_index, batch in enumerate(progress):
            slices = (
                [self.graph_slices[(batch_index + offset) % len(self.graph_slices)]]
                if train
                else self.graph_slices
            )
            for graph_slice in slices:
                selected = self._select_batch(batch, graph_slice)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                with (
                    torch.set_grad_enabled(train),
                    torch.autocast(
                        device_type=self.device.type,
                        enabled=self.grad_scaler.is_enabled(),
                    ),
                ):
                    distribution = self._forecast(selected, graph_slice, active_steps)
                    components = self._loss(
                        distribution, selected, graph_slice, active_steps, train=train
                    )
                if not torch.isfinite(components.total):
                    raise FloatingPointError(f"non-finite v3 loss: {components.detached()}")
                if train:
                    self.grad_scaler.scale(components.total).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip_norm
                    )
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                samples = selected[0].shape[0] * selected[0].shape[2]
                loss_sum += float(components.total.detach()) * samples
                count += samples
                progress.set_postfix(loss=f"{float(components.total.detach()):.4f}")
        if count == 0:
            raise ValueError("empty v3 dataloader")
        return loss_sum / count

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader
    ) -> tuple[float, dict[str, Any], dict[int, np.ndarray]]:
        self.model.eval()
        candidates = (0, 2, 5, 11)
        horizons = tuple(index for index in candidates if index < self.config.data.forecast_steps)
        accumulator = PublicationMetricAccumulator(
            self.data.series.feature_names[: self.model.output_dim],
            horizons,
            self.config.data.sampling_seconds,
        )
        loss_sum = 0.0
        count = 0
        mean = torch.as_tensor(self.data.scaler.mean, device=self.device)
        std = torch.as_tensor(self.data.scaler.std, device=self.device)
        for batch in tqdm(
            loader,
            desc="v3 test",
            leave=False,
            disable=not self.config.training.show_batch_progress,
        ):
            for graph_slice in self.graph_slices:
                selected = self._select_batch(batch, graph_slice)
                distribution = self._forecast(
                    selected, graph_slice, self.config.data.forecast_steps
                )
                components = self._loss(
                    distribution,
                    selected,
                    graph_slice,
                    self.config.data.forecast_steps,
                    train=False,
                )
                target = selected[1][..., : self.model.output_dim]
                original_distribution = rescale_distribution(
                    distribution,
                    mean[..., : self.model.output_dim],
                    std[..., : self.model.output_dim],
                )
                original_target = self.data.scaler.inverse_tensor(target)
                accumulator.update(original_distribution, original_target)
                samples = target.shape[0] * target.shape[2]
                loss_sum += float(components.total) * samples
                count += samples
        metrics, windows = accumulator.compute()
        return loss_sum / max(count, 1), metrics, windows

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        validation_loss: float,
        *,
        best_epoch: int | None = None,
        best_validation_loss: float | None = None,
        patience: int = 0,
        history: list[dict[str, float | int]] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture_version": self.model.architecture_version,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "grad_scaler": self.grad_scaler.state_dict(),
                "epoch": epoch,
                "validation_loss": validation_loss,
                "best_epoch": best_epoch if best_epoch is not None else epoch,
                "best_validation_loss": (
                    best_validation_loss
                    if best_validation_loss is not None
                    else validation_loss
                ),
                "patience": patience,
                "history": history or [],
                "config": asdict(self.config),
                "scaler": self.data.scaler.state_dict(),
                "torch_version": torch.__version__,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                },
            },
            path,
        )

    def _restore_checkpoint(self, path: str | Path) -> tuple[int, int, float, int, list, dict]:
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if int(saved.get("architecture_version", -1)) != self.model.architecture_version:
            raise ValueError("checkpoint is not a DAGNN + PI-STNet v3 checkpoint")
        self.model.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
        epoch = int(saved["epoch"])
        restored: list[str] = ["model", "optimizer"]
        limitations: list[str] = []
        if "scheduler" in saved:
            self.scheduler.load_state_dict(saved["scheduler"])
            restored.append("scheduler")
        else:
            # Legacy checkpoints contain the optimizer LR after scheduler.step().
            # Advancing only the scheduler counters makes the next step epoch+1
            # without changing that restored LR a second time.
            self.scheduler.last_epoch = epoch
            self.scheduler._step_count = epoch + 1  # noqa: SLF001
            self.scheduler._last_lr = [  # noqa: SLF001
                group["lr"] for group in self.optimizer.param_groups
            ]
            limitations.append("scheduler state reconstructed from epoch and optimizer LR")
        if "grad_scaler" in saved:
            self.grad_scaler.load_state_dict(saved["grad_scaler"])
            restored.append("grad_scaler")
        elif self.grad_scaler.is_enabled():
            # A conservative scale avoids overflow when moving a legacy AMP run
            # to a materially faster/different accelerator.
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1.0)
            limitations.append("AMP GradScaler reset to conservative scale=1.0")
        rng = saved.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_cpu"].cpu())
            if self.device.type == "cuda" and rng.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])
            restored.append("rng_state")
        else:
            limitations.append("RNG state unavailable in legacy checkpoint")
        history = list(saved.get("history", []))
        if not history:
            limitations.append("pre-resume epoch history unavailable in legacy checkpoint")
        best_epoch = int(saved.get("best_epoch", epoch))
        best_loss = float(saved.get("best_validation_loss", saved["validation_loss"]))
        patience = int(saved.get("patience", 0))
        report = {
            "source_checkpoint": str(Path(path).resolve()),
            "source_epoch": epoch,
            "restored": restored,
            "limitations": limitations,
            "exact_continuation": not limitations,
        }
        return epoch, best_epoch, best_loss, patience, history, report

    @staticmethod
    def _write_resume_report(output: Path, report: dict[str, Any]) -> None:
        output.mkdir(parents=True, exist_ok=True)
        with (output / "resume_report.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        resume_checkpoint: str | Path | None = None,
    ) -> V3TrainingResult:
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        output = Path(self.config.training.output_dir)
        checkpoint = output / "best_model_v3.pt"
        best_loss, best_epoch, patience = math.inf, -1, 0
        history: list[dict[str, float | int]] = []
        start_epoch = 0
        if resume_checkpoint is not None:
            start_epoch, best_epoch, best_loss, patience, history, report = (
                self._restore_checkpoint(resume_checkpoint)
            )
            self._write_resume_report(output, report)
        last_checkpoint = output / "last_training_state_v3.pt"
        for epoch in range(start_epoch, self.config.training.epochs):
            train_loss = self._run_epoch(train_loader, epoch, train=True)
            validation_loss = self._run_epoch(validation_loader, epoch, train=False)
            self.scheduler.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "active_steps": self._active_steps(epoch),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"v3 epoch={epoch + 1}/{self.config.training.epochs} "
                f"steps={self._active_steps(epoch)} train={train_loss:.6f} "
                f"val={validation_loss:.6f}",
                flush=True,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch + 1
                patience = 0
                self._save_checkpoint(
                    checkpoint,
                    best_epoch,
                    best_loss,
                    best_epoch=best_epoch,
                    best_validation_loss=best_loss,
                    patience=patience,
                    history=history,
                )
            else:
                patience += 1
            self._save_checkpoint(
                last_checkpoint,
                epoch + 1,
                validation_loss,
                best_epoch=best_epoch,
                best_validation_loss=best_loss,
                patience=patience,
                history=history,
            )
            if patience >= self.config.training.patience:
                break
        if best_epoch < 0:
            raise RuntimeError("v3 training did not produce a checkpoint")
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(saved["model"])
        test_loss, metrics, windows = self.evaluate(test_loader)
        latency_ms, throughput = self._benchmark_inference(test_loader)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "history.json").open("w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)
        windows_path = output / "window_errors.npz"
        np.savez_compressed(
            windows_path, **{f"h{key + 1}": value for key, value in windows.items()}
        )
        result = V3TrainingResult(
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            test_loss=test_loss,
            test_metrics=metrics,
            checkpoint_path=str(checkpoint),
            window_errors_path=str(windows_path),
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            training_wall_seconds=time.perf_counter() - started,
            inference_latency_ms=latency_ms,
            inference_sensor_windows_per_second=throughput,
            peak_accelerator_memory_mb=(
                torch.cuda.max_memory_allocated(self.device) / (1024**2)
                if self.device.type == "cuda"
                else 0.0
            ),
        )
        with (output / "result.json").open("w", encoding="utf-8") as stream:
            json.dump(asdict(result), stream, ensure_ascii=False, indent=2)
        return result

    def fit_validation_only(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        resume_checkpoint: str | Path | None = None,
    ) -> V3ValidationResult:
        """Hyperparameter-selection loop that never constructs a test result."""

        output = Path(self.config.training.output_dir)
        checkpoint = output / "best_validation_model_v3.pt"
        best_loss, best_epoch, patience = math.inf, -1, 0
        history: list[dict[str, float | int]] = []
        start_epoch = 0
        if resume_checkpoint is not None:
            start_epoch, best_epoch, best_loss, patience, history, report = (
                self._restore_checkpoint(resume_checkpoint)
            )
            self._write_resume_report(output, report)
        last_checkpoint = output / "last_validation_state_v3.pt"
        for epoch in range(start_epoch, self.config.training.epochs):
            train_loss = self._run_epoch(train_loader, epoch, train=True)
            validation_loss = self._run_epoch(validation_loader, epoch, train=False)
            self.scheduler.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "active_steps": self._active_steps(epoch),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            print(
                f"[validation] epoch={epoch + 1}/{self.config.training.epochs} "
                f"train={train_loss:.6f} validation={validation_loss:.6f}",
                flush=True,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch + 1
                patience = 0
                self._save_checkpoint(
                    checkpoint,
                    best_epoch,
                    best_loss,
                    best_epoch=best_epoch,
                    best_validation_loss=best_loss,
                    patience=patience,
                    history=history,
                )
            else:
                patience += 1
            self._save_checkpoint(
                last_checkpoint,
                epoch + 1,
                validation_loss,
                best_epoch=best_epoch,
                best_validation_loss=best_loss,
                patience=patience,
                history=history,
            )
            if patience >= self.config.training.patience:
                break
        if best_epoch < 0:
            raise RuntimeError("validation-only tuning did not produce a checkpoint")
        output.mkdir(parents=True, exist_ok=True)
        with (output / "validation_history.json").open("w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)
        result = V3ValidationResult(best_epoch, best_loss, str(checkpoint))
        with (output / "validation_result.json").open("w", encoding="utf-8") as stream:
            json.dump(asdict(result), stream, ensure_ascii=False, indent=2)
        return result

    @torch.no_grad()
    def _benchmark_inference(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        batch = next(iter(loader))
        graph_slice = self.graph_slices[0]
        selected = self._select_batch(batch, graph_slice)
        for _ in range(2):
            self._forecast(selected, graph_slice, self.config.data.forecast_steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        repeats = 5
        started = time.perf_counter()
        for _ in range(repeats):
            self._forecast(selected, graph_slice, self.config.data.forecast_steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = (time.perf_counter() - started) / repeats
        sensor_windows = selected[0].shape[0] * selected[0].shape[2]
        return seconds * 1000.0, sensor_windows / max(seconds, 1e-9)


def build_loaders(
    config: V3ExperimentConfig,
    data: PreparedV3Data,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(data.train_dataset, shuffle=True, **common),
        DataLoader(data.validation_dataset, shuffle=False, **common),
        DataLoader(
            data.test_dataset,
            shuffle=False,
            **{
                **common,
                "batch_size": config.training.evaluation_batch_size
                or config.training.batch_size,
            },
        ),
    )


def build_v3_runtime(
    config: V3ExperimentConfig,
    data: PreparedV3Data,
) -> tuple[DAGNNPISTNetV3, V3Trainer, tuple[DataLoader, DataLoader, DataLoader]]:
    if config.training.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(config.training.seed)
    torch.use_deterministic_algorithms(
        config.training.deterministic_algorithms, warn_only=True
    )
    requested = config.training.device.lower()
    device = torch.device(
        "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu"
        if requested == "auto"
        else requested
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = DAGNNPISTNetV3(
        input_dim=data.series.num_features,
        output_dim=data.series.num_features,
        num_nodes=data.series.num_nodes,
        forecast_steps=config.data.forecast_steps,
        calendar_dim=data.calendar_dim,
        event_dim=data.event_dim,
        config=config.model,
        physics=config.physics,
    ).to(device)
    trainer = V3Trainer(model, config, data, device)
    return model, trainer, build_loaders(config, data, device)
