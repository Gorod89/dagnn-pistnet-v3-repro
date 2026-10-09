from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ExperimentConfig
from ..data.pipeline import StandardScaler
from ..metrics import MetricAccumulator
from ..physics import NoPhysicsConstraint, PhysicsConstraint


def build_graph_partitions(
    num_nodes: int,
    edge_index: torch.Tensor,
    max_nodes: int,
    seed: int,
) -> list[torch.Tensor]:
    """Create deterministic locality-preserving graph partitions for node mini-batches."""

    if max_nodes >= num_nodes:
        return [torch.arange(num_nodes, dtype=torch.long)]
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    edges = edge_index.detach().to(device="cpu", dtype=torch.long).numpy()
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in zip(edges[0], edges[1], strict=True):
        source_int, target_int = int(source), int(target)
        if source_int != target_int:
            neighbors[source_int].append(target_int)
            neighbors[target_int].append(source_int)

    rng = np.random.default_rng(seed)
    order = rng.permutation(num_nodes).tolist()
    assigned = np.zeros(num_nodes, dtype=bool)
    cursor = 0
    partitions: list[torch.Tensor] = []
    while not assigned.all():
        while cursor < num_nodes and assigned[order[cursor]]:
            cursor += 1
        queue: deque[int] = deque([order[cursor]])
        nodes: list[int] = []
        queued = np.zeros(num_nodes, dtype=bool)
        queued[order[cursor]] = True
        while len(nodes) < max_nodes:
            if not queue:
                while cursor < num_nodes and assigned[order[cursor]]:
                    cursor += 1
                if cursor >= num_nodes:
                    break
                queue.append(order[cursor])
                queued[order[cursor]] = True
            node = queue.popleft()
            if assigned[node]:
                continue
            assigned[node] = True
            nodes.append(node)
            for neighbor in neighbors[node]:
                if not assigned[neighbor] and not queued[neighbor]:
                    queue.append(neighbor)
                    queued[neighbor] = True
        partitions.append(torch.tensor(nodes, dtype=torch.long))
    return partitions


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    test_loss: float
    test_metrics: dict[str, dict[str, dict[str, float]]]
    checkpoint_path: str


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        scaler: StandardScaler,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        physics_constraint: PhysicsConstraint | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.scaler = scaler
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.physics = physics_constraint or NoPhysicsConstraint()
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(config.training.epochs, 1)
        )
        amp_enabled = config.training.amp and self.device.type == "cuda"
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.graph_partitions = self._prepare_graph_partitions()

    def _prepare_graph_partitions(
        self,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None:
        max_nodes = self.config.training.node_batch_size
        num_nodes = int(getattr(self.model, "num_nodes", 0))
        if max_nodes is None or max_nodes >= num_nodes:
            return None
        node_partitions = build_graph_partitions(
            num_nodes=num_nodes,
            edge_index=self.edge_index,
            max_nodes=max_nodes,
            seed=self.config.training.seed,
        )
        source = self.edge_index[0].detach().cpu().numpy()
        target = self.edge_index[1].detach().cpu().numpy()
        weights = self.edge_weight.detach().cpu().numpy()
        prepared: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for nodes_cpu in node_partitions:
            mapping = np.full(num_nodes, -1, dtype=np.int64)
            mapping[nodes_cpu.numpy()] = np.arange(nodes_cpu.numel())
            mask = (mapping[source] >= 0) & (mapping[target] >= 0)
            local_edges = np.stack([mapping[source[mask]], mapping[target[mask]]], axis=0)
            prepared.append(
                (
                    nodes_cpu,
                    nodes_cpu.to(self.device),
                    torch.as_tensor(local_edges, dtype=torch.long, device=self.device),
                    torch.as_tensor(weights[mask], dtype=torch.float32, device=self.device),
                )
            )
        return prepared

    def _loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.config.training.loss == "mse":
            data_loss = torch.mean((prediction - target) ** 2)
        else:
            original_target = self.scaler.inverse_tensor(target)
            mask = torch.isfinite(original_target) & (
                original_target.abs() > self.config.training.null_threshold
            )
            if mask.any():
                data_loss = (prediction - target).abs()[mask].mean()
            else:
                # Some traffic datasets contain complete sensor outages. Keep a
                # differentiable zero so the batch is ignored instead of being
                # learned as a real all-zero traffic state.
                data_loss = prediction.sum() * 0.0
        physics_prediction = self.scaler.inverse_tensor(prediction)
        physics_loss = self.physics(
            physics_prediction,
            self.edge_index if edge_index is None else edge_index,
            self.edge_weight if edge_weight is None else edge_weight,
        )
        total = data_loss + self.config.physics.weight * physics_loss
        return total, {
            "data": float(data_loss.detach()),
            "physics": float(physics_loss.detach()),
        }

    def _run_epoch(
        self, loader: DataLoader, train: bool, teacher_forcing_ratio: float = 0.0
    ) -> float:
        self.model.train(train)
        loss_sum = 0.0
        sample_count = 0
        progress = tqdm(
            loader,
            desc="train" if train else "validation",
            leave=False,
            disable=not self.config.training.show_batch_progress,
        )
        partition_offset = (
            random.randrange(len(self.graph_partitions)) if train and self.graph_partitions else 0
        )
        for batch_index, (x, target, history_calendar, future_calendar) in enumerate(progress):
            edge_index = self.edge_index
            edge_weight = self.edge_weight
            node_ids = None
            if self.graph_partitions:
                partition = self.graph_partitions[
                    (batch_index + partition_offset) % len(self.graph_partitions)
                ]
                nodes_cpu, node_ids, edge_index, edge_weight = partition
                x = x.index_select(2, nodes_cpu)
                target = target.index_select(2, nodes_cpu)
            x = x.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            history_calendar = history_calendar.to(self.device, non_blocking=True)
            future_calendar = future_calendar.to(self.device, non_blocking=True)
            batch_size = x.shape[0]
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with (
                torch.set_grad_enabled(train),
                torch.autocast(
                    device_type=self.device.type,
                    enabled=self.grad_scaler.is_enabled(),
                ),
            ):
                prediction = self.model(
                    x,
                    edge_index,
                    edge_weight,
                    history_calendar=history_calendar,
                    future_calendar=future_calendar,
                    teacher_forcing_target=target if train else None,
                    teacher_forcing_ratio=teacher_forcing_ratio if train else 0.0,
                    node_ids=node_ids,
                )
                loss, components = self._loss(
                    prediction, target, edge_index=edge_index, edge_weight=edge_weight
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {components}")
            if train:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip_norm
                )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if sample_count == 0:
            raise ValueError("empty dataloader")
        return loss_sum / sample_count

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, dict[str, Any]]:
        self.model.eval()
        horizon_candidates = (0, 2, 5, 11)
        horizon_indices = tuple(
            index for index in horizon_candidates if index < self.config.data.forecast_steps
        )
        accumulator = MetricAccumulator(
            feature_names=self.config.data.feature_names,
            horizon_indices=horizon_indices,
        )
        loss_sum = 0.0
        sample_count = 0
        for x, target, history_calendar, future_calendar in tqdm(
            loader,
            desc="test",
            leave=False,
            disable=not self.config.training.show_batch_progress,
        ):
            x = x.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            history_calendar = history_calendar.to(self.device, non_blocking=True)
            future_calendar = future_calendar.to(self.device, non_blocking=True)
            prediction = self.model(
                x,
                self.edge_index,
                self.edge_weight,
                history_calendar=history_calendar,
                future_calendar=future_calendar,
            )
            loss, _ = self._loss(prediction, target)
            batch_size = x.shape[0]
            loss_sum += float(loss) * batch_size
            sample_count += batch_size
            accumulator.update(
                self.scaler.inverse_tensor(prediction),
                self.scaler.inverse_tensor(target),
            )
        if sample_count == 0:
            raise ValueError("empty test dataloader")
        step_minutes = max(1, self.config.data.sampling_seconds // 60)
        return loss_sum / sample_count, accumulator.compute(step_minutes=step_minutes)

    def _save_checkpoint(self, path: Path, epoch: int, validation_loss: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "validation_loss": validation_loss,
                "scaler": self.scaler.state_dict(),
                "config": asdict(self.config),
                "torch_version": torch.__version__,
                "architecture_version": int(getattr(self.model, "architecture_version", 1)),
            },
            path,
        )

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
    ) -> TrainingResult:
        output_dir = Path(self.config.training.output_dir)
        checkpoint_path = output_dir / "best_model.pt"
        best_loss = math.inf
        best_epoch = -1
        patience = 0
        history: list[dict[str, float | int]] = []

        for epoch in range(self.config.training.epochs):
            denominator = max(self.config.training.epochs - 1, 1)
            schedule_progress = epoch / denominator
            teacher_forcing_ratio = (
                self.config.training.teacher_forcing_start
                + (
                    self.config.training.teacher_forcing_end
                    - self.config.training.teacher_forcing_start
                )
                * schedule_progress
            )
            train_loss = self._run_epoch(
                train_loader,
                train=True,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            validation_loss = self._run_epoch(validation_loader, train=False)
            self.scheduler.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "teacher_forcing_ratio": teacher_forcing_ratio,
                }
            )
            print(
                f"epoch={epoch + 1}/{self.config.training.epochs} "
                f"train_loss={train_loss:.6f} val_loss={validation_loss:.6f} "
                f"lr={self.optimizer.param_groups[0]['lr']:.8f} "
                f"teacher_forcing={teacher_forcing_ratio:.3f}",
                flush=True,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch + 1
                patience = 0
                self._save_checkpoint(checkpoint_path, best_epoch, best_loss)
            else:
                patience += 1
                if patience >= self.config.training.patience:
                    break

        if best_epoch < 0:
            raise RuntimeError("training did not produce a checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        test_loss, test_metrics = self.evaluate(test_loader)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "history.json").open("w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)
        result = TrainingResult(
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            test_loss=test_loss,
            test_metrics=test_metrics,
            checkpoint_path=str(checkpoint_path),
        )
        with (output_dir / "result.json").open("w", encoding="utf-8") as stream:
            json.dump(asdict(result), stream, ensure_ascii=False, indent=2)
        return result
