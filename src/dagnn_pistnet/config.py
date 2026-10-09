from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    data_path: str
    adjacency_path: str | None = None
    feature_names: tuple[str, ...] = ("value",)
    units: tuple[str, ...] = ("unknown",)
    history_steps: int = 12
    forecast_steps: int = 12
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    graph_top_k: int = 10
    graph_threshold: float = 0.0
    graph_directed: bool = False
    sampling_seconds: int = 300
    use_calendar_features: bool = False
    start_timestamp: str | None = None
    missing_value_policy: str = "error"

    def __post_init__(self) -> None:
        if self.history_steps < 1 or self.forecast_steps < 1:
            raise ValueError("history_steps and forecast_steps must be positive")
        if not 0 < self.train_ratio < 1 or not 0 <= self.val_ratio < 1:
            raise ValueError("invalid chronological split ratios")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")
        if len(self.feature_names) != len(self.units):
            raise ValueError("feature_names and units must have equal lengths")
        if self.sampling_seconds < 1:
            raise ValueError("sampling_seconds must be positive")
        if self.missing_value_policy not in {"error", "zero"}:
            raise ValueError("missing_value_policy must be 'error' or 'zero'")


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64
    lstm_layers: int = 2
    adaptive_embedding_dim: int = 16
    adaptive_top_k: int = 10
    diffusion_steps: int = 2
    dropout: float = 0.1
    adaptive_candidate_graph: bool = False

    def __post_init__(self) -> None:
        if self.diffusion_steps < 1:
            raise ValueError("diffusion_steps must be positive")


@dataclass(frozen=True)
class PhysicsConfig:
    kind: str = "none"
    weight: float = 0.0
    dt_seconds: float = 300.0
    density_channel: int | None = None
    flow_channel: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    patience: int = 6
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 5.0
    loss: str = "masked_mae"
    null_threshold: float = 1e-5
    show_batch_progress: bool = False
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.0
    node_batch_size: int | None = None
    evaluation_batch_size: int | None = None
    output_dir: str = "artifacts/default"

    def __post_init__(self) -> None:
        if self.loss not in {"masked_mae", "mse"}:
            raise ValueError("loss must be 'masked_mae' or 'mse'")
        if self.null_threshold < 0:
            raise ValueError("null_threshold must be non-negative")
        if not 0 <= self.teacher_forcing_start <= 1:
            raise ValueError("teacher_forcing_start must be in [0,1]")
        if not 0 <= self.teacher_forcing_end <= 1:
            raise ValueError("teacher_forcing_end must be in [0,1]")
        if self.node_batch_size is not None and self.node_batch_size < 1:
            raise ValueError("node_batch_size must be positive")
        if self.evaluation_batch_size is not None and self.evaluation_batch_size < 1:
            raise ValueError("evaluation_batch_size must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        data_raw = dict(raw["data"])
        data_raw["feature_names"] = tuple(data_raw.get("feature_names", ("value",)))
        data_raw["units"] = tuple(data_raw.get("units", ("unknown",)))
        return cls(
            data=DataConfig(**data_raw),
            model=ModelConfig(**raw.get("model", {})),
            physics=PhysicsConfig(**raw.get("physics", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError(f"configuration must be a mapping: {config_path}")
        return cls.from_dict(raw)
