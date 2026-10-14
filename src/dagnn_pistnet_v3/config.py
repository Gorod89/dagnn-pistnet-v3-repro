from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dagnn_pistnet.config import DataConfig


@dataclass(frozen=True)
class V3DataConfig:
    dataset: str
    data_path: str
    adjacency_path: str | None = None
    feature_names: tuple[str, ...] = ("value",)
    units: tuple[str, ...] = ("unknown",)
    history_steps: int = 12
    forecast_steps: int = 12
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    split_policy: str = "nonoverlapping_timeline"
    graph_top_k: int = 10
    graph_threshold: float = 0.0
    graph_directed: bool = False
    sampling_seconds: int = 300
    use_calendar_features: bool = False
    start_timestamp: str | None = None
    missing_value_policy: str = "error"
    edge_attributes_path: str | None = None
    cluster_index_path: str | None = None
    event_features_path: str | None = None
    segment_volume_path: str | None = None
    boundary_flow_path: str | None = None
    primary_feature_index: int = 0

    def __post_init__(self) -> None:
        # Reuse the stable v2 validation contract.
        self.to_v2()
        if self.split_policy not in {
            "nonoverlapping_timeline",
            "official_sliding_windows",
        }:
            raise ValueError(
                "split_policy must be 'nonoverlapping_timeline' or "
                "'official_sliding_windows'"
            )
        if not 0 <= self.primary_feature_index < len(self.feature_names):
            raise ValueError("primary_feature_index is outside declared features")

    def to_v2(self) -> DataConfig:
        v2_fields = {
            key: value
            for key, value in asdict(self).items()
            if key in DataConfig.__dataclass_fields__
        }
        return DataConfig(**v2_fields)


@dataclass(frozen=True)
class V3ModelConfig:
    hidden_dim: int = 64
    temporal_layers: int = 2
    dropout: float = 0.1
    num_regimes: int = 3
    hierarchy_ratio: float = 0.08
    max_clusters: int = 256
    min_clusters: int = 2
    operator_substeps: int = 1
    operator_order: tuple[str, str, str] = ("transport", "diffusion", "source_sink")
    edge_attr_dim: int = 0
    min_scale: float = 0.01
    max_scale: float = 3.0
    use_delta_decoder: bool = True
    use_hierarchy: bool = True
    use_operator_splitting: bool = True
    use_probabilistic_head: bool = True
    use_event_jump: bool = True
    use_hard_conservation: bool = False
    use_trend_seasonal_residual: bool = True
    moving_average_window: int = 3

    def __post_init__(self) -> None:
        if self.hidden_dim < 4 or self.temporal_layers < 1:
            raise ValueError("hidden_dim >= 4 and temporal_layers >= 1 are required")
        if self.num_regimes < 1:
            raise ValueError("num_regimes must be positive")
        if not 0 < self.hierarchy_ratio <= 1:
            raise ValueError("hierarchy_ratio must be in (0, 1]")
        if self.operator_substeps < 1:
            raise ValueError("operator_substeps must be positive")
        order = tuple(self.operator_order)
        if len(order) != 3 or set(order) != {"transport", "diffusion", "source_sink"}:
            raise ValueError("operator_order must permute transport, diffusion, source_sink")
        object.__setattr__(self, "operator_order", order)
        if not 0 < self.min_scale <= self.max_scale:
            raise ValueError("invalid probabilistic scale bounds")
        if self.moving_average_window < 1 or self.moving_average_window % 2 == 0:
            raise ValueError("moving_average_window must be a positive odd number")


@dataclass(frozen=True)
class V3PhysicsConfig:
    density_channel: int | None = None
    flow_channel: int | None = None
    dt_seconds: float = 300.0
    nonnegative_weight: float = 0.0
    capacity_weight: float = 0.0

    def validate_for_model(self, data: V3DataConfig, model: V3ModelConfig) -> None:
        if not model.use_hard_conservation:
            return
        if self.density_channel is None or self.flow_channel is None:
            raise ValueError("hard conservation requires density_channel and flow_channel")
        if data.segment_volume_path is None:
            raise ValueError("hard conservation requires segment_volume_path")
        if not data.graph_directed or data.adjacency_path is None:
            raise ValueError("hard conservation requires an explicit directed dataset graph")
        names = tuple(name.lower() for name in data.feature_names)
        if self.density_channel >= len(names) or "density" not in names[self.density_channel]:
            raise ValueError("density_channel must reference an independently measured density")
        if self.flow_channel >= len(names) or "flow" not in names[self.flow_channel]:
            raise ValueError("flow_channel must reference an independently measured flow")


@dataclass(frozen=True)
class V3TrainingConfig:
    epochs: int = 30
    batch_size: int = 32
    evaluation_batch_size: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    patience: int = 8
    num_workers: int = 0
    node_batch_size: int | None = None
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 5.0
    curriculum_start_steps: int = 1
    curriculum_full_fraction: float = 0.5
    nll_weight: float = 1.0
    mae_weight: float = 0.2
    semigroup_weight: float = 0.05
    scale_regularization_weight: float = 1e-4
    masked_pretrain_weight: float = 0.0
    pretrained_encoder_path: str | None = None
    mask_ratio: float = 0.4
    output_dir: str = "artifacts/v3/default"
    show_batch_progress: bool = False
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.node_batch_size is not None and self.node_batch_size < 2:
            raise ValueError("node_batch_size must be at least 2")
        if self.curriculum_start_steps < 1:
            raise ValueError("curriculum_start_steps must be positive")
        if not 0 < self.curriculum_full_fraction <= 1:
            raise ValueError("curriculum_full_fraction must be in (0,1]")
        if not 0 <= self.mask_ratio < 1:
            raise ValueError("mask_ratio must be in [0,1)")
        for name in (
            "nll_weight",
            "mae_weight",
            "semigroup_weight",
            "scale_regularization_weight",
            "masked_pretrain_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class V3StudyConfig:
    seeds: tuple[int, ...] = (13, 42, 97)
    bootstrap_samples: int = 2000
    confidence_level: float = 0.95
    primary_metric: str = "mae"
    primary_horizon_minutes: int = 60
    fail_on_dirty_worktree: bool = False

    def __post_init__(self) -> None:
        if len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("study seeds must be non-empty and unique")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0,1)")


@dataclass(frozen=True)
class V3ExperimentConfig:
    data: V3DataConfig
    model: V3ModelConfig = field(default_factory=V3ModelConfig)
    physics: V3PhysicsConfig = field(default_factory=V3PhysicsConfig)
    training: V3TrainingConfig = field(default_factory=V3TrainingConfig)
    study: V3StudyConfig = field(default_factory=V3StudyConfig)

    def __post_init__(self) -> None:
        if self.training.curriculum_start_steps > self.data.forecast_steps:
            raise ValueError("curriculum_start_steps exceeds forecast_steps")
        self.physics.validate_for_model(self.data, self.model)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> V3ExperimentConfig:
        data_raw = dict(raw["data"])
        data_raw["feature_names"] = tuple(data_raw.get("feature_names", ("value",)))
        data_raw["units"] = tuple(data_raw.get("units", ("unknown",)))
        study_raw = dict(raw.get("study", {}))
        if "seeds" in study_raw:
            study_raw["seeds"] = tuple(int(seed) for seed in study_raw["seeds"])
        return cls(
            data=V3DataConfig(**data_raw),
            model=V3ModelConfig(**raw.get("model", {})),
            physics=V3PhysicsConfig(**raw.get("physics", {})),
            training=V3TrainingConfig(**raw.get("training", {})),
            study=V3StudyConfig(**study_raw),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> V3ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("v3 configuration must be a mapping")
        return cls.from_dict(raw)
