from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dagnn_pistnet.training.trainer import seed_everything

from .config import V3ExperimentConfig
from .data import V3WindowDataset, prepare_v3_data
from .modules import TemporalDecompositionEncoder
from .reproducibility import build_reproducibility_manifest, write_manifest


@dataclass(frozen=True)
class CrossDatasetPretrainConfig:
    dataset_configs: tuple[str, ...]
    hidden_dim: int = 64
    temporal_layers: int = 2
    moving_average_window: int = 3
    decompose: bool = True
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 32
    node_batch_size: int = 512
    steps_per_dataset_per_epoch: int | None = 100
    learning_rate: float = 3e-4
    mask_ratio: float = 0.4
    node_mask_probability: float = 0.2
    time_mask_probability: float = 0.2
    seed: int = 42
    device: str = "auto"
    output_dir: str = "artifacts/v3/pretraining"
    resume: bool = True

    def __post_init__(self) -> None:
        if not self.dataset_configs:
            raise ValueError("pretraining requires at least one dataset")
        if self.epochs < 1 or self.batch_size < 1 or self.node_batch_size < 1:
            raise ValueError("pretraining epochs and batch sizes must be positive")
        if (
            self.steps_per_dataset_per_epoch is not None
            and self.steps_per_dataset_per_epoch < 1
        ):
            raise ValueError("steps_per_dataset_per_epoch must be positive")
        if not 0 <= self.mask_ratio < 1:
            raise ValueError("mask_ratio must be in [0,1)")
        if not 0 <= self.node_mask_probability < 1:
            raise ValueError("node_mask_probability must be in [0,1)")
        if not 0 <= self.time_mask_probability < 1:
            raise ValueError("time_mask_probability must be in [0,1)")

    @classmethod
    def from_yaml(cls, path: str | Path) -> CrossDatasetPretrainConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("pretraining configuration must be a mapping")
        raw = dict(raw)
        raw["dataset_configs"] = tuple(raw["dataset_configs"])
        return cls(**raw)


class PrimaryFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        base: V3WindowDataset,
        feature_index: int,
        node_batch_size: int,
        seed: int,
    ) -> None:
        self.base = base
        self.feature_index = feature_index
        self.node_batch_size = node_batch_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        history, *_ = self.base[index]
        if history.shape[1] > self.node_batch_size:
            rng = np.random.default_rng(self.seed + index)
            nodes = np.sort(
                rng.choice(history.shape[1], size=self.node_batch_size, replace=False)
            )
            history = history.index_select(1, torch.as_tensor(nodes, dtype=torch.long))
        return history[..., self.feature_index : self.feature_index + 1], torch.tensor(0)


class CrossDatasetMaskedModel(nn.Module):
    def __init__(self, config: CrossDatasetPretrainConfig, dataset_names: tuple[str, ...]) -> None:
        super().__init__()
        self.encoder = TemporalDecompositionEncoder(
            input_dim=1,
            calendar_dim=0,
            hidden_dim=config.hidden_dim,
            layers=config.temporal_layers,
            dropout=config.dropout,
            moving_average_window=config.moving_average_window,
            decompose=config.decompose,
        )
        self.reconstruction = nn.Linear(config.hidden_dim, 1)
        self.input_adapters = nn.ModuleDict({name: nn.Linear(1, 1) for name in dataset_names})
        self.output_adapters = nn.ModuleDict({name: nn.Linear(1, 1) for name in dataset_names})
        for adapter in (*self.input_adapters.values(), *self.output_adapters.values()):
            nn.init.ones_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, dataset: str) -> torch.Tensor:
        adapted = self.input_adapters[dataset](x)
        _, sequence = self.encoder(adapted.masked_fill(mask, 0.0), None)
        return self.output_adapters[dataset](self.reconstruction(sequence))


def structured_mask(x: torch.Tensor, config: CrossDatasetPretrainConfig) -> torch.Tensor:
    random_mask = torch.rand_like(x) < config.mask_ratio
    node_mask = torch.rand(x.shape[0], 1, x.shape[2], 1, device=x.device)
    time_mask = torch.rand(x.shape[0], x.shape[1], 1, 1, device=x.device)
    return (
        random_mask
        | (node_mask < config.node_mask_probability)
        | (time_mask < config.time_mask_probability)
    )


def run_cross_dataset_pretraining(config: CrossDatasetPretrainConfig) -> Path:
    seed_everything(config.seed)
    requested = config.device.lower()
    device = torch.device(
        "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu"
        if requested == "auto"
        else requested
    )
    prepared: list[tuple[str, V3ExperimentConfig]] = []
    names: list[str] = []
    for config_path in config.dataset_configs:
        experiment_config = V3ExperimentConfig.from_yaml(config_path)
        name = experiment_config.data.dataset.replace("-", "_").lower()
        if name in names:
            raise ValueError(f"duplicate pretraining dataset name: {name}")
        names.append(name)
        prepared.append((name, experiment_config))

    model = CrossDatasetMaskedModel(config, tuple(names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int | str]] = []
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "pretraining_state.pt"
    start_epoch = 0
    start_dataset = 0
    if config.resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("config") != asdict(config):
            raise ValueError("pretraining resume state belongs to another configuration")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = list(state.get("history", []))
        start_epoch = int(state["next_epoch"])
        start_dataset = int(state["next_dataset"])
        print(
            f"pretrain resume epoch={start_epoch + 1} dataset_index={start_dataset}",
            flush=True,
        )
    for epoch in range(start_epoch, config.epochs):
        model.train()
        dataset_offset = start_dataset if epoch == start_epoch else 0
        for dataset_index in range(dataset_offset, len(prepared)):
            name, experiment_config = prepared[dataset_index]
            # Load one corpus at a time: LargeST alone is multi-gigabyte and the
            # publication pipeline must not retain all cities in RAM together.
            data = prepare_v3_data(experiment_config)
            dataset = PrimaryFeatureDataset(
                data.train_dataset,
                experiment_config.data.primary_feature_index,
                config.node_batch_size,
                config.seed,
            )
            generator = torch.Generator().manual_seed(config.seed + epoch)
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=generator,
                num_workers=0,
            )
            loss_sum = 0.0
            count = 0
            for step, (x, _) in enumerate(loader):
                if (
                    config.steps_per_dataset_per_epoch is not None
                    and step >= config.steps_per_dataset_per_epoch
                ):
                    break
                x = x.to(device)
                mask = structured_mask(x, config)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, mask, name)
                loss = (prediction - x).abs()[mask].mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_sum += float(loss.detach()) * x.shape[0]
                count += x.shape[0]
                if (step + 1) % 25 == 0:
                    print(
                        f"pretrain epoch={epoch + 1}/{config.epochs} dataset={name} "
                        f"step={step + 1} masked_mae={loss_sum / max(count, 1):.6f}",
                        flush=True,
                    )
            history.append(
                {
                    "epoch": epoch + 1,
                    "dataset": name,
                    "masked_mae": loss_sum / max(count, 1),
                }
            )
            print(
                f"pretrain epoch={epoch + 1}/{config.epochs} dataset={name} "
                f"masked_mae={loss_sum / max(count, 1):.6f}",
                flush=True,
            )
            del loader, dataset, data
            gc.collect()
            next_dataset = dataset_index + 1
            next_epoch = epoch
            if next_dataset >= len(prepared):
                next_dataset = 0
                next_epoch = epoch + 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "next_epoch": next_epoch,
                    "next_dataset": next_dataset,
                    "config": asdict(config),
                },
                state_path,
            )
    checkpoint = output / "pretrained_encoder.pt"
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "reconstruction": model.reconstruction.state_dict(),
            "adapters": {
                name: {
                    "input": model.input_adapters[name].state_dict(),
                    "output": model.output_adapters[name].state_dict(),
                }
                for name in names
            },
            "config": asdict(config),
            "torch_version": torch.__version__,
        },
        checkpoint,
    )
    with (output / "pretraining_history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, ensure_ascii=False, indent=2)
    manifests = [
        build_reproducibility_manifest(
            experiment_config,
            Path.cwd(),
            command=["pretrain", *config.dataset_configs],
        )
        for _, experiment_config in prepared
    ]
    write_manifest(
        output / "pretraining_manifest.json",
        {
            "pretraining_config": asdict(config),
            "datasets": manifests,
            "checkpoint": str(checkpoint),
        },
    )
    return checkpoint


def load_pretrained_encoder(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["encoder"]
    target = model.temporal_encoder.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    missing, unexpected = model.temporal_encoder.load_state_dict(compatible, strict=False)
    return {
        "loaded": sorted(compatible),
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
    }
