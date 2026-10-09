from pathlib import Path

import numpy as np
import pytest
import torch

from dagnn_pistnet.config import ExperimentConfig
from dagnn_pistnet.experiment import Experiment
from dagnn_pistnet.training import build_graph_partitions


def test_synthetic_experiment_smoke_test(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    data_path = tmp_path / "tiny.npz"
    np.savez(data_path, data=rng.normal(size=(80, 5, 1)).astype(np.float32))
    config = ExperimentConfig.from_dict(
        {
            "data": {
                "dataset": "PEMS07",
                "data_path": str(data_path),
                "feature_names": ["flow"],
                "units": ["vehicles"],
                "history_steps": 4,
                "forecast_steps": 3,
                "graph_top_k": 2,
            },
            "model": {
                "hidden_dim": 8,
                "lstm_layers": 1,
                "adaptive_embedding_dim": 4,
                "adaptive_top_k": 2,
                "dropout": 0.0,
            },
            "training": {"batch_size": 2, "device": "cpu", "amp": False},
        }
    )
    result = Experiment(config).smoke_test()
    assert result["prediction_shape"] == [1, 3, 5, 1]
    assert result["graph_source"] == "train-only-correlation"


def test_synthetic_experiment_trains_and_saves_checkpoint(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    data_path = tmp_path / "train.npz"
    np.savez(data_path, data=rng.normal(size=(60, 3, 1)).astype(np.float32))
    output_dir = tmp_path / "artifacts"
    config = ExperimentConfig.from_dict(
        {
            "data": {
                "dataset": "PEMS07",
                "data_path": str(data_path),
                "feature_names": ["flow"],
                "units": ["vehicles"],
                "history_steps": 3,
                "forecast_steps": 2,
                "graph_top_k": 1,
            },
            "model": {
                "hidden_dim": 4,
                "lstm_layers": 1,
                "adaptive_embedding_dim": 2,
                "adaptive_top_k": 1,
                "dropout": 0.0,
            },
            "training": {
                "epochs": 1,
                "batch_size": 8,
                "patience": 1,
                "device": "cpu",
                "amp": False,
                "output_dir": str(output_dir),
            },
        }
    )
    result = Experiment(config).train()
    assert result.best_epoch == 1
    assert (output_dir / "best_model.pt").exists()
    assert (output_dir / "result.json").exists()
    evaluation = Experiment(config).evaluate()
    assert evaluation["checkpoint_epoch"] == 1
    assert "5m" in evaluation["test_metrics"]

    legacy_path = tmp_path / "legacy.pt"
    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=False)
    checkpoint.pop("architecture_version")
    torch.save(checkpoint, legacy_path)
    with pytest.raises(ValueError, match="architecture v1"):
        Experiment(config).evaluate(legacy_path)


def test_synthetic_experiment_trains_with_local_graph_partitions(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    data_path = tmp_path / "partitioned.npz"
    np.savez(data_path, data=rng.normal(size=(50, 8, 1)).astype(np.float32))
    config = ExperimentConfig.from_dict(
        {
            "data": {
                "dataset": "PEMS07",
                "data_path": str(data_path),
                "feature_names": ["flow"],
                "units": ["vehicles"],
                "history_steps": 3,
                "forecast_steps": 2,
                "graph_top_k": 2,
            },
            "model": {
                "hidden_dim": 4,
                "lstm_layers": 1,
                "adaptive_embedding_dim": 2,
                "adaptive_top_k": 1,
                "adaptive_candidate_graph": True,
                "dropout": 0.0,
            },
            "training": {
                "epochs": 1,
                "batch_size": 4,
                "node_batch_size": 3,
                "patience": 1,
                "device": "cpu",
                "amp": False,
                "output_dir": str(tmp_path / "partition-artifacts"),
            },
        }
    )
    result = Experiment(config).train()
    assert result.best_epoch == 1
    assert "5m" in result.test_metrics


def test_conservation_rejects_correlation_graph(tmp_path: Path) -> None:
    data_path = tmp_path / "physics.npz"
    np.savez(data_path, data=np.ones((40, 4, 2), dtype=np.float32))
    config = ExperimentConfig.from_dict(
        {
            "data": {
                "dataset": "PEMS07",
                "data_path": str(data_path),
                "feature_names": ["density", "flow"],
                "units": ["vehicles_per_km", "vehicles_per_second"],
                "history_steps": 3,
                "forecast_steps": 2,
            },
            "physics": {
                "kind": "directed_conservation",
                "weight": 0.1,
                "density_channel": 0,
                "flow_channel": 1,
            },
            "training": {"device": "cpu", "amp": False},
        }
    )
    with pytest.raises(ValueError, match="directed dataset graph"):
        Experiment(config).smoke_test()


def test_all_zero_outage_batch_is_ignored(tmp_path: Path) -> None:
    data_path = tmp_path / "outage.npz"
    np.savez(data_path, data=np.zeros((40, 4, 1), dtype=np.float32))
    config = ExperimentConfig.from_dict(
        {
            "data": {
                "dataset": "PEMS07",
                "data_path": str(data_path),
                "feature_names": ["flow"],
                "units": ["vehicles"],
                "history_steps": 3,
                "forecast_steps": 2,
            },
            "model": {
                "hidden_dim": 4,
                "lstm_layers": 1,
                "adaptive_embedding_dim": 2,
                "adaptive_top_k": 1,
            },
            "training": {"device": "cpu", "amp": False},
        }
    )
    result = Experiment(config).smoke_test()
    assert result["loss"] == pytest.approx(0.0)


def test_graph_partitions_cover_every_node_once() -> None:
    edges = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    partitions = build_graph_partitions(6, edges, max_nodes=2, seed=3)
    joined = torch.cat(partitions)
    assert sorted(joined.tolist()) == list(range(6))
    assert all(partition.numel() <= 2 for partition in partitions)
