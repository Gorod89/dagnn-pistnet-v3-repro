from __future__ import annotations

import pytest
import torch

from dagnn_pistnet_v3.config import (
    V3ExperimentConfig,
    V3ModelConfig,
    V3PhysicsConfig,
)
from dagnn_pistnet_v3.model import DAGNNPISTNetV3
from dagnn_pistnet_v3.modules import ConservativeEdgeFlux


def test_internal_edge_flux_conserves_total_mass() -> None:
    module = ConservativeEdgeFlux(hidden_dim=8, edge_attr_dim=0, num_regimes=3)
    previous = torch.rand(2, 4)
    state = torch.randn(2, 4, 8)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    next_density, _ = module(
        previous,
        state,
        edges,
        torch.ones(3),
        segment_volume=torch.ones(4),
        dt_seconds=1.0,
    )
    before = previous.sum(dim=1, keepdim=True).expand(-1, 3)
    after = next_density.sum(dim=1)
    assert torch.allclose(before, after, atol=1e-5)


def test_hard_physics_rejects_flow_only_configuration() -> None:
    raw = {
        "data": {
            "dataset": "flow-only",
            "data_path": "unused.npy",
            "adjacency_path": "graph.npy",
            "feature_names": ["flow"],
            "units": ["vehicles_per_step"],
            "graph_directed": True,
            "segment_volume_path": "length.npy",
        },
        "model": {"use_hard_conservation": True},
        "physics": {"density_channel": 0, "flow_channel": 0},
    }
    with pytest.raises(ValueError, match="density"):
        V3ExperimentConfig.from_dict(raw)


def test_full_model_density_update_is_conservative() -> None:
    model_config = V3ModelConfig(
        hidden_dim=8,
        temporal_layers=1,
        num_regimes=2,
        use_hard_conservation=True,
    )
    physics = V3PhysicsConfig(density_channel=0, flow_channel=1, dt_seconds=1.0)
    model = DAGNNPISTNetV3(2, 2, 3, 1, 0, 0, model_config, physics)
    history = torch.rand(2, 4, 3, 2)
    result = model(
        history,
        torch.tensor([[0, 1], [1, 2]]),
        torch.ones(2),
        cluster_index=torch.tensor([0, 0, 1]),
        segment_volume=torch.ones(3),
    )
    before = history[:, -1, :, 0].sum(dim=1, keepdim=True)
    after = result.means[:, 0, :, :, 0].sum(dim=1)
    assert torch.allclose(before.expand_as(after), after, atol=1e-5)
