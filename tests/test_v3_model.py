from __future__ import annotations

import pytest
import torch

from dagnn_pistnet_v3.config import V3ModelConfig, V3PhysicsConfig
from dagnn_pistnet_v3.model import DAGNNPISTNetV3


def _graph() -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.tensor([[0, 1, 2, 3, 1], [1, 2, 3, 4, 0]])
    edge_weight = torch.ones(edge_index.shape[1])
    return edge_index, edge_weight


def test_v3_forward_distribution_and_backward() -> None:
    edge_index, edge_weight = _graph()
    model = DAGNNPISTNetV3(
        input_dim=1,
        output_dim=1,
        num_nodes=5,
        forecast_steps=4,
        calendar_dim=2,
        event_dim=1,
        config=V3ModelConfig(hidden_dim=16, temporal_layers=1, num_regimes=3),
    )
    x = torch.randn(2, 6, 5, 1)
    history_calendar = torch.randn(2, 6, 2)
    future_calendar = torch.randn(2, 4, 2)
    events = torch.randn(2, 4, 5, 1)
    distribution = model(
        x,
        edge_index,
        edge_weight,
        history_calendar=history_calendar,
        future_calendar=future_calendar,
        cluster_index=torch.tensor([0, 0, 1, 1, 1]),
        future_events=events,
    )
    assert distribution.means.shape == (2, 4, 5, 3, 1)
    assert distribution.mean.shape == (2, 4, 5, 1)
    assert distribution.variance.shape == (2, 4, 5, 1)
    assert torch.all(distribution.scales > 0)
    distribution.mean.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_v3_accepts_local_node_partition() -> None:
    edge_index = torch.tensor([[0, 1], [1, 2]])
    model = DAGNNPISTNetV3(
        input_dim=1,
        output_dim=1,
        num_nodes=100,
        forecast_steps=2,
        calendar_dim=0,
        event_dim=0,
        config=V3ModelConfig(hidden_dim=8, temporal_layers=1),
    )
    result = model(
        torch.randn(1, 4, 3, 1),
        edge_index,
        torch.ones(2),
        cluster_index=torch.tensor([0, 0, 1]),
    )
    assert result.mean.shape == (1, 2, 3, 1)


def test_disabled_probabilistic_head_has_one_fixed_scale_mode() -> None:
    edge_index, edge_weight = _graph()
    config = V3ModelConfig(
        hidden_dim=8,
        temporal_layers=1,
        num_regimes=5,
        use_probabilistic_head=False,
        min_scale=0.05,
    )
    model = DAGNNPISTNetV3(1, 1, 5, 2, 0, 0, config, V3PhysicsConfig())
    result = model(
        torch.randn(1, 4, 5, 1),
        edge_index,
        edge_weight,
        cluster_index=torch.tensor([0, 0, 1, 1, 1]),
    )
    assert result.means.shape[-2] == 1
    assert torch.allclose(result.scales, torch.full_like(result.scales, 0.05))


def test_operator_order_is_capacity_matched_and_changes_only_composition() -> None:
    edge_index, edge_weight = _graph()
    default = DAGNNPISTNetV3(
        1, 1, 5, 2, 0, 0,
        V3ModelConfig(hidden_dim=8, temporal_layers=1, dropout=0.0),
    )
    reverse = DAGNNPISTNetV3(
        1, 1, 5, 2, 0, 0,
        V3ModelConfig(
            hidden_dim=8,
            temporal_layers=1,
            dropout=0.0,
            operator_order=("source_sink", "diffusion", "transport"),
        ),
    )
    reverse.load_state_dict(default.state_dict())
    default.eval()
    reverse.eval()
    assert sum(p.numel() for p in default.parameters()) == sum(
        p.numel() for p in reverse.parameters()
    )
    state = torch.randn(2, 5, 8)
    expected = default.source_sink(
        default.diffusion(
            default.transport(state, edge_index, edge_weight, None),
            edge_index,
            edge_weight,
        ),
        None,
    )
    actual_default = default._operator_step(
        state, edge_index, edge_weight, None, None, None
    )
    actual_reverse = reverse._operator_step(
        state, edge_index, edge_weight, None, None, None
    )
    assert torch.allclose(actual_default, expected)
    assert not torch.allclose(actual_default, actual_reverse)


def test_operator_order_rejects_missing_or_repeated_modules() -> None:
    with pytest.raises(ValueError, match="operator_order"):
        V3ModelConfig(operator_order=("transport", "diffusion", "diffusion"))
