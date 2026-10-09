import numpy as np
import pytest
import torch

from dagnn_pistnet.data import adjacency_to_graph, build_correlation_graph
from dagnn_pistnet.models import (
    AdaptiveGraphLearner,
    BidirectionalDiffusionConv,
    EdgeGraphConv,
    SequentialForecastDecoder,
)


def test_adjacency_to_graph_adds_self_loops() -> None:
    graph = adjacency_to_graph(np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float32))
    edges = {tuple(edge) for edge in graph.edge_index.T.tolist()}
    assert {(0, 0), (0, 1), (1, 1)} <= edges


def test_correlation_graph_uses_requested_node_count() -> None:
    rng = np.random.default_rng(4)
    adjacency = build_correlation_graph(rng.normal(size=(50, 7, 1)), top_k=2)
    assert adjacency.shape == (7, 7)
    assert np.allclose(adjacency, adjacency.T)


def test_adaptive_graph_depends_only_on_context() -> None:
    learner = AdaptiveGraphLearner(num_nodes=5, context_dim=4, top_k=2)
    context = torch.randn(3, 5, 4)
    edge_index, edge_weight = learner(context)
    assert edge_index.shape == (2, 10)
    assert edge_weight.shape == (3, 10)
    assert torch.isfinite(edge_weight).all()


def test_adaptive_graph_can_reweight_sparse_candidate_edges() -> None:
    learner = AdaptiveGraphLearner(num_nodes=6, context_dim=4, top_k=2)
    context = torch.randn(2, 3, 4)
    node_ids = torch.tensor([1, 3, 5])
    candidates = torch.tensor([[0, 0, 1, 2], [0, 1, 2, 1]])
    edge_index, edge_weight = learner(context, node_ids=node_ids, candidate_edge_index=candidates)
    assert torch.equal(edge_index, candidates)
    assert edge_weight.shape == (2, 4)
    assert torch.isfinite(edge_weight).all()


def test_edge_graph_conv_rejects_wrong_batch_weights() -> None:
    layer = EdgeGraphConv(3, 4)
    x = torch.randn(2, 3, 3)
    edges = torch.tensor([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="batch"):
        layer(x, edges, torch.ones(4, 2))


def test_bidirectional_diffusion_contains_forward_and_reverse_walks() -> None:
    layer = BidirectionalDiffusionConv(1, 2, diffusion_steps=2)
    x = torch.tensor([[[1.0], [0.0], [0.0]]])
    edges = torch.tensor([[0, 1], [1, 2]])
    features = layer.diffusion_features(x, edges, torch.ones(2))
    assert features.shape == (1, 3, 5)
    assert features[0, 1, 1].item() == pytest.approx(1.0)
    assert features[0, 2, 2].item() == pytest.approx(1.0)
    assert features[0, :, 3:].sum().item() == pytest.approx(0.0)


def test_sequential_decoder_uses_teacher_forcing_only_after_first_step() -> None:
    torch.manual_seed(7)
    decoder = SequentialForecastDecoder(6, 1, forecast_steps=3, calendar_dim=2)
    initial = torch.randn(2, 4, 6)
    last = torch.randn(2, 4, 1)
    calendar = torch.randn(2, 3, 2)
    target = torch.full((2, 3, 4, 1), 10.0)
    autoregressive = decoder(initial, last, calendar)
    teacher_forced = decoder(
        initial,
        last,
        calendar,
        teacher_forcing_target=target,
        teacher_forcing_ratio=1.0,
    )
    assert torch.allclose(autoregressive[:, 0], teacher_forced[:, 0])
    assert not torch.allclose(autoregressive[:, 1], teacher_forced[:, 1])
