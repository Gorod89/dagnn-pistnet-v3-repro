import torch

from dagnn_pistnet.models import DAGNNPISTNet


def test_model_decodes_sequential_future_with_calendar() -> None:
    model = DAGNNPISTNet(
        input_dim=1,
        output_dim=1,
        num_nodes=6,
        forecast_steps=4,
        hidden_dim=8,
        lstm_layers=1,
        adaptive_embedding_dim=4,
        adaptive_top_k=2,
        diffusion_steps=2,
        calendar_dim=5,
        dropout=0.0,
    )
    x = torch.randn(2, 5, 6, 1)
    nodes = torch.arange(6)
    edge_index = torch.stack([nodes, nodes])
    history_calendar = torch.randn(2, 5, 5)
    future_calendar = torch.randn(2, 4, 5)
    target = torch.randn(2, 4, 6, 1)
    prediction = model(
        x,
        edge_index,
        torch.ones(6),
        history_calendar=history_calendar,
        future_calendar=future_calendar,
        teacher_forcing_target=target,
        teacher_forcing_ratio=0.5,
    )
    assert prediction.shape == (2, 4, 6, 1)
    prediction.square().mean().backward()
    assert model.decoder.output_layer.weight.grad is not None


def test_model_accepts_sparse_node_partition() -> None:
    model = DAGNNPISTNet(
        input_dim=1,
        output_dim=1,
        num_nodes=8,
        forecast_steps=2,
        hidden_dim=6,
        lstm_layers=1,
        adaptive_embedding_dim=3,
        adaptive_top_k=2,
        adaptive_candidate_graph=True,
        calendar_dim=0,
    )
    x = torch.randn(2, 4, 3, 1)
    node_ids = torch.tensor([1, 4, 7])
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    prediction = model(x, edges, torch.ones(3), node_ids=node_ids)
    assert prediction.shape == (2, 2, 3, 1)
