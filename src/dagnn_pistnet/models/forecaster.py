from __future__ import annotations

import torch
from torch import nn

from .adaptive_graph import AdaptiveGraphLearner
from .decoder import SequentialForecastDecoder
from .layers import BidirectionalDiffusionConv, EdgeGraphConv, TemporalEncoder


class DAGNNPISTNet(nn.Module):
    """Calendar-aware adaptive-graph autoregressive traffic forecaster."""

    architecture_version = 2

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        forecast_steps: int,
        hidden_dim: int = 64,
        lstm_layers: int = 2,
        adaptive_embedding_dim: int = 16,
        adaptive_top_k: int = 10,
        diffusion_steps: int = 2,
        calendar_dim: int = 0,
        dropout: float = 0.1,
        adaptive_candidate_graph: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_nodes = int(num_nodes)
        self.forecast_steps = int(forecast_steps)
        self.calendar_dim = int(calendar_dim)
        self.adaptive_candidate_graph = bool(adaptive_candidate_graph)
        if output_dim > input_dim:
            raise ValueError("output_dim cannot exceed input_dim for autoregressive decoding")

        self.temporal_encoder = TemporalEncoder(
            input_dim + calendar_dim,
            hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
        )
        self.static_conv = BidirectionalDiffusionConv(
            hidden_dim, hidden_dim, diffusion_steps=diffusion_steps
        )
        self.adaptive_graph = AdaptiveGraphLearner(
            num_nodes=num_nodes,
            context_dim=hidden_dim,
            embedding_dim=adaptive_embedding_dim,
            top_k=adaptive_top_k,
        )
        self.adaptive_conv = EdgeGraphConv(hidden_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.decoder = SequentialForecastDecoder(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            forecast_steps=forecast_steps,
            calendar_dim=calendar_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        static_edge_index: torch.Tensor,
        static_edge_weight: torch.Tensor,
        history_calendar: torch.Tensor | None = None,
        future_calendar: torch.Tensor | None = None,
        teacher_forcing_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        node_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B,T,N,F]")
        if x.shape[3] != self.input_dim:
            raise ValueError(f"model expects F={self.input_dim}; got {tuple(x.shape)}")
        batch, history, nodes, _ = x.shape
        if node_ids is None:
            if nodes != self.num_nodes:
                raise ValueError(f"model expects N={self.num_nodes}; got {nodes}")
            node_ids = torch.arange(self.num_nodes, device=x.device)
        elif node_ids.ndim != 1 or node_ids.shape[0] != nodes:
            raise ValueError("node_ids must have shape [N]")
        if self.calendar_dim:
            expected_calendar = (batch, history, self.calendar_dim)
            if history_calendar is None or tuple(history_calendar.shape) != expected_calendar:
                raise ValueError(f"history_calendar must have shape {expected_calendar}")
            calendar_per_node = history_calendar[:, :, None, :].expand(-1, -1, nodes, -1)
            encoder_input = torch.cat([x, calendar_per_node], dim=-1)
        else:
            encoder_input = x

        temporal = self.temporal_encoder(encoder_input)
        static = torch.relu(self.static_conv(temporal, static_edge_index, static_edge_weight))
        adaptive_index, adaptive_weight = self.adaptive_graph(
            temporal,
            node_ids=node_ids,
            candidate_edge_index=static_edge_index if self.adaptive_candidate_graph else None,
        )
        adaptive = torch.relu(self.adaptive_conv(temporal, adaptive_index, adaptive_weight))
        fused = self.fusion(torch.cat([temporal, static, adaptive], dim=-1))
        return self.decoder(
            initial_state=fused,
            last_observation=x[:, -1, :, : self.output_dim],
            future_calendar=future_calendar,
            teacher_forcing_target=teacher_forcing_target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
