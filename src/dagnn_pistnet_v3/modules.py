from __future__ import annotations

import math
import sys

import torch
from torch import nn
from torch.nn import functional as F


def edge_aggregate(
    messages: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Sum ``[B,E,D]`` messages at target nodes without optional dependencies."""

    output = messages.new_zeros(messages.shape[0], num_nodes, messages.shape[-1])
    output.index_add_(1, target, messages)
    return output


class TemporalDecompositionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        calendar_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        moving_average_window: int,
        decompose: bool,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.calendar_dim = calendar_dim
        self.hidden_dim = hidden_dim
        self.window = moving_average_window
        self.decompose = decompose
        value_dim = input_dim * (3 if decompose else 1)
        self.value_projection = nn.Linear(value_dim, hidden_dim)
        self.calendar_projection = (
            nn.Linear(calendar_dim, hidden_dim, bias=False) if calendar_dim else None
        )
        self.encoder = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, nodes, features = x.shape
        values = x.permute(0, 2, 3, 1).reshape(batch * nodes, features, time)
        pad = self.window // 2
        values = F.pad(values, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(values, kernel_size=self.window, stride=1)
        return trend.reshape(batch, nodes, features, time).permute(0, 3, 1, 2)

    def forward(
        self,
        x: torch.Tensor,
        history_calendar: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("x must have shape [B,T,N,F]")
        batch, time, nodes, features = x.shape
        if features != self.input_dim:
            raise ValueError(f"encoder expected {self.input_dim} features, got {features}")
        if self.decompose:
            trend = self._moving_average(x)
            residual = x - trend
            values = torch.cat([x, trend, residual], dim=-1)
        else:
            values = x
        projected = self.value_projection(values)
        if self.calendar_dim:
            expected = (batch, time, self.calendar_dim)
            if history_calendar is None or tuple(history_calendar.shape) != expected:
                raise ValueError(f"history_calendar must have shape {expected}")
            calendar = history_calendar[:, :, None, :].expand(-1, -1, nodes, -1)
            projected = projected + self.calendar_projection(calendar)
        flattened = projected.permute(0, 2, 1, 3)
        flattened = flattened.reshape(batch * nodes, time, self.hidden_dim)
        use_cudnn = not (sys.platform == "win32" and flattened.is_cuda)
        with torch.backends.cudnn.flags(enabled=use_cudnn):
            sequence, _ = self.encoder(flattened)
        sequence = self.normalization(sequence)
        sequence = sequence.reshape(batch, nodes, time, self.hidden_dim).permute(0, 2, 1, 3)
        return sequence[:, -1], sequence


class EdgeMessageBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_attr_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_attr_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return state
        source, target = edge_index
        if self.edge_attr_dim:
            if edge_attr is None or tuple(edge_attr.shape) != (
                edge_index.shape[1],
                self.edge_attr_dim,
            ):
                raise ValueError("edge_attr shape does not match configured edge_attr_dim")
            attributes = edge_attr[None].expand(state.shape[0], -1, -1)
        else:
            attributes = state.new_empty(state.shape[0], source.numel(), 0)
        weights = edge_weight[None, :, None].expand(state.shape[0], -1, -1)
        messages = self.message(
            torch.cat([state[:, source], state[:, target], attributes, weights], dim=-1)
        )
        aggregate = edge_aggregate(messages, target, state.shape[1])
        degree = edge_aggregate(weights, target, state.shape[1]).clamp_min(1e-6)
        aggregate = aggregate / degree
        return self.norm(state + self.update(torch.cat([state, aggregate], dim=-1)))


class MultiScaleGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_attr_dim: int, dropout: float) -> None:
        super().__init__()
        self.fine = EdgeMessageBlock(hidden_dim, edge_attr_dim, dropout)
        self.coarse = EdgeMessageBlock(hidden_dim, 0, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        cluster_index: torch.Tensor,
        coarse_edge_index: torch.Tensor,
        coarse_edge_weight: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fine = self.fine(state, edge_index, edge_weight, edge_attr)
        num_clusters = int(cluster_index.max()) + 1
        coarse = state.new_zeros(state.shape[0], num_clusters, state.shape[-1])
        coarse.index_add_(1, cluster_index, fine)
        counts = torch.bincount(cluster_index, minlength=num_clusters).to(state.dtype)
        coarse = coarse / counts[None, :, None].clamp_min(1.0)
        coarse = self.coarse(coarse, coarse_edge_index, coarse_edge_weight)
        upsampled = coarse.index_select(1, cluster_index)
        return self.fusion(torch.cat([state, fine, upsampled], dim=-1))


class DirectedTransportOperator(nn.Module):
    def __init__(self, hidden_dim: int, edge_attr_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        self.flux = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_attr_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.Sigmoid())
        self.update = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return state
        source, target = edge_index
        weights = edge_weight[None, :, None].expand(state.shape[0], -1, -1)
        if self.edge_attr_dim:
            if edge_attr is None:
                raise ValueError("transport operator requires edge attributes")
            attributes = edge_attr[None].expand(state.shape[0], -1, -1)
        else:
            attributes = state.new_empty(state.shape[0], source.numel(), 0)
        pair = torch.cat([state[:, source], state[:, target]], dim=-1)
        message = self.flux(torch.cat([pair, attributes, weights], dim=-1))
        message = message * self.gate(torch.cat([pair, weights], dim=-1)) * weights
        aggregate = edge_aggregate(message, target, state.shape[1])
        degree = edge_aggregate(weights, target, state.shape[1]).clamp_min(1e-6)
        return self.norm(state + self.update(aggregate / degree))


class SpillbackDiffusionOperator(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.diffusivity = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return state
        source, target = edge_index
        weights = edge_weight[None, :, None].expand(state.shape[0], -1, -1)
        pair = torch.cat([state[:, source], state[:, target], weights], dim=-1)
        coefficient = self.diffusivity(pair) * weights
        # A learned bounded Laplacian term. Reverse influence represents spillback.
        message = coefficient * (state[:, source] - state[:, target])
        aggregate = edge_aggregate(message, target, state.shape[1])
        return self.norm(state + self.update(aggregate))


class LocalSourceSinkOperator(nn.Module):
    def __init__(self, hidden_dim: int, context_dim: int, dropout: float) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, state: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        if self.context_dim:
            if context is None or context.shape[:2] != state.shape[:2]:
                raise ValueError("source/sink context has invalid shape")
            inputs = torch.cat([state, context], dim=-1)
        else:
            inputs = state
        return self.norm(state + self.net(inputs))


class EventJumpOperator(nn.Module):
    def __init__(self, hidden_dim: int, event_dim: int) -> None:
        super().__init__()
        self.event_dim = event_dim
        if event_dim:
            self.gate = nn.Sequential(nn.Linear(hidden_dim + event_dim, hidden_dim), nn.Sigmoid())
            self.jump = nn.Sequential(nn.Linear(hidden_dim + event_dim, hidden_dim), nn.Tanh())

    def forward(self, state: torch.Tensor, events: torch.Tensor | None) -> torch.Tensor:
        if not self.event_dim:
            return state
        if events is None or tuple(events.shape[:2]) != tuple(state.shape[:2]):
            raise ValueError("event tensor must have shape [B,N,event_dim]")
        joined = torch.cat([state, events], dim=-1)
        return state + self.gate(joined) * self.jump(joined)


class ConservativeEdgeFlux(nn.Module):
    """Update density by an incidence balance, ensuring internal conservation."""

    def __init__(self, hidden_dim: int, edge_attr_dim: int, num_regimes: int) -> None:
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        self.num_regimes = num_regimes
        self.regime_embedding = nn.Parameter(
            torch.randn(num_regimes, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.flux_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_attr_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        previous_density: torch.Tensor,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        segment_volume: torch.Tensor,
        dt_seconds: float,
        edge_attr: torch.Tensor | None = None,
        boundary_flow: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, hidden = state.shape
        source, target = edge_index
        mode_state = state[:, :, None, :] + self.regime_embedding[None, None, :, :]
        source_state = mode_state[:, source]
        target_state = mode_state[:, target]
        weights = edge_weight[None, :, None, None].expand(batch, -1, self.num_regimes, -1)
        if self.edge_attr_dim:
            if edge_attr is None:
                raise ValueError("conservative flux requires edge attributes")
            attributes = edge_attr[None, :, None, :].expand(batch, -1, self.num_regimes, -1)
        else:
            attributes = state.new_empty(batch, source.numel(), self.num_regimes, 0)
        flux = self.flux_head(
            torch.cat([source_state, target_state, attributes, weights], dim=-1)
        ).squeeze(-1)
        flux = flux * edge_weight[None, :, None]
        balance = state.new_zeros(batch, nodes, self.num_regimes)
        balance.index_add_(1, target, flux)
        balance.index_add_(1, source, -flux)
        if boundary_flow is not None:
            if tuple(boundary_flow.shape) != (batch, nodes):
                raise ValueError("boundary_flow must have shape [B,N]")
            balance = balance + boundary_flow[:, :, None]
        volume = segment_volume[None, :, None].clamp_min(1e-8)
        next_density = previous_density[:, :, None] + float(dt_seconds) * balance / volume
        return next_density, flux
