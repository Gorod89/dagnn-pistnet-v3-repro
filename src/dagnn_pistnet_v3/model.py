from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import V3ModelConfig, V3PhysicsConfig
from .graph import induced_coarse_graph
from .modules import (
    ConservativeEdgeFlux,
    DirectedTransportOperator,
    EventJumpOperator,
    LocalSourceSinkOperator,
    MultiScaleGraphBlock,
    SpillbackDiffusionOperator,
    TemporalDecompositionEncoder,
)


@dataclass
class ForecastDistribution:
    means: torch.Tensor  # [B,P,N,K,F]
    scales: torch.Tensor  # [B,P,N,K,F]
    logits: torch.Tensor  # [B,P,N,K]
    edge_fluxes: tuple[torch.Tensor, ...] = ()

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        return (self.weights[..., None] * self.means).sum(dim=-2)

    @property
    def variance(self) -> torch.Tensor:
        weights = self.weights[..., None]
        second = weights * (self.scales.square() + self.means.square())
        return (second.sum(dim=-2) - self.mean.square()).clamp_min(0.0)


class DAGNNPISTNetV3(nn.Module):
    """Research architecture implementing the seven cross-domain recommendations."""

    architecture_version = 3

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        forecast_steps: int,
        calendar_dim: int,
        event_dim: int,
        config: V3ModelConfig,
        physics: V3PhysicsConfig | None = None,
    ) -> None:
        super().__init__()
        if output_dim > input_dim:
            raise ValueError("output_dim cannot exceed input_dim")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_nodes = int(num_nodes)
        self.forecast_steps = int(forecast_steps)
        self.calendar_dim = int(calendar_dim)
        self.event_dim = int(event_dim)
        self.config = config
        self.physics = physics or V3PhysicsConfig()
        self.num_regimes = config.num_regimes if config.use_probabilistic_head else 1
        hidden = config.hidden_dim

        self.temporal_encoder = TemporalDecompositionEncoder(
            input_dim=input_dim,
            calendar_dim=calendar_dim,
            hidden_dim=hidden,
            layers=config.temporal_layers,
            dropout=config.dropout,
            moving_average_window=config.moving_average_window,
            decompose=config.use_trend_seasonal_residual,
        )
        self.history_reconstruction = nn.Linear(hidden, input_dim)
        self.hierarchy = MultiScaleGraphBlock(hidden, config.edge_attr_dim, config.dropout)
        self.transport = DirectedTransportOperator(hidden, config.edge_attr_dim, config.dropout)
        self.diffusion = SpillbackDiffusionOperator(hidden, config.dropout)
        context_dim = calendar_dim + event_dim
        self.source_sink = LocalSourceSinkOperator(hidden, context_dim, config.dropout)
        self.event_jump = EventJumpOperator(hidden, event_dim if config.use_event_jump else 0)
        self.observation_update = nn.GRUCell(input_dim + calendar_dim, hidden)
        self.regime_embedding = nn.Parameter(torch.randn(self.num_regimes, hidden) * 0.02)
        self.regime_logits = nn.Linear(hidden + calendar_dim, self.num_regimes)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, output_dim)
        )
        self.scale_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, output_dim)
        )
        self.conservative_flux = (
            ConservativeEdgeFlux(hidden, config.edge_attr_dim, self.num_regimes)
            if config.use_hard_conservation
            else None
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        history_calendar: torch.Tensor | None,
        cluster_index: torch.Tensor,
        coarse_edge_index: torch.Tensor | None,
        coarse_edge_weight: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state, sequence = self.temporal_encoder(x, history_calendar)
        if self.config.use_hierarchy:
            if coarse_edge_index is None or coarse_edge_weight is None:
                coarse_edge_index, coarse_edge_weight = induced_coarse_graph(
                    cluster_index, edge_index, edge_weight
                )
            state = self.hierarchy(
                state,
                edge_index,
                edge_weight,
                cluster_index,
                coarse_edge_index,
                coarse_edge_weight,
                edge_attr,
            )
        return state, sequence

    def masked_reconstruction(
        self,
        x: torch.Tensor,
        history_calendar: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.shape != x.shape:
            raise ValueError("pretraining mask must match x")
        _, sequence = self.temporal_encoder(x.masked_fill(mask, 0.0), history_calendar)
        return self.history_reconstruction(sequence)

    def _operator_step(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor | None,
        context: torch.Tensor | None,
        events: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.config.use_operator_splitting:
            for _ in range(self.config.operator_substeps):
                for operator in self.config.operator_order:
                    if operator == "transport":
                        state = self.transport(state, edge_index, edge_weight, edge_attr)
                    elif operator == "diffusion":
                        state = self.diffusion(state, edge_index, edge_weight)
                    else:
                        state = self.source_sink(state, context)
        if self.config.use_event_jump:
            state = self.event_jump(state, events)
        return state

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        history_calendar: torch.Tensor | None = None,
        future_calendar: torch.Tensor | None = None,
        cluster_index: torch.Tensor | None = None,
        coarse_edge_index: torch.Tensor | None = None,
        coarse_edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        future_events: torch.Tensor | None = None,
        segment_volume: torch.Tensor | None = None,
        boundary_flow: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> ForecastDistribution:
        if x.ndim != 4 or x.shape[-1] != self.input_dim or x.shape[2] > self.num_nodes:
            raise ValueError(
                f"x must have shape [B,T,N<={self.num_nodes},{self.input_dim}], "
                f"got {tuple(x.shape)}"
            )
        batch, _, nodes, _ = x.shape
        steps = self.forecast_steps if steps is None else int(steps)
        if not 1 <= steps <= self.forecast_steps:
            raise ValueError("steps is outside configured forecast horizon")
        if cluster_index is None:
            cluster_index = torch.zeros(nodes, dtype=torch.long, device=x.device)
        if cluster_index.shape != (nodes,):
            raise ValueError("cluster_index must have shape [N]")
        if self.calendar_dim:
            if (
                future_calendar is None
                or future_calendar.shape[0] != batch
                or future_calendar.shape[1] < steps
            ):
                raise ValueError("future_calendar has invalid shape")
        if self.event_dim:
            expected = (batch, steps, nodes, self.event_dim)
            if future_events is None:
                future_events = x.new_zeros(expected)
            elif (
                future_events.shape[0] != batch
                or future_events.shape[1] < steps
                or tuple(future_events.shape[2:]) != (nodes, self.event_dim)
            ):
                raise ValueError(f"future_events must start with shape {expected}")

        state, _ = self.encode(
            x,
            edge_index,
            edge_weight,
            history_calendar,
            cluster_index,
            coarse_edge_index,
            coarse_edge_weight,
            edge_attr,
        )
        previous = x[:, -1, :, : self.output_dim]
        means: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        fluxes: list[torch.Tensor] = []
        for step in range(steps):
            calendar_step = (
                future_calendar[:, step, None, :].expand(-1, nodes, -1)
                if self.calendar_dim
                else None
            )
            event_step = future_events[:, step] if self.event_dim else None
            context_parts = [value for value in (calendar_step, event_step) if value is not None]
            context = torch.cat(context_parts, dim=-1) if context_parts else None
            state = self._operator_step(
                state, edge_index, edge_weight, edge_attr, context, event_step
            )
            mode_state = state[:, :, None, :] + self.regime_embedding[None, None, :, :]
            delta = self.delta_head(mode_state)
            mode_means = previous[:, :, None, :] + delta if self.config.use_delta_decoder else delta
            raw_scale = self.scale_head(mode_state)
            if self.config.use_probabilistic_head:
                mode_scales = self.config.min_scale + F.softplus(raw_scale)
                mode_scales = mode_scales.clamp_max(self.config.max_scale)
            else:
                mode_scales = torch.full_like(raw_scale, self.config.min_scale)
            logit_input = (
                torch.cat([state, calendar_step], dim=-1) if calendar_step is not None else state
            )
            mode_logits = self.regime_logits(logit_input)

            if self.conservative_flux is not None:
                density_channel = self.physics.density_channel
                if density_channel is None or segment_volume is None:
                    raise ValueError(
                        "hard conservation requires density channel and segment volume"
                    )
                boundary_step = boundary_flow[:, step] if boundary_flow is not None else None
                density, edge_flux = self.conservative_flux(
                    previous[:, :, density_channel],
                    state,
                    edge_index,
                    edge_weight,
                    segment_volume,
                    self.physics.dt_seconds,
                    edge_attr=edge_attr,
                    boundary_flow=boundary_step,
                )
                replacement = mode_means.clone()
                replacement[..., density_channel] = density
                mode_means = replacement
                fluxes.append(edge_flux)

            weights = torch.softmax(mode_logits, dim=-1)
            expected = (weights[..., None] * mode_means).sum(dim=-2)
            observation = x.new_zeros(batch, nodes, self.input_dim)
            observation[..., : self.output_dim] = expected
            update_parts = [observation]
            if calendar_step is not None:
                update_parts.append(calendar_step)
            state = self.observation_update(
                torch.cat(update_parts, dim=-1).reshape(batch * nodes, -1),
                state.reshape(batch * nodes, -1),
            ).reshape(batch, nodes, -1)
            previous = expected
            means.append(mode_means)
            scales.append(mode_scales)
            logits.append(mode_logits)
        return ForecastDistribution(
            means=torch.stack(means, dim=1),
            scales=torch.stack(scales, dim=1),
            logits=torch.stack(logits, dim=1),
            edge_fluxes=tuple(fluxes),
        )
