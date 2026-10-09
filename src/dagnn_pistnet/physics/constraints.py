from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class PhysicsConstraint(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        prediction: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class NoPhysicsConstraint(PhysicsConstraint):
    def forward(
        self,
        prediction: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        del edge_index, edge_weight
        return prediction.new_zeros(())


class DirectedConservationLoss(PhysicsConstraint):
    """Discrete conservation residual for explicitly directed road graphs.

    Predictions must contain independently meaningful density and flow channels.
    The class deliberately does not derive density from speed. Using this loss on
    METR-LA speed or PEMS flow-only files is a configuration error.
    """

    def __init__(
        self,
        density_channel: int,
        flow_channel: int,
        dt_seconds: float = 300.0,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if density_channel == flow_channel:
            raise ValueError("density and flow channels must be different")
        if dt_seconds <= 0 or residual_scale <= 0:
            raise ValueError("dt_seconds and residual_scale must be positive")
        self.density_channel = int(density_channel)
        self.flow_channel = int(flow_channel)
        self.dt_seconds = float(dt_seconds)
        self.residual_scale = float(residual_scale)

    def forward(
        self,
        prediction: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.ndim != 4 or prediction.shape[1] < 2:
            raise ValueError("prediction must have shape [B,T>=2,N,C]")
        channel_count = prediction.shape[-1]
        if max(self.density_channel, self.flow_channel) >= channel_count:
            raise ValueError("configured physics channels are missing from predictions")

        source, target = edge_index
        weights = edge_weight.to(prediction.dtype).clamp_min(0.0)
        node_count = prediction.shape[2]
        out_degree = torch.zeros(node_count, dtype=prediction.dtype, device=prediction.device)
        out_degree.index_add_(0, source, weights)
        normalized = weights / out_degree[source].clamp_min(1e-12)

        density = prediction[..., self.density_channel]
        flow = prediction[..., self.flow_channel]
        edge_flow = flow[:, :-1, source] * normalized[None, None, :]
        divergence = torch.zeros_like(flow[:, :-1])
        divergence.index_add_(2, source, edge_flow)
        divergence.index_add_(2, target, -edge_flow)
        density_rate = (density[:, 1:] - density[:, :-1]) / self.dt_seconds
        residual = density_rate + divergence
        return torch.mean((residual / self.residual_scale) ** 2)
