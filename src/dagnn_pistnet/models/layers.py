from __future__ import annotations

import sys

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """Encode every sensor history with shared recurrent parameters."""

    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,T,N,F], got {tuple(x.shape)}")
        batch, time, nodes, features = x.shape
        per_node = x.permute(0, 2, 1, 3).reshape(batch * nodes, time, features)
        # cuDNN RNN teardown can terminate Windows CUDA processes with
        # 0xC0000409 when the recurrent output feeds graph autograd operators.
        # Native PyTorch kernels avoid that runtime bug while keeping tensors on GPU.
        use_cudnn = not (sys.platform == "win32" and per_node.is_cuda)
        with torch.backends.cudnn.flags(enabled=use_cudnn):
            _, (hidden, _) = self.lstm(per_node)
        return hidden[-1].reshape(batch, nodes, -1)


class EdgeGraphConv(nn.Module):
    """Memory-bounded weighted message passing over an edge list.

    ``edge_index[0]`` is the source and ``edge_index[1]`` is the target. Edge
    weights can be shared ``[E]`` or context-conditioned ``[B,E]``.
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.message = nn.Linear(input_dim, output_dim, bias=False)
        self.root = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B,N,F]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        batch, nodes, _ = x.shape
        source, target = edge_index[0], edge_index[1]
        if source.numel() != edge_weight.shape[-1]:
            raise ValueError("edge count and edge weights differ")
        if source.numel() == 0:
            return self.root(x)

        projected = self.message(x)
        if edge_weight.ndim == 1:
            degree = torch.zeros(nodes, dtype=x.dtype, device=x.device)
            degree.index_add_(0, target, edge_weight.abs().to(x.dtype))
            normalized = edge_weight.to(x.dtype) / degree[target].clamp_min(1e-12)
            messages = projected[:, source, :] * normalized[None, :, None]
        elif edge_weight.ndim == 2:
            if edge_weight.shape[0] != batch:
                raise ValueError("batched edge weights do not match x batch")
            degree = torch.zeros(batch, nodes, dtype=x.dtype, device=x.device)
            degree.scatter_add_(1, target[None, :].expand(batch, -1), edge_weight.abs().to(x.dtype))
            normalized = edge_weight.to(x.dtype) / degree[:, target].clamp_min(1e-12)
            messages = projected[:, source, :] * normalized[..., None]
        else:
            raise ValueError("edge_weight must have shape [E] or [B,E]")

        aggregated = torch.zeros(
            batch, nodes, projected.shape[-1], dtype=projected.dtype, device=projected.device
        )
        aggregated.index_add_(1, target, messages)
        return aggregated + self.root(x)


class BidirectionalDiffusionConv(nn.Module):
    """Forward/reverse random-walk diffusion over multiple graph hops."""

    def __init__(self, input_dim: int, output_dim: int, diffusion_steps: int = 2) -> None:
        super().__init__()
        if diffusion_steps < 1:
            raise ValueError("diffusion_steps must be positive")
        self.diffusion_steps = int(diffusion_steps)
        support_count = 1 + 2 * self.diffusion_steps
        self.projection = nn.Linear(input_dim * support_count, output_dim)

    @staticmethod
    def _propagate(
        x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        if edge_weight.ndim != 1:
            raise ValueError("diffusion edge weights must have shape [E]")
        source, target = edge_index
        nodes = x.shape[1]
        if source.numel() == 0:
            return torch.zeros_like(x)
        out_degree = torch.zeros(nodes, dtype=x.dtype, device=x.device)
        out_degree.index_add_(0, source, edge_weight.abs().to(x.dtype))
        transition = edge_weight.to(x.dtype) / out_degree[source].clamp_min(1e-12)
        messages = x[:, source, :] * transition[None, :, None]
        output = torch.zeros_like(x)
        output.index_add_(1, target, messages)
        return output

    def diffusion_features(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B,N,F]")
        features = [x]
        for support in (edge_index, edge_index.flip(0)):
            state = x
            for _ in range(self.diffusion_steps):
                state = self._propagate(state, support, edge_weight)
                features.append(state)
        return torch.cat(features, dim=-1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        return self.projection(self.diffusion_features(x, edge_index, edge_weight))
