from __future__ import annotations

import math

import torch
from torch import nn


class AdaptiveGraphLearner(nn.Module):
    """Learn a sparse topology and condition its weights on observed history.

    The topology uses one ``N x N`` score matrix, while per-sample dynamic
    weights are represented as ``B x E``. Targets are never accepted, which
    prevents the leakage present in the original error-derived graph.
    """

    def __init__(
        self,
        num_nodes: int,
        context_dim: int,
        embedding_dim: int = 16,
        top_k: int = 10,
        include_self: bool = True,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        self.num_nodes = int(num_nodes)
        self.top_k = min(max(int(top_k), 1), num_nodes)
        self.include_self = include_self
        self.source_embedding = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.target_embedding = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.context_gate = nn.Linear(context_dim, 1)
        nn.init.xavier_uniform_(self.source_embedding)
        nn.init.xavier_uniform_(self.target_embedding)

    @staticmethod
    def _source_softmax(scores: torch.Tensor, source: torch.Tensor, nodes: int) -> torch.Tensor:
        maximum = torch.full((nodes,), float("-inf"), dtype=scores.dtype, device=scores.device)
        maximum.scatter_reduce_(0, source, scores, reduce="amax", include_self=True)
        exponent = torch.exp(scores - maximum[source])
        denominator = torch.zeros(nodes, dtype=scores.dtype, device=scores.device)
        denominator.scatter_add_(0, source, exponent)
        return exponent / denominator[source].clamp_min(1e-12)

    def forward(
        self,
        context: torch.Tensor,
        node_ids: torch.Tensor | None = None,
        candidate_edge_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3:
            raise ValueError(f"context must have shape [B,N,F], got {tuple(context.shape)}")
        local_nodes = context.shape[1]
        if node_ids is None:
            if local_nodes != self.num_nodes:
                raise ValueError("node_ids are required for a graph partition")
            node_ids = torch.arange(self.num_nodes, device=context.device)
        if node_ids.ndim != 1 or node_ids.shape[0] != local_nodes:
            raise ValueError("node_ids must have shape [N]")

        if candidate_edge_index is not None:
            if candidate_edge_index.ndim != 2 or candidate_edge_index.shape[0] != 2:
                raise ValueError("candidate_edge_index must have shape [2,E]")
            source, target = candidate_edge_index
            source_global = node_ids[source]
            target_global = node_ids[target]
            scores = torch.relu(
                (self.source_embedding[source_global] * self.target_embedding[target_global]).sum(
                    -1
                )
                / math.sqrt(self.source_embedding.shape[-1])
            )
            base_weight = self._source_softmax(scores, source, local_nodes)
            gate = 0.5 + torch.sigmoid(self.context_gate(context).squeeze(-1))
            dynamic_weight = base_weight[None, :] * gate[:, source] * gate[:, target]
            return candidate_edge_index, dynamic_weight

        if local_nodes != self.num_nodes:
            raise ValueError("all-pairs adaptive graph requires every model node")
        scores = torch.relu(
            self.source_embedding
            @ self.target_embedding.T
            / math.sqrt(self.source_embedding.shape[-1])
        )
        if not self.include_self and self.num_nodes > 1:
            scores = scores.masked_fill(
                torch.eye(self.num_nodes, dtype=torch.bool, device=scores.device),
                float("-inf"),
            )
        values, targets = torch.topk(scores, k=self.top_k, dim=-1)
        base_weight = torch.softmax(values, dim=-1).reshape(-1)
        source = torch.arange(self.num_nodes, device=context.device).repeat_interleave(self.top_k)
        target = targets.reshape(-1)
        edge_index = torch.stack([source, target], dim=0)

        gate = 0.5 + torch.sigmoid(self.context_gate(context).squeeze(-1))
        dynamic_weight = base_weight[None, :] * gate[:, source] * gate[:, target]
        return edge_index, dynamic_weight
