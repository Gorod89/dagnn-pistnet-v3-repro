from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GraphHierarchy:
    cluster_index: torch.Tensor
    coarse_edge_index: torch.Tensor
    coarse_edge_weight: torch.Tensor
    num_clusters: int

    def to(self, device: torch.device | str) -> GraphHierarchy:
        return GraphHierarchy(
            cluster_index=self.cluster_index.to(device),
            coarse_edge_index=self.coarse_edge_index.to(device),
            coarse_edge_weight=self.coarse_edge_weight.to(device),
            num_clusters=self.num_clusters,
        )


def build_topological_hierarchy(
    num_nodes: int,
    edge_index: np.ndarray | torch.Tensor,
    edge_weight: np.ndarray | torch.Tensor,
    num_clusters: int,
    seed: int = 42,
) -> GraphHierarchy:
    """Deterministic locality-preserving graph coarsening.

    This is deliberately topology based. It never treats sensor indices as a
    spatial coordinate. Published experiments may replace the assignments with
    metadata-derived corridor/region identifiers.
    """

    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    num_clusters = min(max(int(num_clusters), 1), num_nodes)
    edges = torch.as_tensor(edge_index, dtype=torch.long).cpu().numpy()
    weights = torch.as_tensor(edge_weight, dtype=torch.float32).cpu().numpy()
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] != weights.shape[0]:
        raise ValueError("invalid fine graph")

    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in zip(edges[0], edges[1], strict=True):
        source_i, target_i = int(source), int(target)
        if source_i != target_i:
            neighbors[source_i].append(target_i)
            neighbors[target_i].append(source_i)
    for items in neighbors:
        items.sort()

    rng = np.random.default_rng(seed)
    roots = rng.permutation(num_nodes).tolist()
    assignment = np.full(num_nodes, -1, dtype=np.int64)
    target_size = int(np.ceil(num_nodes / num_clusters))
    root_cursor = 0
    for cluster in range(num_clusters):
        while root_cursor < num_nodes and assignment[roots[root_cursor]] >= 0:
            root_cursor += 1
        if root_cursor >= num_nodes:
            break
        queue: deque[int] = deque([roots[root_cursor]])
        while queue and np.count_nonzero(assignment == cluster) < target_size:
            node = queue.popleft()
            if assignment[node] >= 0:
                continue
            assignment[node] = cluster
            queue.extend(neighbor for neighbor in neighbors[node] if assignment[neighbor] < 0)

    # Disconnected leftovers go to the currently smallest cluster.
    counts = np.bincount(assignment[assignment >= 0], minlength=num_clusters)
    for node in np.flatnonzero(assignment < 0):
        cluster = int(np.argmin(counts))
        assignment[node] = cluster
        counts[cluster] += 1

    coarse: dict[tuple[int, int], float] = {}
    for (source, target), weight in zip(edges.T, weights, strict=True):
        coarse_source = int(assignment[int(source)])
        coarse_target = int(assignment[int(target)])
        if coarse_source == coarse_target:
            continue
        key = (coarse_source, coarse_target)
        coarse[key] = coarse.get(key, 0.0) + float(weight)
    if coarse:
        pairs = np.asarray(sorted(coarse), dtype=np.int64).T
        coarse_weights = np.asarray([coarse[tuple(pair)] for pair in pairs.T], dtype=np.float32)
        outgoing = np.bincount(pairs[0], weights=coarse_weights, minlength=num_clusters)
        coarse_weights /= np.maximum(outgoing[pairs[0]], 1e-8)
    else:
        pairs = np.empty((2, 0), dtype=np.int64)
        coarse_weights = np.empty((0,), dtype=np.float32)
    return GraphHierarchy(
        cluster_index=torch.as_tensor(assignment, dtype=torch.long),
        coarse_edge_index=torch.as_tensor(pairs, dtype=torch.long),
        coarse_edge_weight=torch.as_tensor(coarse_weights, dtype=torch.float32),
        num_clusters=num_clusters,
    )


def induced_coarse_graph(
    cluster_index: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a coarse graph on-device for externally supplied clusters."""

    source = cluster_index.index_select(0, edge_index[0])
    target = cluster_index.index_select(0, edge_index[1])
    mask = source != target
    source, target, weights = source[mask], target[mask], edge_weight[mask]
    if source.numel() == 0:
        return edge_index.new_empty((2, 0)), edge_weight.new_empty((0,))
    num_clusters = int(cluster_index.max()) + 1
    flat = source * num_clusters + target
    unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    aggregated = weights.new_zeros(unique.numel())
    aggregated.index_add_(0, inverse, weights)
    coarse_source = torch.div(unique, num_clusters, rounding_mode="floor")
    coarse_target = unique.remainder(num_clusters)
    degree = weights.new_zeros(num_clusters)
    degree.index_add_(0, coarse_source, aggregated)
    aggregated = aggregated / degree.index_select(0, coarse_source).clamp_min(1e-8)
    return torch.stack([coarse_source, coarse_target]), aggregated
