from __future__ import annotations

import numpy as np

from dagnn_pistnet_v3.graph import build_topological_hierarchy


def test_hierarchy_is_deterministic_and_covers_nodes() -> None:
    edges = np.asarray([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    first = build_topological_hierarchy(6, edges, np.ones(5), 2, seed=7)
    second = build_topological_hierarchy(6, edges, np.ones(5), 2, seed=7)
    assert np.array_equal(first.cluster_index.numpy(), second.cluster_index.numpy())
    assert set(first.cluster_index.tolist()) == {0, 1}
    assert first.coarse_edge_index.shape[0] == 2
