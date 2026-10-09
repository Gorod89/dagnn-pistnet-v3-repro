from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dagnn_pistnet.data import (
    ChronologicalSplit,
    GraphData,
    StandardScaler,
    TrafficSeries,
    adjacency_to_graph,
    build_adapter,
    build_correlation_graph,
    encode_calendar_features,
)

from .config import V3ExperimentConfig
from .graph import GraphHierarchy, build_topological_hierarchy, induced_coarse_graph


def _load_array(path: str | None, name: str) -> np.ndarray | None:
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".npy":
        return np.asarray(np.load(source, allow_pickle=False), dtype=np.float32)
    if source.suffix.lower() == ".npz":
        archive = np.load(source, allow_pickle=False)
        key = name if name in archive.files else archive.files[0]
        return np.asarray(archive[key], dtype=np.float32)
    if source.suffix.lower() in {".csv", ".txt"}:
        return np.asarray(np.loadtxt(source, delimiter=","), dtype=np.float32)
    raise ValueError(f"unsupported auxiliary array format: {source}")


class V3WindowDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]
):
    def __init__(
        self,
        values: np.ndarray,
        history_steps: int,
        forecast_steps: int,
        target_start: int,
        target_end: int,
        calendar: np.ndarray | None,
        events: np.ndarray | None,
        boundary_flow: np.ndarray | None,
    ) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.history_steps = int(history_steps)
        self.forecast_steps = int(forecast_steps)
        self.first_target = max(int(target_start), history_steps)
        self.last_target = int(target_end) - forecast_steps
        if self.last_target < self.first_target:
            raise ValueError("split is too short for a v3 window")
        time, nodes, _ = self.values.shape
        self.calendar = (
            np.zeros((time, 0), dtype=np.float32)
            if calendar is None
            else np.asarray(calendar, dtype=np.float32)
        )
        self.events = (
            np.zeros((time, nodes, 0), dtype=np.float32)
            if events is None
            else np.asarray(events, dtype=np.float32)
        )
        self.boundary_flow = (
            np.zeros((time, nodes), dtype=np.float32)
            if boundary_flow is None
            else np.asarray(boundary_flow, dtype=np.float32)
        )
        if self.calendar.shape[0] != time:
            raise ValueError("calendar time dimension does not match values")
        if self.events.shape[:2] != (time, nodes):
            raise ValueError("events must have shape [T,N,E]")
        if self.boundary_flow.shape != (time, nodes):
            raise ValueError("boundary_flow must have shape [T,N]")

    def __len__(self) -> int:
        return self.last_target - self.first_target + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = self.first_target + index
        history = slice(start - self.history_steps, start)
        future = slice(start, start + self.forecast_steps)
        return tuple(
            torch.from_numpy(np.asarray(value, dtype=np.float32))
            for value in (
                self.values[history],
                self.values[future],
                self.calendar[history],
                self.calendar[future],
                self.events[future],
                self.boundary_flow[future],
            )
        )


@dataclass(frozen=True)
class PreparedV3Data:
    series: TrafficSeries
    split: ChronologicalSplit
    scaler: StandardScaler
    graph: GraphData
    graph_source: str
    hierarchy: GraphHierarchy
    edge_attributes: torch.Tensor | None
    segment_volume: torch.Tensor | None
    event_dim: int
    calendar_dim: int
    train_dataset: V3WindowDataset
    validation_dataset: V3WindowDataset
    test_dataset: V3WindowDataset


def _official_sliding_window_split(
    total: int,
    history_steps: int,
    forecast_steps: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[ChronologicalSplit, dict[str, tuple[int, int]], int]:
    """Reproduce LargeST's split of eligible sliding-window anchor indices.

    The official LargeST builder first creates every valid ``12 -> 12`` sample,
    then applies Python ``round`` to the 60% and 20% sample counts. Its index is
    the final history timestamp, whereas :class:`V3WindowDataset` indexes the
    first future timestamp; the one-step conversion is made explicit here.
    """

    eligible = total - history_steps - forecast_steps + 1
    if eligible < 1:
        raise ValueError("time series is too short for an official sliding-window split")
    train_windows = round(eligible * train_ratio)
    val_windows = round(eligible * val_ratio)
    test_windows = eligible - train_windows - val_windows
    if min(train_windows, val_windows, test_windows) < 1:
        raise ValueError("official sliding-window split produced an empty partition")

    first_target = history_steps
    first_val_target = first_target + train_windows
    first_test_target = first_val_target + val_windows
    ranges = {
        "train": (first_target, first_val_target + forecast_steps - 1),
        "val": (first_val_target, first_test_target + forecast_steps - 1),
        "test": (first_test_target, total),
    }
    split = ChronologicalSplit(
        train_end=first_val_target,
        val_end=first_test_target,
        total=total,
    )

    # Exact official expression:
    # x_train = data[:idx_val[0] - seq_length_x, :, 0]
    # idx_val[0] is the final-history anchor, one before first_val_target.
    scaler_end = first_val_target - 1 - history_steps
    if scaler_end < 1:
        raise ValueError("official training prefix is empty")
    return split, ranges, scaler_end


def prepare_v3_data(config: V3ExperimentConfig) -> PreparedV3Data:
    data_config = config.data
    series = build_adapter(data_config.to_v2()).load()
    if data_config.split_policy == "official_sliding_windows":
        split, dataset_bounds, scaler_end = _official_sliding_window_split(
            series.time_steps,
            data_config.history_steps,
            data_config.forecast_steps,
            data_config.train_ratio,
            data_config.val_ratio,
        )
    else:
        split = ChronologicalSplit.from_ratios(
            series.time_steps, data_config.train_ratio, data_config.val_ratio
        )
        dataset_bounds = {
            name: split.target_bounds(name) for name in ("train", "val", "test")
        }
        scaler_end = split.train_end
    scaler = StandardScaler.fit(series.values[:scaler_end])
    normalized = scaler.transform(series.values)
    calendar = None
    if data_config.use_calendar_features:
        if series.timestamps is None:
            raise ValueError("calendar features require timestamps")
        calendar = encode_calendar_features(series.timestamps)

    if series.adjacency is not None:
        adjacency = series.adjacency
        graph_source = "dataset"
    else:
        adjacency = build_correlation_graph(
            series.values[: split.train_end], top_k=data_config.graph_top_k
        )
        graph_source = "train-only-correlation"
    graph = adjacency_to_graph(
        adjacency,
        threshold=data_config.graph_threshold,
        add_self_loops=False,
        remove_self_loops=True,
        directed=data_config.graph_directed,
    )

    edge_attributes_array = _load_array(data_config.edge_attributes_path, "edge_attributes")
    edge_attributes = None
    if edge_attributes_array is not None:
        if edge_attributes_array.shape != (graph.edge_index.shape[1], config.model.edge_attr_dim):
            raise ValueError(
                "edge attributes must align with graph edges and configured edge_attr_dim"
            )
        edge_attributes = torch.as_tensor(edge_attributes_array, dtype=torch.float32)
    elif config.model.edge_attr_dim:
        raise ValueError("edge_attr_dim > 0 requires edge_attributes_path")

    cluster_array = _load_array(data_config.cluster_index_path, "cluster_index")
    if cluster_array is not None:
        cluster = np.asarray(cluster_array, dtype=np.int64).reshape(-1)
        if cluster.shape != (series.num_nodes,) or cluster.min() < 0:
            raise ValueError("cluster_index must contain one non-negative id per node")
        # Normalize arbitrary region identifiers to a contiguous range.
        _, cluster = np.unique(cluster, return_inverse=True)
        clusters = int(cluster.max()) + 1
        hierarchy = build_topological_hierarchy(
            series.num_nodes,
            graph.edge_index,
            graph.edge_weight,
            clusters,
            seed=config.training.seed,
        )
        supplied_cluster = torch.as_tensor(cluster, dtype=torch.long)
        coarse_index, coarse_weight = induced_coarse_graph(
            supplied_cluster,
            torch.as_tensor(graph.edge_index, dtype=torch.long),
            torch.as_tensor(graph.edge_weight, dtype=torch.float32),
        )
        hierarchy = GraphHierarchy(
            cluster_index=supplied_cluster,
            coarse_edge_index=coarse_index,
            coarse_edge_weight=coarse_weight,
            num_clusters=clusters,
        )
    else:
        clusters = round(series.num_nodes * config.model.hierarchy_ratio)
        clusters = min(config.model.max_clusters, max(config.model.min_clusters, clusters))
        hierarchy = build_topological_hierarchy(
            series.num_nodes,
            graph.edge_index,
            graph.edge_weight,
            clusters,
            seed=config.training.seed,
        )

    events = _load_array(data_config.event_features_path, "events")
    if events is not None:
        if events.ndim == 2:
            events = events[..., None]
        if events.shape[:2] != (series.time_steps, series.num_nodes):
            raise ValueError("event features must have shape [T,N,E]")
    boundary = _load_array(data_config.boundary_flow_path, "boundary_flow")
    if boundary is not None and boundary.shape != (series.time_steps, series.num_nodes):
        raise ValueError("boundary flow must have shape [T,N]")
    volume_array = _load_array(data_config.segment_volume_path, "segment_volume")
    segment_volume = None
    if volume_array is not None:
        volume_array = volume_array.reshape(-1)
        if volume_array.shape != (series.num_nodes,) or (volume_array <= 0).any():
            raise ValueError("segment volume must be positive with shape [N]")
        segment_volume = torch.as_tensor(volume_array, dtype=torch.float32)

    datasets: dict[str, V3WindowDataset] = {}
    for name in ("train", "val", "test"):
        start, end = dataset_bounds[name]
        datasets[name] = V3WindowDataset(
            normalized,
            history_steps=data_config.history_steps,
            forecast_steps=data_config.forecast_steps,
            target_start=start,
            target_end=end,
            calendar=calendar,
            events=events,
            boundary_flow=boundary,
        )
    return PreparedV3Data(
        series=series,
        split=split,
        scaler=scaler,
        graph=graph,
        graph_source=graph_source,
        hierarchy=hierarchy,
        edge_attributes=edge_attributes,
        segment_volume=segment_volume,
        event_dim=0 if events is None else int(events.shape[-1]),
        calendar_dim=0 if calendar is None else int(calendar.shape[-1]),
        train_dataset=datasets["train"],
        validation_dataset=datasets["val"],
        test_dataset=datasets["test"],
    )
