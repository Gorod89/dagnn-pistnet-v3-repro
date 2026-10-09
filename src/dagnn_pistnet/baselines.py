from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.pipeline import StandardScaler, TrafficWindowDataset
from .metrics import MetricAccumulator


@torch.no_grad()
def evaluate_naive_baselines(
    test_dataset: TrafficWindowDataset,
    train_values: np.ndarray,
    scaler: StandardScaler,
    feature_names: tuple[str, ...],
    forecast_steps: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate persistence and node-wise historical mean without model training."""

    horizon_indices = tuple(index for index in (0, 2, 5, 11) if index < forecast_steps)
    accumulators = {
        "persistence": MetricAccumulator(feature_names, horizon_indices),
        "historical_mean": MetricAccumulator(feature_names, horizon_indices),
    }
    normalized_train = scaler.transform(train_values)
    historical_mean = torch.from_numpy(normalized_train.mean(axis=0, keepdims=True))
    normalized_mse = {name: 0.0 for name in accumulators}
    sample_count = 0

    for x, target, _, _ in DataLoader(test_dataset, batch_size=batch_size, shuffle=False):
        persistence = x[:, -1:].expand(-1, forecast_steps, -1, -1)
        mean_prediction = historical_mean.expand(x.shape[0], forecast_steps, -1, -1)
        predictions = {
            "persistence": persistence,
            "historical_mean": mean_prediction,
        }
        for name, prediction in predictions.items():
            normalized_mse[name] += float(torch.mean((prediction - target) ** 2)) * x.shape[0]
            accumulators[name].update(
                scaler.inverse_tensor(prediction), scaler.inverse_tensor(target)
            )
        sample_count += x.shape[0]

    if sample_count == 0:
        raise ValueError("empty test dataset")
    return {
        name: {
            "normalized_mse": normalized_mse[name] / sample_count,
            "horizons": accumulator.compute(step_minutes=5),
        }
        for name, accumulator in accumulators.items()
    }
