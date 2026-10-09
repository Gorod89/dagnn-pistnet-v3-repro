from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from .losses import gaussian_mixture_crps, gaussian_mixture_nll, mixture_quantile
from .model import ForecastDistribution


def rescale_distribution(
    distribution: ForecastDistribution,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> ForecastDistribution:
    while mean.ndim < distribution.means.ndim:
        mean = mean.unsqueeze(-2)
        std = std.unsqueeze(-2)
    return ForecastDistribution(
        means=distribution.means * std + mean,
        scales=distribution.scales * std.abs(),
        logits=distribution.logits,
        edge_fluxes=distribution.edge_fluxes,
    )


class PublicationMetricAccumulator:
    def __init__(
        self,
        feature_names: tuple[str, ...],
        horizon_indices: tuple[int, ...],
        sampling_seconds: int,
        null_threshold: float = 1e-5,
    ) -> None:
        self.feature_names = feature_names
        self.horizon_indices = horizon_indices
        self.sampling_seconds = sampling_seconds
        self.null_threshold = null_threshold
        self.sums: dict[tuple[int, int], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.window_errors: dict[int, list[np.ndarray]] = defaultdict(list)

    @torch.no_grad()
    def update(self, distribution: ForecastDistribution, target: torch.Tensor) -> None:
        point = distribution.mean
        for horizon in self.horizon_indices:
            for feature in range(target.shape[-1]):
                y = target[:, horizon, :, feature]
                p = point[:, horizon, :, feature]
                valid = torch.isfinite(y) & (y.abs() > self.null_threshold)
                count = int(valid.sum())
                if count == 0:
                    continue
                error = p - y
                absolute = error.abs()
                key = (horizon, feature)
                stats = self.sums[key]
                stats["count"] += count
                stats["absolute"] += float(absolute[valid].sum())
                stats["squared"] += float(error[valid].square().sum())
                stats["target_absolute"] += float(y[valid].abs().sum())
                stats["smape"] += float(
                    (2.0 * absolute / (p.abs() + y.abs()).clamp_min(1e-5))[valid].sum()
                )

                sub = ForecastDistribution(
                    means=distribution.means[:, horizon : horizon + 1, :, :, feature : feature + 1],
                    scales=distribution.scales[
                        :, horizon : horizon + 1, :, :, feature : feature + 1
                    ],
                    logits=distribution.logits[:, horizon : horizon + 1],
                )
                sub_target = target[:, horizon : horizon + 1, :, feature : feature + 1]
                sub_mask = valid[:, None, :, None]
                stats["nll"] += float(gaussian_mixture_nll(sub, sub_target, sub_mask)) * count
                stats["crps"] += float(gaussian_mixture_crps(sub, sub_target, sub_mask)) * count

                low80 = mixture_quantile(sub.means, sub.scales, sub.logits, 0.10).squeeze((1, 3))
                high80 = mixture_quantile(sub.means, sub.scales, sub.logits, 0.90).squeeze((1, 3))
                low95 = mixture_quantile(sub.means, sub.scales, sub.logits, 0.025).squeeze((1, 3))
                high95 = mixture_quantile(sub.means, sub.scales, sub.logits, 0.975).squeeze((1, 3))
                stats["coverage80"] += float(((y >= low80) & (y <= high80) & valid).sum())
                stats["coverage95"] += float(((y >= low95) & (y <= high95) & valid).sum())
                stats["width80"] += float((high80 - low80)[valid].sum())
                stats["width95"] += float((high95 - low95)[valid].sum())

                per_window_valid = valid.sum(dim=1).clamp_min(1)
                per_window = (absolute * valid).sum(dim=1) / per_window_valid
                self.window_errors[horizon].append(per_window.cpu().numpy())

    def compute(self) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
        step_minutes = self.sampling_seconds / 60.0
        result: dict[str, Any] = {}
        for (horizon, feature), stats in sorted(self.sums.items()):
            count = max(stats["count"], 1.0)
            horizon_key = f"{(horizon + 1) * step_minutes:g}min"
            result.setdefault(horizon_key, {})[self.feature_names[feature]] = {
                "mae": stats["absolute"] / count,
                "rmse": math.sqrt(stats["squared"] / count),
                "wape": stats["absolute"] / max(stats["target_absolute"], 1e-8),
                "smape": stats["smape"] / count,
                "nll": stats["nll"] / count,
                "crps": stats["crps"] / count,
                "coverage80": stats["coverage80"] / count,
                "coverage95": stats["coverage95"] / count,
                "width80": stats["width80"] / count,
                "width95": stats["width95"] / count,
                "count": int(stats["count"]),
            }
        windows = {
            horizon: np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
            for horizon, chunks in self.window_errors.items()
        }
        return result, windows
