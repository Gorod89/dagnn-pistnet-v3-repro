from __future__ import annotations

from dataclasses import dataclass

import torch


def _valid_mask(target: torch.Tensor, null_threshold: float | None = None) -> torch.Tensor:
    mask = torch.isfinite(target)
    if null_threshold is not None:
        mask = mask & (target.abs() > null_threshold)
    return mask


def masked_mae(
    prediction: torch.Tensor, target: torch.Tensor, null_threshold: float | None = None
) -> torch.Tensor:
    mask = _valid_mask(target, null_threshold)
    if not mask.any():
        return prediction.new_zeros(())
    return (prediction - target).abs()[mask].mean()


def masked_rmse(
    prediction: torch.Tensor, target: torch.Tensor, null_threshold: float | None = None
) -> torch.Tensor:
    mask = _valid_mask(target, null_threshold)
    if not mask.any():
        return prediction.new_zeros(())
    return torch.sqrt(((prediction - target) ** 2)[mask].mean())


def masked_mape(
    prediction: torch.Tensor,
    target: torch.Tensor,
    null_threshold: float = 1e-5,
) -> torch.Tensor:
    mask = _valid_mask(target, null_threshold)
    if not mask.any():
        return prediction.new_zeros(())
    return ((prediction - target).abs()[mask] / target.abs()[mask]).mean() * 100.0


@dataclass
class MetricAccumulator:
    feature_names: tuple[str, ...]
    horizon_indices: tuple[int, ...]
    null_threshold: float = 1e-5

    def __post_init__(self) -> None:
        shape = (len(self.horizon_indices), len(self.feature_names))
        self.absolute_sum = torch.zeros(shape, dtype=torch.float64)
        self.square_sum = torch.zeros(shape, dtype=torch.float64)
        self.percentage_sum = torch.zeros(shape, dtype=torch.float64)
        self.count = torch.zeros(shape, dtype=torch.float64)
        self.percentage_count = torch.zeros(shape, dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().to(device="cpu", dtype=torch.float64)
        target = target.detach().to(device="cpu", dtype=torch.float64)
        for horizon_position, horizon_index in enumerate(self.horizon_indices):
            if horizon_index >= prediction.shape[1]:
                continue
            error = prediction[:, horizon_index] - target[:, horizon_index]
            for feature_index in range(len(self.feature_names)):
                error_feature = error[..., feature_index]
                target_feature = target[:, horizon_index, :, feature_index]
                valid = torch.isfinite(target_feature) & (
                    target_feature.abs() > self.null_threshold
                )
                self.absolute_sum[horizon_position, feature_index] += error_feature.abs()[
                    valid
                ].sum()
                self.square_sum[horizon_position, feature_index] += (
                    error_feature[valid] ** 2
                ).sum()
                self.count[horizon_position, feature_index] += valid.sum()
                self.percentage_sum[horizon_position, feature_index] += (
                    error_feature.abs()[valid] / target_feature.abs()[valid]
                ).sum()
                self.percentage_count[horizon_position, feature_index] += valid.sum()

    def compute(self, step_minutes: int = 5) -> dict[str, dict[str, dict[str, float]]]:
        result: dict[str, dict[str, dict[str, float]]] = {}
        for horizon_position, horizon_index in enumerate(self.horizon_indices):
            horizon_key = f"{(horizon_index + 1) * step_minutes}m"
            result[horizon_key] = {}
            for feature_index, feature_name in enumerate(self.feature_names):
                count = self.count[horizon_position, feature_index].clamp_min(1.0)
                percentage_count = self.percentage_count[horizon_position, feature_index].clamp_min(
                    1.0
                )
                result[horizon_key][feature_name] = {
                    "MAE": float(self.absolute_sum[horizon_position, feature_index] / count),
                    "RMSE": float(
                        torch.sqrt(self.square_sum[horizon_position, feature_index] / count)
                    ),
                    "MAPE": float(
                        self.percentage_sum[horizon_position, feature_index]
                        / percentage_count
                        * 100.0
                    ),
                }
        return result
