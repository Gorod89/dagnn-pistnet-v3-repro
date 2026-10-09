from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .model import ForecastDistribution

LOG_TWO_PI = math.log(2.0 * math.pi)


def gaussian_mixture_nll(
    distribution: ForecastDistribution,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if target.shape != distribution.means.shape[:3] + distribution.means.shape[-1:]:
        raise ValueError("target shape does not match forecast distribution")
    expanded = target.unsqueeze(-2)
    log_component = -0.5 * (
        ((expanded - distribution.means) / distribution.scales).square()
        + 2.0 * distribution.scales.log()
        + LOG_TWO_PI
    )
    log_weights = torch.log_softmax(distribution.logits, dim=-1).unsqueeze(-1)
    nll = -torch.logsumexp(log_weights + log_component, dim=-2)
    if mask is not None:
        if mask.shape != target.shape:
            raise ValueError("mask shape must match target")
        return nll[mask].mean() if mask.any() else nll.sum() * 0.0
    return nll.mean()


def mixture_cdf(
    value: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    standardized = (value.unsqueeze(-2) - means) / scales
    component_cdf = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
    return (torch.softmax(logits, dim=-1).unsqueeze(-1) * component_cdf).sum(dim=-2)


def mixture_quantile(
    means: torch.Tensor,
    scales: torch.Tensor,
    logits: torch.Tensor,
    probability: float,
    iterations: int = 32,
) -> torch.Tensor:
    if not 0 < probability < 1:
        raise ValueError("probability must be in (0,1)")
    lower = (means - 8.0 * scales).amin(dim=-2)
    upper = (means + 8.0 * scales).amax(dim=-2)
    for _ in range(iterations):
        midpoint = (lower + upper) * 0.5
        cdf = mixture_cdf(midpoint, means, scales, logits)
        lower = torch.where(cdf < probability, midpoint, lower)
        upper = torch.where(cdf >= probability, midpoint, upper)
    return (lower + upper) * 0.5


def _normal_absolute_moment(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = std.clamp_min(1e-8)
    ratio = mean / std
    phi = torch.exp(-0.5 * ratio.square()) / math.sqrt(2.0 * math.pi)
    return 2.0 * std * phi + mean * torch.erf(ratio / math.sqrt(2.0))


def gaussian_mixture_crps(
    distribution: ForecastDistribution,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact CRPS for a univariate Gaussian mixture, averaged over all scalars."""

    weights = distribution.weights.unsqueeze(-1)
    first = (
        weights
        * _normal_absolute_moment(
            distribution.means - target.unsqueeze(-2), distribution.scales
        )
    ).sum(dim=-2)
    means_i = distribution.means.unsqueeze(-3)
    means_j = distribution.means.unsqueeze(-2)
    scales_i = distribution.scales.unsqueeze(-3)
    scales_j = distribution.scales.unsqueeze(-2)
    pair_std = torch.sqrt(scales_i.square() + scales_j.square())
    pair_moment = _normal_absolute_moment(means_i - means_j, pair_std)
    weights_scalar = distribution.weights
    pair_weights = weights_scalar.unsqueeze(-1) * weights_scalar.unsqueeze(-2)
    second = 0.5 * (pair_weights.unsqueeze(-1) * pair_moment).sum(dim=(-3, -2))
    crps = first - second
    if mask is not None:
        return crps[mask].mean() if mask.any() else crps.sum() * 0.0
    return crps.mean()


@dataclass(frozen=True)
class V3LossComponents:
    total: torch.Tensor
    nll: torch.Tensor
    mae: torch.Tensor
    semigroup: torch.Tensor
    scale_regularization: torch.Tensor
    reconstruction: torch.Tensor
    nonnegative: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach())
            for name in (
                "total",
                "nll",
                "mae",
                "semigroup",
                "scale_regularization",
                "reconstruction",
                "nonnegative",
            )
        }
