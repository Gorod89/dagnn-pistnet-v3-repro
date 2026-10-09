from __future__ import annotations

import math

import torch

from dagnn_pistnet_v3.losses import (
    gaussian_mixture_crps,
    gaussian_mixture_nll,
    mixture_quantile,
)
from dagnn_pistnet_v3.model import ForecastDistribution


def _standard_normal() -> ForecastDistribution:
    return ForecastDistribution(
        means=torch.zeros(2, 1, 3, 1, 1),
        scales=torch.ones(2, 1, 3, 1, 1),
        logits=torch.zeros(2, 1, 3, 1),
    )


def test_standard_normal_nll_and_crps_at_zero() -> None:
    distribution = _standard_normal()
    target = torch.zeros(2, 1, 3, 1)
    assert torch.allclose(
        gaussian_mixture_nll(distribution, target),
        torch.tensor(0.5 * math.log(2 * math.pi)),
        atol=1e-6,
    )
    expected_crps = 2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)
    assert torch.allclose(
        gaussian_mixture_crps(distribution, target),
        torch.tensor(expected_crps),
        atol=1e-6,
    )


def test_mixture_median_is_zero_for_standard_normal() -> None:
    distribution = _standard_normal()
    median = mixture_quantile(
        distribution.means, distribution.scales, distribution.logits, 0.5
    )
    assert torch.allclose(median, torch.zeros_like(median), atol=1e-5)
