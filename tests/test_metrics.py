import pytest
import torch

from dagnn_pistnet.metrics import MetricAccumulator, masked_mae, masked_mape


def test_masked_metrics_ignore_zero_for_mape() -> None:
    target = torch.tensor([0.0, 2.0])
    prediction = torch.tensor([100.0, 3.0])
    assert masked_mae(prediction, target).item() == pytest.approx(50.5)
    assert masked_mape(prediction, target).item() == pytest.approx(50.0)


def test_horizon_metrics_use_exact_step_not_cumulative_average() -> None:
    target = torch.ones(1, 3, 1, 1)
    prediction = torch.tensor([[[[2.0]], [[3.0]], [[4.0]]]])
    accumulator = MetricAccumulator(("flow",), (0, 2))
    accumulator.update(prediction, target)
    metrics = accumulator.compute(step_minutes=5)
    assert metrics["5m"]["flow"]["MAE"] == pytest.approx(1.0)
    assert metrics["15m"]["flow"]["MAE"] == pytest.approx(3.0)
