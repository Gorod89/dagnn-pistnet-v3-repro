import pytest
import torch

from dagnn_pistnet.physics import DirectedConservationLoss


def test_conservation_loss_is_zero_for_constant_balanced_cycle() -> None:
    constraint = DirectedConservationLoss(density_channel=0, flow_channel=1, dt_seconds=1)
    prediction = torch.ones(2, 4, 3, 2)
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    loss = constraint(prediction, edges, torch.ones(3))
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_conservation_requires_independent_channels() -> None:
    with pytest.raises(ValueError, match="different"):
        DirectedConservationLoss(density_channel=0, flow_channel=0)
