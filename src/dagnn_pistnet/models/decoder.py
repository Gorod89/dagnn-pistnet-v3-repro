from __future__ import annotations

import torch
from torch import nn


class SequentialForecastDecoder(nn.Module):
    """Autoregressive GRU decoder with known future calendar covariates."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        forecast_steps: int,
        calendar_dim: int = 0,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.forecast_steps = int(forecast_steps)
        self.calendar_dim = int(calendar_dim)
        self.cell = nn.GRUCell(output_dim + calendar_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        initial_state: torch.Tensor,
        last_observation: torch.Tensor,
        future_calendar: torch.Tensor | None = None,
        teacher_forcing_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        if initial_state.ndim != 3:
            raise ValueError("initial_state must have shape [B,N,H]")
        batch, nodes, hidden_dim = initial_state.shape
        expected_observation = (batch, nodes, self.output_dim)
        if tuple(last_observation.shape) != expected_observation:
            raise ValueError(
                f"last_observation must have shape {expected_observation}, "
                f"got {tuple(last_observation.shape)}"
            )
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0,1]")

        if self.calendar_dim:
            expected_calendar = (batch, self.forecast_steps, self.calendar_dim)
            if future_calendar is None or tuple(future_calendar.shape) != expected_calendar:
                raise ValueError(f"future_calendar must have shape {expected_calendar}")
        else:
            future_calendar = initial_state.new_zeros(batch, self.forecast_steps, 0)

        if teacher_forcing_target is not None:
            expected_target = (batch, self.forecast_steps, nodes, self.output_dim)
            if tuple(teacher_forcing_target.shape) != expected_target:
                raise ValueError(f"teacher_forcing_target must have shape {expected_target}")

        hidden = initial_state
        previous = last_observation
        outputs: list[torch.Tensor] = []
        for step in range(self.forecast_steps):
            calendar_step = future_calendar[:, step, None, :].expand(-1, nodes, -1)
            decoder_input = torch.cat([previous, calendar_step], dim=-1)
            hidden = self.cell(
                decoder_input.reshape(batch * nodes, -1),
                hidden.reshape(batch * nodes, hidden_dim),
            ).reshape(batch, nodes, hidden_dim)
            prediction = self.output_layer(hidden)
            outputs.append(prediction)

            if teacher_forcing_target is not None and teacher_forcing_ratio > 0:
                use_target = (
                    torch.rand(batch, 1, 1, device=prediction.device) < teacher_forcing_ratio
                )
                previous = torch.where(use_target, teacher_forcing_target[:, step], prediction)
            else:
                previous = prediction
        return torch.stack(outputs, dim=1)
