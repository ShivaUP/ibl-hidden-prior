"""Predictive-coding RNN: same I/O as StandardRNN, PE-style state update."""

from __future__ import annotations

import torch
from torch import nn

from src.models.interfaces import ModelOutputs, OutputHeads


class PCCell(nn.Module):
    """One-step predictive-coding style update.

    Predict next input from hidden state; update on prediction error.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.predict = nn.Linear(hidden_size, input_size)
        self.error_to_h = nn.Linear(input_size, hidden_size)
        self.recur = nn.Linear(hidden_size, hidden_size)
        self.act = nn.Tanh()

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_hat = self.predict(h)
        err = x_t - x_hat
        h = self.act(self.recur(h) + self.error_to_h(err))
        return h, err


class PredictiveCodingRNN(nn.Module):
    """RNN with explicit prediction-error hidden updates (fair I/O vs StandardRNN)."""

    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = PCCell(input_size, hidden_size)
        self.heads = OutputHeads(hidden_size)

    def forward(self, batch: dict) -> ModelOutputs:
        x = batch["inputs"]  # (B, T, C)
        mask = batch["mask"]  # (B, T)
        bsz, tmax, _ = x.shape
        h = torch.zeros(bsz, self.hidden_size, device=x.device, dtype=x.dtype)
        # Keep last valid hidden per sequence
        last_h = h.clone()
        for t in range(tmax):
            h_new, _ = self.cell(x[:, t, :], h)
            m = mask[:, t].unsqueeze(-1).float()
            h = h_new * m + h * (1.0 - m)
            last_h = torch.where(mask[:, t].unsqueeze(-1), h, last_h)
        return self.heads(last_h)
