"""Standard task-trained RNN (GRU) with shared output heads."""

from __future__ import annotations

import torch
from torch import nn

from src.models.interfaces import ModelOutputs, OutputHeads


class StandardRNN(nn.Module):
    """Generic GRU over 100 ms binary event channels."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.heads = OutputHeads(hidden_size)

    def forward(self, batch: dict) -> ModelOutputs:
        x = batch["inputs"]  # (B, T, C)
        lengths = batch["lengths"].cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, h_n = self.rnn(packed)
        # Last layer final hidden state
        h = h_n[-1]  # (B, H)
        return self.heads(h)
