"""Shared model I/O contract for Standard RNN, PC-RNN, and Bayesian models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


@dataclass
class ModelOutputs:
    """Common outputs for behavior + latent-prior comparison."""

    choice_logits: torch.Tensor  # (B,) logit for choice_right=1
    rt_log_mean: torch.Tensor  # (B,) mean of log-RT Gaussian
    rt_log_std: torch.Tensor  # (B,) std of log-RT (positive)
    prior: torch.Tensor  # (B,) q_t in (0, 1), P(right)-like subjective prior

    def choice_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.choice_logits)


class BehaviorModel(Protocol):
    """Protocol all v1 models satisfy."""

    def forward(self, batch: dict) -> ModelOutputs: ...


def extract_latent_prior(outputs: ModelOutputs) -> torch.Tensor:
    """Explicit latent-prior extraction (model-agnostic)."""
    return outputs.prior


def choice_nll(outputs: ModelOutputs, choice_right: torch.Tensor) -> torch.Tensor:
    """Bernoulli NLL on choice_right ∈ {0,1}."""
    probs = outputs.choice_probs().clamp(1e-6, 1 - 1e-6)
    y = choice_right.float()
    return -(y * torch.log(probs) + (1 - y) * torch.log(1 - probs)).mean()


def rt_nll(outputs: ModelOutputs, log_rt: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL on log(RT)."""
    mean = outputs.rt_log_mean
    std = outputs.rt_log_std.clamp_min(1e-3)
    z = (log_rt - mean) / std
    return (0.5 * z.pow(2) + std.log() + 0.5 * torch.log(torch.tensor(2 * torch.pi, device=log_rt.device))).mean()


def joint_loss(
    outputs: ModelOutputs,
    choice_right: torch.Tensor,
    log_rt: torch.Tensor,
    lambda_rt: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L = L_choice + lambda_rt * L_RT with lambda_rt < 1."""
    lc = choice_nll(outputs, choice_right)
    lr = rt_nll(outputs, log_rt)
    total = lc + lambda_rt * lr
    return total, {
        "loss": float(total.detach()),
        "choice_nll": float(lc.detach()),
        "rt_nll": float(lr.detach()),
    }


class OutputHeads(nn.Module):
    """Shared linear heads from a latent vector."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.choice = nn.Linear(hidden_size, 1)
        self.rt_mean = nn.Linear(hidden_size, 1)
        self.rt_log_std = nn.Linear(hidden_size, 1)
        self.prior = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor) -> ModelOutputs:
        return ModelOutputs(
            choice_logits=self.choice(h).squeeze(-1),
            rt_log_mean=self.rt_mean(h).squeeze(-1),
            rt_log_std=torch.nn.functional.softplus(self.rt_log_std(h).squeeze(-1)) + 1e-3,
            prior=torch.sigmoid(self.prior(h).squeeze(-1)),
        )
