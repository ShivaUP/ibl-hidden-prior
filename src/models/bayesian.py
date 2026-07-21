"""Bayesian / explicit online-inference model (trial-level)."""

from __future__ import annotations

import torch
from torch import nn

from src.models.interfaces import ModelOutputs


class BayesianOnlineModel(nn.Module):
    """Explicit online prior update + choice/RT observation model.

    Maintains q_t = P(right) across trials within a session (reset on session_start).
    """

    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.input_size = input_size
        self.evidence = nn.Linear(input_size, 1)
        self.choice_stim = nn.Linear(input_size, 1)
        self.rt_net = nn.Sequential(
            nn.Linear(input_size + 1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )
        self.prior_logit0 = nn.Parameter(torch.zeros(()))
        self.leak_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid -> (0,1)
        self.prior_to_choice = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: dict) -> ModelOutputs:
        feats = batch["features"]  # (B, F)
        starts = batch["session_start"]  # (B,)
        bsz = feats.shape[0]
        leak = torch.sigmoid(self.leak_raw)

        qs: list[torch.Tensor] = []
        qi = torch.sigmoid(self.prior_logit0)
        for i in range(bsz):
            if bool(starts[i].item()) or i == 0:
                qi = torch.sigmoid(self.prior_logit0)
            nudge = torch.sigmoid(self.evidence(feats[i : i + 1])).view(())
            qi = (1.0 - leak) * qi + leak * nudge
            qs.append(qi)
        prior = torch.stack(qs)  # (B,)

        choice_logits = (
            self.prior_to_choice * (2.0 * prior - 1.0)
            + self.choice_stim(feats).squeeze(-1)
        )
        rt_params = self.rt_net(torch.cat([feats, prior.unsqueeze(-1)], dim=-1))
        rt_log_mean = rt_params[:, 0]
        rt_log_std = torch.nn.functional.softplus(rt_params[:, 1]) + 1e-3

        return ModelOutputs(
            choice_logits=choice_logits,
            rt_log_mean=rt_log_mean,
            rt_log_std=rt_log_std,
            prior=prior,
        )
