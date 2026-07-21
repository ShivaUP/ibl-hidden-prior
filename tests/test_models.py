"""Smoke tests for model interfaces and forward shapes."""

from __future__ import annotations

import torch

from src.models.bayesian import BayesianOnlineModel
from src.models.interfaces import extract_latent_prior, joint_loss
from src.models.pc_rnn import PredictiveCodingRNN
from src.models.standard_rnn import StandardRNN


def test_standard_rnn_forward():
    model = StandardRNN(input_size=9, hidden_size=16)
    bsz, t, c = 4, 7, 9
    batch = {
        "inputs": torch.randn(bsz, t, c),
        "mask": torch.ones(bsz, t, dtype=torch.bool),
        "lengths": torch.tensor([7, 5, 6, 4]),
        "choice_right": torch.randint(0, 2, (bsz,)),
        "log_rt": torch.randn(bsz),
    }
    out = model(batch)
    assert out.choice_logits.shape == (bsz,)
    assert out.prior.shape == (bsz,)
    probs = out.choice_probs()
    assert torch.all((probs > 0) & (probs < 1))
    loss, _ = joint_loss(out, batch["choice_right"], batch["log_rt"], lambda_rt=0.2)
    loss.backward()


def test_pc_rnn_forward():
    model = PredictiveCodingRNN(input_size=9, hidden_size=16)
    bsz, t, c = 3, 5, 9
    batch = {
        "inputs": torch.randn(bsz, t, c),
        "mask": torch.ones(bsz, t, dtype=torch.bool),
        "lengths": torch.tensor([5, 5, 4]),
        "choice_right": torch.randint(0, 2, (bsz,)),
        "log_rt": torch.randn(bsz),
    }
    out = model(batch)
    assert torch.isfinite(extract_latent_prior(out)).all()


def test_bayes_forward():
    model = BayesianOnlineModel(input_size=5, hidden_size=8)
    bsz = 6
    batch = {
        "features": torch.randn(bsz, 5),
        "session_start": torch.tensor([True, False, False, True, False, False]),
        "choice_right": torch.randint(0, 2, (bsz,)),
        "log_rt": torch.randn(bsz),
    }
    out = model(batch)
    assert out.prior.min() >= 0 and out.prior.max() <= 1
