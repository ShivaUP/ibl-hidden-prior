"""Tests for mouse prior estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.mouse_prior import apply_mouse_prior, fit_mouse_prior
from src.eval.prior_match import prior_match_metrics


def _fake_trials(n: int = 200) -> pd.DataFrame:
    eids = ["a"] * (n // 2) + ["b"] * (n - n // 2)
    tidx = list(range(n // 2)) + list(range(n - n // 2))
    # two blocks
    pleft = np.array([0.2] * (n // 4) + [0.8] * (n // 4) + [0.2] * (n // 4) + [0.8] * (n - 3 * (n // 4)))
    stim = (np.random.default_rng(0).random(n) < pleft).astype(int)
    # choice somewhat follows stim and block
    choice = (np.random.default_rng(1).random(n) < (0.3 + 0.4 * stim + 0.2 * (pleft < 0.5))).astype(int)
    return pd.DataFrame(
        {
            "eid": eids,
            "trial_index": tidx,
            "stimulus_right": stim,
            "abs_contrast": np.where(stim == 1, 0.25, 0.25),
            "choice_right": choice,
            "probabilityLeft": pleft,
            "block_switch": 0,
        }
    )


def test_fit_and_apply_mouse_prior():
    df = _fake_trials()
    params, info = fit_mouse_prior(df, train_eids=["a", "b"])
    assert 0.01 <= params.alpha <= 0.8
    out = apply_mouse_prior(df, params)
    assert out["mouse_prior_hat"].between(0, 1).all()
    assert info["n_trials"] == len(df)


def test_prior_match_metrics():
    df = pd.DataFrame(
        {
            "mouse_prior_hat": [0.2, 0.4, 0.6, 0.8],
            "prior_q": [0.25, 0.35, 0.55, 0.85],
        }
    )
    m = prior_match_metrics(df)
    assert m["corr"] > 0.9
    assert m["rmse"] < 0.1
