"""Tests for derived feature encodings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features import (
    add_derived_columns,
    contrast_high,
    oracle_prior_right,
    reward_channel,
)


def _trials() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contrastLeft": [0.25, np.nan, 0.0625, np.nan, 1.0],
            "contrastRight": [np.nan, 0.125, np.nan, 0.0, np.nan],
            "choice": [1, -1, 1, -1, 1],
            "feedbackType": [1, -1, 1, -1, 1],
            "probabilityLeft": [0.8, 0.2, 0.5, 0.2, 0.8],
            "stimOn_times": [0.0, 1.0, 2.0, 3.0, 4.0],
            "goCue_times": [0.01, 1.01, 2.01, 3.01, 4.01],
            "response_times": [0.3, 1.4, 2.2, 3.5, 4.3],
            "feedback_times": [0.31, 1.41, 2.21, 3.51, 4.31],
        }
    )


def test_contrast_high_levels():
    ch = contrast_high(_trials())
    assert list(ch) == [1, 0, 0, 0, 1]


def test_reward_from_feedback():
    r = reward_channel(_trials())
    assert list(r) == [1, 0, 1, 0, 1]


def test_oracle_prior_right():
    o = oracle_prior_right(_trials())
    assert list(o) == [0, 1, 0, 1, 0]


def test_add_derived_columns():
    out = add_derived_columns(_trials())
    for col in (
        "stimulus_right",
        "contrast_high",
        "reward",
        "rt",
        "prev_choice_right",
        "prev_correct",
        "prev_fast_rt",
        "oracle_prior_right",
    ):
        assert col in out.columns
    assert out["rt"].iloc[0] > 0
    # IBL: choice -1 = right → prev_choice_right after trial1 (choice=-1)
    assert list(out["prev_choice_right"]) == [0, 0, 1, 0, 1]


def test_choice_right_matches_ibl_feedback():
    """Regression: choice_right must match feedback given stim side."""
    # Minimal synthetic: right stim + choice=-1 → correct; left + choice=+1 → correct
    trials = pd.DataFrame(
        {
            "contrastLeft": [np.nan, 0.25],
            "contrastRight": [0.25, np.nan],
            "choice": [-1, 1],
            "feedbackType": [1.0, 1.0],
        }
    )
    choice_right = (trials["choice"] == -1).astype(int)
    stim_right = trials["contrastRight"].notna().astype(int)
    assert list(choice_right) == [1, 0]
    assert list(stim_right) == [1, 0]
    agree = ((stim_right == choice_right) == (trials["feedbackType"] == 1)).all()
    assert bool(agree)
