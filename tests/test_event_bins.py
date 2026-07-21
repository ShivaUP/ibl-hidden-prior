"""Tests for event-bin painting and leakage rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.event_bins import CHANNEL_ORDER_COMMON, trial_to_bins


def _row(**kwargs) -> pd.Series:
    base = dict(
        stimulus_right=1,
        contrast_high=1,
        prev_choice_right=0,
        prev_correct=1,
        prev_fast_rt=0,
        reward=1,
        oracle_prior_right=0,
        stimOn_times=10.0,
        goCue_times=10.02,
        response_times=10.45,
        feedback_times=10.46,
        stimOff_times=11.5,
    )
    base.update(kwargs)
    return pd.Series(base)


def test_bin0_has_stim_not_response_or_reward():
    x, meta = trial_to_bins(_row(), condition="history_only")
    ch = {n: i for i, n in enumerate(meta["channels"])}
    assert x[0, ch["stimulus_right"]] == 1
    assert x[0, ch["contrast_high"]] == 1
    assert x[0, ch["response_made"]] == 0
    assert x[0, ch["reward"]] == 0
    assert x[0, ch["prev_correct"]] == 1


def test_response_and_reward_bins():
    x, meta = trial_to_bins(_row(), condition="history_only")
    ch = {n: i for i, n in enumerate(meta["channels"])}
    assert meta["resp_bin"] is not None
    assert meta["fb_bin"] is not None
    assert x[meta["resp_bin"], ch["response_made"]] == 1
    assert x[meta["fb_bin"], ch["reward"]] == 1
    # No response_made before response bin
    if meta["resp_bin"] > 0:
        assert x[: meta["resp_bin"], ch["response_made"]].sum() == 0


def test_full_info_adds_oracle_channel():
    x, meta = trial_to_bins(_row(oracle_prior_right=1), condition="full_information")
    assert "oracle_prior_right" in meta["channels"]
    assert x.shape[1] == len(CHANNEL_ORDER_COMMON) + 1
    assert np.all(x[:, meta["channels"].index("oracle_prior_right")] == 1)
