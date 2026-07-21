"""Unit tests for session/trial QC (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.qc import score_session, trial_qc_masks


def _clean_trials(n: int = 100) -> pd.DataFrame:
    t0 = np.arange(n, dtype=float) * 10.0
    stim = t0 + 1.0
    go = stim + 0.01
    resp = go + 0.3
    fb = resp + 0.01
    # Alternate left/right contrasts
    contrast_left = np.array([0.25 if i % 2 == 0 else np.nan for i in range(n)])
    contrast_right = np.array([np.nan if i % 2 == 0 else 0.125 for i in range(n)])
    n0 = max(n // 5, 1)
    n1 = max(n // 2, 1)
    n2 = n - n0 - n1
    if n2 < 1:
        n2 = 1
        n1 = n - n0 - n2
    pleft = np.array([0.5] * n0 + [0.2] * n1 + [0.8] * n2, dtype=float)
    assert len(pleft) == n
    return pd.DataFrame(
        {
            "contrastLeft": contrast_left,
            "contrastRight": contrast_right,
            "choice": np.where(np.arange(n) % 3 == 0, -1, 1),
            "feedbackType": np.ones(n),
            "probabilityLeft": pleft,
            "stimOn_times": stim,
            "goCue_times": go,
            "response_times": resp,
            "feedback_times": fb,
        }
    )


def test_clean_session_passes():
    trials = _clean_trials(420)
    result = score_session("fake", trials, min_choice_trials=400)
    assert result.passes_session
    assert result.fraction_pass_rules_1_to_4 >= 0.95


def test_broken_rt_fails_almost_perfect():
    trials = _clean_trials(420)
    # Make half of RTs nonpositive by placing response before goCue
    mid = len(trials) // 2
    trials.loc[:mid, "response_times"] = trials.loc[:mid, "goCue_times"] - 0.1
    result = score_session("fake", trials, min_choice_trials=400)
    assert not result.passes_session
    assert any("pre_trim_fraction" in r or "post_trim_fraction" in r for r in result.fail_reasons)


def test_trial_masks_choice_excludes_nogo():
    trials = _clean_trials(50)
    trials.loc[0, "choice"] = 0
    masks = trial_qc_masks(trials)
    assert not bool(masks.choice_ok.iloc[0])
    assert bool(masks.pass_rules_1_to_4.iloc[1])
