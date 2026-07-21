"""Tests for Phase 6 metric helpers."""

from __future__ import annotations

import pandas as pd

from src.eval.behavior_choice import choice_metrics
from src.eval.switch_centered import extract_switch_windows


def test_choice_metrics_perfect():
    df = pd.DataFrame({"choice_right": [1, 0, 1, 0], "p_right": [0.99, 0.01, 0.99, 0.01]})
    m = choice_metrics(df)
    assert m["choice_acc"] == 1.0
    assert m["choice_nll"] < 0.1


def test_switch_window_extraction():
    rows = []
    # one session with a switch at trial_index 20
    for i in range(50):
        pleft = 0.2 if i < 20 else 0.8
        rows.append(
            {
                "eid": "e1",
                "trial_index": i,
                "probabilityLeft": pleft,
                "block_switch": int(i == 20),
                "choice_right": 1 if pleft < 0.5 else 0,
                "p_right": 0.7 if pleft < 0.5 else 0.3,
                "prior_q": 0.5,
            }
        )
    df = pd.DataFrame(rows)
    sw = extract_switch_windows(df, pre=10, post=30, min_pre=8, min_post=16)
    assert (sw["rel_trial"] == 0).sum() == 1
    assert sw["rel_trial"].min() >= -10
    assert sw["rel_trial"].max() <= 30
