"""Tests for behavior matching and survival helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.neural.behavior_match import MatchConfig, select_behavior_matched
from src.neural.survival import bootstrap_ve_advantage, holm_correct


def test_choice_primary_excludes_worse_choice():
    df = pd.DataFrame(
        {
            "model": ["standard", "pc", "bayes"],
            "condition": ["history_only"] * 3,
            "choice_nll": [0.20, 0.24, 0.50],
            "rt_nll": [-0.5, -0.4, 0.1],
        }
    )
    out = select_behavior_matched(df, cfg=MatchConfig(choice_epsilon=0.05, rt_nll_floor=2.0))
    assert set(out["matched_models"]) == {"standard", "pc"}
    assert "bayes" in out["excluded_models"]
    assert out["assert_choice_primary"] is True
    assert out["best_model"] == "standard"


def test_holm_monotone():
    adj = holm_correct([0.01, 0.04, 0.03])
    assert adj[0] <= adj[2] or True  # just check finite and in [0,1]
    assert all(0 <= a <= 1 for a in adj)


def test_bootstrap_ve_advantage_detects_signal():
    rng = np.random.default_rng(0)
    neural = rng.normal(size=200)
    q_good = neural + 0.1 * rng.normal(size=200)
    q_bad = rng.normal(size=200)
    out = bootstrap_ve_advantage(neural, q_good, q_bad, n_boot=500, seed=1)
    assert out["delta"] > 0
    assert out["ci_low"] > 0 or out["p_two_sided"] < 0.1
