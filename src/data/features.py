"""Derived trial features for v1 (project encodings on top of IBL fields)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.qc import absolute_contrast, compute_rt

CONTRAST_HIGH_LEVELS = frozenset({0.25, 0.5, 1.0})
CONTRAST_LOW_LEVELS = frozenset({0.0, 0.0625, 0.125})


def stimulus_right(trials: pd.DataFrame) -> pd.Series:
    """1 if stimulus on right (contrastRight finite), else 0."""
    return trials["contrastRight"].notna().astype(int)


def contrast_high(trials: pd.DataFrame, atol: float = 1e-6) -> pd.Series:
    """Binary high-contrast channel from absolute contrast."""
    abs_c = absolute_contrast(trials)
    high = pd.Series(0, index=trials.index, dtype=int)
    for level in CONTRAST_HIGH_LEVELS:
        high.loc[np.isclose(abs_c, level, atol=atol, equal_nan=False)] = 1
    # Unknown levels stay 0 but can be flagged by caller via abs_c uniqueness.
    return high


def reward_channel(trials: pd.DataFrame) -> pd.Series:
    """reward=1 iff feedbackType == +1."""
    return (trials["feedbackType"] == 1).astype(int)


def prev_choice_right(trials: pd.Series | pd.DataFrame) -> pd.Series:
    """Previous-trial choice was right (IBL choice==-1). First trial = 0."""
    choice = trials["choice"] if isinstance(trials, pd.DataFrame) else trials
    prev = choice.shift(1)
    out = (prev == -1).astype(int)
    out.iloc[0] = 0
    return out


def prev_correct(trials: pd.DataFrame) -> pd.Series:
    """Previous-trial feedbackType == +1. First trial = 0."""
    prev = trials["feedbackType"].shift(1)
    out = (prev == 1).astype(int)
    out.iloc[0] = 0
    return out


def prev_fast_rt(trials: pd.DataFrame) -> pd.Series:
    """Previous RT below session median (among positive RTs). First trial = 0."""
    rt = compute_rt(trials)
    positive = rt[rt > 0]
    median = float(positive.median()) if len(positive) else np.nan
    prev_rt = rt.shift(1)
    out = (prev_rt < median).astype(int)
    out.iloc[0] = 0
    # If prev RT missing/nonpositive, leave 0
    out.loc[prev_rt.isna() | (prev_rt <= 0)] = 0
    out.iloc[0] = 0
    return out


def oracle_prior_right(trials: pd.DataFrame) -> pd.Series:
    """Full-info oracle: 1 if probabilityLeft < 0.5 (right favored), else 0.

    Unbiased 0.5 → 0 (no right bias). Left-favoring 0.8 → 0. Right-favoring 0.2 → 1.
    """
    p = trials["probabilityLeft"]
    return (p < 0.5).astype(int)


def add_derived_columns(trials: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with v1 derived columns attached."""
    out = trials.copy()
    out["abs_contrast"] = absolute_contrast(out)
    out["stimulus_right"] = stimulus_right(out)
    out["contrast_high"] = contrast_high(out)
    out["reward"] = reward_channel(out)
    out["rt"] = compute_rt(out)
    out["prev_choice_right"] = prev_choice_right(out)
    out["prev_correct"] = prev_correct(out)
    out["prev_fast_rt"] = prev_fast_rt(out)
    out["oracle_prior_right"] = oracle_prior_right(out)
    return out
