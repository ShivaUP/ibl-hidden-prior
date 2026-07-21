"""Session and trial QC for almost-perfect behavior-core inclusion.

Rules follow docs/spec.md and configs/frozen_v1.yaml.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

CORE_FIELDS: tuple[str, ...] = (
    "contrastLeft",
    "contrastRight",
    "choice",
    "feedbackType",
    "probabilityLeft",
    "stimOn_times",
    "goCue_times",
    "response_times",
    "feedback_times",
)


@dataclass
class TrialQCMasks:
    """Boolean masks aligned to trials index."""

    has_core_fields: bool
    finite_events: pd.Series
    monotonic: pd.Series
    rt_positive: pd.Series
    choice_ok: pd.Series
    pass_rules_1_to_4: pd.Series
    pass_after_rt_trim: pd.Series
    rt: pd.Series


@dataclass
class SessionQCResult:
    """QC summary for one eid."""

    eid: str
    n_trials: int
    n_choice_trials: int
    has_core_fields: bool
    missing_fields: list[str]
    probability_left_values: list[float]
    has_expected_probability_set: bool
    has_left_bias_block: bool
    has_right_bias_block: bool
    has_unbiased_block: bool
    n_pass_rules_1_to_4: int
    fraction_pass_rules_1_to_4: float
    n_pass_after_rt_trim: int
    fraction_after_rt_trim: float
    passes_min_choice_trials: bool
    passes_bias_blocks: bool
    passes_almost_perfect_pre: bool
    passes_almost_perfect_post: bool
    passes_session: bool
    fail_reasons: list[str] = field(default_factory=list)
    absolute_contrast_levels: list[float] = field(default_factory=list)
    rt_median: float | None = None
    n_nonpositive_rt: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_rt(trials: pd.DataFrame) -> pd.Series:
    """RT = response - goCue, with stimOn fallback if goCue missing on a trial."""
    response = trials["response_times"]
    if "goCue_times" in trials.columns:
        rt = response - trials["goCue_times"]
        if "stimOn_times" in trials.columns:
            missing_go = trials["goCue_times"].isna()
            rt = rt.where(~missing_go, response - trials["stimOn_times"])
        return rt
    return response - trials["stimOn_times"]


def absolute_contrast(trials: pd.DataFrame) -> pd.Series:
    left = trials["contrastLeft"]
    right = trials["contrastRight"]
    out = pd.Series(np.nan, index=trials.index, dtype=float)
    left_only = left.notna() & right.isna()
    right_only = right.notna() & left.isna()
    both = left.notna() & right.notna()
    out.loc[left_only] = left.loc[left_only].abs()
    out.loc[right_only] = right.loc[right_only].abs()
    out.loc[both] = np.maximum(left.loc[both].abs(), right.loc[both].abs())
    return out


def trial_qc_masks(
    trials: pd.DataFrame,
    rt_percentiles: tuple[float, float] = (1.0, 99.0),
) -> TrialQCMasks:
    """Build trial-level QC masks (rules 1–4, then RT percentile trim)."""
    missing = [c for c in CORE_FIELDS if c not in trials.columns]
    has_core = len(missing) == 0
    if not has_core:
        empty = pd.Series(False, index=trials.index)
        return TrialQCMasks(
            has_core_fields=False,
            finite_events=empty,
            monotonic=empty,
            rt_positive=empty,
            choice_ok=empty,
            pass_rules_1_to_4=empty,
            pass_after_rt_trim=empty,
            rt=pd.Series(np.nan, index=trials.index),
        )

    finite_events = (
        trials["stimOn_times"].notna()
        & trials["goCue_times"].notna()
        & trials["response_times"].notna()
        & trials["feedback_times"].notna()
    )
    monotonic = finite_events & (
        (trials["stimOn_times"] <= trials["goCue_times"])
        & (trials["goCue_times"] <= trials["response_times"])
        & (trials["response_times"] <= trials["feedback_times"])
    )
    rt = compute_rt(trials)
    rt_positive = monotonic & (rt > 0)
    choice_ok = trials["choice"].isin([-1, 1])
    pass_1_to_4 = finite_events & monotonic & rt_positive & choice_ok

    pass_after = pass_1_to_4.copy()
    if pass_1_to_4.any():
        rt_ok = rt.loc[pass_1_to_4]
        lo = np.nanpercentile(rt_ok, rt_percentiles[0])
        hi = np.nanpercentile(rt_ok, rt_percentiles[1])
        in_band = (rt >= lo) & (rt <= hi)
        pass_after = pass_1_to_4 & in_band

    return TrialQCMasks(
        has_core_fields=True,
        finite_events=finite_events,
        monotonic=monotonic,
        rt_positive=rt_positive,
        choice_ok=choice_ok,
        pass_rules_1_to_4=pass_1_to_4,
        pass_after_rt_trim=pass_after,
        rt=rt,
    )


def score_session(
    eid: str,
    trials: pd.DataFrame,
    *,
    min_choice_trials: int = 400,
    min_fraction_pre: float = 0.95,
    min_fraction_post: float = 0.90,
    expected_probability_left: tuple[float, ...] = (0.2, 0.5, 0.8),
    rt_percentiles: tuple[float, float] = (1.0, 99.0),
    require_left_right_bias: bool = True,
) -> SessionQCResult:
    """Score one session against almost-perfect inclusion rules."""
    missing = [c for c in CORE_FIELDS if c not in trials.columns]
    masks = trial_qc_masks(trials, rt_percentiles=rt_percentiles)
    n_trials = int(len(trials))
    n_choice = int(masks.choice_ok.sum()) if masks.has_core_fields else 0

    pleft_vals: list[float] = []
    has_left = has_right = has_unbiased = False
    abs_levels: list[float] = []
    if masks.has_core_fields:
        pleft_vals = sorted({float(v) for v in trials["probabilityLeft"].dropna().unique()})
        has_left = any(abs(v - 0.8) < 1e-9 for v in pleft_vals)
        has_right = any(abs(v - 0.2) < 1e-9 for v in pleft_vals)
        # 0.2 favors right stim (low P(left)); 0.8 favors left.
        has_unbiased = any(abs(v - 0.5) < 1e-9 for v in pleft_vals)
        abs_c = absolute_contrast(trials).dropna().unique()
        abs_levels = sorted({round(float(v), 4) for v in abs_c})

    expected_set = {round(float(v), 4) for v in expected_probability_left}
    observed_set = {round(float(v), 4) for v in pleft_vals}
    has_expected = observed_set.issubset(expected_set) and len(observed_set) > 0

    n_pre = int(masks.pass_rules_1_to_4.sum()) if masks.has_core_fields else 0
    n_post = int(masks.pass_after_rt_trim.sum()) if masks.has_core_fields else 0
    # Denominator: completed choice trials (choice ±1), per spec session gate wording.
    denom = max(n_choice, 1)
    frac_pre = n_pre / denom if n_choice else 0.0
    frac_post = n_post / denom if n_choice else 0.0

    passes_min = n_choice >= min_choice_trials
    passes_bias = (has_left and has_right) if require_left_right_bias else True
    passes_pre = frac_pre >= min_fraction_pre
    passes_post = frac_post >= min_fraction_post

    fail_reasons: list[str] = []
    if missing:
        fail_reasons.append(f"missing_fields:{','.join(missing)}")
    if not passes_min:
        fail_reasons.append(f"too_few_choice_trials:{n_choice}<{min_choice_trials}")
    if not has_expected:
        fail_reasons.append(f"unexpected_probabilityLeft:{pleft_vals}")
    if require_left_right_bias and not passes_bias:
        fail_reasons.append("missing_left_or_right_bias_block")
    if not passes_pre:
        fail_reasons.append(f"pre_trim_fraction:{frac_pre:.4f}<{min_fraction_pre}")
    if not passes_post:
        fail_reasons.append(f"post_trim_fraction:{frac_post:.4f}<{min_fraction_post}")

    passes = (
        masks.has_core_fields
        and passes_min
        and has_expected
        and passes_bias
        and passes_pre
        and passes_post
    )

    rt_median = None
    n_nonpos = 0
    if masks.has_core_fields:
        finite_rt = masks.rt.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite_rt):
            rt_median = float(finite_rt.median())
        n_nonpos = int((masks.rt <= 0).sum())

    return SessionQCResult(
        eid=eid,
        n_trials=n_trials,
        n_choice_trials=n_choice,
        has_core_fields=masks.has_core_fields,
        missing_fields=missing,
        probability_left_values=pleft_vals,
        has_expected_probability_set=has_expected,
        has_left_bias_block=has_left,
        has_right_bias_block=has_right,
        has_unbiased_block=has_unbiased,
        n_pass_rules_1_to_4=n_pre,
        fraction_pass_rules_1_to_4=float(frac_pre),
        n_pass_after_rt_trim=n_post,
        fraction_after_rt_trim=float(frac_post),
        passes_min_choice_trials=passes_min,
        passes_bias_blocks=passes_bias,
        passes_almost_perfect_pre=passes_pre,
        passes_almost_perfect_post=passes_post,
        passes_session=passes,
        fail_reasons=fail_reasons,
        absolute_contrast_levels=abs_levels,
        rt_median=rt_median,
        n_nonpositive_rt=n_nonpos,
    )
