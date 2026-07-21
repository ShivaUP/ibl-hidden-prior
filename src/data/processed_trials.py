"""Build cleaned trial-level tables for behavior-core sessions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.features import add_derived_columns
from src.data.qc import trial_qc_masks


def detect_block_switches(probability_left: pd.Series) -> pd.Series:
    """1 on the first trial of a new probabilityLeft value within a session."""
    prev = probability_left.shift(1)
    switch = probability_left.ne(prev) & prev.notna()
    switch.iloc[0] = False
    return switch.astype(int)


def trials_from_switch(probability_left: pd.Series) -> pd.Series:
    """Trial index relative to most recent switch (0 = switch trial)."""
    switch = detect_block_switches(probability_left).astype(bool)
    # Cumulative count of switches; within each block, count trials
    block_id = switch.cumsum()
    # First block starts at 0 without a switch flag
    pos = probability_left.groupby(block_id).cumcount()
    # For switch trial itself, pos is 0; pre-switch negative encoding is done at analysis time
    return pos.astype(int)


def build_processed_session(
    eid: str,
    trials: pd.DataFrame,
    *,
    rt_percentiles: tuple[float, float] = (1.0, 99.0),
) -> pd.DataFrame:
    """QC-filter a session, add derived features, switch markers, condition flags."""
    masks = trial_qc_masks(trials, rt_percentiles=rt_percentiles)
    if not masks.has_core_fields:
        raise ValueError(f"{eid}: missing core fields")

    # Keep QC-pass trials after RT trim; preserve session order for prev_* features.
    # Compute derived features on full session first so prev_* sees true history,
    # then filter to retained trials.
    featured = add_derived_columns(trials)
    featured["eid"] = eid
    featured["trial_index"] = np.arange(len(featured), dtype=int)
    if "stimOff_times" in trials.columns:
        featured["stimOff_times"] = trials["stimOff_times"].to_numpy()
    featured["qc_pass_rules"] = masks.pass_rules_1_to_4.astype(int)
    featured["qc_pass"] = masks.pass_after_rt_trim.astype(int)
    featured["block_switch"] = detect_block_switches(featured["probabilityLeft"])
    featured["trials_from_block_start"] = trials_from_switch(featured["probabilityLeft"])
    featured["condition_fixed_prior"] = (
        np.isclose(featured["probabilityLeft"], 0.5)
    ).astype(int)
    # Full-info / history-only share the same trial set; channel sets differ later.
    featured["in_full_information"] = featured["qc_pass"]
    featured["in_history_only"] = featured["qc_pass"]
    featured["in_fixed_prior"] = (
        featured["qc_pass"] & featured["condition_fixed_prior"]
    ).astype(int)

    # Choice target helpers. IBL ALF: choice=-1 right, +1 left, 0 no-go.
    featured["choice_right"] = (featured["choice"] == -1).astype(int)
    featured["log_rt"] = np.log(featured["rt"].clip(lower=1e-3))

    kept = featured.loc[featured["qc_pass"] == 1].copy()
    kept.reset_index(drop=True, inplace=True)
    return kept


def schema_columns() -> list[str]:
    """Documented columns expected in processed trial tables."""
    return [
        "eid",
        "trial_index",
        "choice",
        "choice_right",
        "feedbackType",
        "probabilityLeft",
        "contrastLeft",
        "contrastRight",
        "abs_contrast",
        "stimulus_right",
        "contrast_high",
        "reward",
        "rt",
        "log_rt",
        "prev_choice_right",
        "prev_correct",
        "prev_fast_rt",
        "oracle_prior_right",
        "stimOn_times",
        "goCue_times",
        "response_times",
        "feedback_times",
        "block_switch",
        "trials_from_block_start",
        "condition_fixed_prior",
        "in_fixed_prior",
        "in_full_information",
        "in_history_only",
        "qc_pass",
    ]


def summarize_processed(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_trials": int(len(df)),
        "n_eids": int(df["eid"].nunique()) if len(df) else 0,
        "n_fixed_prior": int(df["in_fixed_prior"].sum()) if len(df) else 0,
        "n_switches": int(df["block_switch"].sum()) if len(df) else 0,
        "contrast_high_rate": float(df["contrast_high"].mean()) if len(df) else None,
        "mean_rt": float(df["rt"].mean()) if len(df) else None,
    }
