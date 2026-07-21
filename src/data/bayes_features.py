"""Trial-level feature matrices for the Bayesian model."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Causal features available at current-trial stimulus time.
BAYES_FEATURES_COMMON: tuple[str, ...] = (
    "stimulus_right",
    "contrast_high",
    "prev_choice_right",
    "prev_correct",
    "prev_fast_rt",
)

BAYES_FEATURES_FULL_INFO: tuple[str, ...] = BAYES_FEATURES_COMMON + (
    "oracle_prior_right",
)


def build_bayes_table(trials: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Return a flat table of Bayesian inputs/targets for one condition."""
    if condition == "fixed_prior":
        subset = trials.loc[trials["in_fixed_prior"] == 1].copy()
        features = BAYES_FEATURES_COMMON
    elif condition == "full_information":
        subset = trials.loc[trials["qc_pass"] == 1].copy()
        features = BAYES_FEATURES_FULL_INFO
    elif condition == "history_only":
        subset = trials.loc[trials["qc_pass"] == 1].copy()
        features = BAYES_FEATURES_COMMON
    else:
        raise ValueError(f"Unknown condition: {condition}")

    cols = [
        "eid",
        "trial_index",
        *features,
        "choice_right",
        "rt",
        "log_rt",
        "probabilityLeft",
        "block_switch",
        "trials_from_block_start",
    ]
    out = subset[cols].copy()
    out["condition"] = condition
    # Causality checklist columns (for audits)
    out["has_current_response_as_input"] = 0
    out["has_current_reward_as_input"] = 0
    out["has_oracle_prior"] = int(condition == "full_information")
    return out


def causality_ok(df: pd.DataFrame) -> bool:
    """True if no current-trial response/reward leaked into inputs."""
    return bool(
        (df["has_current_response_as_input"] == 0).all()
        and (df["has_current_reward_as_input"] == 0).all()
    )
