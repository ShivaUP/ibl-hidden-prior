"""Reaction-time evaluation summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rt_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Gaussian log-RT NLL using model rt_log_mean / rt_log_std."""
    log_rt = df["log_rt"].to_numpy(dtype=float)
    mean = df["rt_log_mean"].to_numpy(dtype=float)
    std = df["rt_log_std"].to_numpy(dtype=float).clip(min=1e-3)
    z = (log_rt - mean) / std
    nll = float((0.5 * z**2 + np.log(std) + 0.5 * np.log(2 * np.pi)).mean())
    return {
        "n_trials": int(len(df)),
        "rt_nll": nll,
        "rt_median_mouse": float(np.exp(log_rt).mean()) if False else float(np.median(np.exp(log_rt))),
        "rt_median_model": float(np.median(np.exp(mean))),
        "log_rt_mae": float(np.abs(log_rt - mean).mean()),
    }


def rt_by_strength_and_block(df: pd.DataFrame) -> pd.DataFrame:
    """Median RT summaries by contrast_high and block prior."""
    rows = []
    for (ch, pleft), g in df.groupby(["contrast_high", "probabilityLeft"]):
        rows.append(
            {
                "contrast_high": int(ch),
                "probabilityLeft": float(pleft),
                "n": int(len(g)),
                "rt_median_mouse": float(np.median(np.exp(g["log_rt"]))),
                "rt_iqr_mouse": float(
                    np.subtract(*np.percentile(np.exp(g["log_rt"]), [75, 25]))
                ),
                "rt_median_model": float(np.median(np.exp(g["rt_log_mean"]))),
            }
        )
    return pd.DataFrame(rows)
