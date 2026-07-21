"""Choice metrics and psychometric summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def choice_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Held-out choice metrics from columns choice_right, p_right."""
    y = df["choice_right"].to_numpy(dtype=float)
    p = df["p_right"].to_numpy(dtype=float).clip(1e-6, 1 - 1e-6)
    nll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    acc = float(((p >= 0.5).astype(float) == y).mean())
    # McFadden-style pseudo-R^2 vs intercept-only null
    p_null = float(y.mean().clip(1e-6, 1 - 1e-6))
    nll_null = float(-(y * np.log(p_null) + (1 - y) * np.log(1 - p_null)).mean())
    pseudo_r2 = float(1.0 - (nll / nll_null)) if nll_null > 0 else float("nan")
    return {
        "n_trials": int(len(df)),
        "choice_nll": nll,
        "choice_acc": acc,
        "choice_pseudo_r2": pseudo_r2,
        "p_right_mean": float(p.mean()),
        "choice_right_rate": float(y.mean()),
    }


def psychometric_table(df: pd.DataFrame, by_block: bool = True) -> pd.DataFrame:
    """P(right) vs signed contrast, overall and optionally by probabilityLeft."""
    rows = []
    groups = [("all", df)]
    if by_block and "probabilityLeft" in df.columns:
        for pleft, g in df.groupby("probabilityLeft"):
            groups.append((f"block_{pleft}", g))
    for label, g in groups:
        for sc, gg in g.groupby("signed_contrast"):
            if len(gg) == 0:
                continue
            rows.append(
                {
                    "slice": label,
                    "signed_contrast": float(sc),
                    "n": int(len(gg)),
                    "p_right_mouse": float(gg["choice_right"].mean()),
                    "p_right_model": float(gg["p_right"].mean()) if "p_right" in gg else np.nan,
                }
            )
    return pd.DataFrame(rows)


def mouse_psychometric(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical mouse psychometric only (no model column required beyond choice)."""
    rows = []
    for label, g in [("all", df)] + [
        (f"block_{p}", gg) for p, gg in df.groupby("probabilityLeft")
    ]:
        for sc, gg in g.groupby("signed_contrast"):
            rows.append(
                {
                    "slice": label,
                    "signed_contrast": float(sc),
                    "n": int(len(gg)),
                    "p_right_mouse": float(gg["choice_right"].mean()),
                }
            )
    return pd.DataFrame(rows)
