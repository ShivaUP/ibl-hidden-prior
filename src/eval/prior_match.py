"""Prior-match metrics between mouse latent prior and model q_t."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.switch_centered import extract_switch_windows


def prior_match_metrics(df: pd.DataFrame) -> dict[str, float]:
    """df must contain mouse_prior_hat and prior_q."""
    a = df["mouse_prior_hat"].to_numpy(dtype=float)
    b = df["prior_q"].to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return {"n": float(len(a)), "corr": np.nan, "rmse": np.nan, "mae": np.nan}
    corr = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mae = float(np.mean(np.abs(a - b)))
    return {"n": float(len(a)), "corr": corr, "rmse": rmse, "mae": mae}


def switch_prior_mse(df: pd.DataFrame) -> dict[str, float]:
    """MSE of priors in switch window -10..+30."""
    # Need block_switch etc.
    need = {"mouse_prior_hat", "prior_q", "block_switch", "eid", "trial_index", "probabilityLeft", "choice_right"}
    if not need.issubset(df.columns):
        return {"switch_mse": np.nan, "n_switch_rows": 0.0}
    tmp = df.copy()
    if "p_right" not in tmp.columns:
        tmp["p_right"] = tmp["prior_q"]  # placeholder for extract_switch_windows aligned helper
    sw = extract_switch_windows(tmp)
    if len(sw) == 0:
        return {"switch_mse": np.nan, "n_switch_rows": 0.0}
    mse = float(np.mean((sw["mouse_prior_hat"] - sw["prior_q"]) ** 2))
    return {"switch_mse": mse, "n_switch_rows": float(len(sw))}


def update_asymmetry_prior(df: pd.DataFrame) -> pd.DataFrame:
    """Mean |Δprior| in first 10 post-switch trials by direction."""
    tmp = df.copy()
    if "p_right" not in tmp.columns:
        tmp["p_right"] = tmp.get("prior_q", 0.5)
    sw = extract_switch_windows(tmp)
    rows = []
    if len(sw) == 0:
        return pd.DataFrame(rows)
    for direction, g in sw.groupby("switch_direction"):
        post = g.loc[(g["rel_trial"] >= 1) & (g["rel_trial"] <= 10)]
        if len(post) == 0:
            continue
        # change from switch trial prior
        # compare mean mouse/model prior shift toward new block
        rows.append(
            {
                "switch_direction": direction,
                "n": int((g["rel_trial"] == 0).sum()),
                "mouse_prior_post_mean": float(post["mouse_prior_hat"].mean()),
                "model_prior_post_mean": float(post["prior_q"].mean()),
                "abs_diff_means": float(
                    abs(post["mouse_prior_hat"].mean() - post["prior_q"].mean())
                ),
            }
        )
    return pd.DataFrame(rows)
