"""Switch-centered behavioral analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _aligned_prob(row: pd.Series, use_model: bool) -> float:
    """P(choice aligned with new block). New block from probabilityLeft at switch+."""
    pleft = float(row["probabilityLeft"])
    # New block favors left if pleft>0.5, right if pleft<0.5; 0.5 -> use right by convention
    favor_right = pleft < 0.5
    if use_model:
        p_right = float(row["p_right"])
    else:
        p_right = float(row["choice_right"])
    return p_right if favor_right else (1.0 - p_right)


def extract_switch_windows(
    df: pd.DataFrame,
    *,
    pre: int = 10,
    post: int = 30,
    min_pre: int = 8,
    min_post: int = 16,
    prefer_pre: int = 10,
    prefer_post: int = 20,
) -> pd.DataFrame:
    """Build trial-relative rows around each block switch within each eid."""
    rows: list[dict] = []
    for eid, g in df.groupby("eid"):
        g = g.sort_values("trial_index").reset_index(drop=True)
        switch_idx = g.index[g["block_switch"] == 1].tolist()
        for s in switch_idx:
            # Need previous trial to exist for pre window
            start = max(0, s - pre)
            end = min(len(g), s + post + 1)
            window = g.iloc[start:end]
            n_pre = s - start
            n_post = end - s  # includes switch trial as 0
            # post count of trials after switch (including 0): end-s
            post_count = end - s
            if n_pre < min_pre or post_count < min_post:
                continue
            relaxed = n_pre < prefer_pre or post_count < prefer_post
            new_pleft = float(g.iloc[s]["probabilityLeft"])
            old_pleft = float(g.iloc[s - 1]["probabilityLeft"]) if s > 0 else np.nan
            for _, row in window.iterrows():
                # relative position: trial_index distance from switch trial_index
                rel = int(row["trial_index"] - g.iloc[s]["trial_index"])
                if rel < -pre or rel > post:
                    continue
                base = row.to_dict()
                base.update(
                    {
                        "switch_eid": eid,
                        "switch_trial_index": int(g.iloc[s]["trial_index"]),
                        "rel_trial": rel,
                        "old_probabilityLeft": old_pleft,
                        "new_probabilityLeft": new_pleft,
                        "switch_direction": f"{old_pleft}->{new_pleft}",
                        "relaxed_qc": bool(relaxed),
                        "p_align_mouse": _aligned_prob(row, use_model=False),
                        "p_align_model": _aligned_prob(row, use_model=True)
                        if "p_right" in row and pd.notna(row.get("p_right"))
                        else np.nan,
                    }
                )
                rows.append(base)
    return pd.DataFrame(rows)


def switch_trajectory(switch_df: pd.DataFrame, prefer_strict: bool = False) -> pd.DataFrame:
    """Mean aligned probability vs rel_trial."""
    d = switch_df
    if prefer_strict:
        d = d.loc[~d["relaxed_qc"]]
    rows = []
    for rel, g in d.groupby("rel_trial"):
        rows.append(
            {
                "rel_trial": int(rel),
                "n": int(len(g)),
                "p_align_mouse": float(g["p_align_mouse"].mean()),
                "p_align_model": float(g["p_align_model"].mean())
                if g["p_align_model"].notna().any()
                else np.nan,
                "prior_q_mean": float(g["prior_q"].mean())
                if "prior_q" in g and g["prior_q"].notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("rel_trial")


def _exp_approach(t, a, b, tau):
    # a + (b-a)*(1-exp(-t/tau)) for t>=0
    return a + (b - a) * (1.0 - np.exp(-t / np.maximum(tau, 1e-3)))


def fit_adaptation_half_life(
    traj: pd.DataFrame,
    value_col: str = "p_align_mouse",
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Fit exponential approach on post-switch (rel>=0) and report half-life."""
    post = traj.loc[traj["rel_trial"] >= 0].copy()
    if len(post) < 4 or post[value_col].isna().all():
        return {"ok": False, "reason": "insufficient_post_points"}
    t = post["rel_trial"].to_numpy(dtype=float)
    y = post[value_col].to_numpy(dtype=float)
    a0, b0 = float(y[0]), float(y[-1])
    try:
        popt, _ = curve_fit(
            _exp_approach, t, y, p0=[a0, b0, 5.0], bounds=([-1, -1, 0.1], [2, 2, 100])
        )
        tau = float(popt[2])
        half_life = float(np.log(2) * tau)
        # bootstrap
        rng = np.random.default_rng(seed)
        halves = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(t), size=len(t))
            try:
                pb, _ = curve_fit(
                    _exp_approach,
                    t[idx],
                    y[idx],
                    p0=popt,
                    bounds=([-1, -1, 0.1], [2, 2, 100]),
                    maxfev=5000,
                )
                halves.append(np.log(2) * pb[2])
            except Exception:  # noqa: BLE001
                continue
        ci = (
            (float(np.percentile(halves, 2.5)), float(np.percentile(halves, 97.5)))
            if halves
            else (None, None)
        )
        yhat = _exp_approach(t, *popt)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        return {
            "ok": True,
            "tau": tau,
            "half_life": half_life,
            "half_life_ci95": ci,
            "r2": 1.0 - ss_res / ss_tot,
            "a": float(popt[0]),
            "b": float(popt[1]),
            "value_col": value_col,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)}


def asymmetry_table(switch_df: pd.DataFrame) -> pd.DataFrame:
    """Compare post-switch adaptation for 0.2→0.8 vs 0.8→0.2."""
    rows = []
    for direction, g in switch_df.groupby("switch_direction"):
        post = g.loc[g["rel_trial"] >= 0]
        rows.append(
            {
                "switch_direction": direction,
                "n_switch_trials": int(g.loc[g["rel_trial"] == 0].shape[0]),
                "post_p_align_mouse": float(post["p_align_mouse"].mean()),
                "post_p_align_model": float(post["p_align_model"].mean())
                if post["p_align_model"].notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
