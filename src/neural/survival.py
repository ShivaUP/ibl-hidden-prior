"""Survival tests for matched neural VE advantages."""

from __future__ import annotations

import numpy as np
import pandas as pd


def holm_correct(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        raw = float(pvals[idx])
        # (m - rank) * p_(rank)
        val = min(1.0, (m - rank) * raw)
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def paired_mean_delta_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for mean(a - b) on paired samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 10:
        return {
            "n": n,
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_two_sided": float("nan"),
        }
    obs = float(np.mean(a - b))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = float(np.mean(a[idx] - b[idx]))
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    if abs(obs) < 1e-15:
        p = 1.0
    else:
        p = float(np.mean(deltas * np.sign(obs) <= 0.0))
        p = min(1.0, 2.0 * p)
    return {
        "n": int(n),
        "delta": obs,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_two_sided": p,
    }


def bootstrap_corr_advantage(
    neural: np.ndarray,
    q_better_candidate: np.ndarray,
    q_baseline: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Test whether corr(neural, q_cand) > corr(neural, q_base) via paired trial bootstrap."""
    y = np.asarray(neural, dtype=float)
    c = np.asarray(q_better_candidate, dtype=float)
    b = np.asarray(q_baseline, dtype=float)
    mask = np.isfinite(y) & np.isfinite(c) & np.isfinite(b)
    y, c, b = y[mask], c[mask], b[mask]
    n = len(y)
    if n < 20:
        return {
            "n": n,
            "corr_cand": float("nan"),
            "corr_base": float("nan"),
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_two_sided": float("nan"),
        }

    def _corr(u, v):
        if np.std(u) < 1e-12 or np.std(v) < 1e-12:
            return 0.0
        return float(np.corrcoef(u, v)[0, 1])

    obs_c = _corr(y, c)
    obs_b = _corr(y, b)
    obs = obs_c - obs_b
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = _corr(y[idx], c[idx]) - _corr(y[idx], b[idx])
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    # two-sided: fraction of bootstrap deltas with opposite sign to obs, *2
    if abs(obs) < 1e-15:
        p = 1.0
    else:
        p = float(np.mean(deltas * np.sign(obs) <= 0.0))
        p = min(1.0, 2.0 * p)
    return {
        "n": int(n),
        "corr_cand": obs_c,
        "corr_base": obs_b,
        "delta": float(obs),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_two_sided": p,
    }


def bootstrap_ve_advantage(
    neural: np.ndarray,
    q_cand: np.ndarray,
    q_base: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """VE after per-bootstrap linear recal of q→neural; test VE_cand - VE_base."""
    y = np.asarray(neural, dtype=float)
    c = np.asarray(q_cand, dtype=float)
    b = np.asarray(q_base, dtype=float)
    mask = np.isfinite(y) & np.isfinite(c) & np.isfinite(b)
    y, c, b = y[mask], c[mask], b[mask]
    n = len(y)
    if n < 20:
        return {"n": n, "delta": float("nan"), "p_two_sided": float("nan")}

    def _ve_lin(u, q):
        if np.std(q) < 1e-12:
            pred = np.full_like(u, u.mean())
        else:
            coef = np.polyfit(q, u, 1)
            pred = coef[0] * q + coef[1]
        sst = np.sum((u - u.mean()) ** 2)
        if sst <= 1e-12:
            return 0.0
        return float(1.0 - np.sum((u - pred) ** 2) / sst)

    obs = _ve_lin(y, c) - _ve_lin(y, b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = _ve_lin(y[idx], c[idx]) - _ve_lin(y[idx], b[idx])
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    if abs(obs) < 1e-15:
        p = 1.0
    else:
        p = float(np.mean(deltas * np.sign(obs) <= 0.0))
        p = min(1.0, 2.0 * p)
    return {
        "n": int(n),
        "ve_cand": _ve_lin(y, c),
        "ve_base": _ve_lin(y, b),
        "delta": float(obs),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_two_sided": p,
    }


def session_mean_table(ve_df: pd.DataFrame, value_col: str = "ve_linear_recal") -> pd.DataFrame:
    """Mean VE per model×region across eids (for multi-session summaries)."""
    return (
        ve_df.groupby(["region", "model"], as_index=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
