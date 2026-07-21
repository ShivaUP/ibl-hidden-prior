"""Compact behavior-derived mouse subjective prior (history-only).

Not equal to true probabilityLeft. Template: leaky online estimate of P(right)
updated from experienced stimulus sides, then used as a bias in a logistic
choice model with signed contrast.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


@dataclass
class MousePriorParams:
    alpha: float
    intercept: float
    beta_contrast: float
    beta_prior: float
    p0: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


def _session_prior_path(stim_right: np.ndarray, alpha: float, p0: float = 0.5) -> np.ndarray:
    """Causal prior path: p_t uses only history before trial t."""
    p = float(p0)
    out = np.empty(len(stim_right), dtype=float)
    a = float(np.clip(alpha, 1e-4, 0.999))
    for i, s in enumerate(stim_right.astype(float)):
        out[i] = p
        p = (1.0 - a) * p + a * s
    return out


def compute_prior_column(
    df: pd.DataFrame, alpha: float, p0: float = 0.5
) -> np.ndarray:
    """Compute per-trial mouse prior within each eid (session-ordered)."""
    priors = np.zeros(len(df), dtype=float)
    for _, g in df.groupby("eid", sort=False):
        pos = df.index.get_indexer(g.index)
        order = np.argsort(g["trial_index"].to_numpy())
        stim = g["stimulus_right"].to_numpy()[order]
        ppath = _session_prior_path(stim, alpha=alpha, p0=p0)
        priors[pos[order]] = ppath
    return priors


def _signed_contrast(df: pd.DataFrame) -> np.ndarray:
    return np.where(
        df["stimulus_right"].to_numpy() == 1,
        df["abs_contrast"].to_numpy(dtype=float),
        -df["abs_contrast"].to_numpy(dtype=float),
    )


def _choice_nll_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    p = expit(logits).clip(1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_mouse_prior(
    trials: pd.DataFrame,
    *,
    train_eids: list[str] | None = None,
    alpha_grid: np.ndarray | None = None,
) -> tuple[MousePriorParams, dict]:
    """Fit leaky prior + logistic choice on train sessions (or all if None).

    Strategy: grid over alpha; for each alpha fit L2 logistic on
    [signed_contrast, 2p-1] with sklearn (stable under collinearity).
    """
    df = trials.copy()
    if train_eids is not None:
        df = df[df["eid"].astype(str).isin(set(map(str, train_eids)))].copy()
    df = df.sort_values(["eid", "trial_index"])
    signed = _signed_contrast(df)
    y = df["choice_right"].to_numpy(dtype=int)
    if alpha_grid is None:
        alpha_grid = np.concatenate(
            [np.linspace(0.02, 0.4, 20), np.linspace(0.45, 0.8, 8)]
        )

    best: dict | None = None
    for alpha in alpha_grid:
        prior = compute_prior_column(df, alpha=float(alpha))
        prior_feat = 2.0 * prior - 1.0
        x = np.column_stack([signed, prior_feat])
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            fit_intercept=True,
        )
        clf.fit(x, y)
        logits = clf.decision_function(x)
        nll = _choice_nll_from_logits(logits, y.astype(float))
        cand = {
            "alpha": float(alpha),
            "intercept": float(clf.intercept_[0]),
            "beta_contrast": float(clf.coef_[0, 0]),
            "beta_prior": float(clf.coef_[0, 1]),
            "nll": nll,
        }
        if best is None or cand["nll"] < best["nll"]:
            best = cand

    assert best is not None
    params = MousePriorParams(
        alpha=best["alpha"],
        intercept=best["intercept"],
        beta_contrast=best["beta_contrast"],
        beta_prior=best["beta_prior"],
    )
    prior = compute_prior_column(df, alpha=params.alpha)
    oracle_right = 1.0 - df["probabilityLeft"].to_numpy(dtype=float)
    corr_oracle_right = float(np.corrcoef(prior, oracle_right)[0, 1])
    info = {
        "success": True,
        "nll": float(best["nll"]),
        "n_trials": int(len(df)),
        "corr_with_oracle_prior_right": corr_oracle_right,
        "alpha_grid_size": int(len(alpha_grid)),
        "message": "grid_alpha_plus_logistic",
    }
    return params, info


def apply_mouse_prior(trials: pd.DataFrame, params: MousePriorParams) -> pd.DataFrame:
    """Attach mouse_prior_hat and model-implied choice prob under fitted logistic."""
    out = trials.copy()
    out["mouse_prior_hat"] = compute_prior_column(out, alpha=params.alpha, p0=params.p0)
    signed = _signed_contrast(out)
    prior_feat = 2.0 * out["mouse_prior_hat"].to_numpy() - 1.0
    logits = params.intercept + params.beta_contrast * signed + params.beta_prior * prior_feat
    out["mouse_prior_choice_p_right"] = expit(logits)
    return out
