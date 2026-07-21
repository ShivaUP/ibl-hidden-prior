"""Neural prior readout and model–neural variance explained."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def variance_explained(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 - SSE/SST; nan if y has no variance."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < 3:
        return float("nan")
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 1e-12:
        return float("nan")
    sse = float(np.sum((y - p) ** 2))
    return 1.0 - sse / sst


def fit_prior_readout(
    counts: np.ndarray,
    mouse_prior: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict:
    """Cross-validated Ridge: neural counts → mouse_prior_hat.

    Returns OOF predictions and CV VE / correlation.
    """
    x = np.asarray(counts, dtype=float)
    y = np.asarray(mouse_prior, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x, y = x[mask], y[mask]
    n = len(y)
    oof = np.full(n, np.nan)
    if n < max(10, n_splits * 2) or x.shape[1] == 0:
        return {
            "n": int(n),
            "n_units": int(x.shape[1]) if x.ndim == 2 else 0,
            "ve_cv": float("nan"),
            "corr_cv": float("nan"),
            "oof_pred": oof,
            "mask": mask,
        }

    n_splits = min(n_splits, n // 2)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(x):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x[train_idx])
        x_te = scaler.transform(x[test_idx])
        model = RidgeCV(alphas=np.logspace(-2, 3, 12))
        model.fit(x_tr, y[train_idx])
        oof[test_idx] = model.predict(x_te)

    ve = variance_explained(y, oof)
    corr = float(np.corrcoef(y, oof)[0, 1]) if np.std(oof) > 1e-12 else float("nan")
    return {
        "n": int(n),
        "n_units": int(x.shape[1]),
        "ve_cv": float(ve),
        "corr_cv": corr,
        "oof_pred": oof,
        "mask": mask,
    }


def model_explains_neural_prior(
    neural_prior: np.ndarray,
    model_prior_q: np.ndarray,
) -> dict[str, float]:
    """Primary unmatched metric: VE of neural prior readout by model q_t."""
    ve = variance_explained(neural_prior, model_prior_q)
    a = np.asarray(neural_prior, dtype=float)
    b = np.asarray(model_prior_q, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
    # Also linear refit of q → neural for fair scale
    if len(a) > 2 and np.std(b) > 1e-12:
        coef = np.polyfit(b, a, 1)
        pred = coef[0] * b + coef[1]
        ve_lin = variance_explained(a, pred)
    else:
        ve_lin = float("nan")
    return {
        "n": float(len(a)),
        "ve_raw": float(ve),
        "ve_linear_recal": float(ve_lin),
        "corr": corr,
    }
