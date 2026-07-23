"""Logistic regression decoder for block prior identity from model latent states.

For each model (tanh_bptt, tanh_pc, gru, bayes) we extract the pre-stimulus
hidden state at every trial, then ask: how well does a linear decoder recover
the true block prior (left-biased vs right-biased)?

Binary classification: left block (p_right ≈ 0.2) = 0, right block (p_right ≈ 0.8) = 1.
Unbiased trials (p_right ≈ 0.5) are excluded from the primary binary decode.

A 3-class variant (left / unbiased / right) is also available.

Reference dataset: IBL Brain Wide Map (2025)
https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

BLOCK_LEFT = 0   # p_right ≈ 0.2  (mouse sees mostly left stimuli)
BLOCK_UNBIASED = 1  # p_right ≈ 0.5
BLOCK_RIGHT = 2  # p_right ≈ 0.8  (mouse sees mostly right stimuli)

_BLOCK_PRIORS = {0.2: BLOCK_LEFT, 0.5: BLOCK_UNBIASED, 0.8: BLOCK_RIGHT}


def label_blocks(p_right: np.ndarray, atol: float = 0.05) -> np.ndarray:
    """Map true p_right values to integer block labels.

    Parameters
    ----------
    p_right : array of shape (n_sessions, n_trials) or (n_samples,)
    atol    : tolerance for matching known prior values

    Returns
    -------
    labels : same shape as p_right, dtype int, values in {0, 1, 2, -1}
             -1 means the prior did not match any known level.
    """
    flat = p_right.ravel()
    out = np.full(flat.shape, -1, dtype=np.int64)
    for prior, label in _BLOCK_PRIORS.items():
        out[np.abs(flat - prior) < atol] = label
    return out.reshape(p_right.shape)


# ---------------------------------------------------------------------------
# Latent state extraction
# ---------------------------------------------------------------------------

def extract_latent_states(
    model,
    model_id: str,
    batch,
    cfg: dict,
    *,
    seed: int = 0,
    regime: str = "history_only",
) -> np.ndarray:
    """Run a closed-loop rollout and return pre-stimulus hidden states.

    Parameters
    ----------
    model    : loaded v2 model object
    model_id : one of 'tanh_bptt', 'tanh_pc', 'gru', 'bayes'
    batch    : SyntheticBatch
    cfg      : config dict (from synthetic_v2.yaml)

    Returns
    -------
    hidden : float array of shape (n_sessions * n_trials, hidden_dim)
             For Bayes the single q value is returned as shape (..., 1).
    """
    from src.models_v2.rollout import rollout_closed_loop

    roll = rollout_closed_loop(model, batch, cfg, model_id, seed=seed, regime=regime)
    # pre_stimulus_hidden: (n_sessions, n_trials, hidden_dim) or (S, T, 1) for Bayes
    h = roll["pre_stimulus_hidden"]
    n_sessions, n_trials = h.shape[:2]
    return h.reshape(n_sessions * n_trials, -1)


def extract_labels_flat(batch, *, binary: bool = True) -> np.ndarray:
    """Flatten block labels from a SyntheticBatch.

    Parameters
    ----------
    binary : if True, return only left/right labels (drop unbiased trials).
             If False, return 3-class labels (all trials included, -1 for unknown).

    Returns
    -------
    labels : 1-D int array, length n_sessions * n_trials
             Binary: 0 = left block, 1 = right block
             3-class: 0 = left, 1 = unbiased, 2 = right
    """
    labels = label_blocks(batch.p_right)
    return labels.ravel()


def extract_hidden_by_tick(
    model,
    model_id: str,
    batch,
    cfg: dict,
    *,
    seed: int = 0,
    regime: str = "history_only",
) -> Tuple[np.ndarray, List[str]]:
    """Run a rollout capturing hidden states at every within-trial tick.

    Returns
    -------
    hidden_bt  : float array (n_samples, n_steps, hidden_dim)
                 where n_samples = n_sessions * n_trials.
    tick_phase : list[str] length n_steps, phase name per tick.
    """
    from src.models_v2.rollout import rollout_hidden_by_tick

    roll = rollout_hidden_by_tick(model, batch, cfg, model_id, seed=seed, regime=regime)
    h = roll["hidden_by_tick"]  # (S, T, n_steps, hidden_dim)
    n_sessions, n_trials, n_steps, hidden_dim = h.shape
    hidden_bt = h.reshape(n_sessions * n_trials, n_steps, hidden_dim)
    return hidden_bt, list(roll["tick_phase"])


# ---------------------------------------------------------------------------
# Decoder fitting
# ---------------------------------------------------------------------------

def fit_block_decoder(
    hidden: np.ndarray,
    labels: np.ndarray,
    *,
    binary: bool = True,
    n_folds: int = 5,
    C: float = 1.0,
    random_state: int = 42,
) -> Dict:
    """Fit a logistic regression decoder and evaluate with stratified K-fold CV.

    Parameters
    ----------
    hidden : (n_samples, n_features) latent states
    labels : (n_samples,) block labels (output of extract_labels_flat)
    binary : if True, keep only label ∈ {0, 2} and recode to {0, 1}
    n_folds : number of cross-validation folds
    C       : inverse regularization strength for LogisticRegression
    random_state : for reproducible fold splits

    Returns
    -------
    dict with keys:
        n_samples, n_features, n_folds
        accuracy_mean, accuracy_std, accuracy_folds
        auroc_mean, auroc_std, auroc_folds       (only for binary)
        confusion_matrix (summed across folds)
        coef_mean  : mean LR coefficient vector (n_features,) for binary
        intercept_mean : mean intercept
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for the block decoder. "
            "Install it with: pip install scikit-learn"
        ) from e

    if binary:
        mask = (labels == BLOCK_LEFT) | (labels == BLOCK_RIGHT)
        hidden = hidden[mask]
        # recode: left=0, right=1
        labels = (labels[mask] == BLOCK_RIGHT).astype(np.int64)
    else:
        mask = labels >= 0
        hidden = hidden[mask]
        labels = labels[mask]

    n_samples, n_features = hidden.shape
    classes = np.unique(labels)
    n_classes = len(classes)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    acc_folds: List[float] = []
    auroc_folds: List[float] = []
    cm_sum = np.zeros((n_classes, n_classes), dtype=np.int64)
    coef_folds: List[np.ndarray] = []
    intercept_folds: List[np.ndarray] = []

    for train_idx, test_idx in skf.split(hidden, labels):
        X_train, X_test = hidden[train_idx], hidden[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(C=C, max_iter=1000, random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc_folds.append(float(accuracy_score(y_test, y_pred)))
        cm_sum += confusion_matrix(y_test, y_pred, labels=list(range(n_classes))).astype(np.int64)
        coef_folds.append(clf.coef_.copy())
        intercept_folds.append(clf.intercept_.copy())

        if binary:
            y_prob = clf.predict_proba(X_test)[:, 1]
            try:
                auroc_folds.append(float(roc_auc_score(y_test, y_prob)))
            except ValueError:
                auroc_folds.append(float("nan"))

    result: Dict = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_folds": n_folds,
        "binary": binary,
        "accuracy_mean": float(np.mean(acc_folds)),
        "accuracy_std": float(np.std(acc_folds)),
        "accuracy_folds": [round(a, 4) for a in acc_folds],
        "confusion_matrix": cm_sum.tolist(),
        "coef_mean": np.mean(coef_folds, axis=0).tolist(),
        "intercept_mean": float(np.mean([ic[0] for ic in intercept_folds])),
    }
    if binary:
        result["auroc_mean"] = float(np.nanmean(auroc_folds))
        result["auroc_std"] = float(np.nanstd(auroc_folds))
        result["auroc_folds"] = [round(a, 4) for a in auroc_folds]
    return result


# ---------------------------------------------------------------------------
# Top-level: decode all models
# ---------------------------------------------------------------------------

def decode_all_models(
    cfg: dict,
    batch,
    model_ids: Optional[List[str]] = None,
    *,
    checkpoint_dir=None,
    binary: bool = True,
    n_folds: int = 5,
    C: float = 1.0,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Run block decoder for every model in cfg['models'] (or model_ids).

    Parameters
    ----------
    cfg           : config dict (from synthetic_v2.yaml)
    batch         : SyntheticBatch  (held-out or eval split)
    model_ids     : override list of model IDs; defaults to cfg['models']
    checkpoint_dir: Path to directory containing {model_id}/model.npz;
                    defaults to cfg['paths']['artifacts'] / 'models'
    binary        : True → left vs right blocks; False → 3-class
    n_folds       : CV folds
    C             : LR regularization
    seed          : rollout RNG seed

    Returns
    -------
    dict mapping model_id → result dict from fit_block_decoder
    """
    from pathlib import Path
    from src.models_v2.rollout import load_model

    ids = model_ids or cfg.get("models", ["tanh_bptt", "tanh_pc", "gru", "bayes"])
    if checkpoint_dir is None:
        checkpoint_dir = Path(cfg["paths"]["artifacts"]) / "models"
    else:
        checkpoint_dir = Path(checkpoint_dir)

    labels_flat = extract_labels_flat(batch, binary=False)  # keep all for subsetting inside fit

    results: Dict[str, Dict] = {}
    for model_id in ids:
        ckpt = checkpoint_dir / model_id / "model.npz"
        if not ckpt.exists():
            print(f"[block_decoder] skipping {model_id}: checkpoint not found at {ckpt}")
            continue
        print(f"[block_decoder] extracting latent states for {model_id} ...")
        model = load_model(model_id, ckpt)
        hidden = extract_latent_states(model, model_id, batch, cfg, seed=seed)
        print(f"  hidden shape: {hidden.shape}  |  fitting decoder ...")
        result = fit_block_decoder(
            hidden,
            labels_flat.copy(),
            binary=binary,
            n_folds=n_folds,
            C=C,
        )
        result["model_id"] = model_id
        results[model_id] = result
        _report_one(model_id, result)

    return results


def _report_one(model_id: str, result: Dict) -> None:
    """Print a one-line summary."""
    acc = result["accuracy_mean"]
    acc_std = result["accuracy_std"]
    if result["binary"]:
        auroc = result.get("auroc_mean", float("nan"))
        print(
            f"  {model_id:<14}  acc={acc:.3f}±{acc_std:.3f}  "
            f"AUROC={auroc:.3f}  "
            f"(n={result['n_samples']}, features={result['n_features']})"
        )
    else:
        print(
            f"  {model_id:<14}  acc={acc:.3f}±{acc_std:.3f}  "
            f"(n={result['n_samples']}, features={result['n_features']}, 3-class)"
        )


# ---------------------------------------------------------------------------
# Tick-by-tick ("layer by layer") decoding
# ---------------------------------------------------------------------------

def decode_by_tick(
    hidden_bt: np.ndarray,
    labels: np.ndarray,
    tick_phase: List[str],
    *,
    binary: bool = True,
    n_folds: int = 5,
    C: float = 1.0,
) -> Dict:
    """Fit a separate block decoder at each within-trial tick.

    Parameters
    ----------
    hidden_bt  : (n_samples, n_steps, hidden_dim) hidden states per tick
    labels     : (n_samples,) block labels
    tick_phase : phase name per tick (length n_steps)

    Returns
    -------
    dict with per-tick arrays:
        n_steps, tick_phase, n_features,
        auroc_by_tick, auroc_std_by_tick,
        accuracy_by_tick, accuracy_std_by_tick
    """
    n_samples, n_steps, hidden_dim = hidden_bt.shape

    auroc, auroc_std = [], []
    acc, acc_std = [], []
    for tick in range(n_steps):
        res = fit_block_decoder(
            hidden_bt[:, tick, :],
            labels.copy(),
            binary=binary,
            n_folds=n_folds,
            C=C,
        )
        acc.append(res["accuracy_mean"])
        acc_std.append(res["accuracy_std"])
        if binary:
            auroc.append(res.get("auroc_mean", float("nan")))
            auroc_std.append(res.get("auroc_std", 0.0))

    out: Dict = {
        "n_steps": int(n_steps),
        "n_features": int(hidden_dim),
        "binary": binary,
        "tick_phase": list(tick_phase),
        "accuracy_by_tick": [round(a, 4) for a in acc],
        "accuracy_std_by_tick": [round(a, 4) for a in acc_std],
    }
    if binary:
        out["auroc_by_tick"] = [round(a, 4) for a in auroc]
        out["auroc_std_by_tick"] = [round(a, 4) for a in auroc_std]
    return out


def decode_all_models_by_tick(
    cfg: dict,
    batch,
    model_ids: Optional[List[str]] = None,
    *,
    checkpoint_dir=None,
    binary: bool = True,
    n_folds: int = 5,
    C: float = 1.0,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Run tick-by-tick block decoding for every model.

    Returns dict mapping model_id → result dict from :func:`decode_by_tick`.
    """
    from pathlib import Path
    from src.models_v2.rollout import load_model

    ids = model_ids or cfg.get("models", ["tanh_bptt", "tanh_pc", "gru", "bayes"])
    if checkpoint_dir is None:
        checkpoint_dir = Path(cfg["paths"]["artifacts"]) / "models"
    else:
        checkpoint_dir = Path(checkpoint_dir)

    labels_flat = extract_labels_flat(batch, binary=False)

    results: Dict[str, Dict] = {}
    for model_id in ids:
        ckpt = checkpoint_dir / model_id / "model.npz"
        if not ckpt.exists():
            print(f"[block_decoder] skipping {model_id}: checkpoint not found at {ckpt}")
            continue
        print(f"[block_decoder] tick-by-tick decoding for {model_id} ...")
        model = load_model(model_id, ckpt)
        hidden_bt, tick_phase = extract_hidden_by_tick(model, model_id, batch, cfg, seed=seed)
        print(f"  hidden_by_tick shape: {hidden_bt.shape}  |  fitting {hidden_bt.shape[1]} tick decoders ...")
        result = decode_by_tick(
            hidden_bt,
            labels_flat.copy(),
            tick_phase,
            binary=binary,
            n_folds=n_folds,
            C=C,
        )
        result["model_id"] = model_id
        results[model_id] = result
        _report_by_tick(model_id, result)

    return results


def _report_by_tick(model_id: str, result: Dict) -> None:
    """Print the per-tick AUROC (or accuracy) for one model."""
    phase = result["tick_phase"]
    if result["binary"]:
        vals = result["auroc_by_tick"]
        metric = "AUROC"
    else:
        vals = result["accuracy_by_tick"]
        metric = "acc"
    peak_tick = int(np.argmax(vals))
    cells = "  ".join(f"{p[:4]}={v:.3f}" for p, v in zip(phase, vals))
    print(f"  {model_id:<14} [{metric}]  {cells}")
    print(f"    → peak at tick {peak_tick} ({phase[peak_tick]}): {vals[peak_tick]:.3f}")

