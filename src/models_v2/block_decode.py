"""Post-hoc block decoding from zero-evidence latent trajectories.

Used by switch-centered analyses: logistic / one-hidden MLP probes on
concatenated within-trial recurrent states under a zero-current-evidence probe
(visual / action / reward channels off; go cue retained).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.synthetic.channels import GO_CUE, N_INPUTS

DECODER_NAMES = ("logistic", "mlp")


@dataclass
class DecoderSettings:
    logistic_epochs: int = 80
    mlp_epochs: int = 120
    batch_size: int = 256
    patience: int = 12
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    mlp_hidden_size: int = 64
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4


def make_session_split(
    n_sessions: int,
    seed: int,
    *,
    train_frac: float = 0.6,
    validation_frac: float = 0.2,
) -> dict[str, np.ndarray]:
    """Deterministic train / validation / test session indices."""

    if n_sessions < 3:
        raise ValueError("need at least 3 sessions for a train/val/test split")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n_sessions)
    n_train = max(1, int(round(train_frac * n_sessions)))
    n_val = max(1, int(round(validation_frac * n_sessions)))
    if n_train + n_val >= n_sessions:
        n_train = max(1, n_sessions - 2)
        n_val = 1
    n_test = n_sessions - n_train - n_val
    if n_test < 1:
        raise ValueError("session split left no test sessions")
    return {
        "train": np.sort(order[:n_train]),
        "validation": np.sort(order[n_train : n_train + n_val]),
        "test": np.sort(order[n_train + n_val :]),
    }


def extract_zero_evidence_latents(
    model: Any,
    inputs: np.ndarray,
    phase,
) -> np.ndarray:
    """Roll matched inputs; on a probe branch zero current evidence.

    Parameters
    ----------
    inputs:
        ``(n_sessions, n_trials, n_steps, n_inputs)`` teacher-forced trial tensors.
    Returns
    -------
    latents:
        ``(n_sessions, n_trials, n_steps, hidden_size)`` hidden states on the
        zero-current-evidence probe (go cue only).
    """

    inputs = np.asarray(inputs, dtype=np.float64)
    if inputs.ndim != 4:
        raise ValueError("inputs must be (sessions, trials, steps, channels)")
    n_sessions, n_trials, n_steps, n_channels = inputs.shape
    if n_channels != N_INPUTS:
        raise ValueError(f"expected {N_INPUTS} input channels, got {n_channels}")
    if n_steps != int(phase.n_steps):
        raise ValueError("inputs n_steps does not match phase.n_steps")

    hidden_size = int(model.hidden_size)
    latents = np.empty(
        (n_sessions, n_trials, n_steps, hidden_size),
        dtype=np.float64,
    )
    state = model.zero_state(n_sessions)
    zeros = np.zeros((n_sessions, N_INPUTS), dtype=np.float64)

    for trial in range(n_trials):
        # Probe: current-trial visual / action / reward channels off; go cue only.
        state_cf = state.copy()
        for step in range(n_steps):
            xt = zeros.copy()
            if step == int(phase.go_tick):
                xt[:, GO_CUE] = 1.0
            state_cf = model.step(xt, state_cf)
            latents[:, trial, step] = state_cf

        for step in range(n_steps):
            state = model.step(inputs[:, trial, step], state)

    return latents


def make_decoding_dataset(
    features: np.ndarray,
    p_right: np.ndarray,
    block_id: np.ndarray,
    sessions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Flatten biased-block trials (0.2 / 0.8) for listed sessions."""

    del block_id  # reserved for future block-aware sampling; labels use p_right
    features = np.asarray(features, dtype=np.float64)
    p_right = np.asarray(p_right, dtype=np.float64)
    sessions = np.asarray(sessions, dtype=int)
    rows = []
    labels = []
    session_of = []
    trial_of = []
    for session in sessions:
        prior = p_right[session]
        biased = np.isclose(prior, 0.2) | np.isclose(prior, 0.8)
        if not np.any(biased):
            continue
        x = features[session][biased]
        y = (prior[biased] > 0.5).astype(np.int64)
        trials = np.flatnonzero(biased)
        rows.append(x.reshape(len(trials), -1))
        labels.append(y)
        session_of.append(np.full(len(trials), int(session), dtype=np.int64))
        trial_of.append(trials.astype(np.int64))
    if not rows:
        empty = np.zeros((0, int(np.prod(features.shape[2:]))), dtype=np.float64)
        return {
            "x": empty,
            "y": np.zeros(0, dtype=np.int64),
            "session": np.zeros(0, dtype=np.int64),
            "trial": np.zeros(0, dtype=np.int64),
        }
    return {
        "x": np.concatenate(rows, axis=0),
        "y": np.concatenate(labels, axis=0),
        "session": np.concatenate(session_of, axis=0),
        "trial": np.concatenate(trial_of, axis=0),
    }


def _subsample(
    dataset: dict[str, np.ndarray],
    max_samples: int | None,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    if max_samples is None or len(dataset["y"]) <= max_samples:
        return dataset
    idx = rng.choice(len(dataset["y"]), size=int(max_samples), replace=False)
    return {key: value[idx] for key, value in dataset.items()}


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _minibatches(
    n: int,
    batch_size: int,
    rng: np.random.Generator,
):
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def _fit_logistic(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    *,
    seed: int,
    settings: DecoderSettings,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    train = _subsample(train, settings.max_train_samples, rng)
    validation = _subsample(validation, settings.max_validation_samples, rng)
    x_tr, mean, std = _standardize_fit(train["x"])
    y_tr = train["y"].astype(np.float64)
    x_va = _standardize_apply(validation["x"], mean, std)
    y_va = validation["y"].astype(np.float64)
    x_te = _standardize_apply(test["x"], mean, std)

    n_features = x_tr.shape[1]
    w = rng.normal(0.0, 0.01, size=n_features)
    b = 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_b = 0.0
    v_b = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_w, best_b = w.copy(), b
    best_epoch = 0
    best_val = float("inf")
    stall = 0

    for epoch in range(1, int(settings.logistic_epochs) + 1):
        for idx in _minibatches(len(y_tr), settings.batch_size, rng):
            xb = x_tr[idx]
            yb = y_tr[idx]
            p = _sigmoid(xb @ w + b)
            err = (p - yb) / max(len(yb), 1)
            gw = xb.T @ err + settings.weight_decay * w
            gb = float(err.sum())
            t = epoch  # coarse Adam time
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * (gw**2)
            m_b = beta1 * m_b + (1 - beta1) * gb
            v_b = beta2 * v_b + (1 - beta2) * (gb**2)
            mw_hat = m_w / (1 - beta1**t)
            vw_hat = v_w / (1 - beta2**t)
            mb_hat = m_b / (1 - beta1**t)
            vb_hat = v_b / (1 - beta2**t)
            w -= settings.learning_rate * mw_hat / (np.sqrt(vw_hat) + eps)
            b -= settings.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)

        val_p = _sigmoid(x_va @ w + b) if len(y_va) else np.zeros(0)
        val_ce = _binary_cross_entropy(y_va, val_p) if len(y_va) else float("nan")
        if val_ce < best_val - 1e-5:
            best_val = val_ce
            best_w, best_b = w.copy(), b
            best_epoch = epoch
            stall = 0
        else:
            stall += 1
            if stall >= settings.patience:
                break

    probabilities = _sigmoid(x_te @ best_w + best_b) if len(test["y"]) else np.zeros(0)
    return {
        "probabilities": probabilities.astype(np.float64),
        "best_epoch": int(best_epoch),
        "validation_cross_entropy": float(best_val),
    }


def _fit_mlp(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    *,
    seed: int,
    settings: DecoderSettings,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    train = _subsample(train, settings.max_train_samples, rng)
    validation = _subsample(validation, settings.max_validation_samples, rng)
    x_tr, mean, std = _standardize_fit(train["x"])
    y_tr = train["y"].astype(np.float64)
    x_va = _standardize_apply(validation["x"], mean, std)
    y_va = validation["y"].astype(np.float64)
    x_te = _standardize_apply(test["x"], mean, std)

    n_in = x_tr.shape[1]
    n_h = int(settings.mlp_hidden_size)
    scale1 = (2.0 / max(n_in, 1)) ** 0.5
    scale2 = (2.0 / max(n_h, 1)) ** 0.5
    w1 = rng.normal(0.0, scale1, size=(n_in, n_h))
    b1 = np.zeros(n_h)
    w2 = rng.normal(0.0, scale2, size=(n_h,))
    b2 = 0.0
    params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    moments1 = {k: np.zeros_like(v) if hasattr(v, "shape") else 0.0 for k, v in params.items()}
    moments2 = {k: np.zeros_like(v) if hasattr(v, "shape") else 0.0 for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def forward(x):
        h = np.tanh(x @ params["w1"] + params["b1"])
        logits = h @ params["w2"] + params["b2"]
        return h, logits, _sigmoid(logits)

    best = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in params.items()}
    best_epoch = 0
    best_val = float("inf")
    stall = 0
    t_adam = 0

    for epoch in range(1, int(settings.mlp_epochs) + 1):
        for idx in _minibatches(len(y_tr), settings.batch_size, rng):
            xb = x_tr[idx]
            yb = y_tr[idx]
            h, logits, p = forward(xb)
            err = (p - yb) / max(len(yb), 1)
            gw2 = h.T @ err + settings.weight_decay * params["w2"]
            gb2 = float(err.sum())
            dh = np.outer(err, params["w2"]) * (1.0 - h**2)
            gw1 = xb.T @ dh + settings.weight_decay * params["w1"]
            gb1 = dh.sum(axis=0)
            grads = {"w1": gw1, "b1": gb1, "w2": gw2, "b2": gb2}
            t_adam += 1
            for name, g in grads.items():
                moments1[name] = beta1 * moments1[name] + (1 - beta1) * g
                moments2[name] = beta2 * moments2[name] + (1 - beta2) * (g**2)
                mhat = moments1[name] / (1 - beta1**t_adam)
                vhat = moments2[name] / (1 - beta2**t_adam)
                params[name] = params[name] - settings.learning_rate * mhat / (
                    np.sqrt(vhat) + eps
                )

        _, _, val_p = forward(x_va) if len(y_va) else (None, None, np.zeros(0))
        val_ce = _binary_cross_entropy(y_va, val_p) if len(y_va) else float("nan")
        if val_ce < best_val - 1e-5:
            best_val = val_ce
            best = {
                k: (v.copy() if hasattr(v, "copy") else v) for k, v in params.items()
            }
            best_epoch = epoch
            stall = 0
        else:
            stall += 1
            if stall >= settings.patience:
                break

    params = best
    _, _, probabilities = forward(x_te) if len(test["y"]) else (None, None, np.zeros(0))
    return {
        "probabilities": np.asarray(probabilities, dtype=np.float64),
        "best_epoch": int(best_epoch),
        "validation_cross_entropy": float(best_val),
    }


def fit_decoder(
    decoder_name: str,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    *,
    seed: int,
    settings: DecoderSettings | None = None,
) -> dict[str, Any]:
    settings = settings or DecoderSettings()
    if decoder_name == "logistic":
        return _fit_logistic(
            train, validation, test, seed=seed, settings=settings
        )
    if decoder_name == "mlp":
        return _fit_mlp(train, validation, test, seed=seed, settings=settings)
    raise ValueError(f"unknown decoder_name={decoder_name}")
