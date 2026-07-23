"""Shared rollout / belief helpers for v2 models (Kyan-compatible diagnostics)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.synthetic.channels import (
    ACTION_LEFT,
    ACTION_RIGHT,
    GO_CUE,
    N_INPUTS,
    NOT_REWARDED,
    REWARDED,
    paint_trial,
)
from src.synthetic.generate import SyntheticBatch
from src.synthetic.schema import LEFT, RIGHT

REGIMES = ("history_only", "full_information", "fixed_prior")


def load_model(model_id: str, path) -> Any:
    from src.models_v2.bayes import ExplicitBayes
    from src.models_v2.rnn_cells import GRURNN, TanhRNN

    if model_id in ("tanh_bptt", "tanh_pc"):
        return TanhRNN.load(path)
    if model_id == "gru":
        return GRURNN.load(path)
    if model_id == "bayes":
        return ExplicitBayes.load(path)
    raise ValueError(f"unknown model_id={model_id}")


def _step(model: Any, model_id: str, xt: np.ndarray, state: np.ndarray) -> np.ndarray:
    if model_id == "bayes":
        return model.step_prior(xt, state)
    return model.step(xt, state)


def _logits(model: Any, model_id: str, xt: np.ndarray, state: np.ndarray) -> np.ndarray:
    if model_id == "bayes":
        return model.logits_from_state(xt, state)
    return state @ model.W_hy + model.b_y


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def filter_batch_fixed_prior(batch: SyntheticBatch) -> SyntheticBatch:
    """Keep only trials with true P(right) ≈ 0.5 (unbiased blocks)."""

    mask = np.isclose(batch.p_right, 0.5)
    # Per-session: keep trials where mask; pad by truncating to min count
    n_keep = int(mask.sum(axis=1).min())
    if n_keep < 8:
        raise ValueError("Too few fixed-prior trials in batch for eval")
    idx = []
    for s in range(batch.side.shape[0]):
        where = np.flatnonzero(mask[s])[:n_keep]
        idx.append(where)
    index = np.stack(idx, axis=0)

    def take(arr: np.ndarray) -> np.ndarray:
        out = np.empty((arr.shape[0], n_keep) + arr.shape[2:], dtype=arr.dtype)
        for s in range(arr.shape[0]):
            out[s] = arr[s, index[s]]
        return out

    return SyntheticBatch(
        probability_left=take(batch.probability_left),
        p_right=take(batch.p_right),
        block_id=take(batch.block_id),
        side=take(batch.side),
        contrast=take(batch.contrast),
        phase=batch.phase,
    )


def rollout_closed_loop(
    model: Any,
    batch: SyntheticBatch,
    cfg: dict,
    model_id: str,
    *,
    seed: int = 0,
    regime: str = "history_only",
) -> Dict[str, np.ndarray]:
    """Closed-loop synth eval with Kyan-style zero-evidence probes.

    Readout order matches Kyan: **step the response tick, then** read probs.

    Regimes
    -------
    history_only
        Standard closed-loop (no oracle).
    fixed_prior
        Caller should pass a batch already filtered to p≈0.5 (see
        ``filter_batch_fixed_prior``).
    full_information
        Same dynamics, but choice / zero-evidence logits receive an additive
        bias from log prior odds (eval-time oracle control; gain from config).
    """

    if regime not in REGIMES:
        raise ValueError(f"unknown regime={regime}")

    n_sessions, n_trials = batch.shape
    phase = batch.phase
    noise_std = float(cfg.get("sensory_noise_std_synth", 0.15))
    rng = np.random.default_rng(seed)
    n_steps = phase.n_steps
    hidden_size = int(getattr(model, "hidden_size", 1))
    fi_gain = float(cfg.get("eval", {}).get("fi_oracle_logit_gain", 2.5))
    use_fi = regime == "full_information"

    choice = np.empty((n_sessions, n_trials), dtype=np.int64)
    p_choice_right = np.empty((n_sessions, n_trials), dtype=np.float64)
    zero_evidence_p_right = np.empty((n_sessions, n_trials), dtype=np.float64)
    correct = np.empty((n_sessions, n_trials), dtype=np.bool_)
    pre_stimulus_hidden = np.zeros(
        (n_sessions, n_trials, hidden_size), dtype=np.float64
    )

    state = model.zero_state(n_sessions)
    zeros = np.zeros((n_sessions, N_INPUTS), dtype=np.float64)

    def apply_oracle(logits: np.ndarray, true_p: np.ndarray) -> np.ndarray:
        if not use_fi:
            return logits
        # log-odds bias toward block prior: push right logit up when p high
        p = np.clip(true_p, 1e-4, 1.0 - 1e-4)
        log_odds = np.log(p) - np.log(1.0 - p)
        out = logits.copy()
        out[:, RIGHT] = out[:, RIGHT] + fi_gain * log_odds
        out[:, LEFT] = out[:, LEFT] - fi_gain * log_odds
        return out

    for t in range(n_trials):
        sides = batch.side[:, t]
        contrasts = batch.contrast[:, t]
        true_p_t = batch.p_right[:, t]
        trial_x = np.zeros((n_sessions, n_steps, N_INPUTS), dtype=np.float64)
        for s in range(n_sessions):
            noise = rng.normal(0.0, noise_std, size=2) if noise_std > 0 else None
            tx, _ = paint_trial(
                side=int(sides[s]),
                contrast=float(contrasts[s]),
                action=int(sides[s]),
                rewarded=True,
                phase=phase,
                visual_noise=noise,
            )
            tx[phase.feedback_start :, ACTION_LEFT] = 0.0
            tx[phase.feedback_start :, ACTION_RIGHT] = 0.0
            tx[phase.feedback_start :, REWARDED] = 0.0
            tx[phase.feedback_start :, NOT_REWARDED] = 0.0
            trial_x[s] = tx

        for step in range(phase.stim_start):
            state = _step(model, model_id, trial_x[:, step], state)
        if model_id != "bayes":
            pre_stimulus_hidden[:, t] = state
        else:
            pre_stimulus_hidden[:, t, 0] = state

        # Counterfactual zero-evidence (Kyan: step then read)
        state_cf = state.copy()
        for step in range(phase.stim_start, phase.response_tick + 1):
            xt = zeros.copy()
            if step == phase.go_tick:
                xt[:, GO_CUE] = 1.0
            state_cf = _step(model, model_id, xt, state_cf)
            if step == phase.response_tick:
                logits = apply_oracle(_logits(model, model_id, xt, state_cf), true_p_t)
                zero_evidence_p_right[:, t] = _softmax(logits)[:, RIGHT]

        # Actual path
        acts = np.zeros(n_sessions, dtype=np.int64)
        for step in range(phase.stim_start, phase.response_tick + 1):
            xt = trial_x[:, step]
            state = _step(model, model_id, xt, state)
            if step == phase.response_tick:
                logits = apply_oracle(_logits(model, model_id, xt, state), true_p_t)
                probs = _softmax(logits)
                p_choice_right[:, t] = probs[:, RIGHT]
                acts = np.argmax(probs, axis=1).astype(np.int64)

        for s in range(n_sessions):
            side = int(sides[s])
            act = int(acts[s])
            rew = act == side
            choice[s, t] = act
            correct[s, t] = rew
            for ft in range(phase.feedback_start, n_steps):
                trial_x[s, ft, ACTION_LEFT] = 1.0 if act == LEFT else 0.0
                trial_x[s, ft, ACTION_RIGHT] = 1.0 if act == RIGHT else 0.0
                trial_x[s, ft, REWARDED] = 1.0 if rew else 0.0
                trial_x[s, ft, NOT_REWARDED] = 0.0 if rew else 1.0

        for step in range(phase.feedback_start, n_steps):
            state = _step(model, model_id, trial_x[:, step], state)

    return {
        "choice": choice,
        "p_choice_right": p_choice_right,
        "zero_evidence_p_right": zero_evidence_p_right,
        "correct": correct,
        "pre_stimulus_hidden": pre_stimulus_hidden,
        "side": batch.side.copy(),
        "contrast": batch.contrast.copy(),
        "true_p_right": batch.p_right.copy(),
        "probability_left": batch.probability_left.copy(),
        "block_id": batch.block_id.copy(),
        "regime": np.asarray(regime),
        "p_right": p_choice_right,
        "belief": zero_evidence_p_right,
    }


def rollout_real_session(
    model: Any,
    data: dict,
    cfg: dict,
    model_id: str,
    phase,
    *,
    regime: str = "history_only",
) -> Dict[str, np.ndarray]:
    """Roll one real mapped session (mouse feedback) with Kyan probes + regimes.

    ``data`` is the dict from ``encode_real_session`` / real_v2_ticks npz.
    """

    if regime not in REGIMES:
        raise ValueError(f"unknown regime={regime}")

    x_all = data["inputs"]
    n_trials = int(data["n_trials"])
    n_steps = int(data["n_steps"])
    correct_side = np.asarray(data["correct_side"], dtype=np.int64)
    mouse_choice = np.asarray(data["mouse_choice"], dtype=np.int64)
    contrast = np.asarray(data["contrast"], dtype=np.float64)
    pleft = np.asarray(data["probability_left"], dtype=np.float64)
    true_p = 1.0 - pleft

    if regime == "fixed_prior":
        keep = np.isclose(true_p, 0.5)
        if keep.sum() < 8:
            raise ValueError("Too few fixed-prior trials in session")
    else:
        keep = np.ones(n_trials, dtype=bool)

    fi_gain = float(cfg.get("eval", {}).get("fi_oracle_logit_gain", 2.5))
    use_fi = regime == "full_information"
    hidden_size = int(getattr(model, "hidden_size", 1))
    zeros = np.zeros((1, N_INPUTS), dtype=np.float64)

    # We'll fill only kept trials but run state through full session for causality
    # then subset outputs. Running full session is required for history.
    p_choice_right = np.full(n_trials, np.nan)
    zero_evidence_p_right = np.full(n_trials, np.nan)
    choice = np.full(n_trials, -1, dtype=np.int64)
    pre_h = np.zeros((n_trials, hidden_size), dtype=np.float64)

    state = model.zero_state(1)

    def apply_oracle(logits: np.ndarray, p: float) -> np.ndarray:
        if not use_fi:
            return logits
        pp = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        log_odds = np.log(pp) - np.log(1.0 - pp)
        out = logits.copy()
        out[0, RIGHT] += fi_gain * log_odds
        out[0, LEFT] -= fi_gain * log_odds
        return out

    for t in range(n_trials):
        trial = x_all[t * n_steps : (t + 1) * n_steps]

        for step in range(phase.stim_start):
            state = _step(model, model_id, trial[step : step + 1], state)
        if model_id != "bayes":
            pre_h[t] = state[0]
        else:
            pre_h[t, 0] = state[0]

        state_cf = state.copy()
        for step in range(phase.stim_start, phase.response_tick + 1):
            xt = zeros.copy()
            if step == phase.go_tick:
                xt[0, GO_CUE] = 1.0
            state_cf = _step(model, model_id, xt, state_cf)
            if step == phase.response_tick:
                logits = apply_oracle(
                    _logits(model, model_id, xt, state_cf), float(true_p[t])
                )
                zero_evidence_p_right[t] = float(_softmax(logits)[0, RIGHT])

        for step in range(phase.stim_start, phase.response_tick + 1):
            xt = trial[step : step + 1]
            state = _step(model, model_id, xt, state)
            if step == phase.response_tick:
                logits = apply_oracle(
                    _logits(model, model_id, xt, state), float(true_p[t])
                )
                probs = _softmax(logits)
                p_choice_right[t] = float(probs[0, RIGHT])
                choice[t] = int(np.argmax(probs[0]))

        # Mouse feedback already in trial_x for feedback ticks
        for step in range(phase.feedback_start, n_steps):
            state = _step(model, model_id, trial[step : step + 1], state)

    # Subset to regime mask; reshape as 1×T for figure helpers
    idx = np.flatnonzero(keep)
    block_id = np.zeros(n_trials, dtype=np.int64)
    # block id from changes in true_p
    bid = 0
    for t in range(1, n_trials):
        if not np.isclose(true_p[t], true_p[t - 1]):
            bid += 1
        block_id[t] = bid

    return {
        "choice": choice[idx][None, :],
        "p_choice_right": p_choice_right[idx][None, :],
        "zero_evidence_p_right": zero_evidence_p_right[idx][None, :],
        "correct": (choice[idx] == correct_side[idx])[None, :],
        "pre_stimulus_hidden": pre_h[idx][None, :, :],
        "side": correct_side[idx][None, :],
        "contrast": contrast[idx][None, :],
        "true_p_right": true_p[idx][None, :],
        "probability_left": pleft[idx][None, :],
        "block_id": block_id[idx][None, :],
        "mouse_choice": mouse_choice[idx][None, :],
        "p_right": p_choice_right[idx][None, :],
        "belief": zero_evidence_p_right[idx][None, :],
    }


def pool_real_rollouts(rolls: list[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Pad-pool 1×T session rolls into S×T_max for multipanel figures."""

    if not rolls:
        raise ValueError("no real rollouts to pool")
    n_sess = len(rolls)
    t_max = max(int(r["side"].shape[1]) for r in rolls)
    keys_2d = (
        "choice",
        "p_choice_right",
        "zero_evidence_p_right",
        "correct",
        "side",
        "contrast",
        "true_p_right",
        "probability_left",
        "block_id",
        "mouse_choice",
    )
    out: Dict[str, np.ndarray] = {}
    for key in keys_2d:
        dtype = rolls[0][key].dtype
        fill = -1 if np.issubdtype(dtype, np.integer) else np.nan
        arr = np.full((n_sess, t_max), fill, dtype=np.float64 if fill != -1 else dtype)
        if fill == -1:
            arr = np.full((n_sess, t_max), -1, dtype=dtype)
        for i, r in enumerate(rolls):
            t = r[key].shape[1]
            arr[i, :t] = r[key][0]
        out[key] = arr
    # valid mask for metrics: side >= 0
    out["valid"] = out["side"] >= 0
    out["p_right"] = out["p_choice_right"]
    out["belief"] = out["zero_evidence_p_right"]
    return out


def accuracy_and_ce(roll: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Synth closed-loop accuracy and response cross-entropy vs correct side."""

    side = roll["side"]
    pred = roll["choice"]
    pr = roll["p_choice_right"] if "p_choice_right" in roll else roll["p_right"]
    p_c = np.where(side == RIGHT, pr, 1.0 - pr)
    return {
        "accuracy": float(np.mean(pred == side)),
        "cross_entropy": float(-np.mean(np.log(np.clip(p_c, 1e-12, 1.0)))),
    }


def accuracy_real(roll: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Real transfer: accuracy / CE vs **correct stimulus side only** (not mouse)."""

    valid = roll["valid"] if "valid" in roll else np.ones_like(roll["side"], dtype=bool)
    side = roll["side"][valid]
    pred = roll["choice"][valid]
    pr = roll["p_choice_right"][valid]
    p_c = np.where(side == RIGHT, pr, 1.0 - pr)
    return {
        "accuracy": float(np.mean(pred == side)),
        "acc_vs_correct_side": float(np.mean(pred == side)),
        "cross_entropy": float(-np.mean(np.log(np.clip(p_c, 1e-12, 1.0)))),
        "ce_vs_correct_side": float(-np.mean(np.log(np.clip(p_c, 1e-12, 1.0)))),
    }


def _get(roll, *keys: str) -> np.ndarray:
    files = set(roll.files) if hasattr(roll, "files") else None
    for key in keys:
        if files is not None:
            if key in files:
                return roll[key]
        elif key in roll:
            return roll[key]
    raise KeyError(keys)


def switch_centered_zero_evidence(
    roll,
    *,
    before: int = 20,
    after: int = 30,
) -> Dict[str, np.ndarray]:
    try:
        true_p = _get(roll, "true_p_right")
    except KeyError:
        true_p = 1.0 - _get(roll, "probability_left")
    try:
        pref = _get(roll, "zero_evidence_p_right")
    except KeyError:
        pref = _get(roll, "belief")

    offsets = np.arange(-before, after + 1)
    groups: Dict[str, list] = {"low_to_high": [], "high_to_low": []}
    for session in range(true_p.shape[0]):
        # skip padded / invalid trials
        valid = np.isfinite(true_p[session])
        if "valid" in (getattr(roll, "files", None) or roll):
            try:
                valid = valid & np.asarray(_get(roll, "valid")[session], dtype=bool)
            except Exception:
                pass
        p_sess = true_p[session].copy()
        pref_sess = pref[session].copy()
        # treat invalid as no-switch by freezing last valid prior
        if not valid.any():
            continue
        last = float(p_sess[np.flatnonzero(valid)[0]])
        for t in range(len(p_sess)):
            if not valid[t]:
                p_sess[t] = last
            else:
                last = float(p_sess[t])
        switch_indices = np.flatnonzero(np.diff(p_sess) != 0) + 1
        for switch in switch_indices:
            if switch - before < 0 or switch + after >= p_sess.shape[0]:
                continue
            if not valid[switch - before : switch + after + 1].all():
                continue
            direction = (
                "low_to_high"
                if p_sess[switch] > p_sess[switch - 1]
                else "high_to_low"
            )
            groups[direction].append(
                pref_sess[switch - before : switch + after + 1]
            )
    out: Dict[str, np.ndarray] = {"offsets": offsets}
    for direction, curves in groups.items():
        if curves:
            out[direction] = np.mean(np.stack(curves), axis=0)
        else:
            out[direction] = np.full_like(offsets, np.nan, dtype=np.float64)
    return out


def switch_centered_per_session(
    roll,
    *,
    before: int = 20,
    after: int = 30,
) -> Dict[str, object]:
    """Per-session mean switch curves (for variance-by-color plots)."""

    try:
        true_p = _get(roll, "true_p_right")
    except KeyError:
        true_p = 1.0 - _get(roll, "probability_left")
    try:
        pref = _get(roll, "zero_evidence_p_right")
    except KeyError:
        pref = _get(roll, "belief")

    offsets = np.arange(-before, after + 1)
    n_sess = int(true_p.shape[0])
    per: list[Dict[str, np.ndarray]] = []
    for session in range(n_sess):
        valid = np.isfinite(true_p[session])
        if "valid" in (getattr(roll, "files", None) or roll):
            try:
                valid = valid & np.asarray(_get(roll, "valid")[session], dtype=bool)
            except Exception:
                pass
        groups: Dict[str, list] = {"low_to_high": [], "high_to_low": []}
        if not valid.any():
            per.append(
                {
                    "low_to_high": np.full_like(offsets, np.nan, dtype=np.float64),
                    "high_to_low": np.full_like(offsets, np.nan, dtype=np.float64),
                }
            )
            continue
        p_sess = true_p[session].copy()
        pref_sess = pref[session].copy()
        last = float(p_sess[np.flatnonzero(valid)[0]])
        for t in range(len(p_sess)):
            if not valid[t]:
                p_sess[t] = last
            else:
                last = float(p_sess[t])
        switch_indices = np.flatnonzero(np.diff(p_sess) != 0) + 1
        for switch in switch_indices:
            if switch - before < 0 or switch + after >= p_sess.shape[0]:
                continue
            if not valid[switch - before : switch + after + 1].all():
                continue
            direction = (
                "low_to_high"
                if p_sess[switch] > p_sess[switch - 1]
                else "high_to_low"
            )
            groups[direction].append(
                pref_sess[switch - before : switch + after + 1]
            )
        entry: Dict[str, np.ndarray] = {}
        for direction in ("low_to_high", "high_to_low"):
            curves = groups[direction]
            entry[direction] = (
                np.mean(np.stack(curves), axis=0)
                if curves
                else np.full_like(offsets, np.nan, dtype=np.float64)
            )
        per.append(entry)
    return {"offsets": offsets, "per_session": per}


def summarize_kyan_diagnostics(roll) -> Dict[str, object]:
    try:
        true_p = _get(roll, "true_p_right")
    except KeyError:
        true_p = 1.0 - _get(roll, "probability_left")
    try:
        zero_pref = _get(roll, "zero_evidence_p_right")
    except KeyError:
        zero_pref = _get(roll, "belief")
    try:
        predicted = _get(roll, "p_choice_right")
    except KeyError:
        predicted = _get(roll, "p_right")
    contrast = _get(roll, "contrast")
    block_id = _get(roll, "block_id")

    valid = np.isfinite(true_p) & np.isfinite(zero_pref) & np.isfinite(predicted)
    files = getattr(roll, "files", None)
    has_valid = ("valid" in files) if files is not None else ("valid" in roll)
    if has_valid:
        valid = valid & np.asarray(_get(roll, "valid"), dtype=bool)

    block_age = np.zeros_like(block_id, dtype=np.int64)
    for trial_index in range(1, block_age.shape[1]):
        same = block_id[:, trial_index] == block_id[:, trial_index - 1]
        block_age[:, trial_index] = np.where(same, block_age[:, trial_index - 1] + 1, 0)
    settled = (block_age >= 15) & valid

    low_mask = np.isclose(true_p, 0.2) & valid
    high_mask = np.isclose(true_p, 0.8) & valid
    zero_contrast = np.isclose(contrast, 0.0) & valid
    return {
        "zero_contrast_observed_choice_probability": {
            "low_p_right_block": float(predicted[low_mask & zero_contrast].mean())
            if (low_mask & zero_contrast).any()
            else float("nan"),
            "high_p_right_block": float(predicted[high_mask & zero_contrast].mean())
            if (high_mask & zero_contrast).any()
            else float("nan"),
        },
        "counterfactual_zero_evidence_choice_probability": {
            "low_p_right_block": float(zero_pref[low_mask].mean())
            if low_mask.any()
            else float("nan"),
            "high_p_right_block": float(zero_pref[high_mask].mean())
            if high_mask.any()
            else float("nan"),
            "history_gap": float(
                zero_pref[high_mask].mean() - zero_pref[low_mask].mean()
            )
            if low_mask.any() and high_mask.any()
            else float("nan"),
        },
        "settled_block_zero_evidence_calibration": {
            "definition": "trials at least 15 trials after the most recent switch",
            "low_p_right_block": float(zero_pref[low_mask & settled].mean())
            if (low_mask & settled).any()
            else float("nan"),
            "high_p_right_block": float(zero_pref[high_mask & settled].mean())
            if (high_mask & settled).any()
            else float("nan"),
            "mean_absolute_error_to_true_prior": float(
                np.abs(zero_pref[settled] - true_p[settled]).mean()
            )
            if settled.any()
            else float("nan"),
        },
    }
