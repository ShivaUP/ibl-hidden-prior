"""Shared rollout / belief helpers for v2 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from src.models_v2.bayes import ExplicitBayes
from src.models_v2.rnn_cells import GRURNN, TanhRNN
from src.synthetic.channels import (
    ACTION_LEFT,
    ACTION_RIGHT,
    N_INPUTS,
    NOT_REWARDED,
    REWARDED,
    PhaseTicks,
    paint_trial,
)
from src.synthetic.generate import SyntheticBatch
from src.synthetic.schema import LEFT, RIGHT


def load_model(model_id: str, path) -> Any:
    if model_id in ("tanh_bptt", "tanh_pc"):
        return TanhRNN.load(path)
    if model_id == "gru":
        return GRURNN.load(path)
    if model_id == "bayes":
        return ExplicitBayes.load(path)
    raise ValueError(f"unknown model_id={model_id}")


def rollout_closed_loop(
    model: Any,
    batch: SyntheticBatch,
    cfg: dict,
    model_id: str,
) -> Dict[str, np.ndarray]:
    """Closed-loop synth eval: model chooses; feedback from its own action."""

    n_sessions, n_trials = batch.shape
    phase = batch.phase
    noise_std = float(cfg.get("sensory_noise_std_synth", 0.15))
    rng = np.random.default_rng(0)
    n_steps = phase.n_steps
    choice = np.empty((n_sessions, n_trials), dtype=np.int64)
    p_right = np.empty((n_sessions, n_trials), dtype=np.float64)
    correct = np.empty((n_sessions, n_trials), dtype=np.bool_)
    # Belief proxy: P(right) at response for Bayes; decoded linear for RNN
    belief = np.empty((n_sessions, n_trials), dtype=np.float64)

    is_bayes = model_id == "bayes"
    if is_bayes:
        state = model.zero_state(n_sessions)
    else:
        state = model.zero_state(n_sessions)

    for t in range(n_trials):
        # Build pre-response inputs with placeholder feedback (zeros) for this trial
        # Then run ticks 0..response, take action, then paint feedback and continue
        sides = batch.side[:, t]
        contrasts = batch.contrast[:, t]
        # Allocate per-session trial tensors
        trial_x = np.zeros((n_sessions, n_steps, N_INPUTS), dtype=np.float64)
        for s in range(n_sessions):
            noise = rng.normal(0.0, noise_std, size=2) if noise_std > 0 else None
            # temporary action=side (overwritten after choice for feedback ticks)
            tx, _ = paint_trial(
                side=int(sides[s]),
                contrast=float(contrasts[s]),
                action=int(sides[s]),
                rewarded=True,
                phase=phase,
                visual_noise=noise,
            )
            # zero feedback for closed-loop until we know action
            tx[phase.feedback_start :, ACTION_LEFT] = 0.0
            tx[phase.feedback_start :, ACTION_RIGHT] = 0.0
            tx[phase.feedback_start :, REWARDED] = 0.0
            tx[phase.feedback_start :, NOT_REWARDED] = 0.0
            trial_x[s] = tx

        # Run through response tick
        probs_at_resp = np.empty((n_sessions, 2))
        for step in range(phase.response_tick + 1):
            xt = trial_x[:, step]
            if is_bayes:
                if step == phase.response_tick:
                    probs_at_resp = model.probs(xt, state)
                state = model.step_prior(xt, state)
            else:
                state = model.step(xt, state)
                if step == phase.response_tick:
                    probs_at_resp = model.probs(state)

        acts = (probs_at_resp[:, RIGHT] >= 0.5).astype(np.int64)
        # Optional stochastic — use argmax for eval stability
        for s in range(n_sessions):
            side = int(sides[s])
            act = int(acts[s])
            rew = act == side
            choice[s, t] = act
            p_right[s, t] = float(probs_at_resp[s, RIGHT])
            correct[s, t] = rew
            belief[s, t] = float(probs_at_resp[s, RIGHT])
            for ft in range(phase.feedback_start, n_steps):
                trial_x[s, ft, ACTION_LEFT] = 1.0 if act == LEFT else 0.0
                trial_x[s, ft, ACTION_RIGHT] = 1.0 if act == RIGHT else 0.0
                trial_x[s, ft, REWARDED] = 1.0 if rew else 0.0
                trial_x[s, ft, NOT_REWARDED] = 0.0 if rew else 1.0

        # Continue through feedback ticks to update state
        for step in range(phase.feedback_start, n_steps):
            xt = trial_x[:, step]
            if is_bayes:
                state = model.step_prior(xt, state)
            else:
                state = model.step(xt, state)

    return {
        "choice": choice,
        "p_right": p_right,
        "correct": correct,
        "belief": belief,
        "side": batch.side.copy(),
        "contrast": batch.contrast.copy(),
        "probability_left": batch.probability_left.copy(),
        "block_id": batch.block_id.copy(),
    }


def accuracy_and_ce(roll: Dict[str, np.ndarray]) -> Dict[str, float]:
    yt = roll["side"]
    pr = roll["p_right"]
    pred = roll["choice"]
    acc = float(np.mean(pred == yt))
    # CE vs correct side
    p_correct = np.where(yt == RIGHT, pr, 1.0 - pr)
    ce = float(-np.mean(np.log(np.clip(p_correct, 1e-12, 1.0))))
    return {"accuracy": acc, "cross_entropy": ce}
