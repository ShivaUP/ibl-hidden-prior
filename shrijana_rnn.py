"""Train and evaluate a Gated Recurrent Unit (GRU) on the hidden-prior task.

This is the GRU counterpart to the vanilla-tanh "Standard RNN" script. The task,
input encoding, training loop, and evaluation are identical to that script so the
two models can be compared head-to-head. The only substantive change is the
recurrent cell: a full GRU with update and reset gates, implemented (forward
*and* backprop-through-time) in pure NumPy so it runs without PyTorch.

Task design (synthetic block task):

    - The prior alternates between left (0.2) and right (0.8) blocks.
    - Block lengths are sampled uniformly from [40, 80] trials. Switches are
      not cued.
    - Stimulus contrast is sampled from {0, 0.0625, 0.125, 0.25, 0.5, 1.0}.
    - 0% contrast trials still have a latent side drawn from the block prior, so a
      model can only beat chance on them by using the prior.

NOTE: this uses synthetic episodes. A teammate is analysing the real mouse
dataset; that data can later be fed in place of `generate_trials` while keeping
the rest of the pipeline unchanged.

Each trial is unrolled over exactly eight time steps:

1-2. neutral (zero-input) baseline
3.   visual stimulus
4.   blank delay
5.   go cue
6.   response (the network is read out here)
7-8. action/outcome feedback

Visual encoding:  left -> contrast * [0, 1],  right -> contrast * [1, 0].
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


# Labels ---------------------------------------------------------------------

LEFT = 0
RIGHT = 1

# Zero-based step indices. Their human-readable step numbers are index + 1.
BASELINE_STEPS = (0, 1)
STIMULUS_STEP = 2
BLANK_STEP = 3
GO_STEP = 4
RESPONSE_STEP = 5
FEEDBACK_STEPS = (6, 7)
N_STEPS = 8

# Input channels. Putting RIGHT before LEFT makes the first two entries obey
# the requested left=[0,c], right=[c,0] encoding.
VISUAL_RIGHT = 0
VISUAL_LEFT = 1
GO_CUE = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
REWARDED = 5
NOT_REWARDED = 6
N_INPUTS = 7

INPUT_NAMES = (
    "visual_right",
    "visual_left",
    "go_cue",
    "action_left",
    "action_right",
    "rewarded",
    "not_rewarded",
)


@dataclass(frozen=True)
class TaskConfig:
    """Parameters of the synthetic block task."""

    low_p_right: float = 0.20
    high_p_right: float = 0.80
    min_block_trials: int = 40
    max_block_trials: int = 80
    contrasts: Tuple[float, ...] = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)
    contrast_probabilities: Tuple[float, ...] = (
        0.18,
        0.18,
        0.18,
        0.18,
        0.16,
        0.12,
    )
    sensory_noise_std: float = 0.15
    # Training feedback is teacher-forced with occasional errors. The chosen
    # action plus reward/no-reward still reveals the correct side causally.
    training_feedback_error_rate: float = 0.20

    def validate(self) -> None:
        if not 0.0 < self.low_p_right < 0.5:
            raise ValueError("low_p_right must be between 0 and 0.5")
        if not 0.5 < self.high_p_right < 1.0:
            raise ValueError("high_p_right must be between 0.5 and 1")
        if self.min_block_trials < 2 or self.max_block_trials < self.min_block_trials:
            raise ValueError("Invalid block length limits")
        if len(self.contrasts) != len(self.contrast_probabilities):
            raise ValueError("contrasts and contrast_probabilities must match")
        if any(c < 0 for c in self.contrasts):
            raise ValueError("contrasts must be non-negative")
        if not np.isclose(sum(self.contrast_probabilities), 1.0):
            raise ValueError("contrast_probabilities must sum to 1")
        if self.sensory_noise_std < 0:
            raise ValueError("sensory_noise_std must be non-negative")


@dataclass(frozen=True)
class TrainConfig:
    """Optimization parameters for truncated backpropagation through time."""

    hidden_size: int = 64
    epochs: int = 60
    sessions_per_epoch: int = 24
    trials_per_session: int = 240
    bptt_trials: int = 32
    learning_rate: float = 2e-3
    gradient_clip_norm: float = 1.0
    weight_decay: float = 1e-5
    seed: int = 7


@dataclass
class TrialBatch:
    """A batch of synthetic sessions before action/outcome inputs are added."""

    p_right: np.ndarray  # [sessions, trials]
    block_id: np.ndarray  # [sessions, trials]
    side: np.ndarray  # [sessions, trials], LEFT=0 or RIGHT=1
    contrast: np.ndarray  # [sessions, trials]
    visual_observation: np.ndarray  # [sessions, trials, 2]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.side.shape


def generate_trials(
    n_sessions: int,
    n_trials: int,
    task: TaskConfig,
    rng: np.random.Generator,
) -> TrialBatch:
    """Generate alternating hidden-prior blocks and noisy visual observations."""

    task.validate()
    p_right = np.empty((n_sessions, n_trials), dtype=np.float64)
    block_id = np.empty((n_sessions, n_trials), dtype=np.int64)

    for session in range(n_sessions):
        cursor = 0
        this_prior = task.low_p_right if rng.random() < 0.5 else task.high_p_right
        this_block = 0
        while cursor < n_trials:
            block_length = int(
                rng.integers(task.min_block_trials, task.max_block_trials + 1)
            )
            end = min(cursor + block_length, n_trials)
            p_right[session, cursor:end] = this_prior
            block_id[session, cursor:end] = this_block
            cursor = end
            this_block += 1
            this_prior = (
                task.high_p_right
                if np.isclose(this_prior, task.low_p_right)
                else task.low_p_right
            )

    side = (rng.random((n_sessions, n_trials)) < p_right).astype(np.int64)
    contrast = rng.choice(
        np.asarray(task.contrasts, dtype=np.float64),
        size=(n_sessions, n_trials),
        p=np.asarray(task.contrast_probabilities, dtype=np.float64),
    )

    # Exact signal before observation noise:
    # left -> [0, contrast], right -> [contrast, 0].
    visual_signal = np.zeros((n_sessions, n_trials, 2), dtype=np.float64)
    visual_signal[..., VISUAL_RIGHT] = contrast * (side == RIGHT)
    visual_signal[..., VISUAL_LEFT] = contrast * (side == LEFT)
    sensory_noise = rng.normal(
        loc=0.0,
        scale=task.sensory_noise_std,
        size=visual_signal.shape,
    )
    visual_observation = visual_signal + sensory_noise

    return TrialBatch(
        p_right=p_right,
        block_id=block_id,
        side=side,
        contrast=contrast,
        visual_observation=visual_observation,
    )


def build_training_sequence(
    trials: TrialBatch,
    task: TaskConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a teacher-forced sequence with loss only at response step 6.

    The feedback action is correct on most training trials and intentionally
    wrong on a configurable fraction. Its outcome is then encoded at both
    feedback steps. Nothing from steps 7-8 can affect the current response;
    it can only update the recurrent state for later trials.
    """

    n_sessions, n_trials = trials.shape
    x = np.zeros((n_sessions, n_trials, N_STEPS, N_INPUTS), dtype=np.float64)
    targets = np.full((n_sessions, n_trials, N_STEPS), -1, dtype=np.int64)

    x[:, :, STIMULUS_STEP, :2] = trials.visual_observation
    x[:, :, GO_STEP, GO_CUE] = 1.0
    targets[:, :, RESPONSE_STEP] = trials.side

    make_error = rng.random((n_sessions, n_trials)) < task.training_feedback_error_rate
    feedback_action = np.where(make_error, 1 - trials.side, trials.side)
    rewarded = feedback_action == trials.side

    for step in FEEDBACK_STEPS:
        x[:, :, step, ACTION_LEFT] = feedback_action == LEFT
        x[:, :, step, ACTION_RIGHT] = feedback_action == RIGHT
        x[:, :, step, REWARDED] = rewarded
        x[:, :, step, NOT_REWARDED] = ~rewarded

    return (
        x.reshape(n_sessions, n_trials * N_STEPS, N_INPUTS),
        targets.reshape(n_sessions, n_trials * N_STEPS),
    )


def make_example_trial(
    side: int = LEFT,
    contrast: float = 0.25,
    action: int = LEFT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return one noiseless trial, mainly for inspection and unit tests."""

    if side not in (LEFT, RIGHT) or action not in (LEFT, RIGHT):
        raise ValueError("side and action must be LEFT (0) or RIGHT (1)")
    x = np.zeros((N_STEPS, N_INPUTS), dtype=np.float64)
    x[STIMULUS_STEP, VISUAL_RIGHT] = contrast if side == RIGHT else 0.0
    x[STIMULUS_STEP, VISUAL_LEFT] = contrast if side == LEFT else 0.0
    x[GO_STEP, GO_CUE] = 1.0
    rewarded = action == side
    for step in FEEDBACK_STEPS:
        x[step, ACTION_LEFT if action == LEFT else ACTION_RIGHT] = 1.0
        x[step, REWARDED if rewarded else NOT_REWARDED] = 1.0
    targets = np.full(N_STEPS, -1, dtype=np.int64)
    targets[RESPONSE_STEP] = side
    return x, targets


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 0.5 * (np.tanh(0.5 * value) + 1.0)


class GRU:
    """A Gated Recurrent Unit with a two-action readout and explicit NumPy BPTT.

    Cell equations (per time step, `@` is a matrix product):

        z = sigmoid(x @ W_xz + h @ W_hz + b_z)        # update gate
        r = sigmoid(x @ W_xr + h @ W_hr + b_r)        # reset gate
        n = tanh(x @ W_xn + (r * h) @ W_hn + b_n)     # candidate state
        h_new = (1 - z) * n + z * h                   # blended state

    The update gate lets the unit copy its previous state forward unchanged,
    which is exactly what a persistent block belief needs.
    """

    parameter_names = (
        "W_xz", "W_hz", "b_z",
        "W_xr", "W_hr", "b_r",
        "W_xn", "W_hn", "b_n",
        "W_hy", "b_y",
    )

    def __init__(self, input_size: int, hidden_size: int, rng: np.random.Generator):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

        input_scale = math.sqrt(2.0 / (input_size + hidden_size))
        recurrent_scale = math.sqrt(2.0 / (hidden_size + hidden_size))
        output_scale = math.sqrt(2.0 / (hidden_size + 2))

        def input_weight() -> np.ndarray:
            return rng.normal(0.0, input_scale, (input_size, hidden_size))

        def recurrent_weight() -> np.ndarray:
            return rng.normal(0.0, recurrent_scale, (hidden_size, hidden_size))

        self.W_xz = input_weight()
        self.W_hz = recurrent_weight()
        # Positive update-gate bias makes the cell lean toward keeping its state,
        # which encourages the persistent memory the block prior requires.
        self.b_z = np.ones(hidden_size, dtype=np.float64)

        self.W_xr = input_weight()
        self.W_hr = recurrent_weight()
        self.b_r = np.zeros(hidden_size, dtype=np.float64)

        self.W_xn = input_weight()
        self.W_hn = recurrent_weight()
        self.b_n = np.zeros(hidden_size, dtype=np.float64)

        self.W_hy = rng.normal(0.0, output_scale, (hidden_size, 2))
        self.b_y = np.zeros(2, dtype=np.float64)

    def parameters(self) -> Dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self.parameter_names}

    def zero_state(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.hidden_size), dtype=np.float64)

    def step(self, x_t: np.ndarray, h_previous: np.ndarray) -> np.ndarray:
        z = _sigmoid(x_t @ self.W_xz + h_previous @ self.W_hz + self.b_z)
        r = _sigmoid(x_t @ self.W_xr + h_previous @ self.W_hr + self.b_r)
        n = np.tanh(x_t @ self.W_xn + (r * h_previous) @ self.W_hn + self.b_n)
        return (1.0 - z) * n + z * h_previous

    def response_probabilities(self, hidden: np.ndarray) -> np.ndarray:
        logits = hidden @ self.W_hy + self.b_y
        logits -= logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    def loss_and_gradients(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        weight_decay: float = 0.0,
    ) -> Tuple[float, Dict[str, np.ndarray], np.ndarray]:
        """Forward and backward pass for one truncated sequence chunk."""

        batch_size, n_time, _ = x.shape
        hidden_dim = self.hidden_size

        # Caches for backprop through time.
        h_prev_cache = np.empty((batch_size, n_time, hidden_dim), dtype=np.float64)
        z_cache = np.empty_like(h_prev_cache)
        r_cache = np.empty_like(h_prev_cache)
        n_cache = np.empty_like(h_prev_cache)
        hidden = np.empty_like(h_prev_cache)
        logits = np.empty((batch_size, n_time, 2), dtype=np.float64)

        h = h_initial
        for t in range(n_time):
            x_t = x[:, t]
            z = _sigmoid(x_t @ self.W_xz + h @ self.W_hz + self.b_z)
            r = _sigmoid(x_t @ self.W_xr + h @ self.W_hr + self.b_r)
            n = np.tanh(x_t @ self.W_xn + (r * h) @ self.W_hn + self.b_n)
            h_new = (1.0 - z) * n + z * h

            h_prev_cache[:, t] = h
            z_cache[:, t] = z
            r_cache[:, t] = r
            n_cache[:, t] = n
            hidden[:, t] = h_new
            logits[:, t] = h_new @ self.W_hy + self.b_y
            h = h_new

        shifted = logits - logits.max(axis=2, keepdims=True)
        exp_logits = np.exp(shifted)
        probabilities = exp_logits / exp_logits.sum(axis=2, keepdims=True)

        valid = targets >= 0
        response_rows, response_times = np.nonzero(valid)
        n_responses = len(response_rows)
        if n_responses == 0:
            raise ValueError("A training chunk must contain at least one response step")
        response_targets = targets[response_rows, response_times]
        chosen_probabilities = probabilities[
            response_rows, response_times, response_targets
        ]
        loss = float(-np.mean(np.log(chosen_probabilities + 1e-12)))

        d_logits = np.zeros_like(logits)
        d_logits[response_rows, response_times] = probabilities[
            response_rows, response_times
        ]
        d_logits[response_rows, response_times, response_targets] -= 1.0
        d_logits /= n_responses

        gradients = {
            name: np.zeros_like(value) for name, value in self.parameters().items()
        }
        hidden_2d = hidden.reshape(batch_size * n_time, hidden_dim)
        d_logits_2d = d_logits.reshape(batch_size * n_time, 2)
        gradients["W_hy"] = hidden_2d.T @ d_logits_2d
        gradients["b_y"] = d_logits_2d.sum(axis=0)

        d_h_next = np.zeros((batch_size, hidden_dim), dtype=np.float64)
        for t in range(n_time - 1, -1, -1):
            x_t = x[:, t]
            h_prev = h_prev_cache[:, t]
            z = z_cache[:, t]
            r = r_cache[:, t]
            n = n_cache[:, t]

            # Gradient arriving at h_new from the readout and the future.
            d_h = d_logits[:, t] @ self.W_hy.T + d_h_next

            # h_new = (1 - z) * n + z * h_prev
            d_z = d_h * (h_prev - n)
            d_n = d_h * (1.0 - z)
            d_h_prev = d_h * z  # direct skip-connection path

            # n = tanh(a_n),  a_n = x W_xn + (r * h_prev) W_hn + b_n
            d_a_n = d_n * (1.0 - n * n)
            gated_h = r * h_prev
            gradients["W_xn"] += x_t.T @ d_a_n
            gradients["W_hn"] += gated_h.T @ d_a_n
            gradients["b_n"] += d_a_n.sum(axis=0)
            d_gated_h = d_a_n @ self.W_hn.T
            d_r = d_gated_h * h_prev
            d_h_prev += d_gated_h * r

            # z = sigmoid(a_z),  a_z = x W_xz + h_prev W_hz + b_z
            d_a_z = d_z * z * (1.0 - z)
            gradients["W_xz"] += x_t.T @ d_a_z
            gradients["W_hz"] += h_prev.T @ d_a_z
            gradients["b_z"] += d_a_z.sum(axis=0)
            d_h_prev += d_a_z @ self.W_hz.T

            # r = sigmoid(a_r),  a_r = x W_xr + h_prev W_hr + b_r
            d_a_r = d_r * r * (1.0 - r)
            gradients["W_xr"] += x_t.T @ d_a_r
            gradients["W_hr"] += h_prev.T @ d_a_r
            gradients["b_r"] += d_a_r.sum(axis=0)
            d_h_prev += d_a_r @ self.W_hr.T

            d_h_next = d_h_prev

        if weight_decay:
            for name in ("W_xz", "W_hz", "W_xr", "W_hr", "W_xn", "W_hn", "W_hy"):
                loss += 0.5 * weight_decay * float(np.sum(getattr(self, name) ** 2))
                gradients[name] += weight_decay * getattr(self, name)

        return loss, gradients, hidden[:, -1].copy()

    def save(self, path: Path, metadata: Optional[Mapping[str, object]] = None) -> None:
        payload: Dict[str, np.ndarray] = {
            name: value for name, value in self.parameters().items()
        }
        payload["input_size"] = np.asarray(self.input_size)
        payload["hidden_size"] = np.asarray(self.hidden_size)
        payload["metadata_json"] = np.asarray(json.dumps(metadata or {}))
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "GRU":
        data = np.load(path, allow_pickle=False)
        model = cls.__new__(cls)
        model.input_size = int(data["input_size"])
        model.hidden_size = int(data["hidden_size"])
        for name in cls.parameter_names:
            setattr(model, name, data[name].copy())
        return model


class Adam:
    """Small Adam optimizer for the NumPy model."""

    def __init__(
        self,
        parameters: Mapping[str, np.ndarray],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second_moment = {
            name: np.zeros_like(value) for name, value in parameters.items()
        }
        self.iteration = 0

    def update(self, gradients: Mapping[str, np.ndarray], clip_norm: float) -> float:
        squared_norm = sum(float(np.sum(gradient**2)) for gradient in gradients.values())
        global_norm = math.sqrt(squared_norm)
        scale = min(1.0, clip_norm / (global_norm + 1e-12))
        self.iteration += 1

        for name, parameter in self.parameters.items():
            gradient = gradients[name] * scale
            self.first_moment[name] = (
                self.beta1 * self.first_moment[name] + (1.0 - self.beta1) * gradient
            )
            self.second_moment[name] = (
                self.beta2 * self.second_moment[name]
                + (1.0 - self.beta2) * gradient**2
            )
            corrected_first = self.first_moment[name] / (1.0 - self.beta1**self.iteration)
            corrected_second = self.second_moment[name] / (1.0 - self.beta2**self.iteration)
            parameter -= self.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + self.epsilon
            )
        return global_norm


def train_model(
    task: TaskConfig,
    train: TrainConfig,
    verbose: bool = True,
) -> Tuple[GRU, List[float]]:
    """Train on newly generated sessions each epoch."""

    rng = np.random.default_rng(train.seed)
    model = GRU(N_INPUTS, train.hidden_size, rng)
    optimizer = Adam(model.parameters(), learning_rate=train.learning_rate)
    chunk_steps = train.bptt_trials * N_STEPS
    losses: List[float] = []

    for epoch in range(1, train.epochs + 1):
        trials = generate_trials(
            train.sessions_per_epoch,
            train.trials_per_session,
            task,
            rng,
        )
        x, targets = build_training_sequence(trials, task, rng)
        hidden = model.zero_state(train.sessions_per_epoch)
        chunk_losses: List[float] = []

        for start in range(0, x.shape[1], chunk_steps):
            stop = min(start + chunk_steps, x.shape[1])
            loss, gradients, hidden = model.loss_and_gradients(
                x[:, start:stop],
                targets[:, start:stop],
                hidden,
                weight_decay=train.weight_decay,
            )
            optimizer.update(gradients, train.gradient_clip_norm)
            chunk_losses.append(loss)

        epoch_loss = float(np.mean(chunk_losses))
        losses.append(epoch_loss)
        if verbose and (
            epoch == 1 or epoch == train.epochs or epoch % max(1, train.epochs // 10) == 0
        ):
            print(f"epoch {epoch:4d}/{train.epochs}: response cross-entropy={epoch_loss:.4f}")

    return model, losses


def rollout(
    model: GRU,
    trials: TrialBatch,
    rng: np.random.Generator,
    reset_each_trial: bool = False,
    sample_choices: bool = False,
) -> Dict[str, np.ndarray]:
    """Run closed-loop sessions and feed the model its own action/outcome.

    `zero_evidence_p_right` is a counterfactual readout: from the pre-stimulus
    state on every trial, the model is shown an exactly zero-contrast stimulus.
    It therefore measures block belief without contamination by current sensory
    evidence.
    """

    n_sessions, n_trials = trials.shape
    hidden = model.zero_state(n_sessions)
    p_choice_right = np.empty((n_sessions, n_trials), dtype=np.float64)
    zero_evidence_p_right = np.empty_like(p_choice_right)
    choice = np.empty((n_sessions, n_trials), dtype=np.int64)
    correct = np.empty((n_sessions, n_trials), dtype=bool)
    pre_stimulus_hidden = np.empty(
        (n_sessions, n_trials, model.hidden_size), dtype=np.float64
    )

    zeros = np.zeros((n_sessions, N_INPUTS), dtype=np.float64)
    go_input = zeros.copy()
    go_input[:, GO_CUE] = 1.0

    for trial_index in range(n_trials):
        if reset_each_trial:
            hidden.fill(0.0)

        # Steps 1-2: neutral external input; recurrent memory is not reset.
        hidden = model.step(zeros, hidden)
        hidden = model.step(zeros, hidden)
        pre_stimulus_hidden[:, trial_index] = hidden

        # Counterfactual no-evidence path through steps 3-6.
        counterfactual_hidden = hidden.copy()
        counterfactual_hidden = model.step(zeros, counterfactual_hidden)  # step 3
        counterfactual_hidden = model.step(zeros, counterfactual_hidden)  # step 4
        counterfactual_hidden = model.step(go_input, counterfactual_hidden)  # step 5
        counterfactual_hidden = model.step(zeros, counterfactual_hidden)  # step 6
        zero_evidence_p_right[:, trial_index] = model.response_probabilities(
            counterfactual_hidden
        )[:, RIGHT]

        # Actual stimulus at step 3.
        stimulus_input = zeros.copy()
        stimulus_input[:, :2] = trials.visual_observation[:, trial_index]
        hidden = model.step(stimulus_input, hidden)
        hidden = model.step(zeros, hidden)  # step 4: blank
        hidden = model.step(go_input, hidden)  # step 5: go cue
        hidden = model.step(zeros, hidden)  # step 6: response/readout

        probabilities = model.response_probabilities(hidden)
        p_choice_right[:, trial_index] = probabilities[:, RIGHT]
        if sample_choices:
            this_choice = (rng.random(n_sessions) < probabilities[:, RIGHT]).astype(
                np.int64
            )
        else:
            this_choice = np.argmax(probabilities, axis=1)
        choice[:, trial_index] = this_choice
        this_correct = this_choice == trials.side[:, trial_index]
        correct[:, trial_index] = this_correct

        # Steps 7-8: feed back the model's own response and its outcome.
        feedback = zeros.copy()
        feedback[:, ACTION_LEFT] = this_choice == LEFT
        feedback[:, ACTION_RIGHT] = this_choice == RIGHT
        feedback[:, REWARDED] = this_correct
        feedback[:, NOT_REWARDED] = ~this_correct
        hidden = model.step(feedback, hidden)
        hidden = model.step(feedback, hidden)

    return {
        "p_right": trials.p_right,
        "block_id": trials.block_id,
        "side": trials.side,
        "contrast": trials.contrast,
        "p_choice_right": p_choice_right,
        "zero_evidence_p_right": zero_evidence_p_right,
        "choice": choice,
        "correct": correct,
        "pre_stimulus_hidden": pre_stimulus_hidden,
    }


def summarize_rollout(records: Mapping[str, np.ndarray], task: TaskConfig) -> Dict[str, object]:
    p_right = records["p_right"]
    contrast = records["contrast"]
    target = records["side"]
    predicted = records["p_choice_right"]
    correct = records["correct"]
    zero_pref = records["zero_evidence_p_right"]

    # Exclude the transient immediately after a switch when asking how well the
    # output is calibrated to the settled 0.2/0.8 prior.
    block_age = np.zeros_like(records["block_id"], dtype=np.int64)
    for trial_index in range(1, block_age.shape[1]):
        same_block = (
            records["block_id"][:, trial_index]
            == records["block_id"][:, trial_index - 1]
        )
        block_age[:, trial_index] = np.where(
            same_block, block_age[:, trial_index - 1] + 1, 0
        )
    settled = block_age >= 15

    target_probability = np.where(target == RIGHT, predicted, 1.0 - predicted)
    result: Dict[str, object] = {
        "accuracy": float(correct.mean()),
        "response_cross_entropy": float(-np.log(target_probability + 1e-12).mean()),
        "accuracy_by_contrast": {},
        "mean_choice_probability_right_by_contrast": {},
    }
    for value in task.contrasts:
        mask = np.isclose(contrast, value)
        result["accuracy_by_contrast"][str(value)] = float(correct[mask].mean())
        result["mean_choice_probability_right_by_contrast"][str(value)] = float(
            predicted[mask].mean()
        )

    low_mask = np.isclose(p_right, task.low_p_right)
    high_mask = np.isclose(p_right, task.high_p_right)
    zero_contrast = np.isclose(contrast, 0.0)
    result["zero_contrast_observed_choice_probability"] = {
        "low_p_right_block": float(predicted[low_mask & zero_contrast].mean()),
        "high_p_right_block": float(predicted[high_mask & zero_contrast].mean()),
    }
    result["counterfactual_zero_evidence_choice_probability"] = {
        "low_p_right_block": float(zero_pref[low_mask].mean()),
        "high_p_right_block": float(zero_pref[high_mask].mean()),
        "history_gap": float(zero_pref[high_mask].mean() - zero_pref[low_mask].mean()),
    }
    result["settled_block_zero_evidence_calibration"] = {
        "definition": "trials at least 15 trials after the most recent switch",
        "low_p_right_block": float(zero_pref[low_mask & settled].mean()),
        "high_p_right_block": float(zero_pref[high_mask & settled].mean()),
        "mean_absolute_error_to_true_prior": float(
            np.abs(zero_pref[settled] - p_right[settled]).mean()
        ),
    }
    return result


def decode_block_from_hidden(
    records: Mapping[str, np.ndarray],
    ridge: float = 1e-2,
) -> Dict[str, float]:
    """Fit a linear ridge probe on sessions, then test on held-out sessions."""

    hidden = records["pre_stimulus_hidden"]
    labels = (records["p_right"] > 0.5).astype(np.float64)
    n_sessions = hidden.shape[0]
    split = max(1, int(round(0.7 * n_sessions)))
    if split >= n_sessions:
        split = n_sessions - 1
    if split < 1:
        raise ValueError("At least two evaluation sessions are needed for the probe")

    x_train = hidden[:split].reshape(-1, hidden.shape[-1])
    y_train = labels[:split].reshape(-1)
    x_test = hidden[split:].reshape(-1, hidden.shape[-1])
    y_test = labels[split:].reshape(-1)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_train = np.column_stack((x_train, np.ones(len(x_train))))
    x_test = np.column_stack((x_test, np.ones(len(x_test))))

    penalty = ridge * np.eye(x_train.shape[1])
    penalty[-1, -1] = 0.0  # do not penalize the intercept
    weights = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    predictions = x_test @ weights
    accuracy = np.mean((predictions >= 0.5) == y_test)
    correlation = np.corrcoef(predictions, y_test)[0, 1]
    return {
        "held_out_session_accuracy": float(accuracy),
        "held_out_session_correlation": float(correlation),
        "n_train_sessions": int(split),
        "n_test_sessions": int(n_sessions - split),
    }


def switch_centered_curve(
    records: Mapping[str, np.ndarray],
    before: int = 20,
    after: int = 30,
) -> Dict[str, np.ndarray]:
    """Average the counterfactual zero-evidence preference around switches."""

    p_right = records["p_right"]
    preference = records["zero_evidence_p_right"]
    offsets = np.arange(-before, after + 1)
    groups: Dict[str, List[np.ndarray]] = {"low_to_high": [], "high_to_low": []}

    for session in range(p_right.shape[0]):
        switch_indices = np.flatnonzero(np.diff(p_right[session]) != 0) + 1
        for switch in switch_indices:
            if switch - before < 0 or switch + after >= p_right.shape[1]:
                continue
            direction = (
                "low_to_high"
                if p_right[session, switch] > p_right[session, switch - 1]
                else "high_to_low"
            )
            groups[direction].append(
                preference[session, switch - before : switch + after + 1]
            )

    result: Dict[str, np.ndarray] = {"offsets": offsets}
    for direction, curves in groups.items():
        if curves:
            result[direction] = np.mean(np.stack(curves), axis=0)
        else:
            result[direction] = np.full_like(offsets, np.nan, dtype=np.float64)
    return result


def save_figure(
    records: Mapping[str, np.ndarray],
    losses: Iterable[float],
    task: TaskConfig,
    output_path: Path,
) -> None:
    """Create diagnostic plots for psychometrics and block adaptation."""

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ibl-gru-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    losses = list(losses)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(np.arange(1, len(losses) + 1), losses, color="#3b6ea8")
    axes[0, 0].set(title="GRU training", xlabel="Epoch", ylabel="Response cross-entropy")

    signed_contrast = np.where(
        records["side"] == RIGHT, records["contrast"], -records["contrast"]
    )
    unique_signed = np.unique(signed_contrast)
    for prior, color in ((task.low_p_right, "#d55e00"), (task.high_p_right, "#0072b2")):
        means = []
        for value in unique_signed:
            mask = np.isclose(records["p_right"], prior) & np.isclose(
                signed_contrast, value
            )
            means.append(float(records["p_choice_right"][mask].mean()))
        axes[0, 1].plot(
            unique_signed,
            means,
            marker="o",
            label=f"block P(right)={prior:.1f}",
            color=color,
        )
    axes[0, 1].axhline(0.5, color="0.7", linewidth=1)
    axes[0, 1].axvline(0.0, color="0.7", linewidth=1)
    axes[0, 1].set(
        title="Psychometric curve",
        xlabel="Signed contrast (left negative, right positive)",
        ylabel="Network P(choice right)",
        ylim=(-0.03, 1.03),
    )
    axes[0, 1].legend(frameon=False)

    curve = switch_centered_curve(records)
    axes[1, 0].plot(
        curve["offsets"], curve["low_to_high"], color="#0072b2", label="0.2 -> 0.8"
    )
    axes[1, 0].plot(
        curve["offsets"], curve["high_to_low"], color="#d55e00", label="0.8 -> 0.2"
    )
    axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(
        title="Belief adaptation around block switches",
        xlabel="Trials relative to switch",
        ylabel="P(right) with zero sensory evidence",
        ylim=(-0.03, 1.03),
    )
    axes[1, 0].legend(frameon=False)

    trials = np.arange(records["p_right"].shape[1])
    axes[1, 1].step(
        trials,
        records["p_right"][0],
        where="post",
        color="black",
        linewidth=1.5,
        label="true block P(right)",
    )
    axes[1, 1].plot(
        trials,
        records["zero_evidence_p_right"][0],
        color="#8e44ad",
        alpha=0.9,
        label="GRU zero-evidence preference",
    )
    axes[1, 1].set(
        title="Example held-out session",
        xlabel="Trial",
        ylabel="Probability right",
        ylim=(-0.03, 1.03),
    )
    axes[1, 1].legend(frameon=False)

    figure.suptitle("Hidden-prior GRU diagnostics", fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def print_timing_table() -> None:
    """Print a compact example showing every channel at all eight steps."""

    example, targets = make_example_trial(side=LEFT, contrast=0.25, action=LEFT)
    print("\nEight-step example: left stimulus, contrast=0.25, correct left action")
    header = "step  event       " + "  ".join(f"{name:>14s}" for name in INPUT_NAMES)
    print(header)
    events = ("baseline", "baseline", "stimulus", "blank", "go cue", "response", "reward", "reward")
    for index, event in enumerate(events):
        values = "  ".join(f"{value:14.3f}" for value in example[index])
        target = " <- output target: LEFT" if targets[index] == LEFT else ""
        print(f"{index + 1:>4d}  {event:<10s} {values}{target}")


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a NumPy GRU on an 8-step hidden block-prior task."
    )
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--train-sessions", type=int, default=TrainConfig.sessions_per_epoch)
    parser.add_argument("--train-trials", type=int, default=TrainConfig.trials_per_session)
    parser.add_argument("--test-sessions", type=int, default=48)
    parser.add_argument("--test-trials", type=int, default=320)
    parser.add_argument("--low-p-right", type=float, default=TaskConfig.low_p_right)
    parser.add_argument("--high-p-right", type=float, default=TaskConfig.high_p_right)
    parser.add_argument(
        "--min-block-trials", type=int, default=TaskConfig.min_block_trials
    )
    parser.add_argument(
        "--max-block-trials", type=int, default=TaskConfig.max_block_trials
    )
    parser.add_argument(
        "--sensory-noise", type=float, default=TaskConfig.sensory_noise_std
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to save outputs. Defaults to gru_outputs next to this script.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke-test run (use the defaults for the real experiment).",
    )
    parser.add_argument(
        "--inspect-timing",
        action="store_true",
        help="Print the exact 8-step input schedule before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.inspect_timing:
        print_timing_table()

    task = TaskConfig(
        low_p_right=args.low_p_right,
        high_p_right=args.high_p_right,
        min_block_trials=args.min_block_trials,
        max_block_trials=args.max_block_trials,
        sensory_noise_std=args.sensory_noise,
    )
    train = TrainConfig(
        hidden_size=args.hidden_size,
        epochs=3 if args.quick else args.epochs,
        sessions_per_epoch=6 if args.quick else args.train_sessions,
        trials_per_session=96 if args.quick else args.train_trials,
        bptt_trials=16 if args.quick else TrainConfig.bptt_trials,
        seed=args.seed,
    )
    test_sessions = 8 if args.quick else args.test_sessions
    test_trials = 128 if args.quick else args.test_trials
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "gru_outputs"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Training the history-dependent GRU...")
    model, losses = train_model(task, train)

    test_rng = np.random.default_rng(args.seed + 10_000)
    test_trials_batch = generate_trials(test_sessions, test_trials, task, test_rng)
    print("Evaluating closed-loop sessions...")
    history_records = rollout(
        model,
        test_trials_batch,
        np.random.default_rng(args.seed + 20_000),
        reset_each_trial=False,
    )
    reset_records = rollout(
        model,
        test_trials_batch,
        np.random.default_rng(args.seed + 20_000),
        reset_each_trial=True,
    )

    history_summary = summarize_rollout(history_records, task)
    reset_summary = summarize_rollout(reset_records, task)
    probe_summary = decode_block_from_hidden(history_records)
    results = {
        "model": "GRU",
        "task_config": asdict(task),
        "train_config": asdict(train),
        "history_dependent_evaluation": history_summary,
        "reset_hidden_state_each_trial_control": reset_summary,
        "linear_block_probe": probe_summary,
        "interpretation": {
            "primary_behavioral_test": (
                "The history_gap should be positive: with zero sensory evidence, "
                "P(right) should be higher in 0.8 than in 0.2 blocks."
            ),
            "causal_memory_control": (
                "Resetting hidden state each trial should substantially reduce that gap."
            ),
            "representation_test": (
                "Held-out-session probe accuracy above 0.5 shows that block identity "
                "is linearly decodable from the pre-stimulus recurrent state."
            ),
        },
    }

    model_path = args.output_dir / "hidden_prior_gru.npz"
    results_path = args.output_dir / "metrics.json"
    figure_path = args.output_dir / "diagnostics.png"
    model.save(
        model_path,
        metadata={"task_config": asdict(task), "train_config": asdict(train)},
    )
    results_path.write_text(json.dumps(_json_ready(results), indent=2) + "\n")
    save_figure(history_records, losses, task, figure_path)

    normal_gap = history_summary["counterfactual_zero_evidence_choice_probability"][
        "history_gap"
    ]
    reset_gap = reset_summary["counterfactual_zero_evidence_choice_probability"][
        "history_gap"
    ]
    print("\nKey held-out results")
    print(f"  overall accuracy:                 {history_summary['accuracy']:.3f}")
    print(f"  zero-evidence block history gap:  {normal_gap:.3f}")
    print(f"  gap after hidden-state reset:     {reset_gap:.3f}")
    print(
        "  hidden-state block decoder:      "
        f"{probe_summary['held_out_session_accuracy']:.3f} accuracy"
    )
    print(f"\nSaved model:   {model_path}")
    print(f"Saved metrics: {results_path}")
    print(f"Saved figure:  {figure_path}")


if __name__ == "__main__":
    main()
