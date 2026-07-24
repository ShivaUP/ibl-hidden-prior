#!/usr/bin/env python3
"""
PC_V2_CORRECTED_LOCAL_ONLY_20260723.py
=======================================

A uniquely named, completely standalone implementation of the corrected V2
predictive-coding recurrent network.

This file contains:
  * the nine-tick V2 task;
  * the single-layer, 48-unit tanh recurrent network;
  * iterative predictive-coding inference;
  * local synaptic updates;
  * synthetic training and held-out evaluation;
  * uniquely named model, JSON, and figure outputs.

It deliberately contains NO BPTT model, NO PyTorch/autograd, and NO sklearn.
Only NumPy is required for training. Matplotlib is required for the figure.

Run from the VS Code terminal:

    python PC_V2_CORRECTED_LOCAL_ONLY_20260723.py

For a short wiring check:

    python PC_V2_CORRECTED_LOCAL_ONLY_20260723.py --quick

Corrected PC defaults
---------------------
The response nudge is eight recurrent edges from the earliest previous-trial
feedback state. Synchronous PC therefore needs at least nine inference rounds
to reach the complete previous feedback window. This file uses 32 rounds.

The weak output nudge (precision 0.025) is used only during training inference.
Local updates are divided by that known nudge magnitude so that changing the
nudge does not silently rescale the synaptic learning rate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Unique names: these cannot be confused with earlier model.py/result files.
# ---------------------------------------------------------------------------

UNIQUE_TAG = "PC_V2_CORRECTED_LOCAL_ONLY_20260723"
DEFAULT_OUTPUT_DIRECTORY = UNIQUE_TAG + "_OUTPUT"


# ---------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------

LEFT = 0
RIGHT = 1

CHANNEL_NAMES = (
    "visual_right",
    "visual_left",
    "go_cue",
    "action_left",
    "action_right",
    "rewarded",
    "not_rewarded",
)

VISUAL_RIGHT = 0
VISUAL_LEFT = 1
GO_CUE = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
REWARDED = 5
NOT_REWARDED = 6
N_INPUTS = len(CHANNEL_NAMES)


BLOCK_LENGTH_VALUES = (
    10, 12, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45,
    46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 67, 68, 69, 71, 72, 74, 75, 76, 77, 79,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 93, 94, 97, 98,
)

# Integer counts reproduce the empirical V2 block-length probabilities exactly.
BLOCK_LENGTH_WEIGHTS = (
    5, 1, 1, 5, 4, 3, 4, 5, 2, 3, 6, 7, 4, 3, 3, 4, 2, 2,
    6, 5, 3, 4, 4, 3, 2, 5, 3, 3, 8, 2, 2, 2, 2, 5, 2, 3,
    1, 3, 2, 2, 4, 1, 2, 4, 2, 1, 1, 4, 6, 2, 1, 1, 1, 1,
    4, 1, 1, 2, 1, 3, 2, 4, 1, 2, 5, 2, 1, 1, 1, 1,
)


@dataclass(frozen=True)
class PhaseTicks:
    """The empirical V2 schedule, using zero-based tick numbers."""

    baseline_ticks: int = 2
    go_offset_from_stim_ticks: int = 0
    response_offset_from_go_ticks: int = 4
    feedback_ticks: int = 2
    stim_duration_ticks: int = 15

    @property
    def stim_start(self) -> int:
        return self.baseline_ticks

    @property
    def go_tick(self) -> int:
        return self.stim_start + self.go_offset_from_stim_ticks

    @property
    def response_tick(self) -> int:
        return self.go_tick + self.response_offset_from_go_ticks

    @property
    def feedback_start(self) -> int:
        return self.response_tick + 1

    @property
    def n_steps(self) -> int:
        return self.feedback_start + self.feedback_ticks

    @property
    def stim_end_exclusive(self) -> int:
        return min(
            self.stim_start + self.stim_duration_ticks,
            self.n_steps,
        )


PHASE = PhaseTicks()


@dataclass(frozen=True)
class Configuration:
    """All scientifically important settings in one visible place."""

    hidden_size: int = 48

    # Corrected PC training schedule.
    epochs: int = 8
    sessions_per_epoch: int = 24
    trials_per_session: int = 929
    chunk_trials: int = 32
    inference_steps: int = 32
    inference_learning_rate: float = 0.15
    inference_momentum: float = 0.0
    output_precision: float = 0.025
    value_clip: float = 2.0
    synaptic_learning_rate: float = 0.0004
    normalize_updates_by_nudge: bool = True
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 1.0

    # Synthetic task.
    sensory_noise_std: float = 0.15
    training_feedback_error_rate: float = 0.20
    contrast_levels: Tuple[float, ...] = (
        0.0,
        0.0625,
        0.125,
        0.25,
        1.0,
    )
    contrast_probabilities: Tuple[float, ...] = (
        0.19719042663891778,
        0.2016649323621228,
        0.2125910509885536,
        0.19500520291363163,
        0.1935483870967742,
    )

    # Reproducibility and evaluation.
    seed: int = 7
    evaluation_seed: int = 10007
    evaluation_sessions: int = 48
    evaluation_trials: int = 929


@dataclass
class SyntheticBatch:
    probability_left: np.ndarray
    p_right: np.ndarray
    block_id: np.ndarray
    side: np.ndarray
    contrast: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.side.shape


# ---------------------------------------------------------------------------
# Synthetic V2 task
# ---------------------------------------------------------------------------


def _block_length_probabilities() -> np.ndarray:
    weights = np.asarray(BLOCK_LENGTH_WEIGHTS, dtype=np.float64)
    return weights / weights.sum()


def _sample_next_probability_left(
    current_probability_left: float,
    rng: np.random.Generator,
) -> float:
    # Use rng.choice even for deterministic transitions. This intentionally
    # reproduces the random-number consumption of the modular corrected V2 code,
    # so a given seed produces exactly the same sessions and trained weights.
    candidates = np.asarray([0.2, 0.5, 0.8], dtype=np.float64)
    if np.isclose(current_probability_left, 0.5):
        probabilities = np.asarray([0.4, 0.0, 0.6])
    elif np.isclose(current_probability_left, 0.2):
        probabilities = np.asarray([0.0, 0.0, 1.0])
    else:
        probabilities = np.asarray([1.0, 0.0, 0.0])
    return float(rng.choice(candidates, p=probabilities))


def generate_sessions(
    n_sessions: int,
    n_trials: int,
    cfg: Configuration,
    rng: np.random.Generator,
) -> SyntheticBatch:
    """Generate block-switching sessions from the frozen empirical statistics."""

    probability_left = np.empty((n_sessions, n_trials), dtype=np.float64)
    block_id = np.empty((n_sessions, n_trials), dtype=np.int64)
    lengths = np.asarray(BLOCK_LENGTH_VALUES, dtype=np.int64)
    length_probabilities = _block_length_probabilities()

    for session in range(n_sessions):
        cursor = 0
        current_block = 0
        # The original config represents this as a one-entry categorical
        # distribution. Retaining the rng.choice call makes runs bitwise
        # reproducible with that implementation.
        current_probability_left = float(
            rng.choice(
                np.asarray([0.5], dtype=np.float64),
                p=np.asarray([1.0], dtype=np.float64),
            )
        )
        while cursor < n_trials:
            length = int(rng.choice(lengths, p=length_probabilities))
            stop = min(cursor + length, n_trials)
            probability_left[session, cursor:stop] = current_probability_left
            block_id[session, cursor:stop] = current_block
            cursor = stop
            current_block += 1
            if cursor < n_trials:
                current_probability_left = _sample_next_probability_left(
                    current_probability_left,
                    rng,
                )

    p_right = 1.0 - probability_left
    side = (
        rng.random((n_sessions, n_trials)) < p_right
    ).astype(np.int64)

    contrast_probabilities = np.asarray(
        cfg.contrast_probabilities,
        dtype=np.float64,
    )
    contrast_probabilities /= contrast_probabilities.sum()
    contrast = rng.choice(
        np.asarray(cfg.contrast_levels, dtype=np.float64),
        size=(n_sessions, n_trials),
        p=contrast_probabilities,
    )

    return SyntheticBatch(
        probability_left=probability_left,
        p_right=p_right,
        block_id=block_id,
        side=side,
        contrast=contrast,
    )


def paint_trial(
    side: int,
    contrast: float,
    action: int,
    rewarded: bool,
    visual_noise: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the complete nine-tick input and response target."""

    x = np.zeros((PHASE.n_steps, N_INPUTS), dtype=np.float64)
    targets = np.full(PHASE.n_steps, -1, dtype=np.int64)

    visual_right = float(contrast) if side == RIGHT else 0.0
    visual_left = float(contrast) if side == LEFT else 0.0
    if visual_noise is not None:
        visual_right += float(visual_noise[0])
        visual_left += float(visual_noise[1])

    for tick in range(PHASE.stim_start, PHASE.stim_end_exclusive):
        if tick == PHASE.response_tick:
            continue
        x[tick, VISUAL_RIGHT] = visual_right
        x[tick, VISUAL_LEFT] = visual_left

    x[PHASE.go_tick, GO_CUE] = 1.0
    targets[PHASE.response_tick] = int(side)

    for tick in range(PHASE.feedback_start, PHASE.n_steps):
        x[tick, ACTION_LEFT] = float(action == LEFT)
        x[tick, ACTION_RIGHT] = float(action == RIGHT)
        x[tick, REWARDED] = float(rewarded)
        x[tick, NOT_REWARDED] = float(not rewarded)

    return x, targets


def build_teacher_forced_inputs(
    batch: SyntheticBatch,
    cfg: Configuration,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct training inputs.

    The feedback period supplies chosen action plus rewarded/not-rewarded.
    Therefore the correct side is not directly supplied as a separate label
    input; it can be inferred from action and outcome.
    """

    n_sessions, n_trials = batch.shape
    x = np.zeros(
        (n_sessions, n_trials * PHASE.n_steps, N_INPUTS),
        dtype=np.float64,
    )
    targets = np.full(
        (n_sessions, n_trials * PHASE.n_steps),
        -1,
        dtype=np.int64,
    )

    for session in range(n_sessions):
        for trial in range(n_trials):
            side = int(batch.side[session, trial])
            make_error = rng.random() < cfg.training_feedback_error_rate
            action = 1 - side if make_error else side
            rewarded = action == side
            noise = rng.normal(
                0.0,
                cfg.sensory_noise_std,
                size=2,
            )
            trial_x, trial_targets = paint_trial(
                side=side,
                contrast=float(batch.contrast[session, trial]),
                action=action,
                rewarded=rewarded,
                visual_noise=noise,
            )
            start = trial * PHASE.n_steps
            stop = start + PHASE.n_steps
            x[session, start:stop] = trial_x
            targets[session, start:stop] = trial_targets

    return x, targets


# ---------------------------------------------------------------------------
# The one-layer recurrent network
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


class TanhRecurrentNetwork:
    """One recurrent hidden layer; PC changes training, not this architecture."""

    parameter_names = ("W_xh", "W_hh", "b_h", "W_hy", "b_y")

    def __init__(
        self,
        hidden_size: int,
        rng: np.random.Generator,
        input_size: int = N_INPUTS,
    ):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        input_scale = math.sqrt(2.0 / (input_size + hidden_size))
        output_scale = math.sqrt(2.0 / (hidden_size + 2))
        self.W_xh = rng.normal(
            0.0,
            input_scale,
            (input_size, hidden_size),
        )
        self.W_hh = 0.90 * np.eye(hidden_size) + rng.normal(
            0.0,
            0.01 / math.sqrt(hidden_size),
            (hidden_size, hidden_size),
        )
        self.b_h = np.zeros(hidden_size, dtype=np.float64)
        self.W_hy = rng.normal(
            0.0,
            output_scale,
            (hidden_size, 2),
        )
        self.b_y = np.zeros(2, dtype=np.float64)

    def parameters(self) -> Dict[str, np.ndarray]:
        return {
            name: getattr(self, name)
            for name in self.parameter_names
        }

    def zero_state(self, batch_size: int) -> np.ndarray:
        return np.zeros(
            (batch_size, self.hidden_size),
            dtype=np.float64,
        )

    def step(self, x_t: np.ndarray, previous_hidden: np.ndarray) -> np.ndarray:
        return np.tanh(
            x_t @ self.W_xh
            + previous_hidden @ self.W_hh
            + self.b_h
        )

    def probabilities(self, hidden: np.ndarray) -> np.ndarray:
        return softmax(hidden @ self.W_hy + self.b_y)

    def save(self, path: Path, metadata: Dict[str, object]) -> None:
        payload = {
            name: value
            for name, value in self.parameters().items()
        }
        payload["input_size"] = np.asarray(self.input_size)
        payload["hidden_size"] = np.asarray(self.hidden_size)
        payload["metadata_json"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "TanhRecurrentNetwork":
        data = np.load(path, allow_pickle=False)
        model = cls.__new__(cls)
        model.input_size = int(data["input_size"])
        model.hidden_size = int(data["hidden_size"])
        for name in cls.parameter_names:
            setattr(model, name, data[name].copy())
        return model


class Adam:
    """Optimizer applied to gradients constructed by the local PC rule."""

    def __init__(
        self,
        parameters: Mapping[str, np.ndarray],
        learning_rate: float,
    ):
        self.parameters = parameters
        self.learning_rate = float(learning_rate)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1.0e-8
        self.first = {
            name: np.zeros_like(value)
            for name, value in parameters.items()
        }
        self.second = {
            name: np.zeros_like(value)
            for name, value in parameters.items()
        }
        self.iteration = 0

    def update(
        self,
        gradients: Mapping[str, np.ndarray],
        clip_norm: float,
    ) -> float:
        squared_norm = sum(
            float(np.sum(gradient ** 2))
            for gradient in gradients.values()
        )
        gradient_norm = math.sqrt(squared_norm)
        scale = min(
            1.0,
            float(clip_norm) / (gradient_norm + 1.0e-12),
        )
        self.iteration += 1

        for name, parameter in self.parameters.items():
            gradient = gradients[name] * scale
            self.first[name] = (
                self.beta1 * self.first[name]
                + (1.0 - self.beta1) * gradient
            )
            self.second[name] = (
                self.beta2 * self.second[name]
                + (1.0 - self.beta2) * gradient ** 2
            )
            first_hat = self.first[name] / (
                1.0 - self.beta1 ** self.iteration
            )
            second_hat = self.second[name] / (
                1.0 - self.beta2 ** self.iteration
            )
            parameter -= (
                self.learning_rate
                * first_hat
                / (np.sqrt(second_hat) + self.epsilon)
            )

        return gradient_norm


# ---------------------------------------------------------------------------
# Corrected predictive-coding inference and local learning
# ---------------------------------------------------------------------------


class PredictiveCodingTrainer:
    """
    Iteratively infer hidden values, then update each synapse locally.

    No call in this class performs reverse-mode automatic differentiation or
    backpropagation through time.
    """

    def __init__(self, model: TanhRecurrentNetwork):
        self.model = model

    def _validate(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        initial_hidden: np.ndarray,
    ) -> None:
        if x.ndim != 3 or x.shape[2] != self.model.input_size:
            raise ValueError(
                "x must have shape [batch, time, input_size]"
            )
        if targets.shape != x.shape[:2]:
            raise ValueError("targets must match x batch and time dimensions")
        if initial_hidden.shape != (
            x.shape[0],
            self.model.hidden_size,
        ):
            raise ValueError("initial hidden state has the wrong shape")
        labelled = targets[targets >= 0]
        if labelled.size == 0:
            raise ValueError("each PC chunk needs a response target")
        if np.any((labelled != LEFT) & (labelled != RIGHT)):
            raise ValueError("targets must be LEFT=0 or RIGHT=1")

    def forward_values(
        self,
        x: np.ndarray,
        initial_hidden: np.ndarray,
    ) -> np.ndarray:
        batch_size, n_time, _ = x.shape
        values = np.empty(
            (batch_size, n_time, self.model.hidden_size),
            dtype=np.float64,
        )
        hidden = initial_hidden
        for tick in range(n_time):
            hidden = self.model.step(x[:, tick], hidden)
            values[:, tick] = hidden
        return values

    def _error_terms(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        initial_hidden: np.ndarray,
        values: np.ndarray,
        output_precision: float,
    ) -> Dict[str, object]:
        self._validate(x, targets, initial_hidden)
        if output_precision <= 0.0:
            raise ValueError("output precision must be positive")

        previous_values = np.concatenate(
            (
                initial_hidden[:, None, :],
                values[:, :-1, :],
            ),
            axis=1,
        )
        hidden_prediction = np.tanh(
            x @ self.model.W_xh
            + previous_values @ self.model.W_hh
            + self.model.b_h
        )
        hidden_error = values - hidden_prediction

        response_rows, response_times = np.nonzero(targets >= 0)
        response_targets = targets[response_rows, response_times]
        response_values = values[response_rows, response_times]
        response_probabilities = softmax(
            response_values @ self.model.W_hy + self.model.b_y
        )
        one_hot_targets = np.eye(2, dtype=np.float64)[response_targets]
        output_delta = output_precision * (
            one_hot_targets - response_probabilities
        )
        chosen_probability = response_probabilities[
            np.arange(len(response_targets)),
            response_targets,
        ]
        energy = (
            0.5 * np.sum(hidden_error ** 2)
            - output_precision
            * np.sum(np.log(chosen_probability + 1.0e-12))
        ) / x.shape[0]

        return {
            "previous_values": previous_values,
            "hidden_prediction": hidden_prediction,
            "hidden_error": hidden_error,
            "response_rows": response_rows,
            "response_times": response_times,
            "output_delta": output_delta,
            "energy": float(energy),
        }

    def forward_response_cross_entropy(
        self,
        values: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        rows, times = np.nonzero(targets >= 0)
        response_targets = targets[rows, times]
        probabilities = softmax(
            values[rows, times] @ self.model.W_hy + self.model.b_y
        )
        chosen = probabilities[
            np.arange(len(response_targets)),
            response_targets,
        ]
        return float(-np.mean(np.log(chosen + 1.0e-12)))

    def value_gradients(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        initial_hidden: np.ndarray,
        values: np.ndarray,
        output_precision: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute dE/d(value) for one synchronous PC inference iteration.

        The next-tick prediction error is communicated through the fixed
        generative recurrent weights during inference. This is iterative state
        inference, not a BPTT parameter-gradient calculation.
        """

        terms = self._error_terms(
            x,
            targets,
            initial_hidden,
            values,
            output_precision,
        )
        hidden_error = terms["hidden_error"]
        hidden_prediction = terms["hidden_prediction"]
        value_gradient = hidden_error.copy()

        next_drive = (
            hidden_error[:, 1:]
            * (1.0 - hidden_prediction[:, 1:] ** 2)
        )
        value_gradient[:, :-1] -= next_drive @ self.model.W_hh.T

        rows = terms["response_rows"]
        times = terms["response_times"]
        value_gradient[rows, times] -= (
            terms["output_delta"] @ self.model.W_hy.T
        )
        return value_gradient, float(terms["energy"])

    def infer_values(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        initial_hidden: np.ndarray,
        cfg: Configuration,
        initial_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        values = (
            self.forward_values(x, initial_hidden)
            if initial_values is None
            else np.asarray(initial_values, dtype=np.float64).copy()
        )
        velocity = np.zeros_like(values)
        energy_trace: List[float] = []

        for _ in range(cfg.inference_steps):
            gradient, energy = self.value_gradients(
                x,
                targets,
                initial_hidden,
                values,
                cfg.output_precision,
            )
            energy_trace.append(energy)
            velocity *= cfg.inference_momentum
            velocity += gradient
            values -= cfg.inference_learning_rate * velocity
            np.clip(
                values,
                -cfg.value_clip,
                cfg.value_clip,
                out=values,
            )

        final_terms = self._error_terms(
            x,
            targets,
            initial_hidden,
            values,
            cfg.output_precision,
        )
        energy_trace.append(float(final_terms["energy"]))
        return values, energy_trace

    def local_synaptic_gradients(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        initial_hidden: np.ndarray,
        inferred_values: np.ndarray,
        cfg: Configuration,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Local rule: presynaptic activity times postsynaptic prediction error.

        These weight gradients are assembled directly from local quantities
        after inference. They are not obtained by differentiating through the
        recurrent computation.
        """

        terms = self._error_terms(
            x,
            targets,
            initial_hidden,
            inferred_values,
            cfg.output_precision,
        )
        hidden_delta = (
            terms["hidden_error"]
            * (1.0 - terms["hidden_prediction"] ** 2)
        )
        previous_values = terms["previous_values"]
        rows = terms["response_rows"]
        times = terms["response_times"]
        output_delta = terms["output_delta"]

        if cfg.normalize_updates_by_nudge:
            hidden_delta = hidden_delta / cfg.output_precision
            output_delta = output_delta / cfg.output_precision

        n_responses = len(rows)
        gradients = {
            "W_xh": -np.einsum(
                "btd,bth->dh",
                x,
                hidden_delta,
            ) / n_responses,
            "W_hh": -np.einsum(
                "bth,btk->hk",
                previous_values,
                hidden_delta,
            ) / n_responses,
            "b_h": -hidden_delta.sum(axis=(0, 1)) / n_responses,
            "W_hy": -(
                inferred_values[rows, times].T @ output_delta
            ) / n_responses,
            "b_y": -output_delta.mean(axis=0),
        }

        for name in ("W_xh", "W_hh", "W_hy"):
            gradients[name] += (
                cfg.weight_decay * getattr(self.model, name)
            )
        return gradients, float(terms["energy"])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def minimum_inference_rounds_for_previous_feedback() -> int:
    """
    One round applies the response nudge; later rounds move it backward.

    The earliest feedback tick on trial t-1 is the limiting state for the
    response on trial t.
    """

    temporal_distance = (
        PHASE.n_steps
        + PHASE.response_tick
        - PHASE.feedback_start
    )
    return temporal_distance + 1


def validate_configuration(cfg: Configuration) -> None:
    minimum_rounds = minimum_inference_rounds_for_previous_feedback()
    if cfg.inference_steps < minimum_rounds:
        raise ValueError(
            "inference_steps={0} is too short: at least {1} rounds are "
            "required to reach the complete previous feedback window".format(
                cfg.inference_steps,
                minimum_rounds,
            )
        )
    if cfg.output_precision <= 0.0:
        raise ValueError("output_precision must be positive")
    if cfg.inference_learning_rate <= 0.0:
        raise ValueError("inference_learning_rate must be positive")
    if not 0.0 <= cfg.inference_momentum < 1.0:
        raise ValueError("inference_momentum must be in [0, 1)")
    if cfg.epochs < 1 or cfg.sessions_per_epoch < 1:
        raise ValueError("epochs and sessions_per_epoch must be positive")


def train_predictive_coding(
    cfg: Configuration,
) -> Tuple[TanhRecurrentNetwork, List[Dict[str, float]]]:
    """Train the PC-only model with the corrected local-learning procedure."""

    validate_configuration(cfg)
    rng = np.random.default_rng(cfg.seed)
    model = TanhRecurrentNetwork(cfg.hidden_size, rng)
    pc = PredictiveCodingTrainer(model)
    optimizer = Adam(
        model.parameters(),
        cfg.synaptic_learning_rate,
    )
    chunk_steps = cfg.chunk_trials * PHASE.n_steps
    history: List[Dict[str, float]] = []

    exposures = (
        cfg.epochs
        * cfg.sessions_per_epoch
        * cfg.trials_per_session
    )
    print(
        "Corrected PC training: {0} epochs × {1} sessions × {2} trials "
        "= {3:,} trial exposures".format(
            cfg.epochs,
            cfg.sessions_per_epoch,
            cfg.trials_per_session,
            exposures,
        )
    )
    print(
        "Architecture: 7 inputs -> one 48-unit tanh recurrent layer "
        "-> 2 outputs"
    )
    print(
        "PC inference rounds: {0} (minimum for complete previous feedback: "
        "{1})".format(
            cfg.inference_steps,
            minimum_inference_rounds_for_previous_feedback(),
        )
    )

    for epoch in range(1, cfg.epochs + 1):
        batch = generate_sessions(
            cfg.sessions_per_epoch,
            cfg.trials_per_session,
            cfg,
            rng,
        )
        x, targets = build_teacher_forced_inputs(batch, cfg, rng)
        state = model.zero_state(cfg.sessions_per_epoch)

        cross_entropies: List[float] = []
        gradient_norms: List[float] = []
        final_energies: List[float] = []
        energy_reductions: List[float] = []

        for start in range(0, x.shape[1], chunk_steps):
            stop = min(start + chunk_steps, x.shape[1])
            chunk_x = x[:, start:stop]
            chunk_targets = targets[:, start:stop]

            forward_values = pc.forward_values(chunk_x, state)
            forward_final_state = forward_values[:, -1].copy()
            forward_cross_entropy = (
                pc.forward_response_cross_entropy(
                    forward_values,
                    chunk_targets,
                )
            )

            inferred_values, energy_trace = pc.infer_values(
                chunk_x,
                chunk_targets,
                state,
                cfg,
                initial_values=forward_values,
            )
            gradients, final_energy = pc.local_synaptic_gradients(
                chunk_x,
                chunk_targets,
                state,
                inferred_values,
                cfg,
            )
            gradient_norm = optimizer.update(
                gradients,
                cfg.gradient_clip_norm,
            )

            # Crucial leakage control: carry the pre-update forward state,
            # never the target-nudged inferred state.
            state = forward_final_state

            responses_per_session = max(
                float(np.count_nonzero(chunk_targets >= 0))
                / chunk_x.shape[0],
                1.0,
            )
            cross_entropies.append(forward_cross_entropy)
            gradient_norms.append(gradient_norm)
            final_energies.append(final_energy / responses_per_session)
            energy_reductions.append(
                (energy_trace[0] - energy_trace[-1])
                / responses_per_session
            )

        epoch_result = {
            "epoch": float(epoch),
            "forward_response_cross_entropy": float(
                np.mean(cross_entropies)
            ),
            "gradient_norm": float(np.mean(gradient_norms)),
            "pc_energy_per_response": float(np.mean(final_energies)),
            "pc_energy_reduction_per_response": float(
                np.mean(energy_reductions)
            ),
        }
        history.append(epoch_result)
        print(
            "epoch {0:2d}/{1} | forward CE={2:.5f} | "
            "PC energy/response={3:.6f} | mean gradient norm={4:.4f}".format(
                epoch,
                cfg.epochs,
                epoch_result["forward_response_cross_entropy"],
                epoch_result["pc_energy_per_response"],
                epoch_result["gradient_norm"],
            )
        )

    return model, history


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------


def compute_block_age(block_id: np.ndarray) -> np.ndarray:
    age = np.zeros_like(block_id, dtype=np.int64)
    for trial in range(1, block_id.shape[1]):
        same_block = block_id[:, trial] == block_id[:, trial - 1]
        age[:, trial] = np.where(
            same_block,
            age[:, trial - 1] + 1,
            0,
        )
    return age


def _rollout(
    model: TanhRecurrentNetwork,
    inputs: np.ndarray,
    reset_state_each_trial: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return stimulus-path and counterfactual zero-evidence P(choice right).

    The zero-evidence branch starts after the two baseline ticks, receives the
    go cue but no current visual evidence, and is never carried forward.
    """

    n_sessions, n_trials, _, _ = inputs.shape
    hidden = model.zero_state(n_sessions)
    choice_probability_right = np.empty(
        (n_sessions, n_trials),
        dtype=np.float64,
    )
    zero_evidence_probability_right = np.empty_like(
        choice_probability_right
    )
    zero_input = np.zeros((n_sessions, N_INPUTS), dtype=np.float64)

    for trial in range(n_trials):
        if reset_state_each_trial:
            hidden = model.zero_state(n_sessions)

        # Both branches first pass through the two baseline ticks.
        for tick in range(PHASE.stim_start):
            hidden = model.step(inputs[:, trial, tick], hidden)

        # Counterfactual probe: no visual evidence, go cue retained.
        probe_hidden = hidden.copy()
        for tick in range(
            PHASE.stim_start,
            PHASE.response_tick + 1,
        ):
            probe_input = zero_input.copy()
            if tick == PHASE.go_tick:
                probe_input[:, GO_CUE] = 1.0
            probe_hidden = model.step(probe_input, probe_hidden)
        zero_evidence_probability_right[:, trial] = (
            model.probabilities(probe_hidden)[:, RIGHT]
        )

        # Natural stimulus path and response.
        for tick in range(
            PHASE.stim_start,
            PHASE.response_tick + 1,
        ):
            hidden = model.step(inputs[:, trial, tick], hidden)
            if tick == PHASE.response_tick:
                choice_probability_right[:, trial] = (
                    model.probabilities(hidden)[:, RIGHT]
                )

        # Teacher-forced action and outcome update the next-trial history.
        for tick in range(PHASE.feedback_start, PHASE.n_steps):
            hidden = model.step(inputs[:, trial, tick], hidden)

    return choice_probability_right, zero_evidence_probability_right


def _evaluation_metrics(
    batch: SyntheticBatch,
    choice_probability_right: np.ndarray,
    zero_evidence_probability_right: np.ndarray,
) -> Dict[str, object]:
    target = batch.side
    predicted = (
        choice_probability_right >= 0.5
    ).astype(np.int64)
    chosen_probability = np.where(
        target == RIGHT,
        choice_probability_right,
        1.0 - choice_probability_right,
    )

    accuracy_by_contrast = {}
    for contrast in np.unique(batch.contrast):
        mask = np.isclose(batch.contrast, contrast)
        accuracy_by_contrast[str(float(contrast))] = float(
            np.mean(predicted[mask] == target[mask])
        )

    low_block = np.isclose(batch.p_right, 0.2)
    high_block = np.isclose(batch.p_right, 0.8)
    zero_contrast = np.isclose(batch.contrast, 0.0)
    block_age = compute_block_age(batch.block_id)
    settled = block_age >= 15

    low_zero = float(np.mean(zero_evidence_probability_right[low_block]))
    high_zero = float(np.mean(zero_evidence_probability_right[high_block]))
    settled_low = float(
        np.mean(
            zero_evidence_probability_right[low_block & settled]
        )
    )
    settled_high = float(
        np.mean(
            zero_evidence_probability_right[high_block & settled]
        )
    )

    return {
        "accuracy": float(np.mean(predicted == target)),
        "response_cross_entropy": float(
            -np.mean(np.log(chosen_probability + 1.0e-12))
        ),
        "accuracy_by_contrast": accuracy_by_contrast,
        "zero_contrast_choice_probability_right": {
            "low_p_right_block": float(
                np.mean(
                    choice_probability_right[
                        low_block & zero_contrast
                    ]
                )
            ),
            "high_p_right_block": float(
                np.mean(
                    choice_probability_right[
                        high_block & zero_contrast
                    ]
                )
            ),
        },
        "counterfactual_zero_evidence_choice_probability": {
            "low_p_right_block": low_zero,
            "high_p_right_block": high_zero,
            "history_gap": high_zero - low_zero,
        },
        "settled_block_zero_evidence_calibration": {
            "definition": "trials at least 15 trials after a switch",
            "low_p_right_block": settled_low,
            "high_p_right_block": settled_high,
            "mean_absolute_error_to_true_prior": float(
                0.5
                * (
                    abs(settled_low - 0.2)
                    + abs(settled_high - 0.8)
                )
            ),
        },
    }


def evaluate_predictive_coding(
    model: TanhRecurrentNetwork,
    cfg: Configuration,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Evaluate history-dependent behavior and a reset-memory control."""

    batch = generate_sessions(
        cfg.evaluation_sessions,
        cfg.evaluation_trials,
        cfg,
        np.random.default_rng(cfg.evaluation_seed),
    )
    flat_inputs, _ = build_teacher_forced_inputs(
        batch,
        cfg,
        np.random.default_rng(cfg.evaluation_seed + 1),
    )
    inputs = flat_inputs.reshape(
        cfg.evaluation_sessions,
        cfg.evaluation_trials,
        PHASE.n_steps,
        N_INPUTS,
    )

    history_choice, history_zero = _rollout(
        model,
        inputs,
        reset_state_each_trial=False,
    )
    reset_choice, reset_zero = _rollout(
        model,
        inputs,
        reset_state_each_trial=True,
    )

    metrics = {
        "history_dependent_evaluation": _evaluation_metrics(
            batch,
            history_choice,
            history_zero,
        ),
        "reset_state_each_trial_control": _evaluation_metrics(
            batch,
            reset_choice,
            reset_zero,
        ),
    }
    arrays = {
        "p_right": batch.p_right,
        "side": batch.side,
        "contrast": batch.contrast,
        "block_id": batch.block_id,
        "history_choice_probability_right": history_choice,
        "history_zero_evidence_probability_right": history_zero,
        "reset_choice_probability_right": reset_choice,
        "reset_zero_evidence_probability_right": reset_zero,
    }
    return metrics, arrays


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def make_figure(
    history: List[Dict[str, float]],
    metrics: Dict[str, object],
    arrays: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save one uniquely named four-panel performance figure."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Matplotlib is not installed; metrics and model were saved, "
            "but the figure was skipped."
        )
        return

    history_metrics = metrics["history_dependent_evaluation"]
    reset_metrics = metrics["reset_state_each_trial_control"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True,
    )

    # Training response cross-entropy.
    axes[0, 0].plot(
        [entry["epoch"] for entry in history],
        [
            entry["forward_response_cross_entropy"]
            for entry in history
        ],
        marker="o",
        color="#8e44ad",
    )
    axes[0, 0].set(
        title="Corrected PC training",
        xlabel="Epoch",
        ylabel="Forward response cross-entropy",
    )

    # Accuracy by contrast.
    contrast_keys = [
        str(float(value))
        for value in sorted(np.unique(arrays["contrast"]))
    ]
    contrast_values = np.asarray(
        [float(key) for key in contrast_keys]
    )
    axes[0, 1].plot(
        contrast_values,
        [
            100.0
            * history_metrics["accuracy_by_contrast"][key]
            for key in contrast_keys
        ],
        marker="o",
        label="history carried",
        color="#8e44ad",
    )
    axes[0, 1].plot(
        contrast_values,
        [
            100.0
            * reset_metrics["accuracy_by_contrast"][key]
            for key in contrast_keys
        ],
        marker="s",
        linestyle="--",
        label="state reset each trial",
        color="#777777",
    )
    axes[0, 1].set(
        title="Held-out correct-side accuracy",
        xlabel="Absolute stimulus contrast",
        ylabel="Accuracy (%)",
        ylim=(45.0, 102.0),
    )
    axes[0, 1].legend(frameon=False)

    # Counterfactual zero-evidence preference by true block.
    x = np.asarray([0.2, 0.8])
    history_block = history_metrics[
        "counterfactual_zero_evidence_choice_probability"
    ]
    reset_block = reset_metrics[
        "counterfactual_zero_evidence_choice_probability"
    ]
    axes[1, 0].plot(
        x,
        [
            history_block["low_p_right_block"],
            history_block["high_p_right_block"],
        ],
        marker="o",
        linewidth=2,
        label="history carried",
        color="#8e44ad",
    )
    axes[1, 0].plot(
        x,
        [
            reset_block["low_p_right_block"],
            reset_block["high_p_right_block"],
        ],
        marker="s",
        linestyle="--",
        label="state reset each trial",
        color="#777777",
    )
    axes[1, 0].plot(
        x,
        x,
        linestyle=":",
        color="#333333",
        label="true prior",
    )
    axes[1, 0].set(
        title="Block knowledge with current evidence removed",
        xlabel="True block P(right)",
        ylabel="Model P(choose right)",
        xticks=x,
        ylim=(0.0, 1.0),
    )
    axes[1, 0].legend(frameon=False)

    # Example session.
    n_show = min(300, arrays["p_right"].shape[1])
    trials = np.arange(n_show)
    axes[1, 1].step(
        trials,
        arrays["p_right"][0, :n_show],
        where="post",
        label="true block P(right)",
        color="#222222",
    )
    axes[1, 1].plot(
        trials,
        arrays[
            "history_zero_evidence_probability_right"
        ][0, :n_show],
        label="PC zero-evidence preference",
        color="#8e44ad",
        alpha=0.9,
    )
    axes[1, 1].set(
        title="Example held-out session",
        xlabel="Trial",
        ylabel="P(right)",
        ylim=(0.0, 1.0),
    )
    axes[1, 1].legend(frameon=False)

    for axis in axes.ravel():
        axis.grid(alpha=0.2)

    fig.suptitle(
        "Corrected V2 predictive-coding RNN — local learning only"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate only the corrected V2 predictive-coding RNN."
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Short wiring check; not a scientific training run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Training/model initialization seed.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the corrected default of eight epochs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIRECTORY),
        help="Directory for uniquely named outputs.",
    )
    args = parser.parse_args()

    cfg = Configuration(seed=int(args.seed))
    if args.epochs is not None:
        cfg = replace(cfg, epochs=int(args.epochs))
    if args.quick:
        cfg = replace(
            cfg,
            epochs=2 if args.epochs is None else int(args.epochs),
            sessions_per_epoch=8,
            trials_per_session=240,
            evaluation_sessions=12,
            evaluation_trials=240,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / (
        UNIQUE_TAG + "_MODEL_SEED_{0}.npz".format(cfg.seed)
    )
    metrics_path = args.output_dir / (
        UNIQUE_TAG + "_METRICS_SEED_{0}.json".format(cfg.seed)
    )
    arrays_path = args.output_dir / (
        UNIQUE_TAG + "_EVALUATION_ARRAYS_SEED_{0}.npz".format(cfg.seed)
    )
    figure_path = args.output_dir / (
        UNIQUE_TAG + "_PERFORMANCE_SEED_{0}.png".format(cfg.seed)
    )

    start_time = time.perf_counter()
    model, history = train_predictive_coding(cfg)
    metrics, arrays = evaluate_predictive_coding(model, cfg)
    elapsed_seconds = time.perf_counter() - start_time

    algorithm = {
        "name": "corrected V2 predictive-coding recurrent network",
        "architecture": (
            "7 inputs -> one 48-unit tanh recurrent layer -> 2 outputs"
        ),
        "credit_assignment": "iterative hidden-value inference",
        "synaptic_rule": (
            "local presynaptic activity times postsynaptic prediction error"
        ),
        "backpropagation_used": False,
        "backpropagation_through_time_used": False,
        "automatic_differentiation_used": False,
        "target_nudged_state_carried_between_chunks": False,
    }
    output = {
        "unique_file_tag": UNIQUE_TAG,
        "algorithm": algorithm,
        "configuration": asdict(cfg),
        "phase_ticks": {
            "baseline": [0, 1],
            "stimulus_and_go_start": PHASE.stim_start,
            "go_tick": PHASE.go_tick,
            "response_tick": PHASE.response_tick,
            "feedback_ticks": list(
                range(PHASE.feedback_start, PHASE.n_steps)
            ),
            "n_steps": PHASE.n_steps,
        },
        "minimum_inference_rounds_for_complete_previous_feedback": (
            minimum_inference_rounds_for_previous_feedback()
        ),
        "training_history": history,
        "evaluation": metrics,
        "runtime_seconds": float(elapsed_seconds),
        "outputs": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "evaluation_arrays": str(arrays_path),
            "performance_figure": str(figure_path),
        },
    }

    model.save(
        model_path,
        metadata={
            "unique_file_tag": UNIQUE_TAG,
            "algorithm": algorithm,
            "configuration": asdict(cfg),
        },
    )
    np.savez_compressed(arrays_path, **arrays)
    metrics_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    make_figure(history, metrics, arrays, figure_path)

    history_evaluation = metrics["history_dependent_evaluation"]
    reset_evaluation = metrics["reset_state_each_trial_control"]
    print()
    print("Finished corrected PC-only run.")
    print(
        "Held-out accuracy with history: {0:.2f}%".format(
            100.0 * history_evaluation["accuracy"]
        )
    )
    print(
        "Held-out accuracy with state reset each trial: {0:.2f}%".format(
            100.0 * reset_evaluation["accuracy"]
        )
    )
    print(
        "Zero-evidence history gap: {0:.3f}".format(
            history_evaluation[
                "counterfactual_zero_evidence_choice_probability"
            ]["history_gap"]
        )
    )
    print("Saved uniquely named outputs in:", args.output_dir.resolve())
    print("Model:", model_path.name)
    print("Metrics:", metrics_path.name)
    print("Figure:", figure_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
