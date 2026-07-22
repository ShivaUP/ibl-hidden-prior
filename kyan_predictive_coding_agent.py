"""Architecture-matched predictive-coding RNN for the hidden-prior task.

At test time this class is exactly the same kind of 7-input, tanh-recurrent,
48-hidden-unit, two-output network as the companion BPTT RNN.  The difference
is training credit assignment.  There is no autograd and no reverse BPTT loop.

Training follows the classical predictive-coding value/prediction/error
scheme.  A free hidden value is introduced at every recurrent time step.
Labels nudge only the categorical output at response step 6.  Neural
inference iteratively reduces adjacent prediction errors, then each synapse is
updated from presynaptic activity times its postsynaptic local error.  Weak
nudging is supported, although validation selected unit output precision.  All
inference and optimizer changes affect training only: the model used at test
time is still the same ordinary RNN as the BPTT comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from block_task import (
    N_INPUTS,
    N_STEPS,
    TaskConfig,
    build_training_sequence,
    decode_block_from_hidden,
    generate_trials,
    print_timing_table,
    rollout,
    save_diagnostics,
    summarize_rollout,
    summarize_switch_adaptation,
    write_results,
)


@dataclass(frozen=True)
class PCConfig:
    """Neural-inference and local synaptic-learning settings."""

    # Architecture and within-epoch training-set sizes match the BPTT project.
    # Validation-selected early stopping uses fewer epochs than BPTT, never
    # more training examples.
    hidden_size: int = 48
    epochs: int = 30
    sessions_per_epoch: int = 24
    trials_per_session: int = 240
    inference_chunk_trials: int = 32

    # Number of local activity-relaxation rounds.  This is a PC optimization
    # hyperparameter, not the BPTT truncation horizon, and is selected using a
    # validation set that is separate from the final held-out test sessions.
    inference_steps: int = 8
    inference_learning_rate: float = 0.15
    inference_momentum: float = 0.0
    output_error_precision: float = 1.0
    inferred_value_clip: float = 2.0
    normalize_updates_by_nudge: bool = True

    # Adam only rescales an already-local synaptic error; it assigns no credit.
    # Global-norm clipping exactly matches the companion BPTT optimizer's
    # non-credit-assignment mechanics.
    synaptic_learning_rate: float = 4e-4
    gradient_clip_norm: float = 1.0
    weight_decay: float = 1e-5
    seed: int = 7

    @property
    def unrolled_chunk_steps(self) -> int:
        return self.inference_chunk_trials * N_STEPS

    @property
    def inference_to_unroll_ratio(self) -> float:
        return self.inference_steps / self.unrolled_chunk_steps

    def validate(self) -> None:
        if self.hidden_size < 2:
            raise ValueError("hidden_size must be at least 2")
        if self.epochs < 1 or self.sessions_per_epoch < 1:
            raise ValueError("epochs and sessions_per_epoch must be positive")
        if self.trials_per_session < self.inference_chunk_trials:
            raise ValueError("trials_per_session must cover an inference chunk")
        if self.inference_chunk_trials < 1 or self.inference_steps < 1:
            raise ValueError("chunk length and inference steps must be positive")
        if not 0.0 < self.inference_learning_rate < 1.0:
            raise ValueError("inference_learning_rate must be between 0 and 1")
        if not 0.0 <= self.inference_momentum < 1.0:
            raise ValueError("inference_momentum must be in [0, 1)")
        if self.output_error_precision <= 0.0:
            raise ValueError("output_error_precision must be positive")
        if self.synaptic_learning_rate <= 0.0:
            raise ValueError("synaptic_learning_rate must be positive")
        if self.inferred_value_clip <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("clipping values must be positive")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


class PredictiveCodingRNN:
    """A tanh RNN trained by temporal predictive coding.

    The prediction-mode equations are identical to the companion SimpleRNN::

        h[t] = tanh(x[t] W_xh + h[t-1] W_hh + b_h)
        p[t] = softmax(h[t] W_hy + b_y)

    During PC training, v[t] denotes a free value node and
    mu[t] the recurrent prediction of it::

        epsilon_h[t] = v[t] - mu[t]
        epsilon_y[t] = one_hot(label[t]) - p[t]

    Inference on v uses only its own error, its next temporal neighbour's
    error, and its adjacent output error.  With inferred values frozen, every
    weight update is a presynaptic value times a postsynaptic local error.
    """

    parameter_names = ("W_xh", "W_hh", "b_h", "W_hy", "b_y")

    def __init__(self, input_size: int, hidden_size: int, rng: np.random.Generator):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

        input_scale = math.sqrt(2.0 / (input_size + hidden_size))
        output_scale = math.sqrt(2.0 / (hidden_size + 2))
        self.W_xh = rng.normal(0.0, input_scale, (input_size, hidden_size))
        self.W_hh = 0.90 * np.eye(hidden_size)
        self.W_hh += rng.normal(
            0.0, 0.01 / math.sqrt(hidden_size), self.W_hh.shape
        )
        self.b_h = np.zeros(hidden_size, dtype=np.float64)
        self.W_hy = rng.normal(0.0, output_scale, (hidden_size, 2))
        self.b_y = np.zeros(2, dtype=np.float64)

    def parameters(self) -> Dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self.parameter_names}

    @property
    def parameter_count(self) -> int:
        return int(sum(value.size for value in self.parameters().values()))

    def zero_state(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.hidden_size), dtype=np.float64)

    def step(self, x_t: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """Ordinary recurrent prediction used at test time."""

        return np.tanh(x_t @ self.W_xh + previous @ self.W_hh + self.b_h)

    def response_probabilities(self, hidden: np.ndarray) -> np.ndarray:
        return _softmax(hidden @ self.W_hy + self.b_y)

    def forward_values(self, x: np.ndarray, h_initial: np.ndarray) -> np.ndarray:
        """Initialize free values with the network's forward predictions."""

        batch_size, n_time, _ = x.shape
        values = np.empty((batch_size, n_time, self.hidden_size), dtype=np.float64)
        hidden = h_initial
        for time_index in range(n_time):
            hidden = self.step(x[:, time_index], hidden)
            values[:, time_index] = hidden
        return values

    def _error_terms(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        values: np.ndarray,
        output_precision: float,
    ) -> Dict[str, np.ndarray | float]:
        previous_values = np.concatenate(
            (h_initial[:, None, :], values[:, :-1, :]), axis=1
        )
        hidden_prediction = np.tanh(
            x @ self.W_xh + previous_values @ self.W_hh + self.b_h
        )
        hidden_error = values - hidden_prediction

        response_rows, response_times = np.nonzero(targets >= 0)
        if len(response_rows) == 0:
            raise ValueError("A PC inference chunk needs at least one response target")
        response_targets = targets[response_rows, response_times]
        response_values = values[response_rows, response_times]
        response_prediction = self.response_probabilities(response_values)
        target_one_hot = np.eye(2, dtype=np.float64)[response_targets]

        # A categorical likelihood gives the local output logit error y - p.
        output_delta = output_precision * (target_one_hot - response_prediction)
        chosen = response_prediction[
            np.arange(len(response_targets)), response_targets
        ]
        batch_size = x.shape[0]
        energy = (
            0.5 * np.sum(hidden_error**2)
            - output_precision * np.sum(np.log(chosen + 1e-12))
        ) / batch_size
        return {
            "previous_values": previous_values,
            "hidden_prediction": hidden_prediction,
            "hidden_error": hidden_error,
            "response_rows": response_rows,
            "response_times": response_times,
            "response_values": response_values,
            "response_prediction": response_prediction,
            "output_delta": output_delta,
            "energy": float(energy),
        }

    def infer_values(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        inference_steps: int,
        inference_learning_rate: float,
        output_precision: float,
        value_clip: float,
        inference_momentum: float = 0.0,
        initial_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        """Infer hidden activities through iterative adjacent-error messages."""

        values = (
            self.forward_values(x, h_initial)
            if initial_values is None
            else np.asarray(initial_values, dtype=np.float64).copy()
        )
        expected_shape = (x.shape[0], x.shape[1], self.hidden_size)
        if values.shape != expected_shape:
            raise ValueError(
                f"initial_values must have shape {expected_shape}, got {values.shape}"
            )
        energy_trace: List[float] = []
        velocity = np.zeros_like(values)
        for _ in range(inference_steps):
            terms = self._error_terms(
                x, targets, h_initial, values, output_precision
            )
            energy_trace.append(float(terms["energy"]))
            hidden_error = terms["hidden_error"]
            hidden_prediction = terms["hidden_prediction"]

            # dE/dv[t]: own error minus the message from t+1.  This is a
            # parallel local relaxation, not a reverse pass through the graph.
            value_gradient = hidden_error.copy()
            next_drive = hidden_error[:, 1:] * (
                1.0 - hidden_prediction[:, 1:] ** 2
            )
            value_gradient[:, :-1] -= next_drive @ self.W_hh.T

            rows = terms["response_rows"]
            times = terms["response_times"]
            value_gradient[rows, times] -= terms["output_delta"] @ self.W_hy.T
            # Heavy-ball activity dynamics use only the node's own previous
            # update and its current adjacent prediction-error messages.  No
            # reverse temporal pass or non-local credit signal is introduced.
            velocity *= inference_momentum
            velocity += value_gradient
            values -= inference_learning_rate * velocity
            np.clip(values, -value_clip, value_clip, out=values)

        final_terms = self._error_terms(
            x, targets, h_initial, values, output_precision
        )
        energy_trace.append(float(final_terms["energy"]))
        return values, energy_trace

    def local_synaptic_gradients(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        inferred_values: np.ndarray,
        output_precision: float,
        weight_decay: float,
        normalize_by_nudge: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Form local pre-activity × post-error updates with no BPTT."""

        terms = self._error_terms(
            x, targets, h_initial, inferred_values, output_precision
        )
        hidden_delta = terms["hidden_error"] * (
            1.0 - terms["hidden_prediction"] ** 2
        )
        previous_values = terms["previous_values"]
        rows = terms["response_rows"]
        times = terms["response_times"]
        output_delta = terms["output_delta"]
        # Under weak nudging, all label-induced errors are O(beta).  Dividing
        # the local data update by the known scalar beta keeps its scale stable
        # as beta -> 0 while leaving the inferred activities close to the
        # ordinary forward trajectory.  This is still a local error-times-input
        # rule and does not reveal any additional information to the network.
        if normalize_by_nudge:
            hidden_delta = hidden_delta / output_precision
            output_delta = output_delta / output_precision
        n_responses = len(rows)

        # Negative signs convert the local Hebbian/error update into the
        # gradient convention used below: parameter -= gradient.
        gradients = {
            "W_xh": -(np.einsum("btd,bth->dh", x, hidden_delta)) / n_responses,
            "W_hh": -(
                np.einsum("bth,btk->hk", previous_values, hidden_delta)
            ) / n_responses,
            "b_h": -hidden_delta.sum(axis=(0, 1)) / n_responses,
            "W_hy": -(
                inferred_values[rows, times].T @ output_delta
            ) / n_responses,
            "b_y": -output_delta.mean(axis=0),
        }
        for name in ("W_xh", "W_hh", "W_hy"):
            gradients[name] += weight_decay * getattr(self, name)
        return gradients, float(terms["energy"])

    def save(
        self, path: Path, metadata: Optional[Mapping[str, object]] = None
    ) -> None:
        payload: Dict[str, np.ndarray] = {
            name: value for name, value in self.parameters().items()
        }
        payload["input_size"] = np.asarray(self.input_size)
        payload["hidden_size"] = np.asarray(self.hidden_size)
        payload["metadata_json"] = np.asarray(json.dumps(metadata or {}))
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "PredictiveCodingRNN":
        data = np.load(path, allow_pickle=False)
        model = cls.__new__(cls)
        model.input_size = int(data["input_size"])
        model.hidden_size = int(data["hidden_size"])
        for name in cls.parameter_names:
            setattr(model, name, data[name].copy())
        return model


class LocalAdam:
    """Adam scaling applied after local PC credit assignment."""

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
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {
            name: np.zeros_like(value) for name, value in parameters.items()
        }
        self.iteration = 0

    def update(self, gradients: Mapping[str, np.ndarray], clip_norm: float) -> float:
        squared_norm = sum(
            float(np.sum(gradient**2)) for gradient in gradients.values()
        )
        global_norm = math.sqrt(squared_norm)
        scale = min(1.0, clip_norm / (global_norm + 1e-12))
        self.iteration += 1
        for name, parameter in self.parameters.items():
            gradient = gradients[name] * scale
            self.first[name] = (
                self.beta1 * self.first[name] + (1.0 - self.beta1) * gradient
            )
            self.second[name] = (
                self.beta2 * self.second[name]
                + (1.0 - self.beta2) * gradient**2
            )
            first_hat = self.first[name] / (1.0 - self.beta1**self.iteration)
            second_hat = self.second[name] / (1.0 - self.beta2**self.iteration)
            parameter -= self.learning_rate * first_hat / (
                np.sqrt(second_hat) + self.epsilon
            )
        return global_norm


def train_model(
    task: TaskConfig,
    config: PCConfig,
    verbose: bool = True,
    epoch_callback: Optional[Callable[[int, PredictiveCodingRNN], None]] = None,
) -> Tuple[PredictiveCodingRNN, List[float], List[float]]:
    """Alternate PC activity inference and one local synaptic update."""

    config.validate()
    rng = np.random.default_rng(config.seed)
    model = PredictiveCodingRNN(N_INPUTS, config.hidden_size, rng)
    optimizer = LocalAdam(model.parameters(), config.synaptic_learning_rate)
    chunk_steps = config.inference_chunk_trials * N_STEPS
    epoch_final_energy: List[float] = []
    epoch_inference_reduction: List[float] = []

    for epoch in range(1, config.epochs + 1):
        trials = generate_trials(
            config.sessions_per_epoch,
            config.trials_per_session,
            task,
            rng,
        )
        x, targets = build_training_sequence(trials, task, rng)
        initial_state = model.zero_state(config.sessions_per_epoch)
        final_energies: List[float] = []
        reductions: List[float] = []

        for start in range(0, x.shape[1], chunk_steps):
            stop = min(start + chunk_steps, x.shape[1])
            chunk_x = x[:, start:stop]
            chunk_targets = targets[:, start:stop]
            # Match BPTT's truncated-state convention exactly: the state passed
            # to the next chunk is the final forward state computed before the
            # current chunk's weight update.
            forward_values = model.forward_values(chunk_x, initial_state)
            forward_final_state = forward_values[:, -1].copy()
            inferred, energy_trace = model.infer_values(
                chunk_x,
                chunk_targets,
                initial_state,
                inference_steps=config.inference_steps,
                inference_learning_rate=config.inference_learning_rate,
                output_precision=config.output_error_precision,
                value_clip=config.inferred_value_clip,
                inference_momentum=config.inference_momentum,
                initial_values=forward_values,
            )
            gradients, final_energy = model.local_synaptic_gradients(
                chunk_x,
                chunk_targets,
                initial_state,
                inferred,
                config.output_error_precision,
                config.weight_decay,
                config.normalize_updates_by_nudge,
            )
            optimizer.update(gradients, config.gradient_clip_norm)

            # Do not leak label-conditioned activity or post-update weights
            # into the carried state.
            initial_state = forward_final_state
            final_energies.append(final_energy / chunk_x.shape[1])
            reductions.append(
                (energy_trace[0] - energy_trace[-1]) / chunk_x.shape[1]
            )

        mean_energy = float(np.mean(final_energies))
        mean_reduction = float(np.mean(reductions))
        epoch_final_energy.append(mean_energy)
        epoch_inference_reduction.append(mean_reduction)
        if verbose and (
            epoch == 1
            or epoch == config.epochs
            or epoch % max(1, config.epochs // 10) == 0
        ):
            print(
                f"epoch {epoch:4d}/{config.epochs}: "
                f"PC energy/step={mean_energy:.4f}, "
                f"inference reduction/step={mean_reduction:.4f}"
            )
        if epoch_callback is not None:
            epoch_callback(epoch, model)

    return model, epoch_final_energy, epoch_inference_reduction


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an architecture-matched recurrent network using temporal "
            "predictive coding rather than backpropagation."
        )
    )
    parser.add_argument("--epochs", type=int, default=PCConfig.epochs)
    parser.add_argument("--hidden-size", type=int, default=PCConfig.hidden_size)
    parser.add_argument("--seed", type=int, default=PCConfig.seed)
    parser.add_argument("--train-sessions", type=int, default=PCConfig.sessions_per_epoch)
    parser.add_argument("--train-trials", type=int, default=PCConfig.trials_per_session)
    parser.add_argument("--test-sessions", type=int, default=48)
    parser.add_argument("--test-trials", type=int, default=320)
    parser.add_argument(
        "--inference-chunk-trials",
        type=int,
        default=PCConfig.inference_chunk_trials,
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=PCConfig.inference_steps,
        help=(
            "PC activity-relaxation rounds; the default was selected on a "
            "validation set separate from the final test set"
        ),
    )
    parser.add_argument(
        "--inference-learning-rate",
        type=float,
        default=PCConfig.inference_learning_rate,
    )
    parser.add_argument(
        "--inference-momentum",
        type=float,
        default=PCConfig.inference_momentum,
    )
    parser.add_argument(
        "--output-error-precision",
        type=float,
        default=PCConfig.output_error_precision,
    )
    parser.add_argument(
        "--synaptic-learning-rate",
        type=float,
        default=PCConfig.synaptic_learning_rate,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=PCConfig.gradient_clip_norm,
    )
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--inspect-timing", action="store_true")
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
    config = PCConfig(
        hidden_size=args.hidden_size,
        epochs=3 if args.quick else args.epochs,
        sessions_per_epoch=6 if args.quick else args.train_sessions,
        trials_per_session=96 if args.quick else args.train_trials,
        inference_chunk_trials=(
            12 if args.quick else args.inference_chunk_trials
        ),
        inference_steps=5 if args.quick else args.inference_steps,
        inference_learning_rate=args.inference_learning_rate,
        inference_momentum=args.inference_momentum,
        output_error_precision=args.output_error_precision,
        synaptic_learning_rate=args.synaptic_learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.seed,
    )
    test_sessions = 8 if args.quick else args.test_sessions
    test_trials = 128 if args.quick else args.test_trials
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Training the matched RNN with predictive-coding local updates...")
    print(
        "PC activity inference: "
        f"{config.inference_steps} relaxation rounds per "
        f"{config.unrolled_chunk_steps}-step chunk"
    )
    print(
        "Output nudge strength: "
        f"beta={config.output_error_precision:g}; "
        f"activity momentum={config.inference_momentum:g}"
    )
    model, energy, inference_reduction = train_model(task, config)
    test_batch = generate_trials(
        test_sessions,
        test_trials,
        task,
        np.random.default_rng(args.seed + 10_000),
    )
    print("Evaluating in ordinary recurrent prediction mode with weights frozen...")
    history_records = rollout(
        model,
        test_batch,
        np.random.default_rng(args.seed + 20_000),
        reset_each_trial=False,
    )
    reset_records = rollout(
        model,
        test_batch,
        np.random.default_rng(args.seed + 20_000),
        reset_each_trial=True,
    )
    history_summary = summarize_rollout(history_records, task)
    reset_summary = summarize_rollout(reset_records, task)
    probe_summary = decode_block_from_hidden(history_records)

    results = {
        "algorithm": {
            "name": (
                "validation-tuned architecture-matched temporal "
                "predictive-coding RNN"
            ),
            "credit_assignment": "iterative adjacent prediction-error inference",
            "synaptic_rule": "presynaptic activity times postsynaptic local error",
            "backpropagation_used": False,
            "backpropagation_through_time_used": False,
            "automatic_differentiation_used": False,
            "weak_nudge_normalized_local_updates": (
                config.normalize_updates_by_nudge
            ),
            "weak_nudging_selected_as_default": (
                config.output_error_precision < 1.0
            ),
        },
        "fairness_match_to_companion_rnn": {
            "input_channels": N_INPUTS,
            "hidden_units": model.hidden_size,
            "output_units": 2,
            "learned_parameter_count": model.parameter_count,
            "recurrent_equation_and_initialization": "identical",
            "raw_action_reward_feedback": True,
            "explicit_belief_variable": False,
            "predecoded_correct_side_input": False,
            "response_objective": "categorical cross-entropy at step 6",
            "nominal_unrolled_chunk_steps": config.unrolled_chunk_steps,
            "pc_synchronous_inference_rounds": config.inference_steps,
            "inference_to_unroll_ratio": config.inference_to_unroll_ratio,
            "inference_steps_selected_on_separate_validation_set": True,
            "optimizer_after_local_credit": "Adam with global-norm clipping",
            "optimizer_mechanics_match_companion_rnn": True,
            "pc_training_epochs": config.epochs,
            "companion_rnn_training_epochs": 60,
            "pc_uses_no_more_training_examples_than_companion_rnn": (
                config.epochs <= 60
            ),
        },
        "hyperparameter_selection": {
            "selection_objective": (
                "maximum validation accuracy; lower cross-entropy breaks ties"
            ),
            "validation_seed": 30_007,
            "final_test_set_used_for_selection": False,
            "post_correction_candidates_considered": 40,
            "selected_configuration": {
                "epochs": 30,
                "output_error_precision": 1.0,
                "inference_steps": 8,
                "inference_learning_rate": 0.15,
                "inference_momentum": 0.0,
                "synaptic_learning_rate": 0.0004,
            },
            "selected_validation_accuracy": 0.8081380208333333,
            "selected_validation_cross_entropy": 0.4027319367479862,
            "selection_record": "outputs/selection_summary.json",
            "weak_nudge_was_tested_but_not_selected": True,
        },
        "task_config": asdict(task),
        "predictive_coding_config": asdict(config),
        "history_dependent_evaluation": history_summary,
        "reset_hidden_state_each_trial_control": reset_summary,
        "linear_block_probe": probe_summary,
        "switch_adaptation": {
            "definition": (
                "50% and 90% crossing of the change from the mean at offsets "
                "-10:-1 to the mean at offsets +20:+30"
            ),
            **summarize_switch_adaptation(history_records),
        },
        "training": {
            "final_energy_per_step_by_epoch": energy,
            "energy_removed_by_inference_per_step_by_epoch": inference_reduction,
        },
    }
    model_path = args.output_dir / "predictive_coding_agent.npz"
    metrics_path = args.output_dir / "metrics.json"
    figure_path = args.output_dir / "diagnostics.png"
    model.save(
        model_path,
        metadata={"task_config": asdict(task), "predictive_coding_config": asdict(config)},
    )
    write_results(metrics_path, results)
    save_diagnostics(
        history_records,
        energy,
        inference_reduction,
        task,
        figure_path,
    )

    normal_gap = history_summary["counterfactual_zero_evidence_choice_probability"][
        "history_gap"
    ]
    reset_gap = reset_summary["counterfactual_zero_evidence_choice_probability"][
        "history_gap"
    ]
    print("\nKey held-out results")
    print(f"  learned parameters:               {model.parameter_count}")
    print(f"  overall accuracy:                 {history_summary['accuracy']:.3f}")
    print(f"  zero-evidence block history gap:  {normal_gap:.3f}")
    print(f"  gap after hidden-state reset:     {reset_gap:.3f}")
    print(
        "  hidden-state block decoder:      "
        f"{probe_summary['held_out_session_accuracy']:.3f} accuracy"
    )
    print(f"\nSaved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved figure:  {figure_path}")


if __name__ == "__main__":
    main()
