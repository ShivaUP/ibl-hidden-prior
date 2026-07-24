"""Corrected predictive-coding credit assignment for tanh and GRU RNNs.

Recipe aligned with PC_V2_CORRECTED.py:
  - weak output precision during training inference
  - enough synchronous inference rounds to reach previous-trial feedback
  - nudge-normalized local synaptic updates
  - carry forward (not inferred) state between chunks

GRU PC is gate-aware: prediction errors and local updates use the full GRU step.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.models_v2.rnn_cells import GRURNN, TanhRNN, _softmax
from src.synthetic.channels import N_INPUTS, PhaseTicks

RNNModel = Union[TanhRNN, GRURNN]


def minimum_inference_rounds_for_previous_feedback(phase: PhaseTicks | None = None) -> int:
    """Rounds needed for the response nudge to reach prior-trial feedback."""
    phase = phase or PhaseTicks()
    temporal_distance = phase.n_steps + phase.response_tick - phase.feedback_start
    return int(temporal_distance + 1)


def validate_pc_inference_steps(inference_steps: int, phase: PhaseTicks | None = None) -> None:
    minimum = minimum_inference_rounds_for_previous_feedback(phase)
    if inference_steps < minimum:
        raise ValueError(
            f"pc_inference_steps={inference_steps} is too short: need >= {minimum} "
            "to reach the complete previous feedback window"
        )


def _response_output_terms(
    values: np.ndarray,
    targets: np.ndarray,
    w_hy: np.ndarray,
    b_y: np.ndarray,
    output_precision: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if output_precision <= 0.0:
        raise ValueError("output_precision must be positive")
    response_rows, response_times = np.nonzero(targets >= 0)
    if len(response_rows) == 0:
        raise ValueError("PC chunk needs at least one response target")
    response_targets = targets[response_rows, response_times]
    if np.any((response_targets != 0) & (response_targets != 1)):
        raise ValueError("targets must be LEFT=0 or RIGHT=1")
    response_values = values[response_rows, response_times]
    probs = _softmax(response_values @ w_hy + b_y)
    one_hot = np.eye(2, dtype=np.float64)[response_targets]
    output_delta = output_precision * (one_hot - probs)
    chosen = probs[np.arange(len(response_targets)), response_targets]
    nll_term = -output_precision * float(np.sum(np.log(chosen + 1e-12)))
    return response_rows, response_times, output_delta, chosen, nll_term


class TanhPredictiveCodingTrainer:
    """Iterative PC inference + local updates for TanhRNN (corrected V2)."""

    def __init__(self, model: TanhRNN):
        self.model = model

    def forward_values(self, x: np.ndarray, h_initial: np.ndarray) -> np.ndarray:
        batch_size, n_time, _ = x.shape
        values = np.empty((batch_size, n_time, self.model.hidden_size), dtype=np.float64)
        hidden = h_initial
        for t in range(n_time):
            hidden = self.model.step(x[:, t], hidden)
            values[:, t] = hidden
        return values

    def forward_response_cross_entropy(self, values: np.ndarray, targets: np.ndarray) -> float:
        rows, times = np.nonzero(targets >= 0)
        yt = targets[rows, times]
        probs = _softmax(values[rows, times] @ self.model.W_hy + self.model.b_y)
        chosen = probs[np.arange(len(yt)), yt]
        return float(-np.mean(np.log(chosen + 1e-12)))

    def _error_terms(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        values: np.ndarray,
        output_precision: float,
    ) -> Dict[str, object]:
        previous_values = np.concatenate((h_initial[:, None, :], values[:, :-1, :]), axis=1)
        hidden_prediction = np.tanh(
            x @ self.model.W_xh + previous_values @ self.model.W_hh + self.model.b_h
        )
        hidden_error = values - hidden_prediction
        rows, times, output_delta, _, nll_term = _response_output_terms(
            values, targets, self.model.W_hy, self.model.b_y, output_precision
        )
        energy = (0.5 * float(np.sum(hidden_error**2)) + nll_term) / x.shape[0]
        return {
            "previous_values": previous_values,
            "hidden_prediction": hidden_prediction,
            "hidden_error": hidden_error,
            "response_rows": rows,
            "response_times": times,
            "output_delta": output_delta,
            "energy": float(energy),
        }

    def value_gradients(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        values: np.ndarray,
        output_precision: float,
    ) -> Tuple[np.ndarray, float]:
        terms = self._error_terms(x, targets, h_initial, values, output_precision)
        hidden_error = terms["hidden_error"]
        hidden_prediction = terms["hidden_prediction"]
        value_gradient = hidden_error.copy()
        next_drive = hidden_error[:, 1:] * (1.0 - hidden_prediction[:, 1:] ** 2)
        value_gradient[:, :-1] -= next_drive @ self.model.W_hh.T
        rows = terms["response_rows"]
        times = terms["response_times"]
        value_gradient[rows, times] -= terms["output_delta"] @ self.model.W_hy.T
        return value_gradient, float(terms["energy"])

    def infer_values(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        *,
        inference_steps: int,
        inference_learning_rate: float,
        output_precision: float,
        value_clip: float,
        inference_momentum: float = 0.0,
        initial_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        validate_pc_inference_steps(inference_steps)
        values = (
            self.forward_values(x, h_initial)
            if initial_values is None
            else np.asarray(initial_values, dtype=np.float64).copy()
        )
        velocity = np.zeros_like(values)
        energy_trace: List[float] = []
        for _ in range(inference_steps):
            gradient, energy = self.value_gradients(
                x, targets, h_initial, values, output_precision
            )
            energy_trace.append(energy)
            velocity *= inference_momentum
            velocity += gradient
            values -= inference_learning_rate * velocity
            np.clip(values, -value_clip, value_clip, out=values)
        final_terms = self._error_terms(x, targets, h_initial, values, output_precision)
        energy_trace.append(float(final_terms["energy"]))
        return values, energy_trace

    def local_synaptic_gradients(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        inferred_values: np.ndarray,
        *,
        output_precision: float,
        weight_decay: float,
        normalize_by_nudge: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        terms = self._error_terms(
            x, targets, h_initial, inferred_values, output_precision
        )
        hidden_delta = terms["hidden_error"] * (1.0 - terms["hidden_prediction"] ** 2)
        previous_values = terms["previous_values"]
        rows = terms["response_rows"]
        times = terms["response_times"]
        output_delta = terms["output_delta"]
        if normalize_by_nudge:
            hidden_delta = hidden_delta / output_precision
            output_delta = output_delta / output_precision
        n_responses = len(rows)
        gradients = {
            "W_xh": -(np.einsum("btd,bth->dh", x, hidden_delta)) / n_responses,
            "W_hh": -(np.einsum("bth,btk->hk", previous_values, hidden_delta)) / n_responses,
            "b_h": -hidden_delta.sum(axis=(0, 1)) / n_responses,
            "W_hy": -(inferred_values[rows, times].T @ output_delta) / n_responses,
            "b_y": -output_delta.mean(axis=0),
        }
        for name in ("W_xh", "W_hh", "W_hy"):
            gradients[name] += weight_decay * getattr(self.model, name)
        return gradients, float(terms["energy"])


class GRUPredictiveCodingTrainer:
    """Gate-aware PC inference + local updates for GRURNN."""

    def __init__(self, model: GRURNN):
        self.model = model

    def forward_values(self, x: np.ndarray, h_initial: np.ndarray) -> np.ndarray:
        batch_size, n_time, _ = x.shape
        values = np.empty((batch_size, n_time, self.model.hidden_size), dtype=np.float64)
        hidden = h_initial
        for t in range(n_time):
            hidden = self.model.step(x[:, t], hidden)
            values[:, t] = hidden
        return values

    def forward_response_cross_entropy(self, values: np.ndarray, targets: np.ndarray) -> float:
        rows, times = np.nonzero(targets >= 0)
        yt = targets[rows, times]
        probs = _softmax(values[rows, times] @ self.model.W_hy + self.model.b_y)
        chosen = probs[np.arange(len(yt)), yt]
        return float(-np.mean(np.log(chosen + 1e-12)))

    def _gru_predict_from_previous(
        self, x: np.ndarray, previous_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return pred, z, r, h_hat for each tick from previous hidden values."""
        z = 1.0 / (
            1.0
            + np.exp(
                -(
                    x @ self.model.W_xz
                    + previous_values @ self.model.W_hz
                    + self.model.b_z
                )
            )
        )
        r = 1.0 / (
            1.0
            + np.exp(
                -(
                    x @ self.model.W_xr
                    + previous_values @ self.model.W_hr
                    + self.model.b_r
                )
            )
        )
        h_hat = np.tanh(
            x @ self.model.W_xh
            + (r * previous_values) @ self.model.W_hh
            + self.model.b_h
        )
        pred = (1.0 - z) * previous_values + z * h_hat
        return pred, z, r, h_hat

    def _error_terms(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        values: np.ndarray,
        output_precision: float,
    ) -> Dict[str, object]:
        previous_values = np.concatenate((h_initial[:, None, :], values[:, :-1, :]), axis=1)
        hidden_prediction, z, r, h_hat = self._gru_predict_from_previous(x, previous_values)
        hidden_error = values - hidden_prediction
        rows, times, output_delta, _, nll_term = _response_output_terms(
            values, targets, self.model.W_hy, self.model.b_y, output_precision
        )
        energy = (0.5 * float(np.sum(hidden_error**2)) + nll_term) / x.shape[0]
        return {
            "previous_values": previous_values,
            "hidden_prediction": hidden_prediction,
            "hidden_error": hidden_error,
            "z": z,
            "r": r,
            "h_hat": h_hat,
            "response_rows": rows,
            "response_times": times,
            "output_delta": output_delta,
            "energy": float(energy),
        }

    def _apply_gru_prev_jacobian(
        self,
        dh_new: np.ndarray,
        *,
        h_prev: np.ndarray,
        z: np.ndarray,
        r: np.ndarray,
        h_hat: np.ndarray,
    ) -> np.ndarray:
        """Map ∂E/∂h_new → ∂E/∂h_prev for one GRU step (gate-aware)."""
        dz = dh_new * (h_hat - h_prev)
        dh_hat = dh_new * z
        dh_prev = dh_new * (1.0 - z)
        d_hhat_pre = dh_hat * (1.0 - h_hat**2)
        dh_prev = dh_prev + (d_hhat_pre @ self.model.W_hh.T) * r
        dr = (d_hhat_pre @ self.model.W_hh.T) * h_prev
        dz_pre = dz * z * (1.0 - z)
        dr_pre = dr * r * (1.0 - r)
        return dh_prev + dz_pre @ self.model.W_hz.T + dr_pre @ self.model.W_hr.T

    def value_gradients(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        values: np.ndarray,
        output_precision: float,
    ) -> Tuple[np.ndarray, float]:
        terms = self._error_terms(x, targets, h_initial, values, output_precision)
        hidden_error = terms["hidden_error"]
        value_gradient = hidden_error.copy()
        # Communicate next-tick prediction error through the GRU Jacobian.
        value_gradient[:, :-1] -= self._apply_gru_prev_jacobian(
            hidden_error[:, 1:],
            h_prev=terms["previous_values"][:, 1:],
            z=terms["z"][:, 1:],
            r=terms["r"][:, 1:],
            h_hat=terms["h_hat"][:, 1:],
        )
        rows = terms["response_rows"]
        times = terms["response_times"]
        value_gradient[rows, times] -= terms["output_delta"] @ self.model.W_hy.T
        return value_gradient, float(terms["energy"])

    def infer_values(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        *,
        inference_steps: int,
        inference_learning_rate: float,
        output_precision: float,
        value_clip: float,
        inference_momentum: float = 0.0,
        initial_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        validate_pc_inference_steps(inference_steps)
        values = (
            self.forward_values(x, h_initial)
            if initial_values is None
            else np.asarray(initial_values, dtype=np.float64).copy()
        )
        velocity = np.zeros_like(values)
        energy_trace: List[float] = []
        for _ in range(inference_steps):
            gradient, energy = self.value_gradients(
                x, targets, h_initial, values, output_precision
            )
            energy_trace.append(energy)
            velocity *= inference_momentum
            velocity += gradient
            values -= inference_learning_rate * velocity
            np.clip(values, -value_clip, value_clip, out=values)
        final_terms = self._error_terms(x, targets, h_initial, values, output_precision)
        energy_trace.append(float(final_terms["energy"]))
        return values, energy_trace

    def local_synaptic_gradients(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        inferred_values: np.ndarray,
        *,
        output_precision: float,
        weight_decay: float,
        normalize_by_nudge: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Local rule: one-step GRU PE × presynaptic activities (not BPTT)."""
        terms = self._error_terms(
            x, targets, h_initial, inferred_values, output_precision
        )
        e = terms["hidden_error"]
        z, r, h_hat = terms["z"], terms["r"], terms["h_hat"]
        h_prev = terms["previous_values"]
        if normalize_by_nudge:
            e = e / output_precision
            output_delta = terms["output_delta"] / output_precision
        else:
            output_delta = terms["output_delta"]

        dz = e * (h_hat - h_prev)
        dh_hat = e * z
        d_hhat_pre = dh_hat * (1.0 - h_hat**2)
        dr = (d_hhat_pre @ self.model.W_hh.T) * h_prev
        dz_pre = dz * z * (1.0 - z)
        dr_pre = dr * r * (1.0 - r)

        rows = terms["response_rows"]
        times = terms["response_times"]
        n_responses = len(rows)

        gradients = {
            "W_xh": -(np.einsum("btd,bth->dh", x, d_hhat_pre)) / n_responses,
            "W_hh": -(np.einsum("bth,btk->hk", r * h_prev, d_hhat_pre)) / n_responses,
            "b_h": -d_hhat_pre.sum(axis=(0, 1)) / n_responses,
            "W_xz": -(np.einsum("btd,bth->dh", x, dz_pre)) / n_responses,
            "W_hz": -(np.einsum("bth,btk->hk", h_prev, dz_pre)) / n_responses,
            "b_z": -dz_pre.sum(axis=(0, 1)) / n_responses,
            "W_xr": -(np.einsum("btd,bth->dh", x, dr_pre)) / n_responses,
            "W_hr": -(np.einsum("bth,btk->hk", h_prev, dr_pre)) / n_responses,
            "b_r": -dr_pre.sum(axis=(0, 1)) / n_responses,
            "W_hy": -(inferred_values[rows, times].T @ output_delta) / n_responses,
            "b_y": -output_delta.mean(axis=0),
        }
        for name in self.model.parameter_names:
            if name.startswith("W_"):
                gradients[name] += weight_decay * getattr(self.model, name)
        return gradients, float(terms["energy"])


# Backward-compatible alias used by older imports / docs.
PredictiveCodingTrainer = TanhPredictiveCodingTrainer


def make_pc_trainer(model: RNNModel) -> TanhPredictiveCodingTrainer | GRUPredictiveCodingTrainer:
    if isinstance(model, TanhRNN):
        return TanhPredictiveCodingTrainer(model)
    if isinstance(model, GRURNN):
        return GRUPredictiveCodingTrainer(model)
    raise TypeError(f"unsupported PC model type: {type(model)}")


def make_pc_model(hidden_size: int, rng: np.random.Generator) -> TanhRNN:
    return TanhRNN(hidden_size=hidden_size, rng=rng, input_size=N_INPUTS)
