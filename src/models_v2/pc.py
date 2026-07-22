"""Predictive-coding credit-assignment trainer for the shared tanh RNN (Kyan-adapted)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.models_v2.rnn_cells import Adam, TanhRNN, _softmax
from src.synthetic.channels import N_INPUTS


class PredictiveCodingTrainer:
    """Same test-time tanh RNN as TanhRNN; trains via local PC inference + updates.

    Ported from Kyan's PredictiveCodingRNN. Architecture and forward equations
    match TanhRNN; only credit assignment differs.
    """

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

    def _error_terms(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        h_initial: np.ndarray,
        values: np.ndarray,
        output_precision: float,
    ) -> Dict[str, object]:
        previous_values = np.concatenate(
            (h_initial[:, None, :], values[:, :-1, :]), axis=1
        )
        hidden_prediction = np.tanh(
            x @ self.model.W_xh + previous_values @ self.model.W_hh + self.model.b_h
        )
        hidden_error = values - hidden_prediction

        response_rows, response_times = np.nonzero(targets >= 0)
        if len(response_rows) == 0:
            raise ValueError("PC chunk needs at least one response target")
        response_targets = targets[response_rows, response_times]
        response_values = values[response_rows, response_times]
        response_prediction = _softmax(response_values @ self.model.W_hy + self.model.b_y)
        target_one_hot = np.eye(2, dtype=np.float64)[response_targets]
        output_delta = output_precision * (target_one_hot - response_prediction)
        chosen = response_prediction[np.arange(len(response_targets)), response_targets]
        energy = (
            0.5 * np.sum(hidden_error**2)
            - output_precision * np.sum(np.log(chosen + 1e-12))
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
        values = (
            self.forward_values(x, h_initial)
            if initial_values is None
            else np.asarray(initial_values, dtype=np.float64).copy()
        )
        energy_trace: List[float] = []
        velocity = np.zeros_like(values)
        for _ in range(inference_steps):
            terms = self._error_terms(x, targets, h_initial, values, output_precision)
            energy_trace.append(float(terms["energy"]))
            hidden_error = terms["hidden_error"]
            hidden_prediction = terms["hidden_prediction"]
            value_gradient = hidden_error.copy()
            next_drive = hidden_error[:, 1:] * (1.0 - hidden_prediction[:, 1:] ** 2)
            value_gradient[:, :-1] -= next_drive @ self.model.W_hh.T
            rows = terms["response_rows"]
            times = terms["response_times"]
            value_gradient[rows, times] -= terms["output_delta"] @ self.model.W_hy.T
            velocity *= inference_momentum
            velocity += value_gradient
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
            "W_hh": -(
                np.einsum("bth,btk->hk", previous_values, hidden_delta)
            )
            / n_responses,
            "b_h": -hidden_delta.sum(axis=(0, 1)) / n_responses,
            "W_hy": -(inferred_values[rows, times].T @ output_delta) / n_responses,
            "b_y": -output_delta.mean(axis=0),
        }
        for name in ("W_xh", "W_hh", "W_hy"):
            gradients[name] += weight_decay * getattr(self.model, name)
        return gradients, float(terms["energy"])


def make_pc_model(hidden_size: int, rng: np.random.Generator) -> TanhRNN:
    return TanhRNN(hidden_size=hidden_size, rng=rng, input_size=N_INPUTS)
