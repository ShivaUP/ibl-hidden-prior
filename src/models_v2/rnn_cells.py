"""v2 models: tanh BPTT, tanh PC-CA, GRU, explicit Bayes (adapted from Kyan/Shrijana)."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from src.synthetic.channels import N_INPUTS
from src.synthetic.schema import RIGHT


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


class Adam:
    def __init__(self, parameters: Mapping[str, np.ndarray], learning_rate: float):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1, self.beta2, self.epsilon = 0.9, 0.999, 1e-8
        self.first = {n: np.zeros_like(v) for n, v in parameters.items()}
        self.second = {n: np.zeros_like(v) for n, v in parameters.items()}
        self.iteration = 0

    def update(self, gradients: Mapping[str, np.ndarray], clip_norm: float) -> float:
        sq = sum(float(np.sum(g**2)) for g in gradients.values())
        gnorm = math.sqrt(sq)
        scale = min(1.0, clip_norm / (gnorm + 1e-12))
        self.iteration += 1
        for name, parameter in self.parameters.items():
            g = gradients[name] * scale
            self.first[name] = self.beta1 * self.first[name] + (1 - self.beta1) * g
            self.second[name] = self.beta2 * self.second[name] + (1 - self.beta2) * g**2
            m = self.first[name] / (1 - self.beta1**self.iteration)
            v = self.second[name] / (1 - self.beta2**self.iteration)
            parameter -= self.learning_rate * m / (np.sqrt(v) + self.epsilon)
        return gnorm


class TanhRNN:
    """Vanilla tanh RNN (Kyan standard), adapted to shared v2 inputs."""

    parameter_names = ("W_xh", "W_hh", "b_h", "W_hy", "b_y")

    def __init__(self, hidden_size: int, rng: np.random.Generator, input_size: int = N_INPUTS):
        self.input_size = input_size
        self.hidden_size = hidden_size
        iscale = math.sqrt(2.0 / (input_size + hidden_size))
        oscale = math.sqrt(2.0 / (hidden_size + 2))
        self.W_xh = rng.normal(0.0, iscale, (input_size, hidden_size))
        self.W_hh = 0.90 * np.eye(hidden_size) + rng.normal(
            0.0, 0.01 / math.sqrt(hidden_size), (hidden_size, hidden_size)
        )
        self.b_h = np.zeros(hidden_size)
        self.W_hy = rng.normal(0.0, oscale, (hidden_size, 2))
        self.b_y = np.zeros(2)

    def parameters(self) -> Dict[str, np.ndarray]:
        return {n: getattr(self, n) for n in self.parameter_names}

    def zero_state(self, batch: int) -> np.ndarray:
        return np.zeros((batch, self.hidden_size))

    def step(self, x_t: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)

    def probs(self, h: np.ndarray) -> np.ndarray:
        return _softmax(h @ self.W_hy + self.b_y)

    def loss_and_gradients(
        self, x: np.ndarray, targets: np.ndarray, h0: np.ndarray, weight_decay: float = 0.0
    ) -> Tuple[float, Dict[str, np.ndarray], np.ndarray]:
        bsz, t_len, _ = x.shape
        hidden = np.empty((bsz, t_len, self.hidden_size))
        logits = np.empty((bsz, t_len, 2))
        h = h0
        for t in range(t_len):
            h = self.step(x[:, t], h)
            hidden[:, t] = h
            logits[:, t] = h @ self.W_hy + self.b_y
        probs = _softmax(logits)
        rows, times = np.nonzero(targets >= 0)
        n = len(rows)
        if n == 0:
            raise ValueError("no response targets")
        yt = targets[rows, times]
        loss = float(-np.mean(np.log(probs[rows, times, yt] + 1e-12)))
        dlog = np.zeros_like(logits)
        dlog[rows, times] = probs[rows, times]
        dlog[rows, times, yt] -= 1.0
        dlog /= n
        grads = {n: np.zeros_like(v) for n, v in self.parameters().items()}
        grads["W_hy"] = hidden.reshape(bsz * t_len, -1).T @ dlog.reshape(bsz * t_len, 2)
        grads["b_y"] = dlog.reshape(bsz * t_len, 2).sum(0)
        dh_next = np.zeros((bsz, self.hidden_size))
        for t in range(t_len - 1, -1, -1):
            dh = dlog[:, t] @ self.W_hy.T + dh_next
            dpre = dh * (1.0 - hidden[:, t] ** 2)
            hp = h0 if t == 0 else hidden[:, t - 1]
            grads["W_xh"] += x[:, t].T @ dpre
            grads["W_hh"] += hp.T @ dpre
            grads["b_h"] += dpre.sum(0)
            dh_next = dpre @ self.W_hh.T
        if weight_decay:
            for n in ("W_xh", "W_hh", "W_hy"):
                loss += 0.5 * weight_decay * float(np.sum(getattr(self, n) ** 2))
                grads[n] += weight_decay * getattr(self, n)
        return loss, grads, hidden[:, -1].copy()

    def save(self, path, metadata=None) -> None:
        payload = {n: v for n, v in self.parameters().items()}
        payload["input_size"] = np.asarray(self.input_size)
        payload["hidden_size"] = np.asarray(self.hidden_size)
        import json
        from pathlib import Path

        payload["metadata_json"] = np.asarray(json.dumps(metadata or {}))
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def load(cls, path) -> "TanhRNN":
        data = np.load(path, allow_pickle=False)
        m = cls.__new__(cls)
        m.input_size = int(data["input_size"])
        m.hidden_size = int(data["hidden_size"])
        for n in cls.parameter_names:
            setattr(m, n, data[n].copy())
        return m


class GRURNN:
    """NumPy GRU (Shrijana), adapted to v2 inputs."""

    parameter_names = (
        "W_xz", "W_hz", "b_z",
        "W_xr", "W_hr", "b_r",
        "W_xh", "W_hh", "b_h",
        "W_hy", "b_y",
    )

    def __init__(self, hidden_size: int, rng: np.random.Generator, input_size: int = N_INPUTS):
        self.input_size = input_size
        self.hidden_size = hidden_size
        s = 1.0 / math.sqrt(hidden_size)

        def w_in():
            return rng.normal(0.0, s, (input_size, hidden_size))

        def w_h():
            return rng.normal(0.0, s, (hidden_size, hidden_size))

        self.W_xz, self.W_hz, self.b_z = w_in(), w_h(), np.zeros(hidden_size)
        self.W_xr, self.W_hr, self.b_r = w_in(), w_h(), np.zeros(hidden_size)
        self.W_xh, self.W_hh, self.b_h = w_in(), w_h(), np.zeros(hidden_size)
        oscale = math.sqrt(2.0 / (hidden_size + 2))
        self.W_hy = rng.normal(0.0, oscale, (hidden_size, 2))
        self.b_y = np.zeros(2)

    def parameters(self) -> Dict[str, np.ndarray]:
        return {n: getattr(self, n) for n in self.parameter_names}

    def zero_state(self, batch: int) -> np.ndarray:
        return np.zeros((batch, self.hidden_size))

    def step(self, x_t: np.ndarray, h: np.ndarray) -> np.ndarray:
        z = 1.0 / (1.0 + np.exp(-(x_t @ self.W_xz + h @ self.W_hz + self.b_z)))
        r = 1.0 / (1.0 + np.exp(-(x_t @ self.W_xr + h @ self.W_hr + self.b_r)))
        h_hat = np.tanh(x_t @ self.W_xh + (r * h) @ self.W_hh + self.b_h)
        return (1.0 - z) * h + z * h_hat

    def probs(self, h: np.ndarray) -> np.ndarray:
        return _softmax(h @ self.W_hy + self.b_y)

    def loss_and_gradients(
        self, x: np.ndarray, targets: np.ndarray, h0: np.ndarray, weight_decay: float = 0.0
    ) -> Tuple[float, Dict[str, np.ndarray], np.ndarray]:
        # Forward storing gates for BPTT
        bsz, t_len, _ = x.shape
        h_seq = np.empty((bsz, t_len, self.hidden_size))
        z_seq = np.empty_like(h_seq)
        r_seq = np.empty_like(h_seq)
        hh_seq = np.empty_like(h_seq)
        logits = np.empty((bsz, t_len, 2))
        h = h0
        for t in range(t_len):
            xt = x[:, t]
            z = 1.0 / (1.0 + np.exp(-(xt @ self.W_xz + h @ self.W_hz + self.b_z)))
            r = 1.0 / (1.0 + np.exp(-(xt @ self.W_xr + h @ self.W_hr + self.b_r)))
            h_hat = np.tanh(xt @ self.W_xh + (r * h) @ self.W_hh + self.b_h)
            h_new = (1.0 - z) * h + z * h_hat
            z_seq[:, t], r_seq[:, t], hh_seq[:, t], h_seq[:, t] = z, r, h_hat, h_new
            logits[:, t] = h_new @ self.W_hy + self.b_y
            h = h_new
        probs = _softmax(logits)
        rows, times = np.nonzero(targets >= 0)
        n = len(rows)
        yt = targets[rows, times]
        loss = float(-np.mean(np.log(probs[rows, times, yt] + 1e-12)))
        dlog = np.zeros_like(logits)
        dlog[rows, times] = probs[rows, times]
        dlog[rows, times, yt] -= 1.0
        dlog /= n
        grads = {n: np.zeros_like(v) for n, v in self.parameters().items()}
        grads["W_hy"] = h_seq.reshape(bsz * t_len, -1).T @ dlog.reshape(bsz * t_len, 2)
        grads["b_y"] = dlog.reshape(bsz * t_len, 2).sum(0)
        dh_next = np.zeros((bsz, self.hidden_size))
        for t in range(t_len - 1, -1, -1):
            xt = x[:, t]
            h_prev = h0 if t == 0 else h_seq[:, t - 1]
            z, r, h_hat, h_t = z_seq[:, t], r_seq[:, t], hh_seq[:, t], h_seq[:, t]
            dh = dlog[:, t] @ self.W_hy.T + dh_next
            # h = (1-z)*h_prev + z*h_hat
            dz = dh * (h_hat - h_prev)
            dh_hat = dh * z
            dh_prev = dh * (1.0 - z)
            d_hhat_pre = dh_hat * (1.0 - h_hat**2)
            grads["W_xh"] += xt.T @ d_hhat_pre
            grads["W_hh"] += (r * h_prev).T @ d_hhat_pre
            grads["b_h"] += d_hhat_pre.sum(0)
            dr_h = (d_hhat_pre @ self.W_hh.T) * h_prev
            dh_prev = dh_prev + (d_hhat_pre @ self.W_hh.T) * r
            dr = dr_h
            dz_pre = dz * z * (1.0 - z)
            dr_pre = dr * r * (1.0 - r)
            grads["W_xz"] += xt.T @ dz_pre
            grads["W_hz"] += h_prev.T @ dz_pre
            grads["b_z"] += dz_pre.sum(0)
            grads["W_xr"] += xt.T @ dr_pre
            grads["W_hr"] += h_prev.T @ dr_pre
            grads["b_r"] += dr_pre.sum(0)
            dh_next = dh_prev + dz_pre @ self.W_hz.T + dr_pre @ self.W_hr.T
        if weight_decay:
            for n in self.parameter_names:
                if n.startswith("W_"):
                    loss += 0.5 * weight_decay * float(np.sum(getattr(self, n) ** 2))
                    grads[n] += weight_decay * getattr(self, n)
        return loss, grads, h_seq[:, -1].copy()

    def save(self, path, metadata=None) -> None:
        import json
        from pathlib import Path

        payload = {n: v for n, v in self.parameters().items()}
        payload["input_size"] = np.asarray(self.input_size)
        payload["hidden_size"] = np.asarray(self.hidden_size)
        payload["metadata_json"] = np.asarray(json.dumps(metadata or {}))
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def load(cls, path) -> "GRURNN":
        data = np.load(path, allow_pickle=False)
        m = cls.__new__(cls)
        m.input_size = int(data["input_size"])
        m.hidden_size = int(data["hidden_size"])
        for n in cls.parameter_names:
            setattr(m, n, data[n].copy())
        return m
