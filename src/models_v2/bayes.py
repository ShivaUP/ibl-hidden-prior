"""Explicit online Bayes prior + stimulus readout (NumPy, v2 tick interface)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from src.models_v2.rnn_cells import Adam, _softmax
from src.synthetic.channels import (
    ACTION_LEFT,
    ACTION_RIGHT,
    N_INPUTS,
    NOT_REWARDED,
    REWARDED,
    VISUAL_LEFT,
    VISUAL_RIGHT,
)
from src.synthetic.schema import LEFT, RIGHT


class ExplicitBayes:
    """Online leaky prior + linear stimulus evidence → 2-way logits.

    Capacity is intentionally smaller than hidden-48 RNNs (documented fairness risk
    V2-R2). Parameters are trained by BPTT through the leaky update.
    """

    parameter_names = (
        "prior_logit0",
        "leak_raw",
        "w_evidence",
        "b_evidence",
        "w_stim",
        "b_stim",
        "prior_scale",
        "W_hy",
        "b_y",
    )

    def __init__(self, rng: np.random.Generator, input_size: int = N_INPUTS):
        self.input_size = input_size
        self.prior_logit0 = np.zeros(1)
        self.leak_raw = np.zeros(1)  # sigmoid → (0,1)
        self.w_evidence = rng.normal(0.0, 0.1, (input_size,))
        self.b_evidence = np.zeros(1)
        self.w_stim = rng.normal(0.0, 0.1, (input_size,))
        self.b_stim = np.zeros(1)
        self.prior_scale = np.ones(1)
        self.W_hy = rng.normal(0.0, 0.1, (2, 2))  # [prior_feat, stim_feat] → logits
        self.b_y = np.zeros(2)

    def parameters(self) -> Dict[str, np.ndarray]:
        return {n: getattr(self, n) for n in self.parameter_names}

    def leak(self) -> float:
        return float(1.0 / (1.0 + math.exp(-float(self.leak_raw[0]))))

    def zero_state(self, batch: int) -> np.ndarray:
        """q = P(right), shape (batch,)."""
        q0 = 1.0 / (1.0 + math.exp(-float(self.prior_logit0[0])))
        return np.full(batch, q0, dtype=np.float64)

    def _features(self, x_t: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # evidence nudge from current tick (feedback-heavy)
        e_logit = x_t @ self.w_evidence + self.b_evidence[0]
        nudge = 1.0 / (1.0 + np.exp(-e_logit))
        stim = x_t @ self.w_stim + self.b_stim[0]
        prior_feat = self.prior_scale[0] * (2.0 * q - 1.0)
        feats = np.stack([prior_feat, stim], axis=-1)  # (B, 2)
        return nudge, feats

    def step_prior(self, x_t: np.ndarray, q: np.ndarray) -> np.ndarray:
        leak = self.leak()
        nudge, _ = self._features(x_t, q)
        # Update prior using feedback channels when present; else mild leak to nudge
        has_fb = (x_t[:, REWARDED] + x_t[:, NOT_REWARDED]) > 0.5
        q_new = q.copy()
        # On feedback: hard Bayesian-ish update from action × reward
        act_r = x_t[:, ACTION_RIGHT] > 0.5
        act_l = x_t[:, ACTION_LEFT] > 0.5
        rew = x_t[:, REWARDED] > 0.5
        # If rewarded right action → increase P(right); rewarded left → decrease
        delta = np.zeros_like(q)
        delta = np.where(has_fb & act_r & rew, 1.0, delta)
        delta = np.where(has_fb & act_l & rew, 0.0, delta)
        delta = np.where(has_fb & act_r & ~rew, 0.0, delta)
        delta = np.where(has_fb & act_l & ~rew, 1.0, delta)
        # Soft mix with learned nudge when no clear FB
        target = np.where(has_fb, delta, nudge)
        q_new = (1.0 - leak) * q + leak * target
        return np.clip(q_new, 1e-4, 1.0 - 1e-4)

    def logits_from_state(self, x_t: np.ndarray, q: np.ndarray) -> np.ndarray:
        _, feats = self._features(x_t, q)
        return feats @ self.W_hy + self.b_y

    def probs(self, x_t: np.ndarray, q: np.ndarray) -> np.ndarray:
        return _softmax(self.logits_from_state(x_t, q))

    def loss_and_gradients(
        self, x: np.ndarray, targets: np.ndarray, q0: np.ndarray, weight_decay: float = 0.0
    ) -> Tuple[float, Dict[str, np.ndarray], np.ndarray]:
        """Finite-diff / forward-mode style grads via numeric BP through stored path.

        Uses automatic differentiation via reverse through stored intermediates
        with explicit analytic grads for linear parts + leaky update.
        """
        bsz, t_len, _ = x.shape
        q_seq = np.empty((bsz, t_len), dtype=np.float64)
        nudge_seq = np.empty((bsz, t_len), dtype=np.float64)
        stim_seq = np.empty((bsz, t_len), dtype=np.float64)
        feat_seq = np.empty((bsz, t_len, 2), dtype=np.float64)
        logits = np.empty((bsz, t_len, 2), dtype=np.float64)
        q = q0.copy()
        leak = self.leak()
        for t in range(t_len):
            xt = x[:, t]
            e_logit = xt @ self.w_evidence + self.b_evidence[0]
            nudge = 1.0 / (1.0 + np.exp(-e_logit))
            stim = xt @ self.w_stim + self.b_stim[0]
            has_fb = (xt[:, REWARDED] + xt[:, NOT_REWARDED]) > 0.5
            act_r = xt[:, ACTION_RIGHT] > 0.5
            act_l = xt[:, ACTION_LEFT] > 0.5
            rew = xt[:, REWARDED] > 0.5
            delta = np.zeros(bsz)
            delta = np.where(has_fb & act_r & rew, 1.0, delta)
            delta = np.where(has_fb & act_l & rew, 0.0, delta)
            delta = np.where(has_fb & act_r & ~rew, 0.0, delta)
            delta = np.where(has_fb & act_l & ~rew, 1.0, delta)
            target = np.where(has_fb, delta, nudge)
            q = (1.0 - leak) * q + leak * target
            q = np.clip(q, 1e-4, 1.0 - 1e-4)
            prior_feat = self.prior_scale[0] * (2.0 * q - 1.0)
            feats = np.stack([prior_feat, stim], axis=-1)
            logits[:, t] = feats @ self.W_hy + self.b_y
            q_seq[:, t] = q
            nudge_seq[:, t] = nudge
            stim_seq[:, t] = stim
            feat_seq[:, t] = feats

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

        grads = {name: np.zeros_like(v) for name, v in self.parameters().items()}
        # Output layer
        feats_flat = feat_seq.reshape(bsz * t_len, 2)
        dlog_flat = dlog.reshape(bsz * t_len, 2)
        grads["W_hy"] = feats_flat.T @ dlog_flat
        grads["b_y"] = dlog_flat.sum(0)

        d_feats = dlog @ self.W_hy.T  # (B,T,2)
        d_prior_feat = d_feats[:, :, 0]
        d_stim = d_feats[:, :, 1]
        grads["prior_scale"][0] = float(
            np.sum(d_prior_feat * (2.0 * q_seq - 1.0))
        )
        # d_stim / dw_stim
        for t in range(t_len):
            grads["w_stim"] += x[:, t].T @ d_stim[:, t]
            grads["b_stim"][0] += float(d_stim[:, t].sum())

        # Backprop through leaky q (only prior_feat depends on q)
        dq_next = np.zeros(bsz)
        # Softplus-like: prior_feat = scale * (2q-1) → dq from d_prior_feat
        for t in range(t_len - 1, -1, -1):
            dq = d_prior_feat[:, t] * (2.0 * self.prior_scale[0]) + dq_next
            xt = x[:, t]
            has_fb = (xt[:, REWARDED] + xt[:, NOT_REWARDED]) > 0.5
            # q_t = (1-leak)*q_{t-1} + leak*target; target=nudge if ~fb
            # Grad w.r.t. nudge only when ~has_fb
            dnudge = dq * leak * (~has_fb).astype(float)
            # sigmoid nudge
            nudge = nudge_seq[:, t]
            de = dnudge * nudge * (1.0 - nudge)
            grads["w_evidence"] += xt.T @ de
            grads["b_evidence"][0] += float(de.sum())
            # leak grad (scalar shared)
            q_prev = q0 if t == 0 else q_seq[:, t - 1]
            target = (
                np.where(
                    has_fb,
                    # reconstruct target from q update: approximate
                    (q_seq[:, t] - (1.0 - leak) * q_prev) / max(leak, 1e-8),
                    nudge,
                )
            )
            d_leak = float(np.sum(dq * (target - q_prev)))
            # d sigmoid(leak_raw)/d leak_raw = leak*(1-leak)
            grads["leak_raw"][0] += d_leak * leak * (1.0 - leak)
            dq_next = dq * (1.0 - leak)

        # prior_logit0: only affects q0
        q0_val = 1.0 / (1.0 + math.exp(-float(self.prior_logit0[0])))
        # Approximate: average dq into first step already in dq_next after t=0 loop
        # Recompute contribution to q0
        grads["prior_logit0"][0] += float(
            np.sum(dq_next) * q0_val * (1.0 - q0_val)
        )

        if weight_decay:
            for n in ("w_evidence", "w_stim", "W_hy"):
                loss += 0.5 * weight_decay * float(np.sum(getattr(self, n) ** 2))
                grads[n] += weight_decay * getattr(self, n)
        return loss, grads, q_seq[:, -1].copy()

    def save(self, path, metadata=None) -> None:
        payload = {n: v for n, v in self.parameters().items()}
        payload["input_size"] = np.asarray(self.input_size)
        payload["metadata_json"] = np.asarray(json.dumps(metadata or {}))
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def load(cls, path) -> "ExplicitBayes":
        data = np.load(path, allow_pickle=False)
        m = cls.__new__(cls)
        m.input_size = int(data["input_size"])
        for n in cls.parameter_names:
            setattr(m, n, data[n].copy())
        return m
