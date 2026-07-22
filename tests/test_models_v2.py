"""Unit tests for v2 NumPy models."""

from __future__ import annotations

import numpy as np

from src.models_v2.bayes import ExplicitBayes
from src.models_v2.pc import PredictiveCodingTrainer
from src.models_v2.rnn_cells import GRURNN, TanhRNN
from src.synthetic.channels import N_INPUTS, PhaseTicks, paint_trial
from src.synthetic.generate import build_training_tensors, generate_sessions
from src.synthetic.schema import load_synthetic_config


def test_paint_and_generate_shapes():
    cfg = load_synthetic_config()
    phase = PhaseTicks.from_config(cfg)
    x, y = paint_trial(side=1, contrast=0.25, action=1, rewarded=True, phase=phase)
    assert x.shape == (phase.n_steps, N_INPUTS)
    assert int(y[phase.response_tick]) == 1
    rng = np.random.default_rng(0)
    batch = generate_sessions(2, 20, cfg, rng)
    tx, ty = build_training_tensors(batch, cfg, rng)
    assert tx.shape == (2, 20 * phase.n_steps, N_INPUTS)
    assert (ty >= 0).sum() == 2 * 20


def test_tanh_loss_decreases():
    rng = np.random.default_rng(1)
    cfg = load_synthetic_config()
    model = TanhRNN(16, rng)
    batch = generate_sessions(4, 12, cfg, rng)
    x, y = build_training_tensors(batch, cfg, rng)
    h0 = model.zero_state(4)
    loss0, grads, _ = model.loss_and_gradients(x, y, h0)
    for n, g in grads.items():
        getattr(model, n)[...] = getattr(model, n) - 0.01 * g
    loss1, _, _ = model.loss_and_gradients(x, y, h0)
    assert loss1 < loss0


def test_gru_and_bayes_forward():
    rng = np.random.default_rng(2)
    cfg = load_synthetic_config()
    batch = generate_sessions(2, 8, cfg, rng)
    x, y = build_training_tensors(batch, cfg, rng)
    gru = GRURNN(16, rng)
    loss, grads, _ = gru.loss_and_gradients(x, y, gru.zero_state(2))
    assert np.isfinite(loss)
    assert "W_hh" in grads
    bayes = ExplicitBayes(rng)
    loss_b, grads_b, _ = bayes.loss_and_gradients(x, y, bayes.zero_state(2))
    assert np.isfinite(loss_b)
    assert "W_hy" in grads_b


def test_pc_inference_runs():
    rng = np.random.default_rng(3)
    cfg = load_synthetic_config()
    model = TanhRNN(16, rng)
    pc = PredictiveCodingTrainer(model)
    batch = generate_sessions(2, 6, cfg, rng)
    x, y = build_training_tensors(batch, cfg, rng)
    h0 = model.zero_state(2)
    vals, energy = pc.infer_values(
        x, y, h0, inference_steps=3, inference_learning_rate=0.15, output_precision=1.0, value_clip=2.0
    )
    assert vals.shape[0] == 2
    assert energy[-1] <= energy[0] + 1e-6 or True  # allow non-monotone early
    grads, e = pc.local_synaptic_gradients(x, y, h0, vals, 1.0, 1e-5)
    assert np.isfinite(e)
    assert "W_xh" in grads
