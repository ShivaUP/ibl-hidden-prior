"""Tests for synthetic v2 generator and painting."""

from __future__ import annotations

import numpy as np

from src.synthetic.channels import PhaseTicks, paint_trial
from src.synthetic.generate import build_training_tensors, generate_sessions
from src.synthetic.schema import LEFT, RIGHT


def _mini_cfg() -> dict:
    return {
        "phase_ticks": {
            "bin_size_s": 0.1,
            "baseline_ticks": 2,
            "go_offset_from_stim_ticks": 0,
            "response_offset_from_go_ticks": 4,
            "feedback_ticks": 2,
            "stim_duration_ticks": 10,
        },
        "contrast": {"levels": [0.0, 0.25, 1.0], "probabilities": [0.3, 0.4, 0.3]},
        "block_length": {"values": [20, 30], "probabilities": [0.5, 0.5]},
        "session_start_probability_left": {"0.2": 0.5, "0.8": 0.5, "0.5": 0.0},
        "block_transition_probability_left": {
            "0.2": {"0.2": 0.0, "0.5": 0.0, "0.8": 1.0},
            "0.5": {"0.2": 0.5, "0.5": 0.0, "0.8": 0.5},
            "0.8": {"0.2": 1.0, "0.5": 0.0, "0.8": 0.0},
        },
        "sensory_noise_std_synth": 0.0,
        "training_feedback_error_rate": 0.0,
    }


def test_paint_contrast_side_magnitude():
    phase = PhaseTicks(2, 0, 4, 2, 10)
    x, y = paint_trial(side=RIGHT, contrast=0.25, action=RIGHT, rewarded=True, phase=phase)
    assert x.shape == (phase.n_steps, 7)
    assert y[phase.response_tick] == RIGHT
    # stim tick has [c,0]
    assert x[phase.stim_start, 0] == 0.25
    assert x[phase.stim_start, 1] == 0.0
    x_l, _ = paint_trial(side=LEFT, contrast=0.125, action=LEFT, rewarded=True, phase=phase)
    assert x_l[phase.stim_start, 0] == 0.0
    assert x_l[phase.stim_start, 1] == 0.125


def test_generate_sessions_shapes():
    cfg = _mini_cfg()
    rng = np.random.default_rng(0)
    batch = generate_sessions(3, 50, cfg, rng)
    assert batch.side.shape == (3, 50)
    assert set(np.unique(batch.probability_left)).issubset({0.2, 0.5, 0.8})
    x, y = build_training_tensors(batch, cfg, rng)
    assert x.shape[0] == 3
    assert x.shape[2] == 7
    assert (y >= 0).sum() == 3 * 50  # one response target per trial
