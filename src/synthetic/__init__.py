"""v2 synthetic IBL-like task: empirical blocks + phase-tick trials."""

from __future__ import annotations

from src.synthetic.channels import CHANNEL_NAMES, N_INPUTS, PhaseTicks, paint_trial
from src.synthetic.generate import SyntheticBatch, generate_sessions
from src.synthetic.schema import LEFT, RIGHT, load_synthetic_config

__all__ = [
    "CHANNEL_NAMES",
    "N_INPUTS",
    "PhaseTicks",
    "paint_trial",
    "SyntheticBatch",
    "generate_sessions",
    "LEFT",
    "RIGHT",
    "load_synthetic_config",
]
