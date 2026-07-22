"""Map real IBL processed trials into the shared v2 tick schema."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.synthetic.channels import PhaseTicks, paint_trial
from src.synthetic.schema import LEFT, RIGHT


def row_to_side_contrast(row: pd.Series) -> tuple[int, float]:
    """stimulus_right / abs_contrast from processed trials."""
    side = RIGHT if int(row["stimulus_right"]) == 1 else LEFT
    contrast = float(row["abs_contrast"])
    return side, contrast


def encode_real_session(
    trials: pd.DataFrame,
    phase: PhaseTicks,
) -> dict[str, np.ndarray]:
    """Encode one eid's QC trials into concatenated tick sequences.

    Feedback uses the mouse's actual choice and outcome (transfer protocol).
    """
    g = trials.sort_values("trial_index").reset_index(drop=True)
    n = len(g)
    n_steps = phase.n_steps
    from src.synthetic.channels import N_INPUTS

    x = np.zeros((n * n_steps, N_INPUTS), dtype=np.float64)
    targets_correct = np.full(n * n_steps, -1, dtype=np.int64)
    mouse_choice = np.empty(n, dtype=np.int64)
    correct_side = np.empty(n, dtype=np.int64)
    contrast = np.empty(n, dtype=np.float64)
    pleft = np.empty(n, dtype=np.float64)
    trial_index = g["trial_index"].to_numpy(dtype=np.int64)

    for i, row in g.iterrows():
        side, c = row_to_side_contrast(row)
        # mouse choice_right: 1 = right
        mouse = RIGHT if int(row["choice_right"]) == 1 else LEFT
        rewarded = bool(int(row["reward"]) == 1) or (mouse == side)
        # Prefer explicit reward channel if present
        if "reward" in row.index:
            rewarded = int(row["reward"]) == 1
        trial_x, trial_y = paint_trial(
            side=side,
            contrast=c,
            action=mouse,
            rewarded=rewarded,
            phase=phase,
            visual_noise=None,
        )
        sl = slice(i * n_steps, (i + 1) * n_steps)
        x[sl] = trial_x
        targets_correct[sl] = trial_y
        mouse_choice[i] = mouse
        correct_side[i] = side
        contrast[i] = c
        pleft[i] = float(row["probabilityLeft"])

    return {
        "inputs": x,
        "targets_correct_side": targets_correct,
        "mouse_choice": mouse_choice,
        "correct_side": correct_side,
        "contrast": contrast,
        "probability_left": pleft,
        "trial_index": trial_index,
        "n_trials": n,
        "n_steps": n_steps,
        "response_tick": phase.response_tick,
    }
