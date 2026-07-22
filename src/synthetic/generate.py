"""Generate synthetic sessions from empirical v2 stats."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.synthetic.channels import PhaseTicks, paint_trial
from src.synthetic.schema import LEFT, RIGHT


@dataclass
class SyntheticBatch:
    """Sessions × trials arrays plus optional packed sequences."""

    probability_left: np.ndarray  # [S, T]
    p_right: np.ndarray
    block_id: np.ndarray
    side: np.ndarray  # RIGHT=1 means stimulus on right
    contrast: np.ndarray
    phase: PhaseTicks

    @property
    def shape(self) -> tuple[int, int]:
        return self.side.shape


def _sample_from_dict(d: dict[str, float], rng: np.random.Generator) -> float:
    keys = list(d.keys())
    p = np.asarray([d[k] for k in keys], dtype=float)
    p = p / p.sum()
    return float(keys[int(rng.choice(len(keys), p=p))])


def _sample_block_length(cfg: dict, rng: np.random.Generator) -> int:
    vals = np.asarray(cfg["block_length"]["values"], dtype=int)
    p = np.asarray(cfg["block_length"]["probabilities"], dtype=float)
    p = p / p.sum()
    return int(rng.choice(vals, p=p))


def _next_prior(current: float, trans: dict, rng: np.random.Generator) -> float:
    row = trans[str(round(current, 4))]
    return _sample_from_dict(row, rng)


def generate_sessions(
    n_sessions: int,
    n_trials: int,
    cfg: dict,
    rng: np.random.Generator,
) -> SyntheticBatch:
    phase = PhaseTicks.from_config(cfg)
    start_p = cfg["session_start_probability_left"]
    trans = cfg["block_transition_probability_left"]
    contrast_levels = np.asarray(cfg["contrast"]["levels"], dtype=float)
    contrast_p = np.asarray(cfg["contrast"]["probabilities"], dtype=float)
    contrast_p = contrast_p / contrast_p.sum()

    pleft = np.empty((n_sessions, n_trials), dtype=np.float64)
    block_id = np.empty((n_sessions, n_trials), dtype=np.int64)

    for s in range(n_sessions):
        cursor = 0
        bid = 0
        prior = _sample_from_dict(start_p, rng)
        while cursor < n_trials:
            length = _sample_block_length(cfg, rng)
            end = min(cursor + length, n_trials)
            pleft[s, cursor:end] = prior
            block_id[s, cursor:end] = bid
            cursor = end
            bid += 1
            if cursor < n_trials:
                prior = _next_prior(prior, trans, rng)

    p_right = 1.0 - pleft
    side = (rng.random((n_sessions, n_trials)) < p_right).astype(np.int64)
    contrast = rng.choice(contrast_levels, size=(n_sessions, n_trials), p=contrast_p)

    return SyntheticBatch(
        probability_left=pleft,
        p_right=p_right,
        block_id=block_id,
        side=side,
        contrast=contrast,
        phase=phase,
    )


def build_training_tensors(
    batch: SyntheticBatch,
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forced sequences: [S, T_steps_total, C], targets [S, T_steps_total]."""

    n_sessions, n_trials = batch.shape
    phase = batch.phase
    err_rate = float(cfg.get("training_feedback_error_rate", 0.2))
    noise_std = float(cfg.get("sensory_noise_std_synth", 0.15))
    n_steps = phase.n_steps
    from src.synthetic.channels import N_INPUTS

    x = np.zeros((n_sessions, n_trials * n_steps, N_INPUTS), dtype=np.float64)
    targets = np.full((n_sessions, n_trials * n_steps), -1, dtype=np.int64)

    for s in range(n_sessions):
        for t in range(n_trials):
            side = int(batch.side[s, t])
            c = float(batch.contrast[s, t])
            make_err = rng.random() < err_rate
            action = (1 - side) if make_err else side
            rewarded = action == side
            noise = rng.normal(0.0, noise_std, size=2) if noise_std > 0 else None
            trial_x, trial_y = paint_trial(
                side=side,
                contrast=c,
                action=action,
                rewarded=rewarded,
                phase=phase,
                visual_noise=noise,
            )
            sl = slice(t * n_steps, (t + 1) * n_steps)
            x[s, sl] = trial_x
            targets[s, sl] = trial_y
    return x, targets
