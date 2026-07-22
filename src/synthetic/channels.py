"""Channel layout and within-trial painting for empirical phase ticks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.synthetic.schema import LEFT, RIGHT

CHANNEL_NAMES = (
    "visual_right",
    "visual_left",
    "go_cue",
    "action_left",
    "action_right",
    "rewarded",
    "not_rewarded",
)
N_INPUTS = len(CHANNEL_NAMES)

VISUAL_RIGHT = 0
VISUAL_LEFT = 1
GO_CUE = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
REWARDED = 5
NOT_REWARDED = 6


@dataclass(frozen=True)
class PhaseTicks:
    baseline_ticks: int = 2
    go_offset_from_stim_ticks: int = 0
    response_offset_from_go_ticks: int = 4
    feedback_ticks: int = 2
    stim_duration_ticks: int = 15
    bin_size_s: float = 0.1

    @classmethod
    def from_config(cls, cfg: dict) -> "PhaseTicks":
        p = cfg["phase_ticks"]
        return cls(
            baseline_ticks=int(p["baseline_ticks"]),
            go_offset_from_stim_ticks=int(p["go_offset_from_stim_ticks"]),
            response_offset_from_go_ticks=int(p["response_offset_from_go_ticks"]),
            feedback_ticks=int(p["feedback_ticks"]),
            stim_duration_ticks=int(p["stim_duration_ticks"]),
            bin_size_s=float(p.get("bin_size_s", 0.1)),
        )

    @property
    def stim_start(self) -> int:
        return self.baseline_ticks

    @property
    def go_tick(self) -> int:
        return self.stim_start + self.go_offset_from_stim_ticks

    @property
    def response_tick(self) -> int:
        return self.go_tick + self.response_offset_from_go_ticks

    @property
    def feedback_start(self) -> int:
        return self.response_tick + 1

    @property
    def n_steps(self) -> int:
        return self.feedback_start + self.feedback_ticks

    @property
    def stim_end_exclusive(self) -> int:
        return min(self.stim_start + self.stim_duration_ticks, self.n_steps)


def paint_trial(
    *,
    side: int,
    contrast: float,
    action: int,
    rewarded: bool,
    phase: PhaseTicks,
    visual_noise: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_steps, n_inputs) inputs and (n_steps,) targets (-1 = no loss)."""

    x = np.zeros((phase.n_steps, N_INPUTS), dtype=np.float64)
    targets = np.full(phase.n_steps, -1, dtype=np.int64)

    # Visual: side × magnitude on stim ticks (exclude pure response tick like Kyan)
    vr = float(contrast) if side == RIGHT else 0.0
    vl = float(contrast) if side == LEFT else 0.0
    if visual_noise is not None:
        vr = vr + float(visual_noise[0])
        vl = vl + float(visual_noise[1])
    for t in range(phase.stim_start, phase.stim_end_exclusive):
        if t == phase.response_tick:
            continue
        x[t, VISUAL_RIGHT] = vr
        x[t, VISUAL_LEFT] = vl

    x[phase.go_tick, GO_CUE] = 1.0
    targets[phase.response_tick] = int(side)

    for t in range(phase.feedback_start, phase.n_steps):
        x[t, ACTION_LEFT] = 1.0 if action == LEFT else 0.0
        x[t, ACTION_RIGHT] = 1.0 if action == RIGHT else 0.0
        x[t, REWARDED] = 1.0 if rewarded else 0.0
        x[t, NOT_REWARDED] = 0.0 if rewarded else 1.0

    return x, targets
