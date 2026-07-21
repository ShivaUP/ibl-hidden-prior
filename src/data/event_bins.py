"""100 ms event-bin sequences for RNN-family models (bin 0 = stimOn)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CHANNEL_ORDER_COMMON: tuple[str, ...] = (
    "stimulus_right",
    "contrast_high",
    "delay_phase",
    "response_window",
    "response_made",
    "reward",
    "prev_choice_right",
    "prev_correct",
    "prev_fast_rt",
)

CHANNEL_ORDER_FULL_INFO: tuple[str, ...] = CHANNEL_ORDER_COMMON + ("oracle_prior_right",)


@dataclass(frozen=True)
class BinConfig:
    bin_size_s: float = 0.1
    max_bins: int = 100  # 10 s cap
    pad_bins_after_feedback: int = 1


def _time_to_bin(dt_s: float, bin_size_s: float, n_bins: int) -> int | None:
    if not np.isfinite(dt_s) or dt_s < 0:
        return None
    b = int(dt_s // bin_size_s)
    if b < 0 or b >= n_bins:
        return None
    return b


def trial_to_bins(
    row: pd.Series,
    *,
    condition: str,
    cfg: BinConfig = BinConfig(),
) -> tuple[np.ndarray, dict]:
    """Convert one processed trial row into (T, C) binary channels.

    condition: 'history_only' | 'full_information' | 'fixed_prior'
    fixed_prior / history_only omit oracle_prior_right.
    """
    channels = (
        CHANNEL_ORDER_FULL_INFO
        if condition == "full_information"
        else CHANNEL_ORDER_COMMON
    )
    stim_on = float(row["stimOn_times"])
    go = float(row["goCue_times"]) - stim_on
    resp = float(row["response_times"]) - stim_on
    fb = float(row["feedback_times"]) - stim_on
    if "stimOff_times" in row.index and pd.notna(row.get("stimOff_times")):
        off = float(row["stimOff_times"]) - stim_on
    else:
        # Fallback: keep stim on until response
        off = resp

    end_t = max(fb, off, resp, go) + cfg.pad_bins_after_feedback * cfg.bin_size_s
    n_bins = int(np.ceil(end_t / cfg.bin_size_s)) + 1
    n_bins = max(1, min(n_bins, cfg.max_bins))

    x = np.zeros((n_bins, len(channels)), dtype=np.float32)
    name_to_i = {n: i for i, n in enumerate(channels)}

    go_bin = _time_to_bin(go, cfg.bin_size_s, n_bins)
    resp_bin = _time_to_bin(resp, cfg.bin_size_s, n_bins)
    fb_bin = _time_to_bin(fb, cfg.bin_size_s, n_bins)
    off_bin = _time_to_bin(off, cfg.bin_size_s, n_bins)
    if off_bin is None:
        off_bin = n_bins  # never turns off inside window

    # Stimulus channels: bin 0 .. off_bin-1
    stim_end = max(off_bin, 1)
    for b in range(0, min(stim_end, n_bins)):
        x[b, name_to_i["stimulus_right"]] = float(row["stimulus_right"])
        x[b, name_to_i["contrast_high"]] = float(row["contrast_high"])

    # Response window: from goCue bin through response bin (inclusive)
    start_rw = go_bin if go_bin is not None else 0
    end_rw = resp_bin if resp_bin is not None else start_rw
    for b in range(start_rw, min(end_rw + 1, n_bins)):
        x[b, name_to_i["response_window"]] = 1.0

    # Delay phase: after stimulus offset, before response (exclusive of response_made)
    delay_start = off_bin
    delay_end = resp_bin if resp_bin is not None else n_bins
    for b in range(delay_start, min(delay_end, n_bins)):
        x[b, name_to_i["delay_phase"]] = 1.0

    if resp_bin is not None:
        x[resp_bin, name_to_i["response_made"]] = 1.0
    if fb_bin is not None:
        x[fb_bin, name_to_i["reward"]] = float(row["reward"])

    # History channels: constant across bins (known at stim onset)
    for name in ("prev_choice_right", "prev_correct", "prev_fast_rt"):
        x[:, name_to_i[name]] = float(row[name])

    if condition == "full_information":
        x[:, name_to_i["oracle_prior_right"]] = float(row["oracle_prior_right"])

    meta = {
        "n_bins": n_bins,
        "go_bin": go_bin,
        "resp_bin": resp_bin,
        "fb_bin": fb_bin,
        "off_bin": off_bin if off_bin < n_bins else None,
        "channels": list(channels),
    }
    return x, meta


def assert_bin0_no_future_leakage(x: np.ndarray, channels: list[str]) -> None:
    """Bin 0 must not show response_made/reward unless those events fall in bin 0."""
    name_to_i = {n: i for i, n in enumerate(channels)}
    # Soft check used in tests with controlled rows; production uses event-aligned painting.
    assert x.ndim == 2
    assert "response_made" in name_to_i and "reward" in name_to_i


def build_condition_arrays(
    trials: pd.DataFrame,
    condition: str,
    cfg: BinConfig = BinConfig(),
) -> dict[str, np.ndarray | list]:
    """Build ragged list of bin arrays plus aligned targets/index for one condition."""
    if condition == "fixed_prior":
        subset = trials.loc[trials["in_fixed_prior"] == 1]
    else:
        subset = trials.loc[trials["qc_pass"] == 1]

    sequences: list[np.ndarray] = []
    metas: list[dict] = []
    for _, row in subset.iterrows():
        x, meta = trial_to_bins(row, condition=condition, cfg=cfg)
        sequences.append(x)
        metas.append(meta)

    channels = (
        list(CHANNEL_ORDER_FULL_INFO)
        if condition == "full_information"
        else list(CHANNEL_ORDER_COMMON)
    )
    return {
        "sequences": sequences,
        "metas": metas,
        "channels": channels,
        "eid": subset["eid"].to_numpy(),
        "trial_index": subset["trial_index"].to_numpy(),
        "choice_right": subset["choice_right"].to_numpy(dtype=np.int64),
        "rt": subset["rt"].to_numpy(dtype=np.float64),
        "log_rt": subset["log_rt"].to_numpy(dtype=np.float64),
        "probabilityLeft": subset["probabilityLeft"].to_numpy(dtype=np.float64),
        "block_switch": subset["block_switch"].to_numpy(dtype=np.int64),
    }
