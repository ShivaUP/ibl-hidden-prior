#!/usr/bin/env python3
"""04 — Fit empirical synthetic-task stats from behavior-core trials.

Writes:
  data/manifests/synthetic_stats_v2.json
  configs/synthetic_v2.yaml  (merged defaults + fitted values)

Requires: scripts/03_build_processed_trials.py and scripts/02_audit_event_deltas.py

Usage:
  python scripts/04_fit_synthetic_stats.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _block_runs(probability_left: np.ndarray) -> list[tuple[float, int]]:
    runs: list[tuple[float, int]] = []
    i = 0
    n = len(probability_left)
    while i < n:
        j = i + 1
        while j < n and probability_left[j] == probability_left[i]:
            j += 1
        runs.append((float(probability_left[i]), j - i))
        i = j
    return runs


def fit_stats(trials: pd.DataFrame, event_deltas: dict) -> dict:
    # IBL probabilityLeft -> generative p_right = 1 - probabilityLeft
    block_lengths: list[int] = []
    block_priors_pleft: list[float] = []
    transitions: dict[str, int] = {}
    session_start_pleft: list[float] = []
    trials_per_session: list[int] = []

    for eid, g in trials.groupby("eid", sort=False):
        g = g.sort_values("trial_index")
        p = g["probabilityLeft"].to_numpy(dtype=float)
        trials_per_session.append(int(len(g)))
        runs = _block_runs(p)
        session_start_pleft.append(runs[0][0])
        for prior, length in runs:
            block_lengths.append(int(length))
            block_priors_pleft.append(float(prior))
        seq = [round(prior, 4) for prior, _ in runs]
        for a, b in zip(seq[:-1], seq[1:]):
            key = f"{a}->{b}"
            transitions[key] = transitions.get(key, 0) + 1

    contrast_levels = sorted({round(float(x), 4) for x in trials["abs_contrast"].unique()})
    contrast_p = (
        trials["abs_contrast"]
        .round(4)
        .value_counts(normalize=True)
        .reindex(contrast_levels)
        .fillna(0.0)
        .tolist()
    )

    pooled = event_deltas["pooled"]
    bin_s = 0.1

    def ticks_from_median(seconds: float, minimum: int = 1) -> int:
        return max(minimum, int(round(float(seconds) / bin_s)))

    go_stim = float(pooled["go_minus_stim"]["median"])
    resp_go = float(pooled["resp_minus_go"]["median"])
    fb_resp = float(pooled["fb_minus_resp"]["median"])
    off_stim = float(pooled["off_minus_stim"]["median"])

    # 100 ms resolution: go almost always in same bin as stim
    go_offset_ticks = 0 if go_stim < 0.05 else ticks_from_median(go_stim, 0)
    response_from_go_ticks = ticks_from_median(resp_go, 1)
    feedback_ticks = 2 if fb_resp < 0.05 else ticks_from_median(fb_resp, 1)
    stim_duration_ticks = ticks_from_median(off_stim, 1)

    # Transition matrix over rounded p_left keys
    keys = sorted({float(k.split("->")[0]) for k in transitions} | {float(k.split("->")[1]) for k in transitions})
    # Also include 0.2,0.5,0.8 always
    for v in (0.2, 0.5, 0.8):
        if v not in keys:
            keys.append(v)
    keys = sorted(set(round(k, 4) for k in keys))
    mat = {f"{a}": {f"{b}": 0.0 for b in keys} for a in keys}
    for k, c in transitions.items():
        a, b = k.split("->")
        a, b = str(round(float(a), 4)), str(round(float(b), 4))
        if a in mat and b in mat[a]:
            mat[a][b] += float(c)
    # row-normalize; if a row is empty, uniform to others
    for a in mat:
        s = sum(mat[a].values())
        if s <= 0:
            others = [b for b in mat[a] if b != a]
            if not others:
                mat[a][a] = 1.0
            else:
                for b in others:
                    mat[a][b] = 1.0 / len(others)
        else:
            for b in mat[a]:
                mat[a][b] /= s

    start_counts = pd.Series(session_start_pleft).round(4).value_counts(normalize=True)
    start_p = {str(round(float(k), 4)): float(v) for k, v in start_counts.items()}
    for v in (0.2, 0.5, 0.8):
        start_p.setdefault(str(v), 0.0)
    # renormalize
    z = sum(start_p.values()) or 1.0
    start_p = {k: v / z for k, v in start_p.items()}

    lengths = np.asarray(block_lengths, dtype=float)
    # Empirical length PMF (clipped)
    length_min, length_max = 10, 100
    clipped = np.clip(lengths, length_min, length_max).astype(int)
    length_vals, length_cnt = np.unique(clipped, return_counts=True)
    length_p = (length_cnt / length_cnt.sum()).tolist()

    tps = np.asarray(trials_per_session, dtype=float)

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_trials": "data/processed/trials/all_trials.parquet",
        "n_trials": int(len(trials)),
        "n_eids": int(trials["eid"].nunique()),
        "probability_left_levels": [0.2, 0.5, 0.8],
        "note_p_right": "Generator samples probabilityLeft then sets p_right = 1 - probabilityLeft",
        "session_start_probability_left": start_p,
        "block_transition_probability_left": mat,
        "block_length": {
            "min": int(length_min),
            "max": int(length_max),
            "values": length_vals.tolist(),
            "probabilities": length_p,
            "empirical_mean": float(lengths.mean()),
            "empirical_median": float(np.median(lengths)),
        },
        "contrast": {
            "levels": contrast_levels,
            "probabilities": contrast_p,
        },
        "trials_per_session": {
            "mean": float(tps.mean()),
            "median": float(np.median(tps)),
            "min": int(tps.min()),
            "max": int(tps.max()),
            "default_synth": int(np.median(tps)),
        },
        "timing_s": {
            "bin_size_s": bin_s,
            "go_minus_stim_median": go_stim,
            "resp_minus_go_median": resp_go,
            "fb_minus_resp_median": fb_resp,
            "off_minus_stim_median": off_stim,
        },
        "phase_ticks": {
            "bin_size_s": bin_s,
            "baseline_ticks": 2,
            "go_offset_from_stim_ticks": int(go_offset_ticks),
            "response_offset_from_go_ticks": int(response_from_go_ticks),
            "feedback_ticks": int(feedback_ticks),
            "stim_duration_ticks": int(stim_duration_ticks),
        },
    }


def main() -> int:
    trials_path = ROOT / "data" / "processed" / "trials" / "all_trials.parquet"
    deltas_path = ROOT / "reports" / "inspection" / "event_deltas.json"
    if not trials_path.exists():
        print(f"Missing {trials_path}; run python scripts/03_build_processed_trials.py first.")
        return 1
    if not deltas_path.exists():
        print(f"Missing {deltas_path}; run python scripts/02_audit_event_deltas.py first.")
        return 1
    trials = pd.read_parquet(trials_path)
    event_deltas = json.loads(deltas_path.read_text(encoding="utf-8"))
    stats = fit_stats(trials, event_deltas)

    man = ROOT / "data" / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    stats_path = man / "synthetic_stats_v2.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    cfg = {
        "version": "synthetic_v2",
        "frozen_date": "2026-07-21",
        "spec": "docs/spec_v2.md",
        "stats_path": "data/manifests/synthetic_stats_v2.json",
        "models": ["tanh_bptt", "tanh_pc", "gru", "bayes"],
        "hidden_size": 48,
        "channels": [
            "visual_right",
            "visual_left",
            "go_cue",
            "action_left",
            "action_right",
            "rewarded",
            "not_rewarded",
        ],
        "sensory_noise_std_synth": 0.15,
        "training_feedback_error_rate": 0.20,
        "phase_ticks": stats["phase_ticks"],
        "contrast": stats["contrast"],
        "block_length": {
            "values": stats["block_length"]["values"],
            "probabilities": stats["block_length"]["probabilities"],
        },
        "session_start_probability_left": stats["session_start_probability_left"],
        "block_transition_probability_left": stats["block_transition_probability_left"],
        "trials_per_session_default": stats["trials_per_session"]["default_synth"],
        "train": {
            # Kyan 60×24 schedule; empirical session length (≈929) → ~3.9× Kyan exposure.
            "epochs": 60,
            "sessions_per_epoch": 24,
            "bptt_trials": 32,
            "learning_rate": 0.002,
            "pc_epochs": 60,
            "pc_trials_per_session": 240,
            "pc_synaptic_learning_rate": 0.0004,
            "pc_inference_steps": 8,
            "pc_inference_learning_rate": 0.15,
            "weight_decay": 1.0e-5,
            "gradient_clip_norm": 1.0,
            "seed": 7,
        },
        "eval": {
            "synth_sessions": 48,
            "synth_trials": int(stats["trials_per_session"]["default_synth"]),
            "seed": 10007,
            "regimes": ["history_only", "full_information", "fixed_prior"],
            "fi_oracle_logit_gain": 2.5,
        },
        "paths": {
            "artifacts": "artifacts/v2",
            "reports": "reports/v2",
            "figures": "reports/v2/figures",
        },
    }
    cfg_path = ROOT / "configs" / "synthetic_v2.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Wrote {stats_path}")
    print(f"Wrote {cfg_path}")
    print("phase_ticks:", json.dumps(stats["phase_ticks"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
