#!/usr/bin/env python3
"""16 — Switch-centered block decoding from REAL neural data.

Neural analogue of plot_switch_centered_block_decoding.py (which does this for
the RNN latent trajectories). For each cohort session x ROI:

  1. Bin pre-stimulus spikes per biased trial.
  2. Fit a cross-validated logistic decoder and keep the OUT-OF-FOLD
     P(right-block) for every trial (each trial predicted by a model that
     never saw it).
  3. Align those predictions to genuine 0.2 <-> 0.8 block switches and measure
     balanced hard-decoding success at every trial offset around the switch.

Curves are the mean across cohort sessions per ROI, shading = +/- 1 SEM across
sessions. This shows how quickly the block becomes decodable from each region's
activity after a switch (the neural adaptation curve).

Usage
-----
  conda activate ibl-prior
  python scripts/16_neural_switch_decoding.py                 # cohort, 4 ROIs
  python scripts/16_neural_switch_decoding.py --before 10 --after 20

Output
------
  reports/v2/switch_block_decoding/neural_switch_curves.csv
  reports/v2/figures/switch_block_decoding/neural_switch_block_decoding.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.config import load_frozen_config, repo_root
from src.data.inspect_trials import load_trials_for_eid
from src.eval.block_decoder import BLOCK_LEFT, BLOCK_RIGHT, oof_block_predictions
from src.neural.regions import PRIMARY_ROIS, unit_in_any_decode_region
from src.neural.neural_block_decoder import (
    bin_spikes_per_trial,
    load_session_spikes,
)

ONE_BASE_URL = "https://openalyx.internationalbrainlab.org"
ONE_PASSWORD = "international"

ROI_COLORS = {
    "MOs":   "#3b6ea8",
    "ORBvl": "#8e44ad",
    "ACAd":  "#d9822b",
    "MOp":   "#2f8f6b",
}


def make_one(cache_dir: Path):
    from one.api import ONE
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ONE(base_url=ONE_BASE_URL, password=ONE_PASSWORD, silent=True, cache_dir=str(cache_dir))


def session_switch_windows(prob_right, p_right, before, after):
    """Yield (prob_window, label_window) for isolated 0.2<->0.8 switches.

    prob_right : (T,) out-of-fold P(right-block) per trial, NaN where absent.
    p_right    : (T,) true block P(right) per trial.
    """
    changed = np.flatnonzero(np.diff(p_right) != 0.0) + 1
    for switch in changed:
        prev, cur = float(p_right[switch - 1]), float(p_right[switch])
        genuine = (np.isclose(prev, 0.2) and np.isclose(cur, 0.8)) or \
                  (np.isclose(prev, 0.8) and np.isclose(cur, 0.2))
        if not genuine:
            continue
        start, stop = switch - before, switch + after + 1
        if start < 0 or stop > len(p_right):
            continue
        win_p = p_right[start:stop]
        if not np.all(np.isclose(win_p, 0.2) | np.isclose(win_p, 0.8)):
            continue
        # Keep the window attributable to a single switch.
        if not np.allclose(p_right[start:switch], prev):
            continue
        if not np.allclose(p_right[switch:stop], cur):
            continue
        prob_win = prob_right[start:stop]
        if not np.all(np.isfinite(prob_win)):
            continue
        label_win = (win_p > 0.5).astype(int)
        yield prob_win, label_win


def _balanced_accuracy(labels_col, pred_col):
    vals = []
    for lab in (0, 1):
        sel = labels_col == lab
        if np.any(sel):
            vals.append(float(np.mean(pred_col[sel] == lab)))
    return float(np.mean(vals)) if vals else np.nan


def session_curve(windows, n_offsets):
    """Balanced accuracy per offset for one session's switch windows."""
    if not windows:
        return np.full(n_offsets, np.nan)
    probs = np.stack([w[0] for w in windows])
    labels = np.stack([w[1] for w in windows])
    preds = (probs >= 0.5).astype(int)
    return np.array([_balanced_accuracy(labels[:, j], preds[:, j]) for j in range(n_offsets)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Switch-centered neural block decoding")
    ap.add_argument("--cohort", default="data/manifests/roi_cohort_v2.json")
    ap.add_argument("--regions", nargs="+", default=None,
                    help="ROIs (default: the 4 locked ROIs)")
    ap.add_argument("--before", type=int, default=10, help="Trials before switch (default: 10)")
    ap.add_argument("--after", type=int, default=20, help="Trials after switch (default: 20)")
    ap.add_argument("--t-start", type=float, default=-0.4)
    ap.add_argument("--t-end", type=float, default=0.0)
    ap.add_argument("--min-units", type=int, default=5)
    ap.add_argument("--min-sessions", type=int, default=2,
                    help="Min sessions with >=1 switch to plot a region (default: 2)")
    args = ap.parse_args()

    cfg = load_frozen_config()
    one = make_one(repo_root() / cfg["data"]["cache_dir"])
    from iblatlas.regions import BrainRegions
    brain_regions = BrainRegions()

    manifest = json.loads((ROOT / args.cohort).read_text())
    eids = manifest["curated"]["eids"] or manifest["cohort_union_eids"]
    regions = args.regions or list(PRIMARY_ROIS.keys())
    before, after = args.before, args.after
    offsets = np.arange(-before, after + 1)
    n_off = len(offsets)

    print(f"=== Switch-centered neural block decoding ===")
    print(f"Cohort: {len(eids)} sessions | ROIs: {regions} | window [-{before}, +{after}]\n")

    # region -> list of per-session curves
    region_curves: dict[str, list[np.ndarray]] = {r: [] for r in regions}
    region_nswitch: dict[str, int] = {r: 0 for r in regions}

    for i, eid in enumerate(eids):
        print(f"[{i + 1}/{len(eids)}] {eid}")
        try:
            trials = load_trials_for_eid(one, eid)
        except Exception as exc:  # noqa: BLE001
            print(f"  trials failed: {str(exc)[:50]}")
            continue
        if "probabilityLeft" not in trials.columns or "stimOn_times" not in trials.columns:
            continue

        p_right = 1.0 - trials["probabilityLeft"].to_numpy(dtype=float)
        stim_on = trials["stimOn_times"].to_numpy(dtype=float)
        labels_full = np.full(len(p_right), -1, dtype=np.int64)
        labels_full[np.abs(p_right - 0.2) < 0.05] = BLOCK_LEFT
        labels_full[np.abs(p_right - 0.8) < 0.05] = BLOCK_RIGHT

        try:
            spikes, clusters_df = load_session_spikes(one, eid, brain_regions, good_only=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  spikes failed: {str(exc)[:50]}")
            continue

        # cluster ids per requested ROI (good units)
        roi_units: dict[str, list[int]] = {r: [] for r in regions}
        for _, row in clusters_df.iterrows():
            if not bool(row.get("good", True)):
                continue
            reg = unit_in_any_decode_region(str(row["acronym"]))
            if reg in roi_units:
                roi_units[reg].append(int(row["cluster_id"]))

        biased = (labels_full >= 0) & np.isfinite(stim_on)
        bidx = np.where(biased)[0]
        if len(bidx) < 40:
            continue

        for region, unit_list in roi_units.items():
            if len(unit_list) < args.min_units:
                continue
            cids = np.array(sorted(set(unit_list)), dtype=np.int64)
            counts = bin_spikes_per_trial(
                spikes["times"], spikes["clusters"], cids, stim_on[bidx],
                args.t_start, args.t_end,
            )
            counts = counts[:, counts.std(axis=0) > 1e-6]
            y = (labels_full[bidx] == BLOCK_RIGHT).astype(np.int64)
            if counts.shape[1] < args.min_units or y.min() == y.max():
                continue
            try:
                oof = oof_block_predictions(counts.astype(np.float64), y)
            except Exception:  # noqa: BLE001
                continue
            # Map OOF back into the full trial sequence.
            prob_full = np.full(len(p_right), np.nan)
            prob_full[bidx] = oof
            windows = list(session_switch_windows(prob_full, p_right, before, after))
            if not windows:
                continue
            region_curves[region].append(session_curve(windows, n_off))
            region_nswitch[region] += len(windows)
            print(f"    {region:6s}  units={counts.shape[1]:3d}  switches={len(windows)}")

    # Aggregate: mean +/- SEM across sessions per region.
    rows = []
    plot_regions = []
    for region in regions:
        curves = region_curves[region]
        if len(curves) < args.min_sessions:
            continue
        stack = np.vstack(curves)  # (n_sessions, n_offsets)
        mean = np.nanmean(stack, axis=0)
        n = np.sum(np.isfinite(stack), axis=0)
        sem = np.nanstd(stack, axis=0) / np.sqrt(np.maximum(n, 1))
        plot_regions.append(region)
        for j, off in enumerate(offsets):
            rows.append({
                "region": region, "offset": int(off),
                "balanced_accuracy": float(mean[j]), "sem": float(sem[j]),
                "n_sessions": len(curves), "n_switches": region_nswitch[region],
            })

    if not rows:
        print("\nNo regions had enough sessions with switches.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    out_dir = ROOT / "reports" / "v2" / "switch_block_decoding"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "neural_switch_curves.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCurves saved: {csv_path.relative_to(ROOT)}")

    _plot(df, plot_regions, offsets, before, after)


def _plot(df, regions, offsets, before, after):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for region in regions:
        d = df[df["region"] == region].sort_values("offset")
        mean = 100.0 * d["balanced_accuracy"].to_numpy()
        sem = 100.0 * d["sem"].to_numpy()
        color = ROI_COLORS.get(region, "#555555")
        n_sess = int(d["n_sessions"].iloc[0])
        n_sw = int(d["n_switches"].iloc[0])
        ax.plot(offsets, mean, "-", color=color, lw=2,
                label=f"{region} (n={n_sess} sess, {n_sw} switches)")
        ax.fill_between(offsets, mean - sem, mean + sem, color=color, alpha=0.15)

    ax.axhline(50.0, color="0.4", ls="--", lw=1, label="chance")
    ax.axvline(0, color="black", ls=":", lw=1)
    ax.set_xlabel("Trials relative to block switch")
    ax.set_ylabel("Balanced block-decoding success (%)")
    ax.set_title("Switch-centered block decoding from neural activity\n"
                 "(pre-stimulus window, per-ROI, cohort sessions)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = ROOT / "reports" / "v2" / "figures" / "switch_block_decoding" / "neural_switch_block_decoding.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
