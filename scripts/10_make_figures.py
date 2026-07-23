#!/usr/bin/env python3
"""10 — Make v2 figures (synth + real, per-regime multipanel + comparisons).

Layout
------
  by_model/{model}/{synth|real}/{regime}/multipanel_diagnostics.png
  comparison/{synth|real}_{regime}_accuracy.png
  comparison/{synth|real}_{regime}_history_gap.png
  comparison/synth_vs_real_{regime}_accuracy.png
  comparison/synth_vs_real_{regime}_history_gap.png

Usage:
  python scripts/11_eval_regimes.py   # first, if rollouts missing
  python scripts/10_make_figures.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models_v2.rollout import (
    REGIMES,
    switch_centered_per_session,
    switch_centered_zero_evidence,
)
from src.synthetic.schema import RIGHT, load_synthetic_config

DOMAINS = ("synth", "real")


def _session_colors(n: int) -> list:
    """Distinct colors for n sessions (tab10 / tab20 / hsv fallback)."""

    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(n)]
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(n)]
    cmap = plt.get_cmap("hsv")
    return [cmap(i / max(n, 1)) for i in range(n)]


def _session_labels(domain: str, n_sess: int) -> list[str]:
    if domain == "real":
        man = ROOT / "data" / "manifests" / "real_v2_ticks.json"
        if man.exists():
            eids = [s["eid"][:8] for s in json.loads(man.read_text())["sessions"]]
            if len(eids) == n_sess:
                return [f"S{i + 1}:{e}" for i, e in enumerate(eids)]
    return [f"S{i + 1}" for i in range(n_sess)]


def _rollout_path(cfg: dict, domain: str, regime: str, model_id: str) -> Path:
    base = ROOT / cfg["paths"]["artifacts"]
    if domain == "synth":
        path = base / "synthetic" / "regimes" / regime / model_id / "rollout.npz"
        if not path.exists() and regime == "history_only":
            path = base / "synthetic" / "heldout" / model_id / "rollout.npz"
        return path
    return base / "real" / "regimes" / regime / model_id / "rollout.npz"


def _metrics_path(cfg: dict, domain: str, regime: str, model_id: str) -> Path:
    return ROOT / cfg["paths"]["reports"] / "metrics" / f"{domain}_{regime}_{model_id}.json"


def _true_p_right(roll) -> np.ndarray:
    files = set(roll.files)
    if "true_p_right" in files:
        return roll["true_p_right"]
    return 1.0 - roll["probability_left"]


def _p_choice(roll) -> np.ndarray:
    files = set(roll.files)
    if "p_choice_right" in files:
        return roll["p_choice_right"]
    return roll["p_right"]


def _zero_ev(roll) -> np.ndarray:
    files = set(roll.files)
    if "zero_evidence_p_right" in files:
        return roll["zero_evidence_p_right"]
    return roll["belief"]


def _valid_mask(roll, true_p: np.ndarray) -> np.ndarray:
    files = set(roll.files)
    if "valid" in files:
        return np.asarray(roll["valid"], dtype=bool)
    return np.isfinite(true_p)


def _plot_psychometric(ax, roll, regime: str, domain: str) -> None:
    true_p = _true_p_right(roll)
    p_choice = _p_choice(roll)
    side = roll["side"]
    contrast = roll["contrast"]
    valid = _valid_mask(roll, true_p)
    signed = np.where(side == RIGHT, contrast, -contrast)
    signed = np.where(valid, signed, np.nan)
    unique_signed = np.unique(np.round(signed[np.isfinite(signed)], 5))

    if domain == "synth":
        # Averages only (pool all synthetic sessions).
        if regime == "fixed_prior":
            means, xs = [], []
            for value in unique_signed:
                mask = valid & np.isclose(signed, value)
                if mask.any():
                    xs.append(float(value))
                    means.append(float(np.nanmean(p_choice[mask])))
            if xs:
                ax.plot(xs, means, marker="o", color="#0072b2", label="mean P(choice|contrast)")
        else:
            for prior, color in ((0.2, "#d55e00"), (0.8, "#0072b2")):
                means, xs = [], []
                for value in unique_signed:
                    mask = valid & np.isclose(true_p, prior) & np.isclose(signed, value)
                    if mask.any():
                        xs.append(float(value))
                        means.append(float(np.nanmean(p_choice[mask])))
                if xs:
                    ax.plot(
                        xs,
                        means,
                        marker="o",
                        color=color,
                        label=f"mean, block P(right)={prior:.1f}",
                    )
        title = f"Psychometric (synth average, {regime})"
    else:
        n_sess = int(true_p.shape[0])
        colors = _session_colors(n_sess)
        labels = _session_labels(domain, n_sess)
        for s in range(n_sess):
            color = colors[s]
            v = valid[s]
            if regime == "fixed_prior":
                means, xs = [], []
                for value in unique_signed:
                    mask = v & np.isclose(signed[s], value)
                    if mask.any():
                        xs.append(float(value))
                        means.append(float(np.nanmean(p_choice[s][mask])))
                if xs:
                    ax.plot(
                        xs,
                        means,
                        marker="o",
                        markersize=3,
                        color=color,
                        alpha=0.85,
                        label=labels[s],
                    )
            else:
                for prior, ls in ((0.2, "--"), (0.8, "-")):
                    means, xs = [], []
                    for value in unique_signed:
                        mask = (
                            v
                            & np.isclose(true_p[s], prior)
                            & np.isclose(signed[s], value)
                        )
                        if mask.any():
                            xs.append(float(value))
                            means.append(float(np.nanmean(p_choice[s][mask])))
                    if xs:
                        ax.plot(
                            xs,
                            means,
                            marker="o",
                            markersize=3,
                            linestyle=ls,
                            color=color,
                            alpha=0.85,
                            label=labels[s] if prior == 0.8 else None,
                        )
        title = f"Psychometric by session (real, {regime})"
        if regime != "fixed_prior":
            title += "\nsolid=0.8 prior, dashed=0.2 prior"

    ax.axhline(0.5, color="0.7", linewidth=1)
    ax.axvline(0.0, color="0.7", linewidth=1)
    ax.set(
        title=title,
        xlabel="Signed contrast (left −, right +)",
        ylabel="P(choice right)",
        ylim=(-0.03, 1.03),
    )
    ax.legend(frameon=False, fontsize=6 if domain == "real" else 8, ncol=2, loc="best")


def _plot_switch(ax, roll, regime: str, domain: str) -> None:
    if regime == "fixed_prior":
        ax.text(
            0.5,
            0.5,
            "N/A (no block switches)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set(
            title="Belief adaptation (N/A)",
            xlabel="Trials relative to switch",
            ylabel="P(right) zero evidence",
        )
        return

    mean = switch_centered_zero_evidence(roll, before=20, after=30)
    if domain == "synth":
        ax.plot(
            mean["offsets"],
            mean["low_to_high"],
            color="#0072b2",
            linewidth=2.0,
            label="mean 0.2→0.8",
        )
        ax.plot(
            mean["offsets"],
            mean["high_to_low"],
            color="#d55e00",
            linewidth=2.0,
            label="mean 0.8→0.2",
        )
        title = "Belief adaptation (synth average)"
        leg_fs = 8
    else:
        per = switch_centered_per_session(roll, before=20, after=30)
        offsets = per["offsets"]
        n_sess = len(per["per_session"])
        colors = _session_colors(n_sess)
        labels = _session_labels(domain, n_sess)
        for s, curves in enumerate(per["per_session"]):
            color = colors[s]
            ax.plot(
                offsets,
                curves["low_to_high"],
                color=color,
                alpha=0.7,
                linewidth=1.1,
                label=labels[s],
            )
            ax.plot(
                offsets,
                curves["high_to_low"],
                color=color,
                alpha=0.7,
                linewidth=1.1,
                linestyle="--",
            )
        ax.plot(
            mean["offsets"],
            mean["low_to_high"],
            color="black",
            linewidth=2.2,
            label="mean 0.2→0.8",
        )
        ax.plot(
            mean["offsets"],
            mean["high_to_low"],
            color="black",
            linewidth=2.2,
            linestyle="--",
            label="mean 0.8→0.2",
        )
        title = "Belief adaptation by session (real)\nsolid 0.2→0.8, dashed 0.8→0.2"
        leg_fs = 5.5

    ax.axvline(0, color="0.3", linestyle=":", linewidth=1)
    ax.set(
        title=title,
        xlabel="Trials relative to switch",
        ylabel="P(right) with zero sensory evidence",
        ylim=(-0.03, 1.03),
    )
    ax.legend(frameon=False, fontsize=leg_fs, ncol=2, loc="best")


def _best_session_index(roll, true_p: np.ndarray) -> tuple[int, float]:
    """Session with highest accuracy vs correct side (ties → more valid trials)."""

    valid = _valid_mask(roll, true_p)
    choice = roll["choice"]
    side = roll["side"]
    n_sess = int(true_p.shape[0])
    best_i, best_key = 0, (-1.0, -1)
    for s in range(n_sess):
        v = valid[s]
        if not v.any():
            continue
        acc = float(np.mean(choice[s][v] == side[s][v]))
        n = int(v.sum())
        key = (acc, n)
        if key > best_key:
            best_key = key
            best_i = s
    return best_i, float(best_key[0]) if best_key[0] >= 0 else float("nan")


def _plot_example_session(ax, roll, regime: str, domain: str) -> None:
    true_p = _true_p_right(roll)
    zero_ev = _zero_ev(roll)
    valid = _valid_mask(roll, true_p)
    sess, acc = _best_session_index(roll, true_p)
    labels = _session_labels(domain, int(true_p.shape[0]))
    colors = _session_colors(int(true_p.shape[0]))
    color = colors[sess]

    v = valid[sess]
    last = int(np.flatnonzero(v)[-1]) + 1 if v.any() else 0
    trials = np.arange(last)
    ax.step(
        trials,
        true_p[sess, :last],
        where="post",
        color="black",
        linewidth=1.5,
        label="true block P(right)",
    )
    ax.plot(
        trials,
        zero_ev[sess, :last],
        color=color,
        alpha=0.95,
        linewidth=1.6,
        label=f"model zero-evidence ({labels[sess]})",
    )
    ax.set(
        title=(
            f"Best session {labels[sess]} "
            f"(acc={acc:.3f}, n={last}; {domain}, {regime})"
        ),
        xlabel="Trial (this session only)",
        ylabel="Probability right",
        ylim=(-0.03, 1.03),
        xlim=(0, max(last - 1, 1)),
    )
    ax.legend(frameon=False, fontsize=8)


def _plot_all_sessions_timeline(ax, roll, regime: str, domain: str) -> None:
    true_p = _true_p_right(roll)
    zero_ev = _zero_ev(roll)
    valid = _valid_mask(roll, true_p)
    n_sess = int(true_p.shape[0])
    colors = _session_colors(n_sess)
    labels = _session_labels(domain, n_sess)
    t_max = 0

    for s in range(n_sess):
        v = valid[s]
        if not v.any():
            continue
        last = int(np.flatnonzero(v)[-1]) + 1
        t_max = max(t_max, last)
        trials = np.arange(last)
        color = colors[s]
        ax.step(
            trials,
            true_p[s, :last],
            where="post",
            color=color,
            linewidth=0.8,
            alpha=0.35,
        )
        ax.plot(
            trials,
            zero_ev[s, :last],
            color=color,
            alpha=0.9,
            linewidth=1.2,
            label=labels[s],
        )

    ax.set(
        title=(
            f"All sessions ({domain}, {regime}; n={n_sess})\n"
            "colored: model zero-evidence; faint step: true prior"
        ),
        xlabel="Trial (per session)",
        ylabel="Probability right",
        ylim=(-0.03, 1.03),
        xlim=(0, max(t_max - 1, 1)),
    )
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="best")


def per_model_domain_regime_figure(
    model_id: str, domain: str, regime: str, cfg: dict, fig_root: Path
) -> Path | None:
    hist_path = ROOT / cfg["paths"]["artifacts"] / "models" / model_id / "train_history.json"
    roll_path = _rollout_path(cfg, domain, regime, model_id)
    if not hist_path.exists() or not roll_path.exists():
        print(
            f"SKIP {domain}/{model_id}/{regime}: missing history or rollout",
            file=sys.stderr,
        )
        return None

    hist = json.loads(hist_path.read_text())["history"]
    roll = np.load(roll_path)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], color="#3b6ea8")
    ylab = "PC energy / step" if model_id == "tanh_pc" else "Response cross-entropy"
    ax.set(title="Training (synth)", xlabel="Epoch", ylabel=ylab)

    _plot_psychometric(axes[0, 1], roll, regime, domain)
    _plot_switch(axes[1, 0], roll, regime, domain)
    _plot_example_session(axes[1, 1], roll, regime, domain)

    fig.suptitle(
        f"Hidden-prior diagnostics — {model_id} — {domain} — {regime}", fontsize=13
    )
    out = fig_root / "by_model" / model_id / domain / regime
    out.mkdir(parents=True, exist_ok=True)
    path = out / "multipanel_diagnostics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    # Legacy aliases for history_only synth
    if domain == "synth" and regime == "history_only":
        legacy = fig_root / "by_model" / model_id
        legacy.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, legacy / "multipanel_diagnostics.png")
        legacy_reg = fig_root / "by_model" / model_id / regime
        legacy_reg.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, legacy_reg / "multipanel_diagnostics.png")

    return path


def _load_metric_rows(cfg: dict, domain: str, regime: str) -> list[dict]:
    rows = []
    for mid in cfg["models"]:
        path = _metrics_path(cfg, domain, regime, mid)
        if path.exists():
            rows.append(json.loads(path.read_text()))
    return rows


MODEL_COLORS = {
    "tanh_bptt": "#4c72b0",
    "tanh_pc": "#55a868",
    "gru": "#c44e52",
    "bayes": "#8172b3",
}

MODEL_LABELS = {
    "tanh_bptt": "tanh BPTT",
    "tanh_pc": "tanh PC",
    "gru": "GRU",
    "bayes": "Bayes",
}


def _history_gap(row: dict) -> float:
    g = (
        row.get("kyan_diagnostics", {})
        .get("counterfactual_zero_evidence_choice_probability", {})
        .get("history_gap")
    )
    return float(g) if g is not None else float("nan")


def _acc(row: dict) -> float:
    return float(row.get("accuracy", row.get("acc_vs_correct_side", np.nan)))


def _pretty(mid: str) -> str:
    return MODEL_LABELS.get(mid, mid)


def _model_scorecard(cfg: dict, domain: str, regime: str, out: Path) -> Path | None:
    rows = _load_metric_rows(cfg, domain, regime)
    if not rows:
        return None
    names = [r["model_id"] for r in rows]
    colors = [MODEL_COLORS.get(m, "#888888") for m in names]
    labels = [_pretty(m) for m in names]

    fig = plt.figure(figsize=(12.5, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.15])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    if domain == "synth":
        data_line = (
            "Data: synthetic held-out sessions from the same generator used in training. "
            "Bars are pooled across those sessions (averages)."
        )
    else:
        data_line = (
            "Data: 10 QC behavior-core real sessions (`behavior_core_eids.json`). "
            "Bars are pooled over all valid trials in those sessions."
        )

    if regime == "history_only":
        regime_line = (
            "Regime history_only: the model sees trial history (actions/rewards) but not an "
            "oracle block-prior channel at readout."
        )
    elif regime == "full_information":
        regime_line = (
            "Regime full_information: at readout only, logits are biased by true log-odds of "
            "the block prior (eval-time oracle; not used in training)."
        )
    else:
        regime_line = (
            "Regime fixed_prior: only trials with true P(right)≈0.5. History gap is undefined "
            "(no biased-block switches)."
        )

    glossary = (
        f"Scorecard — {domain} — {regime}\n\n"
        f"{data_line}\n"
        f"{regime_line}\n\n"
        "Scoring (both panels): model choice / probabilities vs the correct stimulus side. "
        "Mouse choice is never the target.\n\n"
        "Left panel — Accuracy:\n"
        "  Fraction of trials where argmax(model choice) equals the true stimulus side. "
        "Range [0, 1]. Chance ≈ 0.5 if sides are balanced.\n"
        "  Black outline marks the highest accuracy among plotted models.\n\n"
        "Right panel — History gap:\n"
        "  Mean zero-evidence P(choice=right) on trials in blocks with true P(right)=0.8, "
        "minus the same quantity on blocks with true P(right)=0.2.\n"
        "  Zero-evidence = model preference when sensory contrast is held at 0 (counterfactual probe).\n"
        "  A gap near 0 means little use of block history; a larger positive gap means the model "
        "systematically prefers right more in right-biased blocks than in left-biased blocks.\n"
        "  Black outline marks the largest absolute gap among plotted models.\n"
        "  Switch dynamics (how fast the preference moves after a switch) are in "
        "`comparison/*_switch_board.png`, not here.\n"
    )
    ax0.text(0.0, 1.0, glossary, transform=ax0.transAxes, va="top", ha="left", fontsize=9.2)

    ax1 = fig.add_subplot(gs[1, 0])
    acc = [_acc(r) for r in rows]
    bars = ax1.bar(labels, acc, color=colors)
    if np.any(np.isfinite(acc)):
        bars[int(np.nanargmax(acc))].set_edgecolor("black")
        bars[int(np.nanargmax(acc))].set_linewidth(2.0)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy vs correct side")
    ax1.set_title("Accuracy")
    for b, v in zip(bars, acc):
        if np.isfinite(v):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    ax2 = fig.add_subplot(gs[1, 1])
    if regime == "fixed_prior":
        ax2.axis("off")
        ax2.text(
            0.5,
            0.5,
            "History gap undefined\n(fixed_prior: no 0.2/0.8 blocks)",
            ha="center",
            va="center",
            fontsize=11,
        )
    else:
        gaps = [_history_gap(r) for r in rows]
        bars2 = ax2.bar(labels, gaps, color=colors)
        if np.any(np.isfinite(gaps)):
            i = int(np.nanargmax(np.abs(gaps)))
            bars2[i].set_edgecolor("black")
            bars2[i].set_linewidth(2.0)
        ax2.axhline(0.0, color="0.5", lw=0.8)
        ax2.set_ylabel("History gap (0.8 − 0.2)")
        ax2.set_title("History gap")
        for b, v in zip(bars2, gaps):
            if np.isfinite(v):
                ax2.text(
                    b.get_x() + b.get_width() / 2,
                    v + (0.02 if v >= 0 else -0.04),
                    f"{v:.3f}",
                    ha="center",
                    fontsize=8,
                )

    path = out / f"{domain}_{regime}_scorecard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _model_switch_board(cfg: dict, domain: str, regime: str, out: Path) -> Path | None:
    if regime == "fixed_prior":
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    any_curve = False
    for mid in cfg["models"]:
        path = _rollout_path(cfg, domain, regime, mid)
        if not path.exists():
            continue
        roll = np.load(path)
        curve = switch_centered_zero_evidence(roll, before=20, after=30)
        color = MODEL_COLORS.get(mid, "#888888")
        axes[0].plot(curve["offsets"], curve["low_to_high"], color=color, lw=2.0, label=_pretty(mid))
        axes[1].plot(curve["offsets"], curve["high_to_low"], color=color, lw=2.0, label=_pretty(mid))
        any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    for ax, title in zip(
        axes,
        (
            "After switch 0.2 → 0.8\n(should rise toward ~0.8)",
            "After switch 0.8 → 0.2\n(should fall toward ~0.2)",
        ),
    ):
        ax.axvline(0, color="0.3", ls=":", lw=1)
        ax.axhline(0.5, color="0.85", lw=1)
        ax.set(
            title=title,
            xlabel="Trials relative to block switch",
            ylabel="Model zero-evidence P(right)",
            ylim=(-0.03, 1.03),
        )
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"How fast each model updates its prior — {domain} — {regime}", fontsize=12)
    path = out / f"{domain}_{regime}_switch_board.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _synth_vs_real_board(cfg: dict, regime: str, out: Path) -> Path | None:
    models = list(cfg["models"])
    names, synth_acc, real_acc, synth_gap, real_gap = [], [], [], [], []
    for mid in models:
        sp = _metrics_path(cfg, "synth", regime, mid)
        rp = _metrics_path(cfg, "real", regime, mid)
        if not (sp.exists() and rp.exists()):
            continue
        s = json.loads(sp.read_text())
        r = json.loads(rp.read_text())
        names.append(_pretty(mid))
        synth_acc.append(_acc(s))
        real_acc.append(_acc(r))
        synth_gap.append(_history_gap(s))
        real_gap.append(_history_gap(r))
    if not names:
        return None

    n_panels = 1 if regime == "fixed_prior" else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.2), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]
    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w / 2, synth_acc, w, label="Synth held-out", color="#4c72b0")
    axes[0].bar(x + w / 2, real_acc, w, label="Real (correct side)", color="#dd8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy vs correct side")
    axes[0].set_title("Does synth ranking transfer to real?")
    axes[0].legend(frameon=False, fontsize=8)

    if regime != "fixed_prior":
        axes[1].bar(x - w / 2, synth_gap, w, label="Synth held-out", color="#55a868")
        axes[1].bar(x + w / 2, real_gap, w, label="Real", color="#c44e52")
        axes[1].axhline(0.0, color="0.5", lw=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names)
        axes[1].set_ylabel("History gap (0.8 − 0.2)")
        axes[1].set_title("Does prior-use strength transfer?")
        axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle(f"Synth vs real transfer — {regime}", fontsize=12)
    path = out / f"synth_vs_real_{regime}_board.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def comparison_figures(cfg: dict, fig_root: Path) -> list[Path]:
    out_cmp = fig_root / "comparison"
    out_score = fig_root / "scorecards"
    out_cmp.mkdir(parents=True, exist_ok=True)
    out_score.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    regimes = list(cfg.get("eval", {}).get("regimes", REGIMES))
    for regime in regimes:
        for domain in DOMAINS:
            p = _model_scorecard(cfg, domain, regime, out_score)
            if p:
                paths.append(p)
            p = _model_switch_board(cfg, domain, regime, out_cmp)
            if p:
                paths.append(p)
        p = _synth_vs_real_board(cfg, regime, out_cmp)
        if p:
            paths.append(p)

    rows = _load_metric_rows(cfg, "real", "history_only")
    if rows:
        path = out_cmp / "real_transfer_accuracy.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [_pretty(r["model_id"]) for r in rows]
        colors = [MODEL_COLORS.get(r["model_id"], "#888") for r in rows]
        acc = [_acc(r) for r in rows]
        ax.bar(labels, acc, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy vs correct side")
        ax.set_title("Real transfer accuracy (history_only)")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> int:
    cfg = load_synthetic_config()
    fig_root = ROOT / cfg["paths"]["figures"]
    fig_root.mkdir(parents=True, exist_ok=True)
    regimes = list(cfg.get("eval", {}).get("regimes", REGIMES))
    made: list[str] = []
    for mid in cfg["models"]:
        for domain in DOMAINS:
            for regime in regimes:
                p = per_model_domain_regime_figure(mid, domain, regime, cfg, fig_root)
                if p:
                    made.append(str(p.relative_to(ROOT)))
    for p in comparison_figures(cfg, fig_root):
        made.append(str(p.relative_to(ROOT)))
    catalog = {
        "figures": made,
        "regenerate": "python scripts/11_eval_regimes.py && python scripts/10_make_figures.py",
        "notes": {
            "by_model/.../multipanel_diagnostics.png": (
                "Synth = averages; real = per-session colors + best-session timeline."
            ),
            "scorecards/{domain}_{regime}_scorecard.png": (
                "Model scorecards with precise reading text above panels; "
                "neutral panel titles (Accuracy / History gap)."
            ),
            "scorecards/SCORECARD_GUIDE.md": "How to read scorecards.",
            "comparison/{domain}_{regime}_switch_board.png": (
                "Side-by-side switch directions with one line per model."
            ),
            "comparison/synth_vs_real_{regime}_board.png": (
                "Synth held-out vs real transfer for the same regime."
            ),
        },
    }
    (fig_root / "figure_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    (out_score := fig_root / "scorecards").mkdir(parents=True, exist_ok=True)
    (out_score / "SCORECARD_GUIDE.md").write_text(
        "\n".join(
            [
                "# How to read scorecards",
                "",
                "Location: `reports/v2/figures/scorecards/{synth|real}_{regime}_scorecard.png`",
                "",
                "Each figure has **reading text on top** and two bar panels below. Panel titles are "
                "neutral (`Accuracy`, `History gap`); interpretation is only in the text block.",
                "",
                "## Header fields",
                "",
                "- **synth / real** — which evaluation domain.",
                "- **regime** — `history_only`, `full_information`, or `fixed_prior` (defined in the text).",
                "",
                "## Left panel: Accuracy",
                "",
                "- Definition: fraction of trials where the model’s discrete choice equals the "
                "**correct stimulus side**.",
                "- Not scored against mouse choice.",
                "- Black outline: model with the highest accuracy on this plot.",
                "",
                "## Right panel: History gap",
                "",
                "- Definition: mean zero-evidence P(choice=right) in true 0.8 blocks minus that in "
                "true 0.2 blocks.",
                "- Zero-evidence: counterfactual probe with sensory contrast held at 0.",
                "- Near 0: little differential prior use across blocks.",
                "- Large positive: model more often prefers right in right-biased blocks than in "
                "left-biased blocks.",
                "- Undefined on `fixed_prior` (no 0.2/0.8 blocks).",
                "- Black outline: largest |gap| among models.",
                "",
                "## Related figures (not scorecards)",
                "",
                "- Switch timing: `comparison/*_switch_board.png`",
                "- Synth vs real transfer: `comparison/synth_vs_real_*_board.png`",
                "- Per-model diagnostics: `by_model/.../multipanel_diagnostics.png`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    guide = fig_root / "COMPARISON_GUIDE.md"
    guide.write_text(
        "# How to read v2 figures\n\n"
        "## Scorecards (separate folder)\n"
        "- Path: `scorecards/{domain}_{regime}_scorecard.png`\n"
        "- Full guide: `scorecards/SCORECARD_GUIDE.md`\n"
        "- Start here for model ranking numbers.\n\n"
        "## Multipanels (`by_model/...`)\n"
        "- **Synth:** psychometric + switch = averages over held-out synthetic sessions.\n"
        "- **Real:** one color per of the 10 core sessions; bottom-right = best session by accuracy.\n\n"
        "## Switch boards (`comparison/*_switch_board.png`)\n"
        "- Left: preference after 0.2→0.8 switches.\n"
        "- Right: preference after 0.8→0.2 switches.\n\n"
        "## Synth vs real boards (`comparison/synth_vs_real_*`)\n"
        "- Same metric and regime: synth held-out vs real transfer.\n\n"
        "All behavioral scores use **correct stimulus side**, not mouse choice.\n",
        encoding="utf-8",
    )
    print(json.dumps(catalog, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
