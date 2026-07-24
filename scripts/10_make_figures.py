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
    switch_centered_correctness_per_session,
    switch_centered_per_session,
)
from src.plot.v2_style import (
    ACCURACY_YLIM,
    CORRECTNESS_YLIM,
    MODEL_COLORS,
    PASTEL,
    PRIOR_COLORS,
    SAVE_DPI,
    apply_style,
    label_above_bars,
    mean_ci95,
    pad_ylim_for_labels,
    save_figure,
    session_colors as _session_colors,
    style_axes_title,
    style_suptitle,
    style_xlabel,
    style_ylabel,
)
from src.synthetic.schema import RIGHT, load_synthetic_config

DOMAINS = ("synth", "real")


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


def _session_accuracies(roll) -> np.ndarray:
    """Per-session correctness vs correct stimulus side (all blocks)."""
    true_p = _true_p_right(roll)
    valid = _valid_mask(roll, true_p)
    choice = np.asarray(roll["choice"])
    side = np.asarray(roll["side"])
    n_sess = int(true_p.shape[0])
    out = np.full(n_sess, np.nan)
    for s in range(n_sess):
        v = valid[s]
        if not np.any(v):
            continue
        out[s] = float(np.mean(choice[s][v] == side[s][v]))
    return out


def _session_correctness_at_prior(roll, prior: float) -> np.ndarray:
    """Per-session correctness restricted to trials with true P(right)≈prior."""
    true_p = _true_p_right(roll)
    valid = _valid_mask(roll, true_p)
    choice = np.asarray(roll["choice"])
    side = np.asarray(roll["side"])
    n_sess = int(true_p.shape[0])
    out = np.full(n_sess, np.nan)
    for s in range(n_sess):
        v = valid[s] & np.isclose(true_p[s], prior)
        if not np.any(v):
            continue
        out[s] = float(np.mean(choice[s][v] == side[s][v]))
    return out


def _correctness_ci_at_prior(
    cfg: dict, domain: str, regime: str, model_id: str, prior: float
) -> tuple[float, float]:
    path = _rollout_path(cfg, domain, regime, model_id)
    if not path.exists():
        return float("nan"), float("nan")
    roll = np.load(path)
    return mean_ci95(_session_correctness_at_prior(roll, prior))


def _session_history_gaps(roll) -> np.ndarray:
    """Per-session history gap from zero-evidence belief."""
    true_p = _true_p_right(roll)
    valid = _valid_mask(roll, true_p)
    belief = _zero_ev(roll)
    n_sess = int(true_p.shape[0])
    out = np.full(n_sess, np.nan)
    for s in range(n_sess):
        v = valid[s]
        hi = v & np.isclose(true_p[s], 0.8)
        lo = v & np.isclose(true_p[s], 0.2)
        if not (hi.any() and lo.any()):
            continue
        out[s] = float(np.nanmean(belief[s][hi]) - np.nanmean(belief[s][lo]))
    return out


def _acc_ci_from_rollout(cfg: dict, domain: str, regime: str, model_id: str) -> tuple[float, float]:
    path = _rollout_path(cfg, domain, regime, model_id)
    if not path.exists():
        return float("nan"), float("nan")
    roll = np.load(path)
    return mean_ci95(_session_accuracies(roll))


def _gap_ci_from_rollout(cfg: dict, domain: str, regime: str, model_id: str) -> tuple[float, float]:
    path = _rollout_path(cfg, domain, regime, model_id)
    if not path.exists():
        return float("nan"), float("nan")
    roll = np.load(path)
    return mean_ci95(_session_history_gaps(roll))


def _curve_mean_sem(per: dict, direction: str):
    offsets = np.asarray(per["offsets"])
    rows = []
    for curves in per["per_session"]:
        rows.append(np.asarray(curves[direction], dtype=float))
    if not rows:
        nan = np.full_like(offsets, np.nan, dtype=float)
        return offsets, nan, nan
    stack = np.vstack(rows)
    mean = np.nanmean(stack, axis=0)
    n = np.sum(np.isfinite(stack), axis=0)
    with np.errstate(invalid="ignore"):
        std = np.nanstd(stack, axis=0, ddof=1)
    sem = np.where(n >= 2, std / np.sqrt(n), 0.0)
    return offsets, mean, sem


def _switch_mean_sem(roll, direction: str, before: int = 30, after: int = 30):
    """Session-mean switch belief curve ± SEM across sessions."""
    per = switch_centered_per_session(roll, before=before, after=after)
    return _curve_mean_sem(per, direction)


def _switch_correctness_mean_sem(
    roll, direction: str, before: int = 30, after: int = 30
):
    """Session-mean switch correctness curve ± SEM across sessions."""
    per = switch_centered_correctness_per_session(roll, before=before, after=after)
    return _curve_mean_sem(per, direction)


def _post_switch_correctness_ci(
    roll, direction: str, *, post_start: int = 0, post_end: int = 15
) -> tuple[float, float]:
    """Mean correctness in post-switch window [post_start, post_end], session 95% CI."""
    per = switch_centered_correctness_per_session(roll, before=5, after=max(post_end, 30))
    offsets = np.asarray(per["offsets"])
    mask = (offsets >= post_start) & (offsets <= post_end)
    sess_means = []
    for curves in per["per_session"]:
        row = np.asarray(curves[direction], dtype=float)
        if not np.any(np.isfinite(row[mask])):
            continue
        sess_means.append(float(np.nanmean(row[mask])))
    return mean_ci95(np.asarray(sess_means, dtype=float))


def _plot_psychometric(ax, roll, regime: str, domain: str) -> None:
    true_p = _true_p_right(roll)
    p_choice = _p_choice(roll)
    side = roll["side"]
    contrast = roll["contrast"]
    valid = _valid_mask(roll, true_p)
    signed = np.where(side == RIGHT, contrast, -contrast)
    signed = np.where(valid, signed, np.nan)
    unique_signed = np.unique(np.round(signed[np.isfinite(signed)], 5))
    n_sess = int(true_p.shape[0])

    def _sess_means(prior: float | None, value: float) -> np.ndarray:
        vals = []
        for s in range(n_sess):
            v = valid[s]
            mask = v & np.isclose(signed[s], value)
            if prior is not None:
                mask = mask & np.isclose(true_p[s], prior)
            if mask.any():
                vals.append(float(np.nanmean(p_choice[s][mask])))
        return np.asarray(vals, dtype=float)

    if domain == "synth":
        if regime == "fixed_prior":
            xs, means, errs = [], [], []
            for value in unique_signed:
                sm = _sess_means(None, float(value))
                if sm.size:
                    m, e = mean_ci95(sm)
                    xs.append(float(value))
                    means.append(m)
                    errs.append(e)
            if xs:
                ax.errorbar(
                    xs,
                    means,
                    yerr=errs,
                    marker="o",
                    color=PASTEL["blue"],
                    ecolor=PASTEL["ink"],
                    capsize=3,
                    label="mean ± 95% CI (sessions)",
                )
        else:
            for prior, color, lab in (
                (0.2, PASTEL["orange"], "block P(right)=0.2"),
                (0.8, PASTEL["blue"], "block P(right)=0.8"),
            ):
                xs, means, errs = [], [], []
                for value in unique_signed:
                    sm = _sess_means(prior, float(value))
                    if sm.size:
                        m, e = mean_ci95(sm)
                        xs.append(float(value))
                        means.append(m)
                        errs.append(e)
                if xs:
                    ax.errorbar(
                        xs,
                        means,
                        yerr=errs,
                        marker="o",
                        color=color,
                        ecolor=PASTEL["ink"],
                        capsize=3,
                        label=f"{lab} (±95% CI)",
                    )
        title = f"Psychometric (synth mean ± 95% CI, {regime})"
    else:
        colors = _session_colors(n_sess)
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
                    )
            else:
                # One color per session; linestyle encodes block prior (0.2 / 0.8 only)
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
                        )
        title = f"Psychometric by session (real, {regime})"
        if regime != "fixed_prior":
            title += "\ncolor=session; solid=0.8, dashed=0.2"

    ax.axhline(0.5, color=PASTEL["gray"], linewidth=1)
    ax.axvline(0.0, color=PASTEL["gray"], linewidth=1)
    ax.set_ylim(-0.03, 1.03)
    style_axes_title(ax, title)
    style_xlabel(ax, "Signed contrast (left −, right +)")
    style_ylabel(ax, "P(choice right)")
    if domain == "synth":
        ax.legend(
            frameon=False,
            fontsize=8,
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            borderaxespad=0.4,
        )
    # Real: colors distinguish sessions; no per-session legend labels


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
        style_axes_title(ax, "Belief adaptation (N/A)")
        style_xlabel(ax, "Trials relative to switch")
        style_ylabel(ax, "P(right) zero evidence")
        return

    if domain == "synth":
        for direction, color, label in (
            ("low_to_high", PASTEL["blue"], "0.2→0.8"),
            ("high_to_low", PASTEL["orange"], "0.8→0.2"),
        ):
            offsets, mean, sem = _switch_mean_sem(roll, direction)
            ax.plot(offsets, mean, color=color, linewidth=2.0, label=f"mean {label}")
            ax.fill_between(
                offsets,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.35,
                linewidth=0,
                label="± SEM" if direction == "low_to_high" else None,
            )
        title = "Belief adaptation (synth mean ± SEM)"
        leg_fs = 8
        show_legend = True
    else:
        per = switch_centered_per_session(roll, before=30, after=30)
        offsets = per["offsets"]
        n_sess = len(per["per_session"])
        colors = _session_colors(n_sess)
        for s, curves in enumerate(per["per_session"]):
            color = colors[s]
            ax.plot(
                offsets,
                curves["low_to_high"],
                color=color,
                alpha=0.55,
                linewidth=1.0,
            )
            ax.plot(
                offsets,
                curves["high_to_low"],
                color=color,
                alpha=0.55,
                linewidth=1.0,
                linestyle="--",
            )
        for direction, ls, label in (
            ("low_to_high", "-", "mean 0.2→0.8"),
            ("high_to_low", "--", "mean 0.8→0.2"),
        ):
            off, m, sem = _switch_mean_sem(roll, direction)
            ax.plot(off, m, color=PASTEL["ink"], linewidth=2.2, linestyle=ls, label=label)
            ax.fill_between(off, m - sem, m + sem, color=PASTEL["ink"], alpha=0.18, linewidth=0)
        title = (
            "Belief adaptation by session (real)\n"
            "color=session; solid 0.2→0.8, dashed 0.8→0.2 (± SEM mean)"
        )
        leg_fs = 8
        show_legend = True

    ax.axvline(0, color=PASTEL["ink"], linestyle=":", linewidth=1)
    ax.set_ylim(-0.03, 1.03)
    style_axes_title(ax, title)
    style_xlabel(ax, "Trials relative to switch")
    style_ylabel(ax, "P(right) with zero sensory evidence")
    if show_legend:
        ax.legend(
            frameon=False,
            fontsize=leg_fs,
            ncol=2,
            loc="lower right",
            borderaxespad=0.5,
        )


def _example_session_index(roll, true_p: np.ndarray) -> tuple[int, float]:
    """Prefer a high-accuracy session that contains 0.2, 0.5, and 0.8 blocks.

    Zero-evidence belief is defined on every trial; showing all three block types
    makes the example readable. Falls back to best accuracy if no session has all
    three priors.
    """
    valid = _valid_mask(roll, true_p)
    choice = roll["choice"]
    side = roll["side"]
    n_sess = int(true_p.shape[0])
    best_i, best_key = 0, (-1, -1.0, -1)  # (n_prior_types, acc, n)
    for s in range(n_sess):
        v = valid[s]
        if not v.any():
            continue
        priors = {float(np.round(p, 1)) for p in true_p[s][v]}
        n_types = sum(1 for p in (0.2, 0.5, 0.8) if any(abs(x - p) < 1e-6 for x in priors))
        acc = float(np.mean(choice[s][v] == side[s][v]))
        n = int(v.sum())
        key = (n_types, acc, n)
        if key > best_key:
            best_key = key
            best_i = s
    return best_i, float(best_key[1]) if best_key[1] >= 0 else float("nan")


def _plot_example_session(ax, roll, regime: str, domain: str) -> None:
    true_p = _true_p_right(roll)
    zero_ev = _zero_ev(roll)
    valid = _valid_mask(roll, true_p)
    sess, acc = _example_session_index(roll, true_p)
    colors = _session_colors(int(true_p.shape[0]))
    color = colors[sess]

    v = valid[sess]
    last = int(np.flatnonzero(v)[-1]) + 1 if v.any() else 0
    trials = np.arange(last)
    # True block prior (0.2 / 0.5 / 0.8) and model zero-evidence on every trial
    ax.step(
        trials,
        true_p[sess, :last],
        where="post",
        color=PASTEL["ink"],
        linewidth=1.5,
        label="true block P(right)",
    )
    ax.plot(
        trials,
        zero_ev[sess, :last],
        color=color,
        alpha=0.95,
        linewidth=1.6,
        label="model zero-evidence",
    )
    # Light guides at each block prior level
    for y, ls in ((0.2, "--"), (0.5, ":"), (0.8, "--")):
        ax.axhline(y, color=PASTEL["gray"], lw=0.6, ls=ls, alpha=0.7)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, max(last - 1, 1))
    n_types = len({float(np.round(p, 1)) for p in true_p[sess][v]}) if v.any() else 0
    style_axes_title(
        ax,
        f"Example session (acc={acc:.3f}, n={last}, {n_types} prior levels; {domain}, {regime})",
    )
    style_xlabel(ax, "Trial (this session only)")
    style_ylabel(ax, "Probability right")
    ax.legend(frameon=False, fontsize=8, loc="best", borderaxespad=0.5)


def _plot_all_sessions_timeline(ax, roll, regime: str, domain: str) -> None:
    true_p = _true_p_right(roll)
    zero_ev = _zero_ev(roll)
    valid = _valid_mask(roll, true_p)
    n_sess = int(true_p.shape[0])
    colors = _session_colors(n_sess)
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
        )

    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, max(t_max - 1, 1))
    style_axes_title(
        ax,
        f"All sessions ({domain}, {regime}; n={n_sess})\n"
        "color=session zero-evidence; faint step=true prior (0.2/0.5/0.8)",
    )
    style_xlabel(ax, "Trial (per session)")
    style_ylabel(ax, "Probability right")
    # No per-session legend: color alone encodes session identity


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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.12)

    ax = axes[0, 0]
    ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], color=PASTEL["blue"])
    ylab = "PC energy / step" if model_id == "tanh_pc" else "Response cross-entropy"
    style_axes_title(ax, "Training (synth)")
    style_xlabel(ax, "Epoch")
    style_ylabel(ax, ylab)

    _plot_psychometric(axes[0, 1], roll, regime, domain)
    _plot_switch(axes[1, 0], roll, regime, domain)
    _plot_example_session(axes[1, 1], roll, regime, domain)

    style_suptitle(
        fig,
        f"Hidden-prior diagnostics — {model_id} — {domain} — {regime}",
        y=1.03,
    )
    out = fig_root / "by_model" / model_id / domain / regime
    out.mkdir(parents=True, exist_ok=True)
    path = out / "multipanel_diagnostics.png"
    save_figure(fig, path)
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


MODEL_LABELS = {
    "tanh_bptt": "tanh BPTT",
    "tanh_pc": "tanh PC",
    "gru": "GRU",
    "gru_pc": "GRU PC",
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


def _label_above_bar(ax, bars, values, errs, *, pad: float = 0.012, fontsize: int = 8) -> None:
    """Place numeric labels just above each bar (not above the CI tip)."""
    label_above_bars(ax, bars, values, errs, pad=pad, fontsize=fontsize)


def _model_scorecard(cfg: dict, domain: str, regime: str, out: Path) -> Path | None:
    rows = _load_metric_rows(cfg, domain, regime)
    if not rows:
        return None
    names = [r["model_id"] for r in rows]
    colors = [MODEL_COLORS.get(m, PASTEL["gray"]) for m in names]
    labels = [_pretty(m) for m in names]

    fig = plt.figure(figsize=(13.0, 9.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.08, wspace=0.06, hspace=0.10)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.35])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    if domain == "synth":
        data_line = (
            "Data: synthetic held-out sessions from the same generator used in training. "
            "Bars are session-means (± 95% CI across sessions)."
        )
    else:
        data_line = (
            "Data: shared behavior+neural cohort "
            "(`shared_behavior_neural_eids.json`). "
            "Bars are session-means (± 95% CI across sessions)."
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
        "Scoring: model choice vs the correct stimulus side (never mouse choice).\n\n"
        "Left — Correctness:\n"
        "  Fraction of trials where argmax(model choice) equals the true stimulus side. "
        "Y-axis 0.50–1.00. Outline = highest correctness.\n"
        "  See also comparison/*_correctness_by_prior.png for P(right)=0.2/0.5/0.8 breakdown "
        "(overall correctness can be driven by one block type).\n\n"
        "Right — History gap:\n"
        "  Mean zero-evidence P(right|block 0.8) − P(right|block 0.2). "
        "Near 0 = little prior use; large positive = strong block-tuned bias.\n"
    )
    ax0.text(0.0, 1.0, glossary, transform=ax0.transAxes, va="top", ha="left", fontsize=9.2)

    ax1 = fig.add_subplot(gs[1, 0])
    acc, acc_err = [], []
    for r in rows:
        mid = r["model_id"]
        m, e = _acc_ci_from_rollout(cfg, domain, regime, mid)
        if not np.isfinite(m):
            m = _acc(r)
            e = 0.0
        acc.append(m)
        acc_err.append(e if np.isfinite(e) else 0.0)
    bars = ax1.bar(
        labels,
        acc,
        color=colors,
        yerr=acc_err,
        capsize=4,
        ecolor=PASTEL["ink"],
        error_kw={"linewidth": 1.0},
    )
    if np.any(np.isfinite(acc)):
        bars[int(np.nanargmax(acc))].set_edgecolor(PASTEL["ink"])
        bars[int(np.nanargmax(acc))].set_linewidth(2.0)
    pad_ylim_for_labels(ax1, acc, acc_err, floor=CORRECTNESS_YLIM[0], headroom=0.07)
    style_ylabel(ax1, "Correctness vs correct stimulus side")
    style_axes_title(ax1, "Correctness (mean ± 95% CI)")
    _label_above_bar(ax1, bars, acc, acc_err, pad=0.010)

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
        gaps, gap_err = [], []
        for r in rows:
            mid = r["model_id"]
            m, e = _gap_ci_from_rollout(cfg, domain, regime, mid)
            if not np.isfinite(m):
                m = _history_gap(r)
                e = 0.0
            gaps.append(m)
            gap_err.append(e if np.isfinite(e) else 0.0)
        bars2 = ax2.bar(
            labels,
            gaps,
            color=colors,
            yerr=gap_err,
            capsize=4,
            ecolor=PASTEL["ink"],
            error_kw={"linewidth": 1.0},
        )
        if np.any(np.isfinite(gaps)):
            i = int(np.nanargmax(np.abs(gaps)))
            bars2[i].set_edgecolor(PASTEL["ink"])
            bars2[i].set_linewidth(2.0)
        ax2.axhline(0.0, color=PASTEL["gray"], lw=0.8)
        pad_ylim_for_labels(ax2, gaps, gap_err, headroom=0.08)
        style_ylabel(ax2, "History gap (0.8 − 0.2)")
        style_axes_title(ax2, "History gap (mean ± 95% CI)")
        for b, v, e in zip(bars2, gaps, gap_err):
            if not np.isfinite(v):
                continue
            y = v + (0.014 if v >= 0 else -0.020)
            ax2.text(
                b.get_x() + b.get_width() / 2,
                y,
                f"{v:.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
                clip_on=True,
            )

    path = out / f"{domain}_{regime}_scorecard.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def _correctness_by_prior_board(cfg: dict, domain: str, regime: str, out: Path) -> Path | None:
    """Grouped bars: correctness in P(right)=0.2 / 0.5 / 0.8 blocks (+ balanced mean).

    Presentation choice: models on x-axis; three prior-colored bars per model so you can
    see whether overall correctness is driven by one block type. A dashed marker shows
    the equal-weight mean across available priors (balanced correctness).
    """
    models = [m for m in cfg["models"] if _rollout_path(cfg, domain, regime, m).exists()]
    if not models:
        return None
    priors = (0.5,) if regime == "fixed_prior" else (0.2, 0.5, 0.8)
    labels = [_pretty(m) for m in models]
    x = np.arange(len(models))
    n_priors = len(priors)
    width = 0.72 / n_priors

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.0),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.12, wspace=0.10, hspace=0.10)
    ax, axb = axes

    balanced = []
    balanced_err = []
    all_prior_means = []
    all_prior_errs = []
    for i, mid in enumerate(models):
        prior_means = []
        prior_errs = []
        for j, prior in enumerate(priors):
            m, e = _correctness_ci_at_prior(cfg, domain, regime, mid, prior)
            prior_means.append(m)
            prior_errs.append(e if np.isfinite(e) else 0.0)
            xpos = x[i] - 0.36 + width / 2 + j * width
            ax.bar(
                xpos,
                m if np.isfinite(m) else 0.0,
                width * 0.90,
                color=PRIOR_COLORS.get(prior, PASTEL["gray"]),
                yerr=e if np.isfinite(e) else 0.0,
                capsize=3,
                ecolor=PASTEL["ink"],
                error_kw={"linewidth": 0.9},
                label=f"P(right)={prior:.1f}" if i == 0 else None,
            )
            if np.isfinite(m):
                # Slightly above bar face; ylim padded below so this never clips
                ax.text(
                    xpos,
                    m + 0.012,
                    f"{m:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    clip_on=True,
                )
        all_prior_means.extend(prior_means)
        all_prior_errs.extend(prior_errs)
        finite = [v for v in prior_means if np.isfinite(v)]
        if finite:
            bal = float(np.mean(finite))
            errs = [e for e, v in zip(prior_errs, prior_means) if np.isfinite(v)]
            bal_e = float(np.mean(errs) / np.sqrt(len(errs))) if errs else 0.0
        else:
            bal, bal_e = float("nan"), 0.0
        balanced.append(bal)
        balanced_err.append(bal_e)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    pad_ylim_for_labels(
        ax,
        all_prior_means,
        all_prior_errs,
        floor=CORRECTNESS_YLIM[0],
        headroom=0.08,
    )
    style_ylabel(ax, "Correctness vs correct stimulus side")
    style_axes_title(
        ax,
        f"Correctness by block prior — {domain} — {regime}\n(grouped bars; mean ± 95% CI)",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right", borderaxespad=0.6)
    ax.axhline(0.5, color=PASTEL["gray"], lw=0.8, ls=":")

    bcols = [MODEL_COLORS.get(m, PASTEL["gray"]) for m in models]
    bars = axb.bar(
        labels,
        balanced,
        color=bcols,
        yerr=balanced_err,
        capsize=4,
        ecolor=PASTEL["ink"],
    )
    if np.any(np.isfinite(balanced)):
        bars[int(np.nanargmax(balanced))].set_edgecolor(PASTEL["ink"])
        bars[int(np.nanargmax(balanced))].set_linewidth(2.0)
    pad_ylim_for_labels(
        axb,
        balanced,
        balanced_err,
        floor=CORRECTNESS_YLIM[0],
        headroom=0.08,
    )
    style_ylabel(axb, "Balanced correctness")
    style_axes_title(
        axb,
        "Balanced correctness\n(equal weight 0.2/0.5/0.8)",
    )
    _label_above_bar(axb, bars, balanced, balanced_err, pad=0.010)

    style_suptitle(
        fig,
        "Does one block type drive overall correctness? Compare priors + balanced score",
        y=1.05,
    )
    path = out / f"{domain}_{regime}_correctness_by_prior.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def _model_switch_board(cfg: dict, domain: str, regime: str, out: Path) -> Path | None:
    if regime == "fixed_prior":
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.10, hspace=0.10)
    any_curve = False
    for mid in cfg["models"]:
        path = _rollout_path(cfg, domain, regime, mid)
        if not path.exists():
            continue
        roll = np.load(path)
        color = MODEL_COLORS.get(mid, PASTEL["gray"])
        for ax, direction in zip(axes, ("low_to_high", "high_to_low")):
            offsets, mean, sem = _switch_mean_sem(roll, direction)
            ax.plot(offsets, mean, color=color, lw=2.0, label=_pretty(mid))
            ax.fill_between(
                offsets,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.30,
                linewidth=0,
            )
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
        ax.axvline(0, color=PASTEL["ink"], ls=":", lw=1)
        ax.axhline(0.5, color=PASTEL["gray"], lw=1)
        ax.set_ylim(-0.03, 1.03)
        style_axes_title(ax, title)
        style_xlabel(ax, "Trials relative to block switch")
        style_ylabel(ax, "Model zero-evidence P(right)")
        ax.legend(frameon=False, fontsize=8, loc="best", borderaxespad=0.5)
    style_suptitle(
        fig,
        f"How fast each model updates its prior — {domain} — {regime} (mean ± SEM)",
        y=1.05,
    )
    path = out / f"{domain}_{regime}_switch_board.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def _synth_vs_real_board(cfg: dict, regime: str, out: Path) -> Path | None:
    models = list(cfg["models"])
    names, synth_acc, real_acc, synth_gap, real_gap = [], [], [], [], []
    synth_acc_e, real_acc_e, synth_gap_e, real_gap_e = [], [], [], []
    for mid in models:
        sp = _metrics_path(cfg, "synth", regime, mid)
        rp = _metrics_path(cfg, "real", regime, mid)
        if not (sp.exists() and rp.exists()):
            continue
        names.append(_pretty(mid))
        sa, se = _acc_ci_from_rollout(cfg, "synth", regime, mid)
        ra, re = _acc_ci_from_rollout(cfg, "real", regime, mid)
        if not np.isfinite(sa):
            sa, se = _acc(json.loads(sp.read_text())), 0.0
        if not np.isfinite(ra):
            ra, re = _acc(json.loads(rp.read_text())), 0.0
        synth_acc.append(sa)
        real_acc.append(ra)
        synth_acc_e.append(se if np.isfinite(se) else 0.0)
        real_acc_e.append(re if np.isfinite(re) else 0.0)
        if regime != "fixed_prior":
            sg, sge = _gap_ci_from_rollout(cfg, "synth", regime, mid)
            rg, rge = _gap_ci_from_rollout(cfg, "real", regime, mid)
            if not np.isfinite(sg):
                sg, sge = _history_gap(json.loads(sp.read_text())), 0.0
            if not np.isfinite(rg):
                rg, rge = _history_gap(json.loads(rp.read_text())), 0.0
            synth_gap.append(sg)
            real_gap.append(rg)
            synth_gap_e.append(sge if np.isfinite(sge) else 0.0)
            real_gap_e.append(rge if np.isfinite(rge) else 0.0)
    if not names:
        return None

    n_panels = 1 if regime == "fixed_prior" else 2
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6.0 * n_panels, 4.8), constrained_layout=True
    )
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.10, hspace=0.10)
    if n_panels == 1:
        axes = [axes]
    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(
        x - w / 2,
        synth_acc,
        w,
        yerr=synth_acc_e,
        capsize=3,
        label="Synth held-out",
        color=PASTEL["blue"],
        ecolor=PASTEL["ink"],
    )
    axes[0].bar(
        x + w / 2,
        real_acc,
        w,
        yerr=real_acc_e,
        capsize=3,
        label="Real (correct side)",
        color=PASTEL["orange"],
        ecolor=PASTEL["ink"],
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    pad_ylim_for_labels(
        axes[0],
        list(synth_acc) + list(real_acc),
        list(synth_acc_e) + list(real_acc_e),
        floor=CORRECTNESS_YLIM[0],
        headroom=0.06,
    )
    style_ylabel(axes[0], "Correctness vs correct stimulus side")
    style_axes_title(axes[0], "Does synth ranking transfer to real?\n(mean ± 95% CI)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", borderaxespad=0.5)

    if regime != "fixed_prior":
        axes[1].bar(
            x - w / 2,
            synth_gap,
            w,
            yerr=synth_gap_e,
            capsize=3,
            label="Synth held-out",
            color=PASTEL["green"],
            ecolor=PASTEL["ink"],
        )
        axes[1].bar(
            x + w / 2,
            real_gap,
            w,
            yerr=real_gap_e,
            capsize=3,
            label="Real",
            color=PASTEL["rose"],
            ecolor=PASTEL["ink"],
        )
        axes[1].axhline(0.0, color=PASTEL["gray"], lw=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names)
        pad_ylim_for_labels(
            axes[1],
            list(synth_gap) + list(real_gap),
            list(synth_gap_e) + list(real_gap_e),
            headroom=0.08,
        )
        style_ylabel(axes[1], "History gap (0.8 − 0.2)")
        style_axes_title(axes[1], "Does prior-use strength transfer?\n(mean ± 95% CI)")
        axes[1].legend(frameon=False, fontsize=8, loc="best", borderaxespad=0.5)

    style_suptitle(fig, f"Synth vs real transfer — {regime}", y=1.05)
    path = out / f"synth_vs_real_{regime}_board.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def _history_gap_board(cfg: dict, domain: str, regime: str, out: Path) -> Path | None:
    """Bar chart of history gap (0.8−0.2 zero-evidence belief) with session 95% CI."""
    if regime == "fixed_prior":
        return None
    models, gaps, errs, colors = [], [], [], []
    for mid in cfg["models"]:
        m, e = _gap_ci_from_rollout(cfg, domain, regime, mid)
        if not np.isfinite(m):
            continue
        models.append(_pretty(mid))
        gaps.append(m)
        errs.append(e if np.isfinite(e) else 0.0)
        colors.append(MODEL_COLORS.get(mid, PASTEL["gray"]))
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    bars = ax.bar(models, gaps, color=colors, yerr=errs, capsize=4, ecolor=PASTEL["ink"])
    ax.axhline(0.0, color=PASTEL["gray"], lw=0.8)
    pad_ylim_for_labels(ax, gaps, errs, headroom=0.08)
    style_ylabel(ax, "History gap\n(mean zero-evidence P(right): 0.8 − 0.2)")
    style_axes_title(
        ax,
        f"Prior-use strength — {domain} — {regime}\n(mean ± 95% CI across sessions)",
    )
    _label_above_bar(ax, bars, gaps, errs, pad=0.012)
    path = out / f"{domain}_{regime}_history_gap.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def _overall_correctness_board(cfg: dict, domain: str, regime: str, out: Path) -> Path | None:
    """Overall correctness vs correct side with session 95% CI."""
    models, acc, errs, colors = [], [], [], []
    for mid in cfg["models"]:
        m, e = _acc_ci_from_rollout(cfg, domain, regime, mid)
        if not np.isfinite(m):
            continue
        models.append(_pretty(mid))
        acc.append(m)
        errs.append(e if np.isfinite(e) else 0.0)
        colors.append(MODEL_COLORS.get(mid, PASTEL["gray"]))
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    bars = ax.bar(models, acc, color=colors, yerr=errs, capsize=4, ecolor=PASTEL["ink"])
    pad_ylim_for_labels(ax, acc, errs, floor=CORRECTNESS_YLIM[0], headroom=0.07)
    style_ylabel(ax, "Correctness vs correct stimulus side")
    style_axes_title(
        ax,
        f"Overall correctness — {domain} — {regime}\n(mean ± 95% CI across sessions)",
    )
    _label_above_bar(ax, bars, acc, errs, pad=0.010)
    path = out / f"{domain}_{regime}_overall_correctness.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def _switch_correctness_board(cfg: dict, domain: str, regime: str, out: Path) -> Path | None:
    """Two-panel switch-centered correctness (0.2→0.8 and 0.8→0.2) with SEM bands."""
    if regime == "fixed_prior":
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.10, wspace=0.10, hspace=0.10)
    any_curve = False
    for mid in cfg["models"]:
        path = _rollout_path(cfg, domain, regime, mid)
        if not path.exists():
            continue
        roll = np.load(path)
        color = MODEL_COLORS.get(mid, PASTEL["gray"])
        for ax, direction in zip(axes, ("low_to_high", "high_to_low")):
            offsets, mean, sem = _switch_correctness_mean_sem(roll, direction)
            if not np.any(np.isfinite(mean)):
                continue
            ax.plot(offsets, mean, color=color, lw=2.0, label=_pretty(mid))
            ax.fill_between(
                offsets,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.28,
                linewidth=0,
            )
            any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    for ax, title in zip(
        axes,
        (
            "Switch 0.2 → 0.8\n(correctness around update)",
            "Switch 0.8 → 0.2\n(correctness around update)",
        ),
    ):
        ax.axvline(0, color=PASTEL["ink"], ls=":", lw=1)
        ax.axhline(0.5, color=PASTEL["gray"], lw=1)
        ax.set_ylim(0.45, 1.02)
        style_axes_title(ax, title)
        style_xlabel(ax, "Trials relative to block switch")
        style_ylabel(ax, "Correctness (choice = correct side)")
        ax.legend(frameon=False, fontsize=8, loc="best", borderaxespad=0.5)
    style_suptitle(
        fig,
        f"Correctness around hidden-prior switches — {domain} — {regime} (mean ± SEM)",
        y=1.05,
    )
    path = out / f"{domain}_{regime}_switch_correctness.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def _switch_correctness_summary_board(
    cfg: dict, domain: str, regime: str, out: Path
) -> Path | None:
    """Post-switch correctness (trials 0–15) by direction, with 95% CI bars."""
    if regime == "fixed_prior":
        return None
    models = []
    low_hi, low_hi_e = [], []
    hi_lo, hi_lo_e = [], []
    colors = []
    for mid in cfg["models"]:
        path = _rollout_path(cfg, domain, regime, mid)
        if not path.exists():
            continue
        roll = np.load(path)
        m1, e1 = _post_switch_correctness_ci(roll, "low_to_high", post_start=0, post_end=15)
        m2, e2 = _post_switch_correctness_ci(roll, "high_to_low", post_start=0, post_end=15)
        if not (np.isfinite(m1) or np.isfinite(m2)):
            continue
        models.append(_pretty(mid))
        low_hi.append(m1 if np.isfinite(m1) else np.nan)
        low_hi_e.append(e1 if np.isfinite(e1) else 0.0)
        hi_lo.append(m2 if np.isfinite(m2) else np.nan)
        hi_lo_e.append(e2 if np.isfinite(e2) else 0.0)
        colors.append(MODEL_COLORS.get(mid, PASTEL["gray"]))
    if not models:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    x = np.arange(len(models))
    for ax, vals, errs, title in zip(
        axes,
        (low_hi, hi_lo),
        (low_hi_e, hi_lo_e),
        (
            "Post-switch correctness\n0.2 → 0.8 (trials 0–15)",
            "Post-switch correctness\n0.8 → 0.2 (trials 0–15)",
        ),
    ):
        bars = ax.bar(x, vals, color=colors, yerr=errs, capsize=4, ecolor=PASTEL["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        pad_ylim_for_labels(ax, vals, errs, floor=0.5, headroom=0.07)
        style_ylabel(ax, "Correctness vs correct side")
        style_axes_title(ax, title)
        _label_above_bar(ax, bars, vals, errs, pad=0.010)
    style_suptitle(
        fig,
        f"Online updating shows up in post-switch correctness — {domain} — {regime}",
        y=1.05,
    )
    path = out / f"{domain}_{regime}_switch_correctness_summary.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def _accuracy_switch_story_board(
    cfg: dict, domain: str, regime: str, out: Path
) -> Path | None:
    """Story figure: overall correctness, then switch-direction correctness curves."""
    if regime == "fixed_prior":
        return None
    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.08, wspace=0.08, hspace=0.12)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    ax_overall = fig.add_subplot(gs[0, :])
    ax_l2h = fig.add_subplot(gs[1, 0])
    ax_h2l = fig.add_subplot(gs[1, 1])

    models, acc, errs, colors = [], [], [], []
    for mid in cfg["models"]:
        m, e = _acc_ci_from_rollout(cfg, domain, regime, mid)
        if not np.isfinite(m):
            continue
        models.append(_pretty(mid))
        acc.append(m)
        errs.append(e if np.isfinite(e) else 0.0)
        colors.append(MODEL_COLORS.get(mid, PASTEL["gray"]))
    if not models:
        plt.close(fig)
        return None
    bars = ax_overall.bar(models, acc, color=colors, yerr=errs, capsize=4, ecolor=PASTEL["ink"])
    pad_ylim_for_labels(ax_overall, acc, errs, floor=CORRECTNESS_YLIM[0], headroom=0.07)
    style_ylabel(ax_overall, "Correctness")
    style_axes_title(
        ax_overall,
        "1. Overall correctness (full task; mean ± 95% CI)",
    )
    _label_above_bar(ax_overall, bars, acc, errs, pad=0.010)

    any_curve = False
    for mid in cfg["models"]:
        path = _rollout_path(cfg, domain, regime, mid)
        if not path.exists():
            continue
        roll = np.load(path)
        color = MODEL_COLORS.get(mid, PASTEL["gray"])
        for ax, direction in ((ax_l2h, "low_to_high"), (ax_h2l, "high_to_low")):
            offsets, mean, sem = _switch_correctness_mean_sem(roll, direction)
            if not np.any(np.isfinite(mean)):
                continue
            ax.plot(offsets, mean, color=color, lw=2.0, label=_pretty(mid))
            ax.fill_between(
                offsets, mean - sem, mean + sem, color=color, alpha=0.28, linewidth=0
            )
            any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    for ax, title in (
        (ax_l2h, "2a. Switch 0.2 → 0.8 (mean ± SEM)"),
        (ax_h2l, "2b. Switch 0.8 → 0.2 (mean ± SEM)"),
    ):
        ax.axvline(0, color=PASTEL["ink"], ls=":", lw=1)
        ax.axhline(0.5, color=PASTEL["gray"], lw=1)
        ax.set_ylim(0.45, 1.02)
        style_axes_title(ax, title)
        style_xlabel(ax, "Trials relative to block switch")
        style_ylabel(ax, "Correctness")
        ax.legend(frameon=False, fontsize=8, loc="best", borderaxespad=0.5)

    style_suptitle(
        fig,
        f"From overall accuracy to switch-centered updating — {domain} — {regime}",
        y=1.02,
    )
    path = out / f"{domain}_{regime}_accuracy_to_switch_story.png"
    save_figure(fig, path)
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
            for maker in (
                _model_scorecard,
                _overall_correctness_board,
                _history_gap_board,
                _model_switch_board,
                _switch_correctness_board,
                _switch_correctness_summary_board,
                _accuracy_switch_story_board,
                _correctness_by_prior_board,
            ):
                if maker is _model_scorecard:
                    p = maker(cfg, domain, regime, out_score)
                else:
                    p = maker(cfg, domain, regime, out_cmp)
                if p:
                    paths.append(p)
        p = _synth_vs_real_board(cfg, regime, out_cmp)
        if p:
            paths.append(p)

    rows = _load_metric_rows(cfg, "real", "history_only")
    if rows:
        path = out_cmp / "real_transfer_correctness.png"
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        labels = [_pretty(r["model_id"]) for r in rows]
        colors = [MODEL_COLORS.get(r["model_id"], PASTEL["gray"]) for r in rows]
        acc, err = [], []
        for r in rows:
            m, e = _acc_ci_from_rollout(cfg, "real", "history_only", r["model_id"])
            if not np.isfinite(m):
                m, e = _acc(r), 0.0
            acc.append(m)
            err.append(e if np.isfinite(e) else 0.0)
        bars = ax.bar(labels, acc, color=colors, yerr=err, capsize=4, ecolor=PASTEL["ink"])
        pad_ylim_for_labels(ax, acc, err, floor=CORRECTNESS_YLIM[0], headroom=0.07)
        style_ylabel(ax, "Correctness vs correct stimulus side")
        style_axes_title(ax, "Real transfer correctness (history_only; mean ± 95% CI)")
        _label_above_bar(ax, bars, acc, err, pad=0.010)
        save_figure(fig, path)
        plt.close(fig)
        paths.append(path)
        legacy = out_cmp / "real_transfer_accuracy.png"
        shutil.copy(path, legacy)
        paths.append(legacy)
    return paths


def main() -> int:
    apply_style()
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
                "Correctness + history gap; bold titles; mean ± 95% CI."
            ),
            "scorecards/SCORECARD_GUIDE.md": "How to read scorecards.",
            "comparison/{domain}_{regime}_overall_correctness.png": (
                "Overall correctness vs correct side; mean ± 95% CI."
            ),
            "comparison/{domain}_{regime}_history_gap.png": (
                "History gap (0.8−0.2 zero-evidence belief); mean ± 95% CI."
            ),
            "comparison/{domain}_{regime}_switch_board.png": (
                "Zero-evidence belief around 0.2↔0.8 switches; mean ± SEM."
            ),
            "comparison/{domain}_{regime}_switch_correctness.png": (
                "Correctness around 0.2↔0.8 switches; mean ± SEM."
            ),
            "comparison/{domain}_{regime}_switch_correctness_summary.png": (
                "Post-switch (0–15) correctness by direction; mean ± 95% CI."
            ),
            "comparison/{domain}_{regime}_accuracy_to_switch_story.png": (
                "Story board: overall correctness then switch-centered correctness."
            ),
            "comparison/{domain}_{regime}_correctness_by_prior.png": (
                "Correctness stratified by block prior P(right)=0.2/0.5/0.8 "
                "plus balanced (equal-weight) correctness."
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
        "- Real domain = shared behavior+neural cohort (same sessions as neural VE).\n"
        "- Start here for model ranking numbers.\n\n"
        "## Multipanels (`by_model/...`)\n"
        "- **Synth:** psychometric + switch = averages over held-out synthetic sessions.\n"
        "- **Real:** one color per shared-cohort session; bottom-right = best session by accuracy.\n\n"
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
