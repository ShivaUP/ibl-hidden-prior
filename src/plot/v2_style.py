"""Shared v2 figure style: pastel colors, DPI 600, bold titles/labels, CI helpers."""

from __future__ import annotations

import numpy as np

# Publication-quality raster export
SAVE_DPI = 600

# Darker pastel model palette (still soft, higher chroma than before)
MODEL_COLORS = {
    "tanh_bptt": "#6B9BD2",  # blue
    "tanh_pc": "#6FAF72",  # green
    "gru": "#D47A82",  # rose
    "gru_pc": "#5EAEA6",  # teal
    "bayes": "#9B7DB8",  # lavender (legacy)
}

# Darker pastel accents for paired / direction plots
PASTEL = {
    "blue": "#6B9BD2",
    "orange": "#D49A5C",
    "green": "#6FAF72",
    "rose": "#D47A82",
    "lavender": "#9B7DB8",
    "teal": "#5EAEA6",
    "yellow": "#C4B04A",
    "gray": "#B0B0B0",
    "ink": "#3A3A3A",
}

# Session colors (cycle) — darker pastels
SESSION_PASTELS = [
    "#6B9BD2",
    "#D49A5C",
    "#6FAF72",
    "#D47A82",
    "#9B7DB8",
    "#5EAEA6",
    "#C4B04A",
    "#B8896A",
    "#7A96C4",
    "#8AAD6A",
    "#C488A8",
    "#6AAD9C",
]

# Block-prior colors for grouped correctness bars
PRIOR_COLORS = {
    0.2: "#D49A5C",  # orange — left-biased (P(right)=0.2)
    0.5: "#5EAEA6",  # teal — unbiased
    0.8: "#6B9BD2",  # blue — right-biased
}

ACCURACY_YLIM = (0.50, 1.00)
CORRECTNESS_YLIM = ACCURACY_YLIM

TITLE_FONTSIZE = 13
SUPTITLE_FONTSIZE = 15
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 9
TITLE_PAD = 12
LABEL_PAD = 8
SAVE_PAD_INCHES = 0.35


def session_colors(n: int) -> list:
    if n <= 0:
        return []
    if n <= len(SESSION_PASTELS):
        return SESSION_PASTELS[:n]
    return [SESSION_PASTELS[i % len(SESSION_PASTELS)] for i in range(n)]


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    """Return (mean, SEM) over finite values. SEM=0 if n<2."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(v))
    if v.size < 2:
        return mean, 0.0
    return mean, float(np.std(v, ddof=1) / np.sqrt(v.size))


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    """Return (mean, half-width of approx 95% CI = 1.96 * SEM)."""
    mean, sem = mean_sem(values)
    if not np.isfinite(sem):
        return mean, float("nan")
    return mean, 1.96 * sem


def style_axes_title(ax, text: str, *, fontsize: int | None = None, pad: float | None = None) -> None:
    ax.set_title(
        text,
        fontweight="bold",
        fontsize=fontsize or TITLE_FONTSIZE,
        pad=TITLE_PAD if pad is None else pad,
    )


def style_xlabel(ax, text: str, *, fontsize: int | None = None) -> None:
    ax.set_xlabel(
        text,
        fontweight="bold",
        fontsize=fontsize or LABEL_FONTSIZE,
        labelpad=LABEL_PAD,
    )


def style_ylabel(ax, text: str, *, fontsize: int | None = None) -> None:
    ax.set_ylabel(
        text,
        fontweight="bold",
        fontsize=fontsize or LABEL_FONTSIZE,
        labelpad=LABEL_PAD,
    )


def style_axes_labels(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel is not None:
        style_xlabel(ax, xlabel)
    if ylabel is not None:
        style_ylabel(ax, ylabel)


def style_suptitle(fig, text: str, *, fontsize: int | None = None, y: float = 1.04) -> None:
    fig.suptitle(
        text,
        fontweight="bold",
        fontsize=fontsize or SUPTITLE_FONTSIZE,
        y=y,
    )


def finalize_axes(ax) -> None:
    """Ensure title + axis labels are bold even if set via ax.set(...)."""
    t = ax.get_title()
    if t:
        ax.set_title(t, fontweight="bold", fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    xl = ax.get_xlabel()
    if xl:
        ax.set_xlabel(xl, fontweight="bold", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    yl = ax.get_ylabel()
    if yl:
        ax.set_ylabel(yl, fontweight="bold", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.tick_params(labelsize=TICK_FONTSIZE)


def finalize_figure(fig) -> None:
    for ax in fig.axes:
        finalize_axes(ax)
    st = fig._suptitle  # noqa: SLF001
    if st is not None and st.get_text():
        st.set_fontweight("bold")
        st.set_fontsize(SUPTITLE_FONTSIZE)


def save_figure(fig, path, *, dpi: int | None = None) -> None:
    """Save with tight padding so titles/labels are not clipped."""
    finalize_figure(fig)
    fig.savefig(
        path,
        dpi=dpi or SAVE_DPI,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )


def pad_ylim_for_labels(
    ax,
    values,
    errs=None,
    *,
    floor: float | None = None,
    ceil: float | None = None,
    headroom: float = 0.06,
) -> None:
    """Expand y-limits so bar labels / error bars sit inside the axes."""
    vals = np.asarray(values, dtype=float)
    if errs is None:
        errs = np.zeros_like(vals)
    else:
        errs = np.asarray(errs, dtype=float)
        errs = np.where(np.isfinite(errs), errs, 0.0)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return
    tops = vals[finite] + errs[finite]
    bots = vals[finite] - errs[finite]
    ymin = float(np.min(bots))
    ymax = float(np.max(tops))
    span = max(ymax - ymin, 1e-3)
    pad = max(headroom, 0.08 * span)
    lo = ymin - 0.02 * span
    hi = ymax + pad
    if floor is not None:
        lo = min(lo, floor) if lo < floor else floor
        # keep floor as lower bound when all values are above it
        lo = floor if ymin >= floor else lo
    if ceil is not None:
        hi = max(hi, min(ceil, ymax + pad))
        if ymax <= ceil:
            hi = max(hi, min(ceil, ymax + pad))
            # Prefer not clipping labels: allow slightly above ceil if needed
            if ymax + pad > ceil:
                hi = ymax + pad
    ax.set_ylim(lo, hi)


def label_above_bars(ax, bars, values, errs=None, *, pad: float = 0.012, fontsize: int = 8) -> None:
    """Place value labels just above bar faces (not above CI tips), clipped to axes."""
    if errs is None:
        errs = [0.0] * len(values)
    for b, v, e in zip(bars, values, errs):
        if not np.isfinite(v):
            continue
        y = float(v) + float(pad)
        ax.text(
            b.get_x() + b.get_width() / 2,
            y,
            f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=True,
            fontweight="normal",
        )


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": SAVE_DPI,
            "font.size": 10,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.titleweight": "bold",
            "axes.titlepad": TITLE_PAD,
            "axes.labelsize": LABEL_FONTSIZE,
            "axes.labelweight": "bold",
            "axes.labelpad": LABEL_PAD,
            "figure.titlesize": SUPTITLE_FONTSIZE,
            "figure.titleweight": "bold",
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "axes.facecolor": "#FBFBFB",
            "figure.facecolor": "white",
            "axes.edgecolor": "#888888",
            "axes.labelcolor": PASTEL["ink"],
            "xtick.color": PASTEL["ink"],
            "ytick.color": PASTEL["ink"],
            "text.color": PASTEL["ink"],
            "grid.color": "#E8E8E8",
            "legend.frameon": False,
            "figure.constrained_layout.use": False,
            "figure.subplot.hspace": 0.35,
            "figure.subplot.wspace": 0.30,
        }
    )
