"""Shared v2 figure style: pastel colors, DPI 600, bold titles/labels, CI helpers."""

from __future__ import annotations

import numpy as np

# Publication-quality raster export
SAVE_DPI = 600

# Fixed display / ranking order for all v2 figures
MODEL_ORDER = ("tanh_bptt", "tanh_pc", "gru", "gru_pc")

# Twin-complement pastel palette:
#   tanh BPTT (blue) ↔ tanh PC (amber)
#   GRU (rose) ↔ GRU PC (teal)
MODEL_COLORS = {
    "tanh_bptt": "#5B8FD9",  # blue
    "tanh_pc": "#E39B3A",  # amber (complement of blue)
    "gru": "#D45C6A",  # rose
    "gru_pc": "#2FA89A",  # teal (complement of rose)
    "bayes": "#9B7DB8",  # lavender (legacy)
}

# Darker pastel accents for paired / direction plots
PASTEL = {
    "blue": "#5B8FD9",
    "orange": "#E39B3A",
    "green": "#6FAF72",
    "rose": "#D45C6A",
    "lavender": "#9B7DB8",
    "teal": "#2FA89A",
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
    0.2: "#E39B3A",  # amber — left-biased (P(right)=0.2)
    0.5: "#2FA89A",  # teal — unbiased
    0.8: "#5B8FD9",  # blue — right-biased
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


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        *[max(0, min(255, int(round(c * 255)))) for c in rgb]
    )


def _mix(hex_color: str, toward: str, amount: float) -> str:
    a = np.asarray(_hex_to_rgb(hex_color), dtype=float)
    b = np.asarray(_hex_to_rgb(toward), dtype=float)
    return _rgb_to_hex(tuple(a * (1.0 - amount) + b * amount))  # type: ignore[arg-type]


def model_bar_shades(model_id: str) -> dict[str, str]:
    """Slight shade variants around the model color for grouped bars.

    overall = darker, low_to_high (0.2→0.8) = base, high_to_low (0.8→0.2) = lighter.
    """
    base = MODEL_COLORS.get(model_id, PASTEL["gray"])
    return {
        "overall": _mix(base, "#1A1A1A", 0.22),
        "low_to_high": base,
        "high_to_low": _mix(base, "#FFFFFF", 0.28),
    }


def ordered_models(model_ids) -> list[str]:
    """Return active models in canonical order, then any extras."""
    ids = [str(m) for m in model_ids]
    primary = [m for m in MODEL_ORDER if m in ids]
    extras = [m for m in ids if m not in MODEL_ORDER]
    return primary + extras


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
    """Save with tight padding so titles/labels are not clipped (publication DPI)."""
    finalize_figure(fig)
    fig.savefig(
        path,
        dpi=dpi or SAVE_DPI,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
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
