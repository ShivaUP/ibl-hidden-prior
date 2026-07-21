"""Inspect IBL trial fields relevant to this project.

This module is exploratory only. It does not freeze preprocessing contracts.

Runnable directly (writes reports under reports/inspection/):

    python src/data/inspect_trials.py
    python src/data/inspect_trials.py --n-sessions 3
    python src/data/inspect_trials.py --eids 4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a

Sources for official IBL field definitions (the public data portal is a hub only):
- https://www.internationalbrainlab.com/data
- https://docs.internationalbrainlab.org/notebooks_external/loading_trials_data.html
- https://docs.internationalbrainlab.org/notebooks_external/data_structure.html
- https://docs.internationalbrainlab.org/_modules/ibllib/io/extractors/training_trials.html
- brainbox.behavior.training (signed contrast; biased blocks {0.2, 0.5, 0.8})
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Official IBL trials columns (raw ALF / SessionLoader fields)
# ---------------------------------------------------------------------------

IBL_FIELD_DEFINITIONS: dict[str, dict[str, str]] = {
    "contrastLeft": {
        "definition": (
            "Contrast of left-side stimulus in [0, 1]; NaN if the stimulus is on "
            "the other side."
        ),
        "source": "IBL trials ALF / ContrastLR extractor",
    },
    "contrastRight": {
        "definition": (
            "Contrast of right-side stimulus in [0, 1]; NaN if the stimulus is on "
            "the other side."
        ),
        "source": "IBL trials ALF / ContrastLR extractor",
    },
    "choice": {
        "definition": (
            "Response ID (IBL ALF spatial): -1 = right, +1 = left, 0 = no-go. "
            "Verified against feedbackType × contrast side in this repo."
        ),
        "source": "IBL trials ALF / Choice extractor",
    },
    "feedbackType": {
        "definition": (
            "Feedback sign: +1 = positive / reward (correct), -1 = negative "
            "(error or no-go), 0 = no feedback."
        ),
        "source": "IBL trials ALF / FeedbackType extractor",
    },
    "rewardVolume": {
        "definition": "Volume of reward delivered on each trial (µl); 0 if incorrect.",
        "source": "IBL trials ALF / RewardVolume extractor",
    },
    "probabilityLeft": {
        "definition": (
            "Hidden block prior: probability that the stimulus appears on the left "
            "(from stim_probability_left). Biased-choice sessions typically use "
            "{0.2, 0.5, 0.8}."
        ),
        "source": "IBL trials ALF / ProbabilityLeft; brainbox.behavior.training",
    },
    "stimOn_times": {
        "definition": "Stimulus onset time in seconds relative to session start.",
        "source": "IBL trials ALF",
    },
    "goCue_times": {
        "definition": "Go-cue tone time in seconds relative to session start.",
        "source": "IBL trials ALF / GoCueTimes extractor",
    },
    "response_times": {
        "definition": (
            "Time when a response was recorded (end of closed-loop state), in "
            "seconds relative to session start."
        ),
        "source": "IBL trials ALF / ResponseTimes extractor",
    },
    "feedback_times": {
        "definition": (
            "Time of feedback delivery (reward or error noise), in seconds "
            "relative to session start."
        ),
        "source": "IBL trials ALF / FeedbackTimes extractor",
    },
    "firstMovement_times": {
        "definition": "Time of first movement in seconds relative to session start.",
        "source": "IBL trials ALF",
    },
    "intervals_0": {
        "definition": "Trial start time (first column of trials.intervals).",
        "source": "IBL trials ALF / Intervals extractor",
    },
    "intervals_1": {
        "definition": "Trial end time (second column of trials.intervals).",
        "source": "IBL trials ALF / Intervals extractor",
    },
    "goCueTrigger_times": {
        "definition": (
            "Time the go-cue trigger command was sent (may precede measured "
            "goCue_times)."
        ),
        "source": "IBL trials ALF",
    },
    "stimOnTrigger_times": {
        "definition": "Time the stimulus-onset trigger command was sent.",
        "source": "IBL trials ALF",
    },
    "stimOff_times": {
        "definition": "Stimulus offset time in seconds relative to session start.",
        "source": "IBL trials ALF",
    },
    "quiescencePeriod": {
        "definition": "Required quiescence duration before the trial proceeds.",
        "source": "IBL trials ALF",
    },
}

# Core fields required by README / grilling for behavior modeling.
CORE_IBL_TRIAL_FIELDS: tuple[str, ...] = (
    "contrastLeft",
    "contrastRight",
    "choice",
    "feedbackType",
    "probabilityLeft",
    "stimOn_times",
    "goCue_times",
    "response_times",
    "feedback_times",
    "rewardVolume",
    "firstMovement_times",
    "intervals_0",
    "intervals_1",
)

# Extra trials columns often present; inspect but not required for v1 core.
OPTIONAL_IBL_TRIAL_FIELDS: tuple[str, ...] = (
    "goCueTrigger_times",
    "stimOnTrigger_times",
    "stimOff_times",
    "stimOffTrigger_times",
    "quiescencePeriod",
    "intervals_bpod_0",
    "intervals_bpod_1",
)

# Backward-compatible alias used by the inspection script.
PROJECT_TRIAL_FIELDS: tuple[str, ...] = CORE_IBL_TRIAL_FIELDS

# Expected coding checks against official IBL definitions.
EXPECTED_CHOICE_VALUES: frozenset[int] = frozenset({-1, 0, 1})
EXPECTED_FEEDBACK_TYPE_VALUES: frozenset[int] = frozenset({-1, 0, 1})
EXPECTED_PROBABILITY_LEFT_BIASED: frozenset[float] = frozenset({0.2, 0.5, 0.8})

# ---------------------------------------------------------------------------
# Project concepts: which IBL columns supply them
# ---------------------------------------------------------------------------

CONCEPT_TO_CANDIDATE_COLUMNS: dict[str, dict[str, Any]] = {
    "stimulus_side": {
        "ibl_columns": ("contrastLeft", "contrastRight"),
        "ibl_rule": (
            "Stim on left iff contrastLeft is finite and contrastRight is NaN; "
            "stim on right iff contrastRight is finite and contrastLeft is NaN."
        ),
        "project_derived": False,
    },
    "stimulus_contrast": {
        "ibl_columns": ("contrastLeft", "contrastRight"),
        "ibl_rule": (
            "Absolute contrast = non-NaN side value in [0, 1]. IBL also uses "
            "signed contrast = contrastRight - contrastLeft (left negative) via "
            "brainbox.behavior.training.get_signed_contrast."
        ),
        "project_derived": False,
    },
    "choice": {
        "ibl_columns": ("choice",),
        "ibl_rule": "-1 right, +1 left, 0 no-go (IBL ALF spatial convention).",
        "project_derived": False,
    },
    "correctness_or_reward": {
        "ibl_columns": ("feedbackType", "rewardVolume"),
        "ibl_rule": (
            "feedbackType +1 correct/reward, -1 error/no-go, 0 no feedback; "
            "rewardVolume > 0 only on rewarded trials."
        ),
        "project_derived": False,
    },
    "block_prior": {
        "ibl_columns": ("probabilityLeft",),
        "ibl_rule": (
            "probabilityLeft is the true hidden prior P(stim left). Biased blocks "
            "typically in {0.2, 0.5, 0.8}."
        ),
        "project_derived": False,
    },
    "stimulus_onset": {
        "ibl_columns": ("stimOn_times",),
        "ibl_rule": "Seconds from session start.",
        "project_derived": False,
    },
    "go_cue_response_permission": {
        "ibl_columns": ("goCue_times",),
        "ibl_rule": "Go-cue tone marks response permission.",
        "project_derived": False,
    },
    "response_time": {
        "ibl_columns": ("response_times",),
        "ibl_rule": "Absolute session time when response was recorded.",
        "project_derived": False,
    },
    "feedback_time": {
        "ibl_columns": ("feedback_times",),
        "ibl_rule": "Absolute session time of reward or error feedback.",
        "project_derived": False,
    },
    "reaction_time_ingredients": {
        "ibl_columns": ("goCue_times", "stimOn_times", "response_times"),
        "ibl_rule": (
            "IBL provides the ingredient times only. Project RT target is derived: "
            "response_times - goCue_times, fallback response_times - stimOn_times."
        ),
        "project_derived": True,
    },
}

# Encodings named in README that are NOT IBL columns (must be defined by us).
PROJECT_DERIVED_ENCODINGS: dict[str, str] = {
    "stimulus_right": (
        "Binary channel: 1 if stim on right (contrastRight finite), else 0. "
        "Not an IBL column."
    ),
    "contrast_high": (
        "Binary high/low contrast split of absolute contrast. Cutoff is a project "
        "decision; IBL only stores continuous contrastLeft/Right."
    ),
    "delay_phase": "Within-trial binary phase channel for RNN event bins. Project-defined.",
    "response_window": "Within-trial binary phase channel. Project-defined.",
    "response_made": "Within-trial binary event at response. Project-defined.",
    "reward": (
        "Binary reward-present channel. May be derived from feedbackType==1 or "
        "rewardVolume>0; incorrect-as-separate-channel is still open."
    ),
    "prev_choice_right": "Previous-trial history channel. Project-defined.",
    "prev_correct": "Previous-trial history channel. Project-defined.",
    "prev_fast_rt": (
        "Binary previous-RT summary (below session median). Project-defined; uses "
        "derived RT, not an IBL column."
    ),
    "oracle_prior_right": (
        "Full-information oracle prior channel from probabilityLeft. Project-defined "
        "binarization of an IBL field."
    ),
    "rt_target": (
        "Training target: response_times - goCue_times (fallback stimOn). "
        "Project-defined from IBL times."
    ),
}


@dataclass(frozen=True)
class FieldSummary:
    """Compact summary for one trials column."""

    name: str
    dtype: str
    n_total: int
    n_missing: int
    n_unique: int
    sample_values: list[Any]
    value_counts_top: dict[str, int] | None
    ibl_definition: str | None = None


def summarize_series(name: str, series: pd.Series, max_samples: int = 8) -> FieldSummary:
    """Summarize a single trials column for inspection reports."""
    n_total = int(len(series))
    n_missing = int(series.isna().sum())
    non_null = series.dropna()
    n_unique = int(non_null.nunique(dropna=True))

    samples = non_null.head(max_samples).tolist()
    value_counts_top: dict[str, int] | None = None
    if n_unique > 0 and n_unique <= 20:
        counts = non_null.value_counts(dropna=True).head(10)
        value_counts_top = {str(k): int(v) for k, v in counts.items()}

    ibl_def = None
    if name in IBL_FIELD_DEFINITIONS:
        ibl_def = IBL_FIELD_DEFINITIONS[name]["definition"]

    return FieldSummary(
        name=name,
        dtype=str(series.dtype),
        n_total=n_total,
        n_missing=n_missing,
        n_unique=n_unique,
        sample_values=samples,
        value_counts_top=value_counts_top,
        ibl_definition=ibl_def,
    )


def compute_rt_preview(trials: pd.DataFrame) -> pd.Series:
    """Preview project RT: goCue→response, else stimOn→response.

    IBL does not store RT as a column; only the ingredient times exist.
    """
    response = trials["response_times"]
    if "goCue_times" in trials.columns:
        rt = response - trials["goCue_times"]
        missing_go = trials["goCue_times"].isna()
        if missing_go.any() and "stimOn_times" in trials.columns:
            rt = rt.where(~missing_go, response - trials["stimOn_times"])
        return rt
    if "stimOn_times" in trials.columns:
        return response - trials["stimOn_times"]
    raise KeyError("Cannot compute RT preview without goCue_times or stimOn_times")


def absolute_contrast(trials: pd.DataFrame) -> pd.Series:
    """Absolute contrast on the stimulated side (IBL: other side is NaN)."""
    left = trials["contrastLeft"]
    right = trials["contrastRight"]
    abs_left = left.abs()
    abs_right = right.abs()
    both_nan = left.isna() & right.isna()
    out = pd.Series(np.nan, index=trials.index, dtype=float)
    left_only = left.notna() & right.isna()
    right_only = right.notna() & left.isna()
    both = left.notna() & right.notna()
    out.loc[left_only] = abs_left.loc[left_only]
    out.loc[right_only] = abs_right.loc[right_only]
    # Both finite should be rare; take max abs and flag in inspection separately.
    out.loc[both] = np.maximum(abs_left.loc[both], abs_right.loc[both])
    out.loc[both_nan] = np.nan
    return out


def signed_contrast(trials: pd.DataFrame) -> pd.Series:
    """IBL-style signed contrast: right positive, left negative (fraction, not %)."""
    left = trials["contrastLeft"].fillna(0.0)
    right = trials["contrastRight"].fillna(0.0)
    return right - left


def stimulus_side_label(trials: pd.DataFrame) -> pd.Series:
    """Label stim side from contrast NaN pattern: 'left', 'right', 'both', or 'none'."""
    left_ok = trials["contrastLeft"].notna()
    right_ok = trials["contrastRight"].notna()
    out = pd.Series("none", index=trials.index, dtype=object)
    out.loc[left_ok & ~right_ok] = "left"
    out.loc[right_ok & ~left_ok] = "right"
    out.loc[left_ok & right_ok] = "both"
    return out


def _coding_checks(trials: pd.DataFrame) -> dict[str, Any]:
    """Validate observed values against official IBL encodings."""
    checks: dict[str, Any] = {}

    if "choice" in trials.columns:
        vals = set(int(v) for v in trials["choice"].dropna().unique())
        checks["choice"] = {
            "observed_values": sorted(vals),
            "expected_values": sorted(EXPECTED_CHOICE_VALUES),
            "matches_ibl_encoding": vals.issubset(EXPECTED_CHOICE_VALUES),
        }

    if "feedbackType" in trials.columns:
        vals = set(int(v) for v in trials["feedbackType"].dropna().unique())
        checks["feedbackType"] = {
            "observed_values": sorted(vals),
            "expected_values": sorted(EXPECTED_FEEDBACK_TYPE_VALUES),
            "matches_ibl_encoding": vals.issubset(EXPECTED_FEEDBACK_TYPE_VALUES),
        }

    if "probabilityLeft" in trials.columns:
        observed = {
            float(v) for v in trials["probabilityLeft"].dropna().unique()
        }
        checks["probabilityLeft"] = {
            "observed_values": sorted(observed),
            "expected_biased_set": sorted(EXPECTED_PROBABILITY_LEFT_BIASED),
            "subset_of_expected_biased_set": observed.issubset(
                EXPECTED_PROBABILITY_LEFT_BIASED
            ),
        }

    if {"contrastLeft", "contrastRight"}.issubset(trials.columns):
        side = stimulus_side_label(trials)
        checks["stimulus_side_pattern"] = {
            str(k): int(v) for k, v in side.value_counts().items()
        }
        checks["n_both_sides_finite"] = int((side == "both").sum())
        checks["n_neither_side_finite"] = int((side == "none").sum())

    return checks


def inspect_trials_table(trials: pd.DataFrame) -> dict[str, Any]:
    """Build a structured inspection payload for one session trials table."""
    columns = list(trials.columns)
    present_core = [c for c in CORE_IBL_TRIAL_FIELDS if c in trials.columns]
    missing_core = [c for c in CORE_IBL_TRIAL_FIELDS if c not in trials.columns]
    present_optional = [c for c in OPTIONAL_IBL_TRIAL_FIELDS if c in trials.columns]

    concept_coverage: dict[str, dict[str, Any]] = {}
    for concept, meta in CONCEPT_TO_CANDIDATE_COLUMNS.items():
        candidates = tuple(meta["ibl_columns"])
        found = [c for c in candidates if c in trials.columns]
        concept_coverage[concept] = {
            "ibl_columns": list(candidates),
            "found_columns": found,
            "covered": len(found) == len(candidates),
            "project_derived": bool(meta["project_derived"]),
            "ibl_rule": meta["ibl_rule"],
        }

    fields_to_summarize = list(
        dict.fromkeys(present_core + present_optional)
    )
    field_summaries = {
        name: summarize_series(name, trials[name]).__dict__
        for name in fields_to_summarize
    }

    extras: dict[str, Any] = {
        "n_trials": int(len(trials)),
        "all_columns": columns,
        "present_project_fields": present_core,  # alias for script compatibility
        "missing_project_fields": missing_core,
        "present_core_ibl_fields": present_core,
        "missing_core_ibl_fields": missing_core,
        "present_optional_ibl_fields": present_optional,
        "ibl_field_definitions": {
            name: IBL_FIELD_DEFINITIONS[name]
            for name in fields_to_summarize
            if name in IBL_FIELD_DEFINITIONS
        },
        "project_derived_encodings": PROJECT_DERIVED_ENCODINGS,
        "concept_coverage": concept_coverage,
        "field_summaries": field_summaries,
        "coding_checks": _coding_checks(trials),
    }

    if {"contrastLeft", "contrastRight"}.issubset(trials.columns):
        abs_c = absolute_contrast(trials)
        extras["absolute_contrast_value_counts"] = {
            str(k): int(v) for k, v in abs_c.value_counts(dropna=False).items()
        }
        signed = signed_contrast(trials)
        extras["signed_contrast_unique_rounded"] = sorted(
            {round(float(v), 4) for v in signed.dropna().unique()}
        )

    if "probabilityLeft" in trials.columns:
        extras["probabilityLeft_value_counts"] = {
            str(k): int(v)
            for k, v in trials["probabilityLeft"].value_counts(dropna=False).items()
        }

    if "response_times" in trials.columns and (
        "goCue_times" in trials.columns or "stimOn_times" in trials.columns
    ):
        rt = compute_rt_preview(trials)
        finite = rt.replace([np.inf, -np.inf], np.nan).dropna()
        extras["rt_preview_seconds"] = {
            "definition": (
                "project-derived: response_times - goCue_times "
                "(fallback: response_times - stimOn_times)"
            ),
            "n_finite": int(finite.shape[0]),
            "n_nonpositive": int((finite <= 0).sum()),
            "min": float(finite.min()) if len(finite) else None,
            "median": float(finite.median()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "p01": float(finite.quantile(0.01)) if len(finite) else None,
            "p99": float(finite.quantile(0.99)) if len(finite) else None,
        }

    if {"stimOn_times", "goCue_times", "response_times", "feedback_times"}.issubset(
        trials.columns
    ):
        order_ok = (
            trials["stimOn_times"].notna()
            & trials["goCue_times"].notna()
            & trials["response_times"].notna()
            & trials["feedback_times"].notna()
            & (trials["stimOn_times"] <= trials["goCue_times"])
            & (trials["goCue_times"] <= trials["response_times"])
            & (trials["response_times"] <= trials["feedback_times"])
        )
        extras["event_order_check"] = {
            "n_complete_timing": int(
                trials[
                    ["stimOn_times", "goCue_times", "response_times", "feedback_times"]
                ]
                .notna()
                .all(axis=1)
                .sum()
            ),
            "n_monotonic_stim_go_resp_fb": int(order_ok.sum()),
        }

    return extras


def format_inspection_text(eid: str, payload: dict[str, Any]) -> str:
    """Render a human-readable inspection report for one eid."""
    lines: list[str] = []
    lines.append(f"=== Session {eid} ===")
    lines.append(f"n_trials: {payload['n_trials']}")
    lines.append("")
    lines.append("All trials columns:")
    for col in payload["all_columns"]:
        lines.append(f"  - {col}")
    lines.append("")
    lines.append("Core IBL fields present:")
    for col in payload["present_core_ibl_fields"]:
        lines.append(f"  - {col}")
    lines.append("")
    lines.append("Core IBL fields missing:")
    if payload["missing_core_ibl_fields"]:
        for col in payload["missing_core_ibl_fields"]:
            lines.append(f"  - {col}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Optional IBL fields present:")
    if payload["present_optional_ibl_fields"]:
        for col in payload["present_optional_ibl_fields"]:
            lines.append(f"  - {col}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Official IBL definitions (for present fields):")
    for name, meta in payload.get("ibl_field_definitions", {}).items():
        lines.append(f"  {name}: {meta['definition']}")
        lines.append(f"    source: {meta['source']}")
    lines.append("")
    lines.append("Project-derived encodings (NOT IBL columns; still open decisions):")
    for name, desc in payload.get("project_derived_encodings", {}).items():
        lines.append(f"  - {name}: {desc}")
    lines.append("")
    lines.append("Concept → IBL column coverage:")
    for concept, info in payload["concept_coverage"].items():
        status = "OK" if info["covered"] else "MISSING"
        derived = "derived" if info["project_derived"] else "raw"
        lines.append(
            f"  [{status}|{derived}] {concept}: found={info['found_columns']}"
        )
        lines.append(f"    rule: {info['ibl_rule']}")
    lines.append("")
    lines.append("Coding checks vs IBL encodings:")
    for key, info in payload.get("coding_checks", {}).items():
        lines.append(f"  {key}: {info}")
    lines.append("")
    lines.append("Field summaries:")
    for name, summary in payload["field_summaries"].items():
        lines.append(
            f"  {name}: dtype={summary['dtype']} missing={summary['n_missing']}/"
            f"{summary['n_total']} unique={summary['n_unique']}"
        )
        if summary.get("ibl_definition"):
            lines.append(f"    ibl: {summary['ibl_definition']}")
        if summary["value_counts_top"] is not None:
            lines.append(f"    value_counts: {summary['value_counts_top']}")
        else:
            lines.append(f"    samples: {summary['sample_values']}")

    if "absolute_contrast_value_counts" in payload:
        lines.append("")
        lines.append(
            f"Absolute contrast value counts: {payload['absolute_contrast_value_counts']}"
        )
    if "signed_contrast_unique_rounded" in payload:
        lines.append(
            f"Signed contrast unique (rounded): {payload['signed_contrast_unique_rounded']}"
        )
    if "probabilityLeft_value_counts" in payload:
        lines.append(
            f"probabilityLeft value counts: {payload['probabilityLeft_value_counts']}"
        )
    if "rt_preview_seconds" in payload:
        lines.append(f"RT preview (seconds): {payload['rt_preview_seconds']}")
    if "event_order_check" in payload:
        lines.append(f"Event order check: {payload['event_order_check']}")
    lines.append("")
    return "\n".join(lines)


def resolve_example_eids(one: Any, n: int = 3) -> list[str]:
    """Resolve a few real public eids; prefer biased-block behavior sessions."""
    seed = ["4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a"]
    found: list[str] = []
    try:
        eids = one.search(task_protocol="_iblrig_tasks_biasedChoiceWorld", limit=20)
        if isinstance(eids, tuple):
            eids = eids[0]
        found = [str(e) for e in list(eids)[: max(n, 1)]]
    except Exception as exc:  # noqa: BLE001 - inspection script should continue
        print(f"[warn] one.search failed ({exc}); falling back to doc example eid(s).")

    out: list[str] = []
    for eid in seed + found:
        if eid not in out:
            out.append(eid)
        if len(out) >= n:
            break
    return out[:n]


def load_trials_for_eid(one: Any, eid: str) -> pd.DataFrame:
    """Load trials for one eid via SessionLoader, with ONE object fallback."""
    try:
        from brainbox.io.one import SessionLoader

        loader = SessionLoader(one=one, eid=eid)
        loader.load_trials()
        trials = loader.trials
        if trials is None or len(trials) == 0:
            raise RuntimeError("SessionLoader.trials empty")
        return trials
    except Exception as exc:
        print(f"[warn] SessionLoader failed for {eid} ({exc}); trying one.load_object.")
        trials = one.load_object(eid, "trials", collection="alf")
        if isinstance(trials, dict):
            return pd.DataFrame(trials)
        return trials


def _missing_deps_message(missing: list[str]) -> str:
    return (
        "Missing required packages: "
        + ", ".join(missing)
        + "\n\nInstall into the project virtualenv, then re-run:\n"
        "  pip install ONE-api ibllib pandas numpy\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a few public IBL sessions and write trial-field inspection reports."
        )
    )
    parser.add_argument(
        "--n-sessions",
        type=int,
        default=3,
        help="Number of sessions to inspect when --eids is not provided.",
    )
    parser.add_argument(
        "--eids",
        nargs="+",
        default=None,
        help="Optional explicit experiment IDs (eids) to inspect.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "inspection",
        help="Directory for text/json inspection reports.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "one_cache",
        help="ONE download/cache directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: download/inspect sessions and write summary files."""
    missing: list[str] = []
    try:
        from one.api import ONE
    except ImportError:
        missing.append("ONE-api")
        ONE = None  # type: ignore[assignment]
    try:
        import brainbox  # noqa: F401
    except ImportError:
        missing.append("ibllib (provides brainbox)")

    if missing:
        print(_missing_deps_message(missing), file=sys.stderr)
        return 1

    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(args.cache_dir),
    )

    eids = args.eids if args.eids else resolve_example_eids(one, n=args.n_sessions)
    print(f"Inspecting {len(eids)} session(s):")
    for eid in eids:
        print(f"  - {eid}")

    session_reports: list[dict[str, Any]] = []
    text_blocks: list[str] = []
    failures: list[dict[str, str]] = []

    for eid in eids:
        print(f"\nLoading trials for {eid} ...")
        try:
            trials = load_trials_for_eid(one, eid)
            payload = inspect_trials_table(trials)
            payload["eid"] = eid
            session_reports.append(payload)
            text_blocks.append(format_inspection_text(eid, payload))
            print(
                f"  loaded {payload['n_trials']} trials; "
                f"columns={len(payload['all_columns'])}"
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {msg}")
            failures.append({"eid": eid, "error": msg})

    stamp = datetime.now(timezone.utc).isoformat()
    summary = {
        "created_utc": stamp,
        "n_requested": len(eids),
        "n_succeeded": len(session_reports),
        "eids": eids,
        "failures": failures,
        "sessions": session_reports,
        "notes": [
            "Discovery only; does not freeze the preprocessing contract.",
            "Raw IBL field definitions come from IBL docs/extractors, not the /data portal page.",
            "RT preview is project-derived: goCue→response with stimOn→response fallback.",
            "Contrast preview uses abs contrast on the non-NaN stimulated side.",
            "Project-derived encodings (contrast_high, prev_fast_rt, etc.) are listed separately.",
        ],
    }

    txt_path = args.out_dir / "ibl_trial_fields_summary.txt"
    json_path = args.out_dir / "ibl_trial_fields_summary.json"

    header = [
        "IBL trial-field inspection",
        f"created_utc: {stamp}",
        f"cache_dir: {args.cache_dir}",
        f"succeeded: {len(session_reports)} / {len(eids)}",
        "",
    ]
    if failures:
        header.append("Failures:")
        for item in failures:
            header.append(f"  - {item['eid']}: {item['error']}")
        header.append("")

    txt_path.write_text("\n".join(header) + "\n".join(text_blocks), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nWrote:\n  {txt_path}\n  {json_path}")
    if not session_reports:
        print("No sessions loaded successfully.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
