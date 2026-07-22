#!/usr/bin/env python3
"""01 — Search candidate eids, score QC, pin behavior-core eid list.

Usage:
  python scripts/01_run_session_qc.py
  python scripts/01_run_session_qc.py --n-candidates 30 --max-core 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.config import load_frozen_config, repo_root
from src.data.inspect_trials import load_trials_for_eid
from src.data.qc import score_session


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run almost-perfect session QC.")
    p.add_argument("--n-candidates", type=int, default=30)
    p.add_argument("--max-core", type=int, default=10, help="Max eids in behavior-core.")
    p.add_argument(
        "--seed-eid",
        default="4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a",
        help="Always include this doc example eid in candidates.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="ONE cache dir (default from config / data/raw/one_cache).",
    )
    return p.parse_args()


def make_one(cache_dir: Path):
    from one.api import ONE

    cache_dir.mkdir(parents=True, exist_ok=True)
    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(cache_dir),
    )


def search_candidates(one, n: int, seed_eid: str) -> list[str]:
    found: list[str] = []
    try:
        eids = one.search(task_protocol="_iblrig_tasks_biasedChoiceWorld", limit=max(n * 2, 20))
        if isinstance(eids, tuple):
            eids = eids[0]
        found = [str(e) for e in list(eids)]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] search failed: {exc}")

    out: list[str] = []
    for eid in [seed_eid, *found]:
        if eid and eid not in out:
            out.append(eid)
        if len(out) >= n:
            break
    return out


def main() -> int:
    args = parse_args()
    cfg = load_frozen_config()
    root = repo_root()
    cache_dir = args.cache_dir or (root / cfg["data"]["cache_dir"])
    manifest_dir = root / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    reports_qc = root / "reports" / "qc"
    reports_qc.mkdir(parents=True, exist_ok=True)

    inc = cfg["data"]["session_inclusion"]
    almost = inc["almost_perfect"]
    trial_inc = cfg["data"]["trial_inclusion"]
    rt_pct = tuple(trial_inc["rt_percentile_trim"])

    stamp = datetime.now(timezone.utc).isoformat()
    one = make_one(Path(cache_dir))
    candidates = search_candidates(one, args.n_candidates, args.seed_eid)

    candidates_payload = {
        "created_utc": stamp,
        "query": {
            "task_protocol": "_iblrig_tasks_biasedChoiceWorld",
            "n_requested": args.n_candidates,
            "seed_eid": args.seed_eid,
        },
        "eids": candidates,
    }
    candidates_path = manifest_dir / "candidates_raw.json"
    candidates_path.write_text(json.dumps(candidates_payload, indent=2), encoding="utf-8")
    print(f"Wrote {candidates_path} ({len(candidates)} eids)")

    rows: list[dict] = []
    for i, eid in enumerate(candidates, start=1):
        print(f"[{i}/{len(candidates)}] QC {eid} ...")
        try:
            trials = load_trials_for_eid(one, eid)
            result = score_session(
                eid,
                trials,
                min_choice_trials=int(inc["min_choice_trials"]),
                min_fraction_pre=float(almost["min_fraction_pass_timing_choice_rules"]),
                min_fraction_post=float(almost["min_fraction_after_rt_percentile_trim"]),
                expected_probability_left=tuple(inc["probability_left_set"]),
                rt_percentiles=(float(rt_pct[0]), float(rt_pct[1])),
                require_left_right_bias=bool(inc["require_left_and_right_bias_blocks"]),
            )
            row = result.to_dict()
            row["error"] = ""
            rows.append(row)
            status = "PASS" if result.passes_session else "FAIL"
            print(
                f"  {status} choice={result.n_choice_trials} "
                f"pre={result.fraction_pass_rules_1_to_4:.3f} "
                f"post={result.fraction_after_rt_trim:.3f} "
                f"reasons={result.fail_reasons}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            rows.append(
                {
                    "eid": eid,
                    "passes_session": False,
                    "fail_reasons": [f"load_error:{type(exc).__name__}"],
                    "error": str(exc),
                    "n_trials": 0,
                    "n_choice_trials": 0,
                    "fraction_pass_rules_1_to_4": 0.0,
                    "fraction_after_rt_trim": 0.0,
                }
            )

    df = pd.DataFrame(rows)
    qc_csv = manifest_dir / "qc_scores.csv"
    df.to_csv(qc_csv, index=False)
    qc_json = reports_qc / "qc_scores.json"
    qc_json.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"Wrote {qc_csv}")

    passed = df[df.get("passes_session", False) == True]  # noqa: E712
    # Prefer higher post-trim fraction, then more choice trials.
    if "fraction_after_rt_trim" in passed.columns:
        passed = passed.sort_values(
            by=["fraction_after_rt_trim", "n_choice_trials"],
            ascending=[False, False],
        )
    core_eids = passed["eid"].head(args.max_core).tolist()

    core_payload = {
        "created_utc": stamp,
        "config_version": cfg.get("version"),
        "n_candidates": len(candidates),
        "n_passed": int(len(passed)),
        "n_core": len(core_eids),
        "gates": {
            "min_choice_trials": inc["min_choice_trials"],
            "min_fraction_pre": almost["min_fraction_pass_timing_choice_rules"],
            "min_fraction_post": almost["min_fraction_after_rt_percentile_trim"],
            "rt_percentile_trim": list(rt_pct),
        },
        "eids": core_eids,
        "per_eid": [
            {
                "eid": r["eid"],
                "n_choice_trials": int(r.get("n_choice_trials", 0) or 0),
                "fraction_pass_rules_1_to_4": float(r.get("fraction_pass_rules_1_to_4", 0) or 0),
                "fraction_after_rt_trim": float(r.get("fraction_after_rt_trim", 0) or 0),
                "probability_left_values": r.get("probability_left_values", []),
                "absolute_contrast_levels": r.get("absolute_contrast_levels", []),
            }
            for _, r in passed.head(args.max_core).iterrows()
        ],
        "notes": [
            "Almost-perfect sessions only.",
            "Re-run scripts/01_run_session_qc.py to refresh.",
        ],
    }
    core_path = manifest_dir / "behavior_core_eids.json"
    core_path.write_text(json.dumps(core_payload, indent=2), encoding="utf-8")
    print(f"Wrote {core_path} ({len(core_eids)} core eids)")

    summary_path = reports_qc / "qc_summary.txt"
    lines = [
        f"created_utc: {stamp}",
        f"candidates: {len(candidates)}",
        f"passed: {len(passed)}",
        f"core: {len(core_eids)}",
        "",
        "Core eids:",
    ]
    for eid in core_eids:
        lines.append(f"  - {eid}")
    if not core_eids:
        lines.append("  (none — loosen search or inspect failures in qc_scores.csv)")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0 if core_eids else 2


if __name__ == "__main__":
    raise SystemExit(main())
