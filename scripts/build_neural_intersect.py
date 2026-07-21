#!/usr/bin/env python3
"""Phase 8.1: build neural-intersect manifest (MOs / vlOFC).

Strict intersect: behavior-core ∩ BWM sessions with spikes in MOs and/or ORBvl.
If empty (expected for behavior-first core), also QC-screen ROI candidates into
an expanded `neural_behavior_pool` for Phase 8 preprocessing.

Usage:
    python scripts/build_neural_intersect.py
    python scripts/build_neural_intersect.py --max-qc 20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.config import load_frozen_config, repo_root
from src.data.inspect_trials import load_trials_for_eid
from src.data.qc import score_session
from src.neural.regions import REGION_ALIASES, acronyms_for_spec_region


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build neural-intersect manifests.")
    p.add_argument("--max-qc", type=int, default=40, help="Max ROI eids to QC-screen.")
    p.add_argument("--skip-qc", action="store_true", help="Only query; do not download/QC.")
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


def search_roi_eids(one) -> dict[str, list[str]]:
    """BWM sessions with spikes.times tagged to each Allen acronym."""
    out: dict[str, list[str]] = {}
    for spec, acronyms in REGION_ALIASES.items():
        eids: list[str] = []
        for acr in acronyms:
            found = one.search(
                projects="ibl_neuropixel_brainwide_01",
                atlas_acronym=acr,
                dataset_types="spikes.times",
            )
            if isinstance(found, tuple):
                found = found[0]
            for e in found:
                s = str(e)
                if s not in eids:
                    eids.append(s)
        out[spec] = eids
    return out


def main() -> int:
    args = parse_args()
    cfg = load_frozen_config()
    root = repo_root()
    stamp = datetime.now(timezone.utc).isoformat()
    cache_dir = root / cfg["data"]["cache_dir"]
    man_dir = root / "data" / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    reports = root / "reports" / "neural"
    reports.mkdir(parents=True, exist_ok=True)

    core_path = man_dir / "behavior_core_eids.json"
    core = set(json.loads(core_path.read_text(encoding="utf-8"))["eids"])

    one = make_one(Path(cache_dir))
    by_region = search_roi_eids(one)
    mos = set(by_region["MOs"])
    orbvl = set(by_region["vlOFC_orbvl"])
    union = sorted(mos | orbvl)

    strict = sorted(core & (mos | orbvl))
    per_region_core = {k: sorted(core & set(v)) for k, v in by_region.items()}

    payload = {
        "created_utc": stamp,
        "config_version": cfg.get("version", "frozen_v1"),
        "region_aliases": {k: list(v) for k, v in REGION_ALIASES.items()},
        "query": {
            "projects": "ibl_neuropixel_brainwide_01",
            "dataset_types": "spikes.times",
            "atlas_acronyms": {k: list(acronyms_for_spec_region(k)) for k in REGION_ALIASES},
        },
        "n_bwm_by_region": {k: len(v) for k, v in by_region.items()},
        "n_bwm_union_MOs_ORBvl": len(union),
        "n_both_MOs_and_ORBvl": len(mos & orbvl),
        "behavior_core_n": len(core),
        "strict_intersect_eids": strict,
        "strict_intersect_n": len(strict),
        "strict_per_region": per_region_core,
        "blocker": None
        if strict
        else (
            "behavior-core ∩ (BWM spikes in MOs|ORBvl) is empty. "
            "Behavior-first core was selected without requiring ephys. "
            "Use neural_behavior_pool (same almost-perfect QC on ROI BWM eids) for Phase 8."
        ),
        "eids": strict,  # primary key expected by frozen path; may be empty
        "bwm_roi_union_eids": union,
    }

    # Expanded pool: QC ROI candidates
    pool_scores: list[dict] = []
    pool_pass: list[str] = []
    if not args.skip_qc:
        inc = cfg["data"]["session_inclusion"]
        almost = inc["almost_perfect"]
        trial_inc = cfg["data"]["trial_inclusion"]
        rt_pct = tuple(trial_inc["rt_percentile_trim"])
        to_score = union[: args.max_qc]
        print(f"QC-screening {len(to_score)} / {len(union)} ROI BWM eids...")
        for i, eid in enumerate(to_score, 1):
            print(f"  [{i}/{len(to_score)}] {eid}")
            try:
                trials = load_trials_for_eid(one, eid)
                res = score_session(
                    eid,
                    trials,
                    min_choice_trials=int(inc["min_choice_trials"]),
                    min_fraction_pre=float(almost["min_fraction_pass_timing_choice_rules"]),
                    min_fraction_post=float(almost["min_fraction_after_rt_percentile_trim"]),
                    expected_probability_left=tuple(inc["probability_left_set"]),
                    rt_percentiles=(float(rt_pct[0]), float(rt_pct[1])),
                    require_left_right_bias=bool(inc["require_left_and_right_bias_blocks"]),
                )
                row = res.to_dict() if hasattr(res, "to_dict") else dict(res.__dict__)
                row["has_MOs"] = eid in mos
                row["has_ORBvl"] = eid in orbvl
                pool_scores.append(row)
                if row.get("passes_session"):
                    pool_pass.append(eid)
                    print(f"    PASS n_choice={row.get('n_choice_trials')}")
                else:
                    print(f"    fail {row.get('fail_reasons')}")
            except Exception as exc:  # noqa: BLE001
                pool_scores.append({"eid": eid, "passes_session": False, "error": str(exc)})
                print(f"    ERROR {exc}")

    payload["neural_behavior_pool"] = {
        "n_qc_attempted": len(pool_scores),
        "n_pass": len(pool_pass),
        "eids": pool_pass,
        "note": (
            "Sessions with BWM spikes in MOs and/or ORBvl that pass the same "
            "almost-perfect behavior QC as behavior-core. Use for Phase 8 when "
            "strict intersect is empty."
        ),
    }

    out_path = man_dir / "neural_intersect_eids.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (reports / "neural_intersect_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if pool_scores:
        (reports / "neural_roi_qc_scores.json").write_text(
            json.dumps(pool_scores, indent=2, default=str), encoding="utf-8"
        )

    print(f"Wrote {out_path}")
    print(
        json.dumps(
            {
                "strict_n": len(strict),
                "blocker": payload["blocker"],
                "pool_pass_n": len(pool_pass),
                "pool_pass_eids": pool_pass,
                "bwm_counts": payload["n_bwm_by_region"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
