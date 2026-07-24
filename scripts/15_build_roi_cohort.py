#!/usr/bin/env python3
"""15 — Build the locked-ROI neural cohort (MOs, ORBvl, ACAd, MOp).

Constructs the shared cohort that does not yet exist in the repo:
  1. Query BWM insertions for each locked ROI (MOs, ORBvl, ACAd, MOp).
  2. Union the candidate sessions, tagging which ROIs each contains.
  3. Apply the project's almost-perfect behavior QC (same gates as
     scripts/01_run_session_qc.py) to each candidate.
  4. Freeze a manifest with per-ROI passing sessions and coverage counts.

Per-region inclusion (NOT anchored on MOs): each ROI's analysis uses the
sessions that contain that ROI and pass QC — matching "per-region VE uses
sessions that contain that region."

Usage
-----
  conda activate ibl-prior
  python scripts/15_build_roi_cohort.py                 # full build
  python scripts/15_build_roi_cohort.py --max-candidates 60   # quick pilot

Output
------
  data/manifests/roi_cohort_v2.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.config import load_frozen_config, repo_root
from src.data.inspect_trials import load_trials_for_eid
from src.data.qc import score_session
from src.neural.regions import PRIMARY_ROIS

ONE_BASE_URL = "https://openalyx.internationalbrainlab.org"
ONE_PASSWORD = "international"
BWM_PROJECT = "ibl_neuropixel_brainwide_01"
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

TARGET_COVERAGE = {"MOs": 8, "ORBvl": 5, "ACAd": 4, "MOp": 3}  # user's reported numbers


def make_one(cache_dir: Path):
    from one.api import ONE
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ONE(base_url=ONE_BASE_URL, password=ONE_PASSWORD, silent=True, cache_dir=str(cache_dir))


def _eid(session_field: str):
    m = _UUID.search(str(session_field))
    return m.group(0).lower() if m else None


def sessions_by_roi(one) -> dict[str, set[str]]:
    """For each locked ROI, the set of BWM sessions with an insertion there."""
    out: dict[str, set[str]] = {}
    for roi, prefixes in PRIMARY_ROIS.items():
        acr = prefixes[0]
        try:
            ins = one.alyx.rest("insertions", "list", atlas_acronym=acr, project=BWM_PROJECT)
            eids = {e for e in (_eid(i.get("session", "")) for i in ins) if e}
        except Exception as exc:  # noqa: BLE001
            print(f"  {roi:6s}: insertion query failed ({exc})")
            eids = set()
        out[roi] = eids
        print(f"  {roi:6s}: {len(eids)} BWM sessions with insertion")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build locked-ROI neural cohort")
    ap.add_argument("--max-candidates", type=int, default=200, metavar="N",
                    help="Cap the number of candidate sessions to QC (default: 200)")
    ap.add_argument("--curate", type=int, default=8, metavar="N",
                    help="Curate a cohort of N sessions anchored on MOs (default: 8; 0 = skip)")
    ap.add_argument("--anchor", default="MOs",
                    help="ROI that must be present in every curated session (default: MOs)")
    ap.add_argument("--cache-dir", default=None, help="ONE cache dir")
    args = ap.parse_args()

    cfg = load_frozen_config()
    inc = cfg["data"]["session_inclusion"]
    almost = inc["almost_perfect"]
    min_choice = int(inc["min_choice_trials"])
    min_pre = float(almost["min_fraction_pass_timing_choice_rules"])
    min_post = float(almost["min_fraction_after_rt_percentile_trim"])
    rt_pct = tuple(cfg["data"]["trial_inclusion"]["rt_percentile_trim"])

    cache_dir = Path(args.cache_dir) if args.cache_dir else repo_root() / cfg["data"]["cache_dir"]
    one = make_one(cache_dir)

    print("=== Locked-ROI cohort build (MOs, ORBvl, ACAd, MOp) ===\n")
    print("Querying BWM insertions per ROI ...")
    roi_sessions = sessions_by_roi(one)

    # Candidate union, tagged with ROIs present.
    candidate_rois: dict[str, list[str]] = {}
    for roi, eids in roi_sessions.items():
        for eid in eids:
            candidate_rois.setdefault(eid, []).append(roi)
    candidates = sorted(candidate_rois)
    # Prioritise multi-ROI sessions (more coverage) when capping.
    candidates.sort(key=lambda e: (-len(candidate_rois[e]), e))
    if args.max_candidates:
        candidates = candidates[: args.max_candidates]
    print(f"\nCandidate union: {len(candidate_rois)} sessions → QC-ing {len(candidates)}\n")

    passing: dict[str, dict] = {}
    for i, eid in enumerate(candidates):
        try:
            trials = load_trials_for_eid(one, eid)
            res = score_session(
                eid, trials,
                min_choice_trials=min_choice,
                min_fraction_pre=min_pre,
                min_fraction_post=min_post,
                rt_percentiles=rt_pct,
            )
            ok = bool(res.passes_session)
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[{i+1}/{len(candidates)}] {eid}  QC error: {str(exc)[:50]}")
            continue
        tag = "PASS" if ok else "fail"
        print(f"[{i+1}/{len(candidates)}] {eid}  {tag}  ROIs={candidate_rois[eid]}")
        if ok:
            passing[eid] = {"rois": candidate_rois[eid], "n_choice_trials": int(res.n_choice_trials)}

    # Per-ROI coverage among QC-passing sessions.
    per_roi: dict[str, list[str]] = {roi: [] for roi in PRIMARY_ROIS}
    for eid, info in passing.items():
        for roi in info["rois"]:
            per_roi[roi].append(eid)

    coverage = {roi: len(eids) for roi, eids in per_roi.items()}

    # Curate an anchored cohort of ~N sessions: every session contains the
    # anchor ROI (default MOs); greedily add sessions to cover the other ROIs
    # first (set-cover), then fill remaining slots by trial count.
    curated: list[str] = []
    curated_coverage: dict[str, int] = {}
    if args.curate and args.curate > 0:
        curated = curate_cohort(passing, anchor=args.anchor, n=args.curate)
        cur_per_roi: dict[str, int] = {roi: 0 for roi in PRIMARY_ROIS}
        for e in curated:
            for roi in passing[e]["rois"]:
                cur_per_roi[roi] += 1
        curated_coverage = cur_per_roi

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Locked-ROI neural cohort (per-region inclusion). "
                       "Sessions with BWM insertion in each ROI passing almost-perfect behavior QC.",
        "rois": list(PRIMARY_ROIS.keys()),
        "qc_gates": {
            "min_choice_trials": min_choice,
            "min_fraction_pre": min_pre,
            "min_fraction_post": min_post,
            "rt_percentile_trim": list(rt_pct),
        },
        "cohort_union_eids": sorted(passing),
        "cohort_union_n": len(passing),
        "per_roi_eids": {roi: sorted(eids) for roi, eids in per_roi.items()},
        "coverage": coverage,
        "eid_rois": {e: passing[e]["rois"] for e in sorted(passing)},
        "curated": {
            "anchor": args.anchor,
            "n_requested": args.curate,
            "eids": curated,
            "coverage": curated_coverage,
        },
        "target_coverage": TARGET_COVERAGE,
        "max_candidates": args.max_candidates,
    }
    out_path = ROOT / "data" / "manifests" / "roi_cohort_v2.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== Coverage (QC-passing union) ===")
    print(f"{'ROI':<8}{'union':>7}{'target':>8}")
    for roi in PRIMARY_ROIS:
        print(f"{roi:<8}{coverage[roi]:>7}{TARGET_COVERAGE.get(roi, '-'):>8}")
    print(f"Union cohort: {len(passing)} sessions")

    if curated:
        print(f"\n=== Curated cohort (anchor={args.anchor}, n={len(curated)}) ===")
        print(f"{'ROI':<8}{'curated':>8}{'target':>8}")
        for roi in PRIMARY_ROIS:
            print(f"{roi:<8}{curated_coverage.get(roi, 0):>8}{TARGET_COVERAGE.get(roi, '-'):>8}")
        print("eids:")
        for e in curated:
            print(f"  {e}  {passing[e]['rois']}")

    print(f"\nManifest: {out_path.relative_to(ROOT)}")


def curate_cohort(passing: dict, *, anchor: str = "MOs", n: int = 8) -> list[str]:
    """Select N anchor-containing sessions, greedily maximizing ROI coverage.

    Phase 1 (set cover): add anchor sessions that introduce a not-yet-covered ROI.
    Phase 2 (fill): add the remaining highest-trial-count anchor sessions up to N.
    """
    anchor_sessions = [e for e in passing if anchor in passing[e]["rois"]]
    # Priority: more ROIs first, then more choice trials.
    anchor_sessions.sort(key=lambda e: (-len(passing[e]["rois"]), -passing[e]["n_choice_trials"]))

    selected: list[str] = []
    covered: set[str] = set()
    for e in anchor_sessions:
        if len(selected) >= n:
            break
        if set(passing[e]["rois"]) - covered:
            selected.append(e)
            covered |= set(passing[e]["rois"])
    for e in anchor_sessions:
        if len(selected) >= n:
            break
        if e not in selected:
            selected.append(e)
    return selected[:n]


if __name__ == "__main__":
    main()
