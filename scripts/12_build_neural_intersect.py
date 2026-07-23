#!/usr/bin/env python3
"""12 — Rebuild shared behavior+neural cohort for the expanded ROI set.

Queries BWM sessions with spikes in any ROI from `src.neural.regions`,
scores behavior QC (almost-perfect), then selects a shared cohort whose
**union** covers as many ROIs as possible.

Important: no single Neuropixels session contains all ROIs. The locked rule
is therefore:
  1) every session has behavior QC + ≥1 ROI,
  2) cohort union maximizes ROI coverage (greedy set cover),
  3) prefer multi-ROI sessions and primary cortex (MOs, ORBvl, ACAd).

Usage:
  python scripts/12_build_neural_intersect.py
  python scripts/12_build_neural_intersect.py --reuse-existing   # copy old manifests only
  python scripts/12_build_neural_intersect.py --max-sessions 20
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

from src.data.config import load_frozen_config
from src.data.inspect_trials import load_trials_for_eid
from src.data.qc import score_session
from src.neural.regions import NEURAL_REGIONS, REGION_ALIASES, REGION_TIERS, atlas_query_acronyms


PRIMARY = ("MOs", "vlOFC_orbvl", "ACAd", "MOp")


def _score_trials(eid: str, trials, fcfg: dict):
    """Call score_session with frozen_v1 inclusion thresholds."""
    inc = fcfg["data"]["session_inclusion"]
    almost = inc["almost_perfect"]
    trial_inc = fcfg["data"]["trial_inclusion"]
    rt_pct = trial_inc["rt_percentile_trim"]
    return score_session(
        eid,
        trials,
        min_choice_trials=int(inc["min_choice_trials"]),
        min_fraction_pre=float(almost["min_fraction_pass_timing_choice_rules"]),
        min_fraction_post=float(almost["min_fraction_after_rt_percentile_trim"]),
        expected_probability_left=tuple(inc["probability_left_set"]),
        rt_percentiles=(float(rt_pct[0]), float(rt_pct[1])),
        require_left_right_bias=bool(inc["require_left_and_right_bias_blocks"]),
    )


def make_one(cache_dir: Path):
    from one.api import ONE

    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(cache_dir),
    )


def _search_region_eids(one, acronyms: list[str]) -> set[str]:
    """Return session eids with spikes.times in any of the given Allen acronyms."""
    eids: set[str] = set()
    for acr in acronyms:
        try:
            found = one.search(
                projects="ibl_neuropixel_brainwide_01",
                dataset_types="spikes.times",
                atlas_acronym=acr,
            )
            if found is not None and len(found):
                eids.update(str(x) for x in found)
        except Exception as exc:  # noqa: BLE001
            print(f"  search failed for {acr}: {exc}", flush=True)
    return eids


def _region_flags_for_eid(one, eid: str) -> dict[str, bool]:
    """Best-effort: which ROIs have units for this session (via channels acronyms)."""
    flags = {r: False for r in NEURAL_REGIONS}
    try:
        pids = one.alyx.rest("insertions", "list", session=eid)
    except Exception:
        return flags
    acronyms_seen: set[str] = set()
    for ins in pids or []:
        try:
            ch = one.load_object(eid, "channels", collection=f"alf/{ins['name']}")
            if isinstance(ch, dict) and "brainLocationAcronyms_ccf_2017" in ch:
                acr = ch["brainLocationAcronyms_ccf_2017"]
            else:
                # try datasets
                acr = one.load_dataset(
                    eid,
                    "channels.brainLocationAcronyms_ccf_2017",
                    collection=f"alf/{ins.get('name', '')}",
                )
            for a in list(acr):
                acronyms_seen.add(str(a))
        except Exception:
            try:
                # Alternative: probe description json
                det = ins.get("json") or {}
                for a in (det.get("regions") or []):
                    acronyms_seen.add(str(a))
            except Exception:
                continue
    for spec, targets in REGION_ALIASES.items():
        for t in targets:
            if any(a == t or a.startswith(t) for a in acronyms_seen):
                flags[spec] = True
                break
    return flags


def _greedy_cover(rows: list[dict], *, max_sessions: int) -> list[dict]:
    """Greedy set cover over ROI flags, preferring primary + multi-ROI + QC quality."""
    remaining = set(NEURAL_REGIONS)
    selected: list[dict] = []
    pool = list(rows)

    def score(row: dict, rem: set[str]) -> tuple:
        flags = {r: bool(row.get(f"has_{r}", False)) for r in NEURAL_REGIONS}
        # also accept has_ORBvl alias for vlOFC
        if "has_ORBvl" in row:
            flags["vlOFC_orbvl"] = flags["vlOFC_orbvl"] or bool(row["has_ORBvl"])
        if "has_MOs" in row:
            flags["MOs"] = flags["MOs"] or bool(row["has_MOs"])
        n_new = sum(1 for r in rem if flags.get(r))
        n_primary = sum(1 for r in PRIMARY if flags.get(r))
        n_all = sum(1 for r in NEURAL_REGIONS if flags.get(r))
        return (
            n_new,
            n_primary,
            n_all,
            float(row.get("fraction_after_rt_trim", 0.0)),
            float(row.get("fraction_pass_rules_1_to_4", 0.0)),
            int(row.get("n_choice_trials", 0)),
        )

    while pool and len(selected) < max_sessions and remaining:
        pool.sort(key=lambda r: score(r, remaining), reverse=True)
        best = pool.pop(0)
        flags = {r: bool(best.get(f"has_{r}", False)) for r in NEURAL_REGIONS}
        if "has_ORBvl" in best:
            flags["vlOFC_orbvl"] = flags["vlOFC_orbvl"] or bool(best["has_ORBvl"])
        if "has_MOs" in best:
            flags["MOs"] = flags["MOs"] or bool(best["has_MOs"])
        if not any(flags.values()):
            continue
        selected.append(best)
        for r, ok in flags.items():
            if ok and r in remaining:
                remaining.discard(r)

    # Fill remaining slots with best QC sessions still unused
    if len(selected) < max_sessions:
        used = {s["eid"] for s in selected}
        leftovers = [r for r in rows if r["eid"] not in used]
        leftovers.sort(
            key=lambda r: (
                sum(1 for k in NEURAL_REGIONS if r.get(f"has_{k}"))
                + int(bool(r.get("has_MOs")))
                + int(bool(r.get("has_ORBvl"))),
                float(r.get("fraction_after_rt_trim", 0.0)),
                float(r.get("fraction_pass_rules_1_to_4", 0.0)),
                int(r.get("n_choice_trials", 0)),
            ),
            reverse=True,
        )
        for r in leftovers:
            if len(selected) >= max_sessions:
                break
            selected.append(r)
    return selected


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reuse-existing", action="store_true")
    p.add_argument("--max-sessions", type=int, default=20)
    p.add_argument("--skip-download-qc", action="store_true", help="Only use cached ROI QC JSON")
    args = p.parse_args()

    out = ROOT / "reports" / "v2" / "neural"
    out.mkdir(parents=True, exist_ok=True)
    man_dir = ROOT / "data" / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_existing:
        src = man_dir / "neural_intersect_eids.json"
        if not src.exists():
            print(f"Missing {src}", file=sys.stderr)
            return 1
        payload = json.loads(src.read_text())
        payload["copied_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "neural_intersect_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"mode": "reuse-existing", "n_pool": len((payload.get("neural_behavior_pool") or {}).get("eids") or [])}, indent=2))
        return 0

    fcfg = load_frozen_config()
    one = make_one(ROOT / fcfg["data"]["cache_dir"])
    atlas = atlas_query_acronyms()
    stamp = datetime.now(timezone.utc).isoformat()

    # 1) BWM eid sets per ROI
    by_region: dict[str, list[str]] = {}
    union: set[str] = set()
    print("Querying BWM sessions per ROI…", flush=True)
    for spec, acrs in atlas.items():
        eids = sorted(_search_region_eids(one, acrs))
        by_region[spec] = eids
        union.update(eids)
        print(f"  {spec}: {len(eids)}", flush=True)

    # Seed with previous MOs/ORBvl union if Alyx search returns sparse
    legacy = man_dir / "neural_intersect_eids.json"
    if legacy.exists():
        leg = json.loads(legacy.read_text())
        union.update(leg.get("bwm_roi_union_eids") or [])
        for e in (leg.get("neural_behavior_pool") or {}).get("eids") or []:
            union.add(e)

    eid_to_regions: dict[str, set[str]] = {e: set() for e in union}
    for spec, eids in by_region.items():
        for e in eids:
            eid_to_regions.setdefault(e, set()).add(spec)

    # 2) Behavior QC — reuse cache; only download trials for sessions needed for coverage
    qc_cache_path = ROOT / "reports" / "neural" / "neural_roi_qc_scores.json"
    expanded_cache = ROOT / "reports" / "neural" / "neural_roi_qc_scores_expanded.json"
    cached: dict[str, dict] = {}
    for path in (qc_cache_path, expanded_cache):
        if path.exists():
            for row in json.loads(path.read_text()):
                cached[row["eid"]] = row

    def _passes(row: dict) -> bool:
        if row.get("passes_session"):
            return True
        return bool(
            row.get("passes_almost_perfect_post")
            and row.get("passes_bias_blocks")
            and row.get("passes_min_choice_trials")
        )

    def _annotate(eid: str, row: dict) -> dict:
        row = dict(row)
        row["eid"] = eid
        known = eid_to_regions.get(eid, set())
        for r in NEURAL_REGIONS:
            row[f"has_{r}"] = (r in known) or bool(row.get(f"has_{r}"))
        if "vlOFC_orbvl" in known or row.get("has_ORBvl"):
            row["has_vlOFC_orbvl"] = True
            row["has_ORBvl"] = True
        if "MOs" in known or row.get("has_MOs"):
            row["has_MOs"] = True
        return row

    def _qc_eid(eid: str) -> dict:
        if eid in cached and "passes_session" in cached[eid]:
            return _annotate(eid, cached[eid])
        if args.skip_download_qc:
            return _annotate(eid, cached.get(eid) or {"eid": eid, "passes_session": False})
        try:
            trials = load_trials_for_eid(one, eid)
            qc = _score_trials(eid, trials, fcfg).to_dict()
            row = qc.to_dict()
        except Exception as exc:  # noqa: BLE001
            row = {"eid": eid, "passes_session": False, "fail_reasons": [f"load_error:{exc}"]}
        cached[eid] = row
        return _annotate(eid, row)

    # Prefer already-QC'd sessions; then QC additional eids only for uncovered ROIs
    rows: list[dict] = [_annotate(eid, row) for eid, row in cached.items() if eid in union]
    print(f"Cached QC rows in union: {len(rows)}", flush=True)

    covered_now = {
        r
        for row in rows
        if _passes(row)
        for r in NEURAL_REGIONS
        if row.get(f"has_{r}")
    }
    missing_now = [r for r in NEURAL_REGIONS if r not in covered_now]
    print(f"ROIs already coverable from cached QC-pass: {sorted(covered_now)}", flush=True)
    print(f"ROIs needing extra QC: {missing_now}", flush=True)

    # For each missing ROI, QC up to 15 candidates from that region's BWM list
    per_region_budget = 15
    for reg in missing_now:
        cands = [e for e in by_region.get(reg, []) if e not in cached or "passes_session" not in cached.get(e, {})]
        # also try cached fails skipped; prioritize unknown
        unknown = [e for e in by_region.get(reg, []) if e not in {r["eid"] for r in rows}]
        to_try = unknown[:per_region_budget]
        print(f"  QC for {reg}: trying {len(to_try)} sessions…", flush=True)
        for j, eid in enumerate(to_try, 1):
            row = _qc_eid(eid)
            rows.append(row)
            if j % 5 == 0:
                print(f"    {reg} {j}/{len(to_try)}", flush=True)

    # Also ensure we have enough multi-ROI primary sessions: QC top dual primary candidates
    primary_cands = []
    for eid, regs in eid_to_regions.items():
        n_pri = sum(1 for r in PRIMARY if r in regs)
        if n_pri >= 2 and eid not in {r["eid"] for r in rows}:
            primary_cands.append((n_pri, len(regs), eid))
    primary_cands.sort(reverse=True)
    for _, __, eid in primary_cands[:10]:
        rows.append(_qc_eid(eid))

    # Dedupe rows by eid (keep latest)
    by_eid = {r["eid"]: r for r in rows}
    rows = list(by_eid.values())

    qc_out = ROOT / "reports" / "neural" / "neural_roi_qc_scores_expanded.json"
    qc_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    pool = [r for r in rows if _passes(r)]
    pool = [
        r
        for r in pool
        if any(r.get(f"has_{reg}") for reg in NEURAL_REGIONS)
        or r.get("has_MOs")
        or r.get("has_ORBvl")
    ]
    print(f"QC-pass pool with ≥1 ROI: {len(pool)}", flush=True)

    selected = _greedy_cover(pool, max_sessions=args.max_sessions)
    covered = {
        r
        for row in selected
        for r in NEURAL_REGIONS
        if row.get(f"has_{r}") or (r == "vlOFC_orbvl" and row.get("has_ORBvl")) or (r == "MOs" and row.get("has_MOs"))
    }
    missing = [r for r in NEURAL_REGIONS if r not in covered]

    pool_manifest = {
        "created_utc": stamp,
        "name": "shared_behavior_neural_pool_expanded",
        "description": (
            "Almost-perfect behavior QC sessions with BWM spikes in any expanded ROI. "
            "Parent pool for the shared behavior+neural cohort."
        ),
        "regions": list(NEURAL_REGIONS),
        "region_tiers": REGION_TIERS,
        "n_sessions": len(pool),
        "eids": [r["eid"] for r in pool],
        "n_bwm_by_region": {k: len(v) for k, v in by_region.items()},
        "papers": {
            "prior_findling_2025": "https://www.nature.com/articles/s41586-025-09226-1",
            "bwm_ibl_2025": "https://www.nature.com/articles/s41586-025-09235-0",
            "behavior_ibl_2021": "https://elifesciences.org/articles/63711",
        },
    }
    (man_dir / "shared_behavior_neural_pool_expanded_eids.json").write_text(
        json.dumps(pool_manifest, indent=2), encoding="utf-8"
    )

    shared = {
        "created_utc": stamp,
        "name": "shared_behavior_neural",
        "description": (
            f"Shared real-eval + neural VE cohort (n={len(selected)}): greedy ROI coverage "
            "over expanded belief-updating regions, then QC-ranked fillers. "
            "Union of sessions covers listed ROIs; individual sessions typically have 1–few ROIs."
        ),
        "source": "shared_behavior_neural_pool_expanded_eids.json",
        "parent_pool": "data/manifests/shared_behavior_neural_pool_expanded_eids.json",
        "n_sessions": len(selected),
        "max_sessions": args.max_sessions,
        "regions_in_scope": list(NEURAL_REGIONS),
        "regions_covered_in_cohort": sorted(covered),
        "regions_missing_in_cohort": missing,
        "selection_rule": (
            "greedy set-cover over ROI flags; prefer new ROIs, then primary cortex "
            "(MOs/ORBvl/ACAd), then multi-ROI count, then fraction_after_rt_trim / "
            "fraction_pass_rules_1_to_4 / n_choice_trials"
        ),
        "eids": [r["eid"] for r in selected],
        "sessions": [
            {
                "eid": r["eid"],
                "n_choice_trials": r.get("n_choice_trials"),
                "fraction_after_rt_trim": r.get("fraction_after_rt_trim"),
                "fraction_pass_rules_1_to_4": r.get("fraction_pass_rules_1_to_4"),
                **{f"has_{reg}": bool(r.get(f"has_{reg}") or (reg == "vlOFC_orbvl" and r.get("has_ORBvl")) or (reg == "MOs" and r.get("has_MOs"))) for reg in NEURAL_REGIONS},
            }
            for r in selected
        ],
        "papers": pool_manifest["papers"],
        "note": (
            "No Neuropixels session has all ROIs. Behavior transfer and neural VE share "
            "these eids; per-region VE uses the subset of sessions that contain that region."
        ),
    }
    (man_dir / "shared_behavior_neural_eids.json").write_text(json.dumps(shared, indent=2), encoding="utf-8")

    # Keep pool25 name as alias of expanded pool for docs compatibility
    (man_dir / "shared_behavior_neural_pool25_eids.json").write_text(
        json.dumps({**pool_manifest, "name": "shared_behavior_neural_pool_expanded_alias"}, indent=2),
        encoding="utf-8",
    )

    summary = {
        "created_utc": stamp,
        "region_aliases": {k: list(v) for k, v in REGION_ALIASES.items()},
        "n_bwm_by_region": {k: len(v) for k, v in by_region.items()},
        "n_bwm_union": len(union),
        "n_qc_pass_pool": len(pool),
        "n_shared_cohort": len(selected),
        "regions_covered": sorted(covered),
        "regions_missing": missing,
        "shared_manifest": "data/manifests/shared_behavior_neural_eids.json",
        "pool_manifest": "data/manifests/shared_behavior_neural_pool_expanded_eids.json",
        "qc_table": str(qc_out.relative_to(ROOT)),
    }
    (out / "neural_intersect_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (man_dir / "neural_intersect_eids.json").write_text(
        json.dumps(
            {
                **summary,
                "bwm_roi_union_eids": sorted(union),
                "neural_behavior_pool": {"eids": [r["eid"] for r in pool], "n_pass": len(pool)},
                "eids": [r["eid"] for r in selected],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    if missing:
        print(
            f"WARNING: cohort still missing ROI coverage for: {missing}. "
            "Those regions will be skipped in neural VE until more BWM+QC sessions are available.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
