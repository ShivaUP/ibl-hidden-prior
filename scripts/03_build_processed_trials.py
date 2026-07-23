#!/usr/bin/env python3
"""03 — Build processed trial tables for a real-eval eid manifest.

Default: shared behavior+neural cohort (sessions with behavior QC AND ROI spikes).

Usage:
  python scripts/03_build_processed_trials.py
  python scripts/03_build_processed_trials.py --manifest data/manifests/behavior_core_eids.json
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
from src.data.processed_trials import build_processed_session, summarize_processed

DEFAULT_MANIFEST = "data/manifests/shared_behavior_neural_eids.json"


def make_one(cache_dir: Path):
    from one.api import ONE

    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(cache_dir),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = p.parse_args()
    cfg = load_frozen_config()
    root = repo_root()
    man_path = root / args.manifest
    core = json.loads(man_path.read_text())
    eids = core["eids"]
    rt_pct = tuple(cfg["data"]["trial_inclusion"]["rt_percentile_trim"])

    trials_dir = root / "data" / "processed" / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    one = make_one(root / cfg["data"]["cache_dir"])
    stamp = datetime.now(timezone.utc).isoformat()

    parts: list[pd.DataFrame] = []
    failed = []
    for i, eid in enumerate(eids, start=1):
        print(f"[{i}/{len(eids)}] process {eid}", flush=True)
        try:
            raw = load_trials_for_eid(one, eid)
            proc = build_processed_session(
                eid, raw, rt_percentiles=(float(rt_pct[0]), float(rt_pct[1]))
            )
            out_pq = trials_dir / f"{eid}.parquet"
            proc.to_parquet(out_pq, index=False)
            parts.append(proc)
            print(f"  kept {len(proc)} trials -> {out_pq.name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {eid}: {exc}", flush=True)
            failed.append({"eid": eid, "error": str(exc)})

    if not parts:
        print("ERROR: no sessions processed", file=sys.stderr)
        return 1

    all_trials = pd.concat(parts, ignore_index=True)
    all_path = trials_dir / "all_trials.parquet"
    all_trials.to_parquet(all_path, index=False)
    # also keep a cohort-specific copy
    cohort_path = trials_dir / "all_trials_shared_behavior_neural.parquet"
    all_trials.to_parquet(cohort_path, index=False)
    summary = summarize_processed(all_trials)
    summary_path = trials_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "created_utc": stamp,
                "manifest": args.manifest,
                "n_ok": len(parts),
                "n_failed": len(failed),
                "failed": failed,
                **summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {all_path} ({summary})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
