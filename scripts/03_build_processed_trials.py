#!/usr/bin/env python3
"""03 — Build processed trial tables for behavior-core eids (v2 data prep).

Writes per-eid and pooled parquet under data/processed/trials/.
Does **not** build v1 RNN bins or Bayes feature tables (removed from pipeline).

Usage:
  python scripts/03_build_processed_trials.py
"""

from __future__ import annotations

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


def make_one(cache_dir: Path):
    from one.api import ONE

    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(cache_dir),
    )


def main() -> int:
    cfg = load_frozen_config()
    root = repo_root()
    core = json.loads((root / cfg["data"]["manifests"]["behavior_core"]).read_text())
    eids = core["eids"]
    rt_pct = tuple(cfg["data"]["trial_inclusion"]["rt_percentile_trim"])

    trials_dir = root / "data" / "processed" / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    one = make_one(root / cfg["data"]["cache_dir"])
    stamp = datetime.now(timezone.utc).isoformat()

    parts: list[pd.DataFrame] = []
    for i, eid in enumerate(eids, start=1):
        print(f"[{i}/{len(eids)}] process {eid}")
        raw = load_trials_for_eid(one, eid)
        proc = build_processed_session(
            eid, raw, rt_percentiles=(float(rt_pct[0]), float(rt_pct[1]))
        )
        out_pq = trials_dir / f"{eid}.parquet"
        proc.to_parquet(out_pq, index=False)
        parts.append(proc)
        print(f"  kept {len(proc)} trials -> {out_pq.name}")

    all_trials = pd.concat(parts, ignore_index=True)
    all_path = trials_dir / "all_trials.parquet"
    all_trials.to_parquet(all_path, index=False)
    summary = summarize_processed(all_trials)
    summary_path = trials_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"created_utc": stamp, **summary}, indent=2), encoding="utf-8"
    )
    print(f"Wrote {all_path} ({summary})")

    report = {
        "created_utc": stamp,
        "n_eids": len(eids),
        "trials_summary": summary,
        "artifacts": {"trials": str(all_path.relative_to(root))},
        "note": "v2: trials only. RNN bins / Bayes tables / splits are not built.",
    }
    report_path = root / "reports" / "qc" / "processed_trials.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
