#!/usr/bin/env python3
"""Phase 1.1: ONE connectivity smoke test (one public eid)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.config import load_frozen_config, repo_root
from src.data.inspect_trials import load_trials_for_eid
from src.data.qc import CORE_FIELDS


def main() -> int:
    cfg = load_frozen_config()
    root = repo_root()
    cache_dir = root / cfg["data"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = root / "reports" / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    from one.api import ONE

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(cache_dir),
    )
    eid = "4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a"
    trials = load_trials_for_eid(one, eid)
    missing = [c for c in CORE_FIELDS if c not in trials.columns]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eid": eid,
        "n_trials": int(len(trials)),
        "columns": list(trials.columns),
        "missing_core_fields": missing,
        "ok": len(missing) == 0 and len(trials) > 0,
    }
    path = out_dir / "smoke_one_connection.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {path}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
