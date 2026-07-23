#!/usr/bin/env python3
"""12 — Refresh neural∩behavior intersect manifest (optional if already present).

Usage:
  python scripts/12_build_neural_intersect.py
  python scripts/12_build_neural_intersect.py --reuse-existing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Copy existing data/manifests/neural_intersect_eids.json into reports/v2/neural/",
    )
    args = p.parse_args()
    src = ROOT / "data" / "manifests" / "neural_intersect_eids.json"
    out = ROOT / "reports" / "v2" / "neural"
    out.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(
            f"Missing {src}. Recover with: git show 02fcb76:scripts/build_neural_intersect.py",
            file=sys.stderr,
        )
        return 1

    if args.reuse_existing or True:
        # Default: reuse frozen v1 intersect (strict core∩ROI was empty; pool=25).
        payload = json.loads(src.read_text())
        payload["copied_utc"] = datetime.now(timezone.utc).isoformat()
        payload["note"] = (
            "Reused v1 neural_intersect_eids.json for v2. "
            "Strict behavior-core ∩ ROI is empty; use neural_behavior_pool for pilots."
        )
        (out / "neural_intersect_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        shutil.copy(src, out / "neural_intersect_eids.json")
        print(
            json.dumps(
                {
                    "n_strict": len(payload.get("eids") or []),
                    "n_pool": len(
                        (payload.get("neural_behavior_pool") or {}).get("eids") or []
                    ),
                    "out": str((out / "neural_intersect_summary.json").relative_to(ROOT)),
                },
                indent=2,
            )
        )
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
