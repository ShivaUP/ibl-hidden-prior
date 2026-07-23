#!/usr/bin/env python3
"""06 — Map real sessions into shared v2 tick tensors.

Default eid list: shared behavior+neural cohort.

Usage:
  python scripts/06_map_real_to_v2_ticks.py
  python scripts/06_map_real_to_v2_ticks.py --manifest data/manifests/behavior_core_eids.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synthetic.channels import PhaseTicks
from src.synthetic.mapper_real import encode_real_session
from src.synthetic.schema import load_synthetic_config

DEFAULT_MANIFEST = "data/manifests/shared_behavior_neural_eids.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = p.parse_args()
    cfg = load_synthetic_config()
    phase = PhaseTicks.from_config(cfg)
    core = json.loads((ROOT / args.manifest).read_text())
    eids = core["eids"]

    trials_path = ROOT / "data" / "processed" / "trials" / "all_trials.parquet"
    if not trials_path.exists():
        trials_path = ROOT / "data" / "processed" / "trials" / "all_trials.csv"
    if not trials_path.exists():
        print(
            f"ERROR: missing processed trials. Run: python scripts/03_build_processed_trials.py",
            file=sys.stderr,
        )
        return 1

    if trials_path.suffix == ".parquet":
        df = pd.read_parquet(trials_path)
    else:
        df = pd.read_csv(trials_path)

    out = ROOT / "data" / "processed" / "real_v2_ticks"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for eid in eids:
        g = df[df["eid"] == eid]
        if g.empty:
            print(f"SKIP missing trials for {eid}", file=sys.stderr)
            continue
        enc = encode_real_session(g, phase)
        path = out / f"{eid}.npz"
        np.savez_compressed(path, **enc)
        written.append(
            {"eid": eid, "n_trials": int(enc["n_trials"]), "path": str(path.relative_to(ROOT))}
        )

    man = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest,
        "cohort": "shared_behavior_neural",
        "n_sessions": len(written),
        "phase_n_steps": phase.n_steps,
        "sessions": written,
        "rebuild": [
            "python scripts/03_build_processed_trials.py",
            "python scripts/06_map_real_to_v2_ticks.py",
        ],
    }
    (ROOT / "data" / "manifests" / "real_v2_ticks.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8"
    )
    print(json.dumps({"n_sessions": len(written), "n_steps": phase.n_steps}, indent=2))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
