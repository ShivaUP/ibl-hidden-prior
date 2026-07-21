#!/usr/bin/env python3
"""Phase 3: build processed trials, RNN bins, Bayesian tables, and splits.

Usage:
    python scripts/build_processed_datasets.py
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.bayes_features import build_bayes_table, causality_ok
from src.data.config import load_frozen_config, repo_root
from src.data.event_bins import BinConfig, build_condition_arrays
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


def make_splits(eids: list[str], seed: int = 0) -> dict:
    """Session-level 70/15/15 split with no overlap."""
    rng = np.random.default_rng(seed)
    eids = list(eids)
    rng.shuffle(eids)
    n = len(eids)
    n_test = max(1, int(round(0.15 * n)))
    n_val = max(1, int(round(0.15 * n)))
    n_train = n - n_val - n_test
    if n_train < 1:
        # Tiny cohort fallback
        n_train = max(1, n - 2)
        n_val = 1 if n > 1 else 0
        n_test = n - n_train - n_val
    train = eids[:n_train]
    val = eids[n_train : n_train + n_val]
    test = eids[n_train + n_val :]
    return {
        "seed": seed,
        "train": train,
        "val": val,
        "test": test,
        "note": "Session-level split; no eid overlap. Switch metrics use held-out sessions only.",
    }


def main() -> int:
    cfg = load_frozen_config()
    root = repo_root()
    core = json.loads((root / cfg["data"]["manifests"]["behavior_core"]).read_text())
    eids = core["eids"]
    rt_pct = tuple(cfg["data"]["trial_inclusion"]["rt_percentile_trim"])

    trials_dir = root / "data" / "processed" / "trials"
    bins_dir = root / "data" / "processed" / "rnn_bins"
    bayes_dir = root / "data" / "processed" / "bayes_trials"
    for d in (trials_dir, bins_dir, bayes_dir):
        d.mkdir(parents=True, exist_ok=True)

    one = make_one(root / cfg["data"]["cache_dir"])
    stamp = datetime.now(timezone.utc).isoformat()

    parts: list[pd.DataFrame] = []
    for i, eid in enumerate(eids, start=1):
        print(f"[{i}/{len(eids)}] process {eid}")
        raw = load_trials_for_eid(one, eid)
        proc = build_processed_session(eid, raw, rt_percentiles=(float(rt_pct[0]), float(rt_pct[1])))
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

    bin_cfg = BinConfig(
        bin_size_s=cfg["rnn"]["bin_size_ms"] / 1000.0,
        max_bins=100,
        pad_bins_after_feedback=1,
    )
    for condition in ("history_only", "full_information", "fixed_prior"):
        print(f"Building RNN bins: {condition}")
        payload = build_condition_arrays(all_trials, condition, cfg=bin_cfg)
        out = bins_dir / f"{condition}.pkl"
        with out.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Also save a small index CSV for inspection
        idx = pd.DataFrame(
            {
                "eid": payload["eid"],
                "trial_index": payload["trial_index"],
                "choice_right": payload["choice_right"],
                "rt": payload["rt"],
                "n_bins": [m["n_bins"] for m in payload["metas"]],
            }
        )
        idx.to_csv(bins_dir / f"{condition}_index.csv", index=False)
        print(f"  {len(payload['sequences'])} sequences -> {out.name}")

    for condition in ("history_only", "full_information", "fixed_prior"):
        print(f"Building Bayesian table: {condition}")
        tab = build_bayes_table(all_trials, condition)
        assert causality_ok(tab)
        path = bayes_dir / f"{condition}.parquet"
        tab.to_parquet(path, index=False)
        print(f"  {len(tab)} rows -> {path.name}")

    splits = make_splits(eids, seed=0)
    # Assert no overlap
    sets = [set(splits[k]) for k in ("train", "val", "test") if splits[k]]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j]), "Split eid overlap"
    splits["created_utc"] = stamp
    splits["config_version"] = cfg.get("version")
    splits_path = root / cfg["data"]["manifests"]["splits"]
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"Wrote {splits_path}: { {k: len(splits[k]) for k in ('train','val','test')} }")

    report = {
        "created_utc": stamp,
        "n_eids": len(eids),
        "trials_summary": summary,
        "splits": {k: splits[k] for k in ("train", "val", "test")},
        "artifacts": {
            "trials": str(all_path.relative_to(root)),
            "rnn_bins": str(bins_dir.relative_to(root)),
            "bayes_trials": str(bayes_dir.relative_to(root)),
            "splits": str(splits_path.relative_to(root)),
        },
    }
    report_path = root / "reports" / "qc" / "processed_datasets.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
