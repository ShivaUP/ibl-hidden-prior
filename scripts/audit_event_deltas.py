#!/usr/bin/env python3
"""Phase 2.1: event-delta audit for behavior-core sessions.

Writes reports/inspection/event_deltas.md and event_deltas.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.config import load_frozen_config, repo_root
from src.data.inspect_trials import load_trials_for_eid
from src.data.qc import trial_qc_masks


def summarize_delta(name: str, series: pd.Series) -> dict:
    x = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(len(x)),
        "median": float(x.median()),
        "mean": float(x.mean()),
        "p05": float(x.quantile(0.05)),
        "p25": float(x.quantile(0.25)),
        "p75": float(x.quantile(0.75)),
        "p95": float(x.quantile(0.95)),
        "frac_within_100ms": float((x.abs() <= 0.1).mean()),
        "frac_negative": float((x < 0).mean()),
    }


def main() -> int:
    cfg = load_frozen_config()
    root = repo_root()
    core_path = root / cfg["data"]["manifests"]["behavior_core"]
    core = json.loads(core_path.read_text(encoding="utf-8"))
    eids = core["eids"]
    if not eids:
        print("No behavior-core eids found.", file=sys.stderr)
        return 2

    from one.api import ONE

    cache_dir = root / cfg["data"]["cache_dir"]
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(cache_dir),
    )

    per_eid = []
    pooled = {k: [] for k in ("go_minus_stim", "resp_minus_go", "fb_minus_resp", "off_minus_stim")}

    for eid in eids:
        trials = load_trials_for_eid(one, eid)
        masks = trial_qc_masks(trials)
        ok = masks.pass_rules_1_to_4
        t = trials.loc[ok]
        deltas = {
            "go_minus_stim": t["goCue_times"] - t["stimOn_times"],
            "resp_minus_go": t["response_times"] - t["goCue_times"],
            "fb_minus_resp": t["feedback_times"] - t["response_times"],
        }
        if "stimOff_times" in t.columns:
            deltas["off_minus_stim"] = t["stimOff_times"] - t["stimOn_times"]

        eid_summary = {"eid": eid, "n_ok_trials": int(ok.sum()), "deltas": {}}
        for name, series in deltas.items():
            eid_summary["deltas"][name] = summarize_delta(name, series)
            pooled[name].append(series)
        per_eid.append(eid_summary)
        print(f"{eid}: n_ok={ok.sum()}")

    pooled_summary = {}
    for name, parts in pooled.items():
        if not parts:
            continue
        pooled_summary[name] = summarize_delta(name, pd.concat(parts, ignore_index=True))

    # Suggested 100 ms phase sketch from pooled medians
    go_med = pooled_summary.get("go_minus_stim", {}).get("median")
    resp_med = pooled_summary.get("resp_minus_go", {}).get("median")
    off_med = pooled_summary.get("off_minus_stim", {}).get("median")
    suggestion = {
        "bin_size_ms": 100,
        "bin0": "stimOn_times",
        "notes": [
            "If median(go-stim) << 100ms, stimulus_right/contrast_high and response_window may both start in bin 0.",
            "response_made occupies the bin containing response_times - stimOn.",
            "reward occupies the bin containing feedback_times - stimOn.",
            "stimOff relative delay informs when stimulus channels turn off.",
        ],
        "pooled_median_go_minus_stim_s": go_med,
        "pooled_median_resp_minus_go_s": resp_med,
        "pooled_median_off_minus_stim_s": off_med,
    }

    stamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "created_utc": stamp,
        "n_eids": len(eids),
        "eids": eids,
        "pooled": pooled_summary,
        "per_eid": per_eid,
        "phase_map_suggestion": suggestion,
    }

    out_dir = root / "reports" / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "event_deltas.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Event-delta audit (behavior-core)",
        "",
        f"created_utc: {stamp}",
        f"n_eids: {len(eids)}",
        "",
        "## Pooled deltas (QC-pass trials only)",
        "",
    ]
    for name, stats in pooled_summary.items():
        md_lines.append(f"### {name}")
        md_lines.append("")
        for k, v in stats.items():
            if k == "name":
                continue
            md_lines.append(f"- {k}: {v}")
        md_lines.append("")
    md_lines.append("## Phase-map suggestion")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps(suggestion, indent=2))
    md_lines.append("```")
    md_lines.append("")

    md_path = out_dir / "event_deltas.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(suggestion, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
