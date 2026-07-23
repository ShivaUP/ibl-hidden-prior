#!/usr/bin/env python3
"""13 — Decode the mouse block-prior belief from real IBL neural data.

Downloads spikes for Brain-Wide-Map sessions that recorded the choice-selective
subcortical regions highlighted in IBL et al. 2025 (Nature) Fig 5, decodes the
block prior (left- vs right-biased block) from a pre-stimulus window per region,
and compares the region-level decodability to the artificial models' hidden
states (from scripts/12_block_decoder.py).

Regions
-------
  Thalamus:   CL, SPF
  Midbrain:   SCm, MRN, SNr, RPF, NPC
  Hindbrain / cerebellar nuclei: GRN, IRN, SOC, VII, TRN, FOTU
  Context (comparison): MOs, VISp

Usage
-----
  conda activate ibl-prior
  python scripts/13_neural_block_decoder.py --max-sessions 3        # quick pilot
  python scripts/13_neural_block_decoder.py --regions GRN MRN SCm
  python scripts/13_neural_block_decoder.py --load-existing          # replot only

Output
------
  reports/v2/block_decoder/neural_region_results.csv     (per session x region)
  reports/v2/block_decoder/neural_region_agg.csv         (per region)
  reports/v2/block_decoder/neural_block_decoder.png      (region ranking)
  reports/v2/block_decoder/neural_vs_model_prior.png     (regions vs models)

Notes
-----
- Public IBL data; no credentials needed. Data downloads on first run
  (~hundreds of MB per session) - start with a small --max-sessions.
- Spikes are resolved by explicit dataset path because ONE cache tables are
  empty on fresh checkouts (SpikeSortingLoader would otherwise return nothing).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.config import load_frozen_config, repo_root
from src.neural.regions import ALL_DECODE_REGIONS
from src.neural.neural_block_decoder import (
    NeuralDecoderConfig,
    aggregate_by_region,
    decode_session,
)

ONE_BASE_URL = "https://openalyx.internationalbrainlab.org"
ONE_PASSWORD = "international"
BWM_PROJECT = "ibl_neuropixel_brainwide_01"
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def make_one(cache_dir: Path):
    from one.api import ONE
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ONE(base_url=ONE_BASE_URL, password=ONE_PASSWORD, silent=True,
               cache_dir=str(cache_dir))


def _eid(session_field: str):
    m = _UUID.search(str(session_field))
    return m.group(0).lower() if m else None


def find_sessions(one, regions: list[str], max_sessions: int | None) -> list[str]:
    """BWM sessions with insertions in any target region (via Alyx REST)."""
    print(f"Searching BWM insertions for {len(regions)} regions ...")
    eids: set[str] = set()
    for region in regions:
        try:
            ins = one.alyx.rest("insertions", "list",
                                atlas_acronym=region, project=BWM_PROJECT)
            hits = {e for e in (_eid(i.get("session", "")) for i in ins) if e}
            print(f"  {region:6s}: {len(hits)} sessions")
            eids.update(hits)
        except Exception as exc:  # noqa: BLE001
            print(f"  {region:6s}: search failed ({exc})")
    # Deterministic hash-shuffle so we don't always hit the same early sessions.
    import hashlib
    ordered = sorted(eids, key=lambda e: hashlib.md5(e.encode()).hexdigest())
    if max_sessions:
        ordered = ordered[:max_sessions]
    print(f"Total unique sessions: {len(eids)} -> decoding {len(ordered)}")
    return ordered


def load_trials(one, eid: str):
    from src.data.inspect_trials import load_trials_for_eid
    try:
        return load_trials_for_eid(one, eid)
    except Exception as exc:  # noqa: BLE001
        print(f"  [trials] {eid}: {exc}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Neural block-prior decoder (IBL BWM)")
    ap.add_argument("--regions", nargs="+", default=None,
                    help="Subset of regions (default: all highlighted + context)")
    ap.add_argument("--max-sessions", type=int, default=3, metavar="N",
                    help="Cap sessions to decode (default: 3; use more for full run)")
    ap.add_argument("--t-start", type=float, default=-0.4,
                    help="Window start rel. stimOn, s (default: -0.4)")
    ap.add_argument("--t-end", type=float, default=0.0,
                    help="Window end rel. stimOn, s (default: 0.0)")
    ap.add_argument("--min-units", type=int, default=5,
                    help="Min units per region per session (default: 5)")
    ap.add_argument("--load-existing", action="store_true",
                    help="Skip download; replot from saved CSV")
    args = ap.parse_args()

    out_dir = ROOT / "reports" / "v2" / "block_decoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "neural_region_results.csv"
    agg_path = out_dir / "neural_region_agg.csv"
    model_json = out_dir / "block_decoder_results.json"

    if args.load_existing:
        if not csv_path.exists():
            print(f"No existing results at {csv_path}")
            sys.exit(1)
        session_df = pd.read_csv(csv_path)
        _finalize(session_df, out_dir, agg_path, model_json)
        return

    cfg = load_frozen_config()
    cache_dir = repo_root() / cfg["data"]["cache_dir"]
    one = make_one(cache_dir)

    from iblatlas.regions import BrainRegions
    brain_regions = BrainRegions()

    dcfg = NeuralDecoderConfig(t_start=args.t_start, t_end=args.t_end, min_units=args.min_units)
    regions = args.regions or list(ALL_DECODE_REGIONS.keys())
    print("\n=== Neural block-prior decoder ===")
    print(f"Regions: {regions}")
    print(f"Window: [{dcfg.t_start:.2f}, {dcfg.t_end:.2f}] s rel. stimOn\n")

    eids = find_sessions(one, regions, args.max_sessions)
    if not eids:
        print("No sessions found.")
        sys.exit(1)

    all_rows: list[dict] = []
    for i, eid in enumerate(eids):
        print(f"\n[{i + 1}/{len(eids)}] {eid}")
        trials = load_trials(one, eid)
        if trials is None:
            continue
        all_rows.extend(decode_session(one, eid, trials, dcfg, brain_regions))

    if not all_rows:
        print("\nNo decodable session x region pairs found. Try more sessions.")
        sys.exit(1)

    session_df = pd.DataFrame(all_rows)
    session_df.to_csv(csv_path, index=False)
    print(f"\nSession x region results: {csv_path.relative_to(ROOT)}")
    _finalize(session_df, out_dir, agg_path, model_json)


def _finalize(session_df: pd.DataFrame, out_dir: Path, agg_path: Path, model_json: Path) -> None:
    agg = aggregate_by_region(session_df.to_dict("records"))
    agg.to_csv(agg_path, index=False)

    print("\n--- Region AUROC ranking ---")
    print(agg[["region", "n_sessions", "auroc_mean", "auroc_sem", "units_mean"]]
          .to_string(index=False))

    from src.plot.phase10_figures import (
        fig_neural_block_decoder,
        fig_neural_vs_model_prior,
    )
    fig1 = out_dir / "neural_block_decoder.png"
    fig_neural_block_decoder(session_df, agg, fig1)
    print(f"\nRegion figure:      {fig1.relative_to(ROOT)}")

    fig2 = out_dir / "neural_vs_model_prior.png"
    fig_neural_vs_model_prior(agg, model_json, fig2)
    print(f"Comparison figure:  {fig2.relative_to(ROOT)}")
    if not model_json.exists():
        print("  (run scripts/12_block_decoder.py first to add model bars)")


if __name__ == "__main__":
    main()
