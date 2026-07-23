"""Decode the mouse's block-prior belief from real IBL neural activity.

Pipeline per session × region:
  1. Load spikes/clusters by explicit dataset path (revision-aware; the ONE cache
     tables are empty on fresh checkouts, so SpikeSortingLoader returns nothing —
     we resolve the exact `alf/{probe}/pykilosort/#rev#/…` paths instead).
  2. Map each cluster to an Allen CCF region via channels.brainLocationIds_ccf_2017.
  3. Bin spikes in a pre-stimulus window per trial (the running prior belief).
  4. Label each trial by block: left-biased (pL=0.8 → p_right≈0.2) vs right-biased.
  5. Fit a logistic-regression decoder (shared with the model decoder) with CV.

Reference regions: choice-selective subcortical areas highlighted in
IBL et al. 2025 (Nature) Fig 5 — thalamus (CL, SPF), midbrain (SCm, MRN, SNr,
RPF, NPC), hindbrain/cerebellar nuclei (GRN, IRN, SOC, VII, TRN, FOTU).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.neural.regions import ALL_DECODE_REGIONS, unit_in_any_decode_region


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NeuralDecoderConfig:
    # Pre-stimulus spike-count window relative to stimOn_times (seconds).
    # Captures the ongoing block-prior belief before stimulus/choice.
    t_start: float = -0.4
    t_end: float = 0.0
    min_units: int = 5             # min clusters in a region to attempt decoding
    min_trials_per_class: int = 20  # min left- and right-block trials
    n_folds: int = 5
    C: float = 1.0
    random_state: int = 42


# ---------------------------------------------------------------------------
# Dataset-path resolution (revision-aware, cache-free)
# ---------------------------------------------------------------------------

_SPIKE_DATASETS = (
    "spikes.times.npy",
    "spikes.clusters.npy",
    "clusters.channels.npy",
    "channels.brainLocationIds_ccf_2017.npy",
)


def _parse_dataset_path(path: str) -> Tuple[str, str, Optional[str]]:
    """Split a relative ALF path into (name, collection, revision)."""
    parts = path.split("/")
    name = parts[-1]
    if len(parts) >= 2 and parts[-2].startswith("#") and parts[-2].endswith("#"):
        return name, "/".join(parts[:-2]), parts[-2].strip("#")
    return name, "/".join(parts[:-1]), None


def _collection_base(path: str) -> str:
    """Directory of a dataset path with any ``#revision#`` folder removed."""
    parts = path.rsplit("/", 1)[0].split("/")
    return "/".join(p for p in parts if not (p.startswith("#") and p.endswith("#")))


def discover_spike_collections(all_ds: List[str]) -> List[str]:
    """Unique ``alf/{probe}/pykilosort`` collections that contain spikes.times."""
    colls = {
        _collection_base(d)
        for d in all_ds
        if d.rsplit("/", 1)[-1] == "spikes.times.npy"
    }
    return sorted(colls)


def _find_dataset(all_ds: List[str], coll_base: str, name: str) -> Optional[str]:
    """Full path for ``name`` under ``coll_base``, preferring the revision copy."""
    cands = [
        d
        for d in all_ds
        if d.rsplit("/", 1)[-1] == name and _collection_base(d) == coll_base
    ]
    if not cands:
        return None
    # Prefer the revisioned dataset (latest spike-sorting re-run).
    rev = [d for d in cands if "#" in d]
    return (rev or cands)[0]


def _load_by_path(one, eid: str, path: str):
    """Load a dataset given its full relative path (works with empty cache)."""
    name, collection, revision = _parse_dataset_path(path)
    # Loading by exact full path is the only reliable route with empty cache
    # tables; fall back to collection+revision if needed.
    try:
        return one.load_dataset(eid, path)
    except Exception:
        return one.load_dataset(eid, name, collection=collection, revision=revision)


def load_session_spikes(one, eid: str, brain_regions) -> Tuple[dict, pd.DataFrame]:
    """Load and merge spikes across probes for one session.

    Returns
    -------
    spikes      : {"times": (N,), "clusters": (N,)}  global cluster ids
    clusters_df : DataFrame [cluster_id, acronym]
    """
    all_ds = [str(d) for d in one.list_datasets(eid)]
    collections = discover_spike_collections(all_ds)
    if not collections:
        raise RuntimeError("no spike-sorting collections found")

    times_list: List[np.ndarray] = []
    clusters_list: List[np.ndarray] = []
    cluster_rows: List[dict] = []
    offset = 0
    errors: List[str] = []

    for coll in collections:
        try:
            paths = {}
            for name in _SPIKE_DATASETS:
                p = _find_dataset(all_ds, coll, name)
                if p is None:
                    raise RuntimeError(f"missing {name}")
                paths[name] = p

            st = np.asarray(_load_by_path(one, eid, paths["spikes.times.npy"]))
            sc = np.asarray(_load_by_path(one, eid, paths["spikes.clusters.npy"])).astype(np.int64)
            cl_ch = np.asarray(_load_by_path(one, eid, paths["clusters.channels.npy"])).astype(np.int64)
            ch_reg = np.asarray(_load_by_path(one, eid, paths["channels.brainLocationIds_ccf_2017.npy"]))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{coll}: {exc}")
            continue

        # Cluster → Allen acronym via its peak channel's region id.
        cl_ch = np.clip(cl_ch, 0, len(ch_reg) - 1)
        region_ids = ch_reg[cl_ch]
        acronyms = np.asarray(brain_regions.id2acronym(region_ids)).astype(str)

        n_clusters = len(cl_ch)
        for local_id in range(n_clusters):
            cluster_rows.append(
                {"cluster_id": local_id + offset, "acronym": acronyms[local_id]}
            )
        times_list.append(st)
        clusters_list.append(sc + offset)
        offset += n_clusters

    if not cluster_rows:
        raise RuntimeError("; ".join(errors) or "no probes loaded")

    spikes = {
        "times": np.concatenate(times_list),
        "clusters": np.concatenate(clusters_list),
    }
    return spikes, pd.DataFrame(cluster_rows)


# ---------------------------------------------------------------------------
# Spike binning
# ---------------------------------------------------------------------------

def bin_spikes_per_trial(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    cluster_ids: np.ndarray,
    stim_on: np.ndarray,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """Count spikes in [stimOn+t_start, stimOn+t_end) for each trial × unit.

    Returns (n_trials, n_units) float32.
    """
    order = np.argsort(spike_times, kind="mergesort")
    st = spike_times[order]
    sc = spike_clusters[order]

    lut = np.full(int(sc.max()) + 2 if len(sc) else 1, -1, dtype=np.int64)
    for idx, cid in enumerate(cluster_ids):
        lut[cid] = idx

    counts = np.zeros((len(stim_on), len(cluster_ids)), dtype=np.float32)
    for t, t0 in enumerate(stim_on):
        i0 = np.searchsorted(st, t0 + t_start, side="left")
        i1 = np.searchsorted(st, t0 + t_end, side="left")
        if i1 <= i0:
            continue
        mapped = lut[sc[i0:i1]]
        mapped = mapped[mapped >= 0]
        if len(mapped):
            counts[t] += np.bincount(mapped, minlength=len(cluster_ids)).astype(np.float32)
    return counts


# ---------------------------------------------------------------------------
# Single-session decoding
# ---------------------------------------------------------------------------

def decode_session(
    one,
    eid: str,
    trials: pd.DataFrame,
    cfg: NeuralDecoderConfig,
    brain_regions,
) -> List[Dict]:
    """Decode block prior from each highlighted region in one session.

    Returns one result dict per region with enough units/trials (or []).
    """
    from src.eval.block_decoder import BLOCK_LEFT, BLOCK_RIGHT, fit_block_decoder

    if "probabilityLeft" not in trials.columns or "stimOn_times" not in trials.columns:
        return []

    p_right = 1.0 - trials["probabilityLeft"].to_numpy(dtype=float)
    labels = np.full(len(p_right), -1, dtype=np.int64)
    labels[np.abs(p_right - 0.2) < 0.05] = BLOCK_LEFT
    labels[np.abs(p_right - 0.8) < 0.05] = BLOCK_RIGHT

    stim_on = trials["stimOn_times"].to_numpy(dtype=float)
    keep = (labels >= 0) & np.isfinite(stim_on)
    if int((labels[keep] == BLOCK_LEFT).sum()) < cfg.min_trials_per_class or \
       int((labels[keep] == BLOCK_RIGHT).sum()) < cfg.min_trials_per_class:
        return []

    try:
        spikes, clusters_df = load_session_spikes(one, eid, brain_regions)
    except Exception as exc:  # noqa: BLE001
        print(f"  [neural] {eid}: spike load failed — {exc}")
        return []

    # Assign clusters to the highlighted decode-regions.
    region_units: Dict[str, List[int]] = {r: [] for r in ALL_DECODE_REGIONS}
    for _, row in clusters_df.iterrows():
        region = unit_in_any_decode_region(str(row["acronym"]))
        if region is not None:
            region_units[region].append(int(row["cluster_id"]))

    t_idx = np.where(keep)[0]
    stim_on_sel = stim_on[t_idx]
    labels_sel = labels[t_idx]

    results: List[Dict] = []
    for region, unit_list in region_units.items():
        if len(unit_list) < cfg.min_units:
            continue
        cids = np.array(sorted(set(unit_list)), dtype=np.int64)
        counts = bin_spikes_per_trial(
            spikes["times"], spikes["clusters"], cids, stim_on_sel, cfg.t_start, cfg.t_end
        )
        # Drop silent/constant units.
        counts = counts[:, counts.std(axis=0) > 1e-6]
        if counts.shape[1] < cfg.min_units:
            continue

        try:
            dec = fit_block_decoder(
                counts.astype(np.float64), labels_sel, binary=True,
                n_folds=cfg.n_folds, C=cfg.C, random_state=cfg.random_state,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [neural] {eid}/{region}: decode failed — {exc}")
            continue

        results.append({
            "eid": eid,
            "region": region,
            "n_units": int(counts.shape[1]),
            "n_trials": int(len(labels_sel)),
            "accuracy": dec["accuracy_mean"],
            "auroc": dec.get("auroc_mean", float("nan")),
        })
        print(f"    {region:6s}  units={counts.shape[1]:3d}  AUROC={results[-1]['auroc']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_region(session_results: List[Dict]) -> pd.DataFrame:
    """Per-region mean AUROC ± SEM across sessions."""
    if not session_results:
        return pd.DataFrame()
    df = pd.DataFrame(session_results)
    rows = []
    for region, g in df.groupby("region"):
        a = g["auroc"].dropna().to_numpy()
        if len(a) == 0:
            continue
        rows.append({
            "region": region,
            "n_sessions": len(a),
            "auroc_mean": float(np.mean(a)),
            "auroc_sem": float(np.std(a) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
            "accuracy_mean": float(g["accuracy"].mean()),
            "units_mean": float(g["n_units"].mean()),
        })
    return pd.DataFrame(rows).sort_values("auroc_mean", ascending=False).reset_index(drop=True)
