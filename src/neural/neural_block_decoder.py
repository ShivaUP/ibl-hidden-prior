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
    # IBL-style quality gate: keep only units passing single-unit QC.
    good_only: bool = True
    min_quality: float = 1.0        # clusters.metrics 'label' threshold (1.0 = all metrics passed)
    # Pseudosession significance test (0 = disabled; IBL uses this instead of a
    # fixed AUROC cutoff to control for block autocorrelation).
    n_pseudo: int = 0
    # Variance-explained prior readout (frozen v1 primary metric): CV Ridge
    # neural -> mouse subjective prior (behavior-derived leaky stim integration).
    compute_ve: bool = True
    prior_alpha: float = 0.05        # leak rate for the mouse subjective prior


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


def load_session_spikes(one, eid: str, brain_regions, *, good_only: bool = True,
                        min_quality: float = 1.0) -> Tuple[dict, pd.DataFrame]:
    """Load and merge spikes across probes for one session.

    Returns
    -------
    spikes      : {"times": (N,), "clusters": (N,)}  global cluster ids
    clusters_df : DataFrame [cluster_id, acronym, good]

    ``good`` marks units passing IBL single-unit QC (clusters.metrics 'label'
    >= ``min_quality``). If metrics are unavailable, all units are marked good.
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

        # Single-unit QC label (optional). IBL 'label' in {0, 1/3, 2/3, 1};
        # good = all metrics passed. Missing metrics → treat all as good.
        good_flags = np.ones(n_clusters, dtype=bool)
        if good_only:
            metrics_path = _find_dataset(all_ds, coll, "clusters.metrics.pqt")
            if metrics_path is not None:
                try:
                    metrics = _load_by_path(one, eid, metrics_path)
                    if hasattr(metrics, "columns") and "label" in metrics.columns:
                        lbl = np.asarray(metrics["label"].to_numpy(dtype=float))
                        m = min(len(lbl), n_clusters)
                        good_flags = np.zeros(n_clusters, dtype=bool)
                        good_flags[:m] = lbl[:m] >= min_quality
                except Exception:  # noqa: BLE001
                    pass  # keep all-good fallback

        for local_id in range(n_clusters):
            cluster_rows.append({
                "cluster_id": local_id + offset,
                "acronym": acronyms[local_id],
                "good": bool(good_flags[local_id]),
            })
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
    valid = np.isfinite(stim_on)                       # all usable trials (for pseudo null)
    biased = (labels >= 0) & valid                     # real left/right-block trials
    if int((labels[biased] == BLOCK_LEFT).sum()) < cfg.min_trials_per_class or \
       int((labels[biased] == BLOCK_RIGHT).sum()) < cfg.min_trials_per_class:
        return []

    try:
        spikes, clusters_df = load_session_spikes(
            one, eid, brain_regions, good_only=cfg.good_only, min_quality=cfg.min_quality
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [neural] {eid}: spike load failed — {exc}")
        return []

    # Assign clusters to the highlighted decode-regions (good units only if gated).
    region_units: Dict[str, List[int]] = {r: [] for r in ALL_DECODE_REGIONS}
    for _, row in clusters_df.iterrows():
        if cfg.good_only and not bool(row.get("good", True)):
            continue
        region = unit_in_any_decode_region(str(row["acronym"]))
        if region is not None:
            region_units[region].append(int(row["cluster_id"]))

    valid_idx = np.where(valid)[0]
    stim_on_valid = stim_on[valid_idx]
    labels_valid = labels[valid_idx]              # 0/1/-1 aligned to valid trials
    real_mask = labels_valid >= 0
    rng = np.random.default_rng(cfg.random_state)

    # Mouse subjective prior (behavior-derived leaky stim integration) for VE.
    mouse_prior_valid = None
    if cfg.compute_ve and "contrastRight" in trials.columns:
        from src.eval.mouse_prior import _session_prior_path
        stim_right = trials["contrastRight"].notna().to_numpy().astype(float)
        mouse_prior_full = _session_prior_path(stim_right, alpha=cfg.prior_alpha)
        mouse_prior_valid = mouse_prior_full[valid_idx]

    results: List[Dict] = []
    for region, unit_list in region_units.items():
        if len(unit_list) < cfg.min_units:
            continue
        cids = np.array(sorted(set(unit_list)), dtype=np.int64)
        # Bin ALL valid trials once; real decode + pseudo null reuse this matrix.
        counts_valid = bin_spikes_per_trial(
            spikes["times"], spikes["clusters"], cids, stim_on_valid, cfg.t_start, cfg.t_end
        )
        counts_valid = counts_valid[:, counts_valid.std(axis=0) > 1e-6]
        if counts_valid.shape[1] < cfg.min_units:
            continue

        try:
            dec = fit_block_decoder(
                counts_valid[real_mask].astype(np.float64), labels_valid[real_mask],
                binary=True, n_folds=cfg.n_folds, C=cfg.C, random_state=cfg.random_state,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [neural] {eid}/{region}: decode failed — {exc}")
            continue

        row = {
            "eid": eid,
            "region": region,
            "n_units": int(counts_valid.shape[1]),
            "n_trials": int(real_mask.sum()),
            "accuracy": dec["accuracy_mean"],
            "auroc": dec.get("auroc_mean", float("nan")),
        }

        # VE prior readout: CV Ridge neural -> mouse subjective prior (all valid trials).
        if cfg.compute_ve and mouse_prior_valid is not None:
            from src.neural.prior_readout import fit_prior_readout
            ve_res = fit_prior_readout(
                counts_valid.astype(np.float64), mouse_prior_valid,
                n_splits=min(5, cfg.n_folds),
            )
            row["ve"] = ve_res["ve_cv"]
            row["ve_corr"] = ve_res["corr_cv"]

        # Pseudosession significance (IBL-style): compare the real AUROC to a
        # null built from block sequences with the same autocorrelation.
        if cfg.n_pseudo > 0:
            p_val, n_null = pseudosession_pvalue(
                counts_valid, labels_valid, row["auroc"], cfg, rng
            )
            row["p_value"] = p_val
            row["n_pseudo"] = n_null

        results.append(row)
        sig = ""
        if cfg.n_pseudo > 0 and np.isfinite(row.get("p_value", np.nan)):
            sig = f"  p={row['p_value']:.3f}"
        ve_str = f"  VE={row['ve']:.3f}" if "ve" in row and np.isfinite(row.get("ve", np.nan)) else ""
        print(f"    {region:6s}  units={row['n_units']:3d}  AUROC={row['auroc']:.3f}{ve_str}{sig}")

    return results


def generate_pseudo_block_labels(
    n_trials: int,
    rng: np.random.Generator,
    *,
    unbiased_start: int = 90,
    mean_len: float = 60.0,
    min_len: int = 20,
    max_len: int = 100,
) -> np.ndarray:
    """Generate a pseudo block-label sequence with IBL biasedChoiceWorld structure.

    First ``unbiased_start`` trials are unbiased (label -1), then blocks alternate
    between left-biased (pL=0.8 → BLOCK_LEFT) and right-biased (pL=0.2 → BLOCK_RIGHT)
    with lengths from a truncated exponential. Preserves the long-run
    autocorrelation that inflates naive decoding scores.
    """
    from src.eval.block_decoder import BLOCK_LEFT, BLOCK_RIGHT

    out = np.full(n_trials, -1, dtype=np.int64)
    i = min(unbiased_start, n_trials)
    p_left = float(rng.choice([0.2, 0.8]))
    while i < n_trials:
        block_len = int(np.clip(rng.exponential(mean_len), min_len, max_len))
        end = min(i + block_len, n_trials)
        out[i:end] = BLOCK_LEFT if p_left == 0.8 else BLOCK_RIGHT
        p_left = 0.2 if p_left == 0.8 else 0.8
        i = end
    return out


def pseudosession_pvalue(
    counts_valid: np.ndarray,
    labels_valid: np.ndarray,
    real_auroc: float,
    cfg: NeuralDecoderConfig,
    rng: np.random.Generator,
) -> Tuple[float, int]:
    """One-sided p-value: P(null AUROC >= real) over pseudosessions.

    ``counts_valid`` are spike counts for ALL valid trials; pseudo labels are
    generated over the full trial sequence and re-decoded. Uses a lighter
    (3-fold) decode for speed. Returns (p_value, n_null_used).
    """
    from src.eval.block_decoder import BLOCK_LEFT, BLOCK_RIGHT, fit_block_decoder

    if not np.isfinite(real_auroc):
        return float("nan"), 0

    n_trials = len(labels_valid)
    null_folds = min(3, cfg.n_folds)
    null: List[float] = []
    for _ in range(cfg.n_pseudo):
        pl = generate_pseudo_block_labels(n_trials, rng)
        keep = pl >= 0
        if int((pl[keep] == BLOCK_LEFT).sum()) < cfg.min_trials_per_class or \
           int((pl[keep] == BLOCK_RIGHT).sum()) < cfg.min_trials_per_class:
            continue
        try:
            d = fit_block_decoder(
                counts_valid[keep].astype(np.float64), pl[keep],
                binary=True, n_folds=null_folds, C=cfg.C, random_state=cfg.random_state,
            )
            a = d.get("auroc_mean", float("nan"))
            if np.isfinite(a):
                null.append(a)
        except Exception:  # noqa: BLE001
            continue

    if not null:
        return float("nan"), 0
    null_arr = np.asarray(null)
    p = (1 + int((null_arr >= real_auroc).sum())) / (1 + len(null_arr))
    return float(p), len(null_arr)



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
        entry = {
            "region": region,
            "n_sessions": len(a),
            "auroc_mean": float(np.mean(a)),
            "auroc_sem": float(np.std(a) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
            "accuracy_mean": float(g["accuracy"].mean()),
            "units_mean": float(g["n_units"].mean()),
        }
        # VE prior-readout summary (only if computed).
        if "ve" in g.columns:
            ve = g["ve"].dropna().to_numpy()
            if len(ve):
                entry["ve_mean"] = float(np.mean(ve))
                entry["ve_sem"] = float(np.std(ve) / np.sqrt(len(ve))) if len(ve) > 1 else 0.0
        # Pseudosession significance summary (only if the null was run).
        if "p_value" in g.columns:
            pv = g["p_value"].dropna().to_numpy()
            if len(pv):
                entry["p_median"] = float(np.median(pv))
                entry["frac_sig_p05"] = float(np.mean(pv < 0.05))
        rows.append(entry)
    return pd.DataFrame(rows).sort_values("auroc_mean", ascending=False).reset_index(drop=True)
