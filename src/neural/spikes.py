"""Load and trial-align spikes for v1 neural regions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.neural.regions import unit_in_region


@dataclass
class NeuralWindow:
    """Peri-stimulus spike-count window (seconds relative to stimOn)."""

    t_start: float = -0.1
    t_end: float = 0.3
    align_event: str = "stimOn_times"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegionSpikeMatrix:
    eid: str
    region: str
    cluster_ids: np.ndarray
    acronyms: np.ndarray
    trial_index: np.ndarray
    counts: np.ndarray  # (n_trials, n_units)
    window: dict
    n_units: int
    n_trials: int


def load_clusters_and_spikes(one, eid: str) -> tuple[pd.DataFrame, dict]:
    """Load clusters table + spikes dict via ONE (may download)."""
    # Prefer SpikeSortingLoader when available; fall back to load_object.
    try:
        from brainbox.io.one import SpikeSortingLoader
        from iblatlas.atlas import AllenAtlas

        atlas = AllenAtlas()
        pids = one.alyx.rest("insertions", "list", session=eid)
        if not pids:
            raise RuntimeError(f"No insertions for {eid}")
        cluster_rows = []
        spike_times_list = []
        spike_clusters_list = []
        offset = 0
        for ins in pids:
            pid = ins["id"]
            ssl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
            spikes, clusters, channels = ssl.load_spike_sorting()
            clusters = ssl.merge_clusters(spikes, clusters, channels)
            # assign global cluster ids by offsetting
            local_ids = np.asarray(clusters["cluster_id"] if "cluster_id" in clusters else np.arange(len(clusters)))
            n = len(local_ids)
            global_ids = local_ids + offset
            acr = np.asarray(
                clusters.get(
                    "acronym",
                    clusters.get("brainLocationAcronyms_ccf_2017", ["unknown"] * n),
                )
            )
            if acr.ndim > 1:
                acr = acr.ravel()
            for i in range(n):
                cluster_rows.append(
                    {
                        "cluster_id": int(global_ids[i]),
                        "acronym": str(acr[i]) if i < len(acr) else "unknown",
                        "pid": pid,
                    }
                )
            st = np.asarray(spikes["times"])
            sc = np.asarray(spikes["clusters"]) + offset
            spike_times_list.append(st)
            spike_clusters_list.append(sc)
            offset += int(local_ids.max()) + 1 if len(local_ids) else 0
        clusters_df = pd.DataFrame(cluster_rows)
        spikes_out = {
            "times": np.concatenate(spike_times_list) if spike_times_list else np.array([]),
            "clusters": np.concatenate(spike_clusters_list) if spike_clusters_list else np.array([]),
        }
        return clusters_df, spikes_out
    except Exception:
        # Fallback: session-level alf
        clusters = one.load_object(eid, "clusters")
        spikes = one.load_object(eid, "spikes")
        n = len(next(iter(clusters.values())))
        acr_key = "brainLocationAcronyms_ccf_2017" if "brainLocationAcronyms_ccf_2017" in clusters else "acronym"
        clusters_df = pd.DataFrame(
            {
                "cluster_id": np.arange(n),
                "acronym": np.asarray(clusters[acr_key]).astype(str).ravel()[:n],
            }
        )
        return clusters_df, {"times": np.asarray(spikes["times"]), "clusters": np.asarray(spikes["clusters"])}


def filter_region_clusters(clusters_df: pd.DataFrame, region: str) -> pd.DataFrame:
    mask = clusters_df["acronym"].map(lambda a: unit_in_region(a, region))
    return clusters_df.loc[mask].copy()


def trial_spike_counts(
    spikes: dict,
    cluster_ids: np.ndarray,
    align_times: np.ndarray,
    trial_index: np.ndarray,
    window: NeuralWindow,
) -> np.ndarray:
    """Return (n_trials, n_units) spike counts in [t_start, t_end) relative to align_times."""
    times = np.asarray(spikes["times"], dtype=float)
    clusters = np.asarray(spikes["clusters"], dtype=int)
    id_to_col = {int(c): i for i, c in enumerate(cluster_ids)}
    n_trials = len(align_times)
    n_units = len(cluster_ids)
    counts = np.zeros((n_trials, n_units), dtype=np.float64)
    if n_units == 0 or len(times) == 0:
        return counts

    # Restrict spikes to clusters of interest
    keep = np.isin(clusters, cluster_ids)
    times = times[keep]
    clusters = clusters[keep]
    order = np.argsort(times)
    times = times[order]
    clusters = clusters[order]

    for t_i, t0 in enumerate(align_times):
        if not np.isfinite(t0):
            continue
        lo = t0 + window.t_start
        hi = t0 + window.t_end
        left = np.searchsorted(times, lo, side="left")
        right = np.searchsorted(times, hi, side="left")
        for c in clusters[left:right]:
            col = id_to_col.get(int(c))
            if col is not None:
                counts[t_i, col] += 1.0
    return counts


def build_region_matrix(
    one,
    eid: str,
    trials: pd.DataFrame,
    region: str,
    window: NeuralWindow | None = None,
) -> RegionSpikeMatrix:
    """End-to-end region spike-count matrix aligned to stimOn."""
    window = window or NeuralWindow()
    clusters_df, spikes = load_clusters_and_spikes(one, eid)
    reg = filter_region_clusters(clusters_df, region)
    cluster_ids = reg["cluster_id"].to_numpy(dtype=int)
    acronyms = reg["acronym"].to_numpy()
    align = trials[window.align_event].to_numpy(dtype=float)
    tidx = trials["trial_index"].to_numpy(dtype=int) if "trial_index" in trials.columns else np.arange(len(trials))
    counts = trial_spike_counts(spikes, cluster_ids, align, tidx, window)
    return RegionSpikeMatrix(
        eid=eid,
        region=region,
        cluster_ids=cluster_ids,
        acronyms=acronyms,
        trial_index=tidx,
        counts=counts,
        window=window.to_dict(),
        n_units=int(len(cluster_ids)),
        n_trials=int(len(trials)),
    )
