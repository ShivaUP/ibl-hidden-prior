"""Run model predictions on processed datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.eval.checkpoint import load_checkpoint
from src.models.datasets import (
    BayesTrialDataset,
    RnnBinDataset,
    bayes_collate,
    rnn_collate,
)
from src.train.loop import normalize_model_name


@torch.no_grad()
def predict_split(
    repo_root: Path,
    model_name: str,
    condition: str,
    eids: list[str],
    *,
    batch_size: int = 64,
    run_name: str = "default",
) -> pd.DataFrame:
    """Return per-trial predictions for the given eids."""
    model_name = normalize_model_name(model_name)
    model, payload = load_checkpoint(repo_root, model_name, condition, run_name)
    device = next(model.parameters()).device
    allow = set(eids)

    if model_name == "bayes":
        path = repo_root / "data" / "processed" / "bayes_trials" / f"{condition}.parquet"
        ds = BayesTrialDataset(path, eid_allowlist=allow)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=bayes_collate)
    else:
        path = repo_root / "data" / "processed" / "rnn_bins" / f"{condition}.pkl"
        ds = RnnBinDataset(path, eid_allowlist=allow)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=rnn_collate)

    rows: list[dict] = []
    for batch in loader:
        batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch_t)
        probs = out.choice_probs().cpu().numpy()
        prior = out.prior.cpu().numpy()
        rt_mean = out.rt_log_mean.cpu().numpy()
        rt_std = out.rt_log_std.cpu().numpy()
        for i in range(len(batch["eid"])):
            rows.append(
                {
                    "eid": batch["eid"][i],
                    "trial_index": int(batch["trial_index"][i]),
                    "choice_right": int(batch["choice_right"][i]),
                    "log_rt": float(batch["log_rt"][i]),
                    "p_right": float(probs[i]),
                    "prior_q": float(prior[i]),
                    "rt_log_mean": float(rt_mean[i]),
                    "rt_log_std": float(rt_std[i]),
                    "model": model_name,
                    "condition": condition,
                }
            )
    return pd.DataFrame(rows)


def attach_trial_meta(preds: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Join predictions with processed trial metadata."""
    meta_cols = [
        "eid",
        "trial_index",
        "probabilityLeft",
        "abs_contrast",
        "stimulus_right",
        "contrast_high",
        "block_switch",
        "trials_from_block_start",
        "choice_right",
        "rt",
        "log_rt",
        "feedbackType",
    ]
    meta = trials[meta_cols].copy()
    # signed contrast: right positive
    meta["signed_contrast"] = np.where(
        meta["stimulus_right"] == 1, meta["abs_contrast"], -meta["abs_contrast"]
    )
    out = preds.merge(meta, on=["eid", "trial_index"], how="inner", suffixes=("", "_meta"))
    if "choice_right_meta" in out.columns:
        out["choice_right"] = out["choice_right_meta"]
        out.drop(columns=["choice_right_meta"], inplace=True)
    if "log_rt_meta" in out.columns:
        out["log_rt"] = out["log_rt_meta"]
        out.drop(columns=["log_rt_meta"], inplace=True)
    return out


@torch.no_grad()
def predict_arrays(
    repo_root: Path,
    model_name: str,
    condition: str,
    *,
    rnn_payload: dict | None = None,
    bayes_df: pd.DataFrame | None = None,
    batch_size: int = 64,
    run_name: str = "default",
) -> pd.DataFrame:
    """Run checkpoint on in-memory RNN payload or Bayesian table (neural eids)."""
    import pickle

    model_name = normalize_model_name(model_name)
    model, _payload = load_checkpoint(repo_root, model_name, condition, run_name)
    device = next(model.parameters()).device
    tmp_dir = repo_root / "data" / "processed" / "neural"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    if model_name == "bayes":
        assert bayes_df is not None
        path = tmp_dir / "_tmp_bayes.parquet"
        bayes_df.to_parquet(path, index=False)
        ds = BayesTrialDataset(path)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=bayes_collate)
    else:
        assert rnn_payload is not None
        path = tmp_dir / "_tmp_rnn.pkl"
        with path.open("wb") as f:
            pickle.dump(rnn_payload, f)
        ds = RnnBinDataset(path)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=rnn_collate)

    for batch in loader:
        batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch_t)
        probs = out.choice_probs().cpu().numpy()
        prior = out.prior.cpu().numpy()
        for i in range(len(batch["eid"])):
            rows.append(
                {
                    "eid": batch["eid"][i],
                    "trial_index": int(batch["trial_index"][i]),
                    "p_right": float(probs[i]),
                    "prior_q": float(prior[i]),
                    "model": model_name,
                    "condition": condition,
                }
            )
    return pd.DataFrame(rows)
