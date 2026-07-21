"""Dataset adapters for RNN event-bins and Bayesian trial tables."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class RnnBinDataset(Dataset):
    """Ragged RNN sequences from a condition pickle."""

    def __init__(
        self,
        pkl_path: Path,
        max_trials: int | None = None,
        eid_allowlist: set[str] | None = None,
    ):
        with Path(pkl_path).open("rb") as f:
            data = pickle.load(f)
        n = len(data["sequences"])
        eids = [str(e) for e in data["eid"][:n]]
        keep = list(range(n))
        if eid_allowlist is not None:
            allow = {str(e) for e in eid_allowlist}
            keep = [i for i in keep if eids[i] in allow]
        if max_trials is not None:
            keep = keep[:max_trials]
        self.sequences = [data["sequences"][i] for i in keep]
        self.choice_right = np.asarray(data["choice_right"], dtype=np.int64)[keep]
        self.log_rt = np.asarray(data["log_rt"], dtype=np.float32)[keep]
        self.eid = [eids[i] for i in keep]
        self.trial_index = np.asarray(data["trial_index"], dtype=np.int64)[keep]
        self.channels = list(data["channels"])
        self.input_size = len(self.channels)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        x = torch.from_numpy(self.sequences[idx])  # (T, C)
        return {
            "inputs": x,
            "choice_right": torch.tensor(self.choice_right[idx], dtype=torch.long),
            "log_rt": torch.tensor(self.log_rt[idx], dtype=torch.float32),
            "length": torch.tensor(x.shape[0], dtype=torch.long),
            "eid": self.eid[idx],
            "trial_index": int(self.trial_index[idx]),
        }


def rnn_collate(batch: list[dict]) -> dict:
    """Pad variable-length sequences; mask True = valid timestep."""
    inputs = [b["inputs"] for b in batch]
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    padded = pad_sequence(inputs, batch_first=True)  # (B, T, C)
    bsz, tmax, _ = padded.shape
    arange = torch.arange(tmax).unsqueeze(0)
    mask = arange < lengths.unsqueeze(1)
    return {
        "inputs": padded,
        "mask": mask,
        "lengths": lengths,
        "choice_right": torch.stack([b["choice_right"] for b in batch]),
        "log_rt": torch.stack([b["log_rt"] for b in batch]),
        "eid": [b["eid"] for b in batch],
        "trial_index": torch.tensor([b["trial_index"] for b in batch], dtype=torch.long),
    }


class BayesTrialDataset(Dataset):
    """Flat Bayesian trial feature rows from a parquet table."""

    FEATURE_CANDIDATES = (
        "stimulus_right",
        "contrast_high",
        "prev_choice_right",
        "prev_correct",
        "prev_fast_rt",
        "oracle_prior_right",
    )

    def __init__(
        self,
        parquet_path: Path,
        max_trials: int | None = None,
        eid_allowlist: set[str] | None = None,
    ):
        df = pd.read_parquet(parquet_path)
        if eid_allowlist is not None:
            allow = {str(e) for e in eid_allowlist}
            df = df[df["eid"].astype(str).isin(allow)].copy()
        # Keep session order for online prior updates
        df = df.sort_values(["eid", "trial_index"]).reset_index(drop=True)
        if max_trials is not None:
            df = df.iloc[:max_trials].copy()
        self.feature_names = [c for c in self.FEATURE_CANDIDATES if c in df.columns]
        self.features = df[self.feature_names].to_numpy(dtype=np.float32)
        self.choice_right = df["choice_right"].to_numpy(dtype=np.int64)
        self.log_rt = df["log_rt"].to_numpy(dtype=np.float32)
        self.eid = df["eid"].astype(str).tolist()
        self.trial_index = df["trial_index"].to_numpy(dtype=np.int64)
        self.session_start = np.zeros(len(df), dtype=bool)
        prev = None
        for i, e in enumerate(self.eid):
            if e != prev:
                self.session_start[i] = True
                prev = e

    def __len__(self) -> int:
        return len(self.choice_right)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": torch.from_numpy(self.features[idx].copy()),
            "choice_right": torch.tensor(self.choice_right[idx], dtype=torch.long),
            "log_rt": torch.tensor(self.log_rt[idx], dtype=torch.float32),
            "session_start": torch.tensor(self.session_start[idx], dtype=torch.bool),
            "eid": self.eid[idx],
            "trial_index": int(self.trial_index[idx]),
        }


def bayes_collate(batch: list[dict]) -> dict:
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "choice_right": torch.stack([b["choice_right"] for b in batch]),
        "log_rt": torch.stack([b["log_rt"] for b in batch]),
        "session_start": torch.stack([b["session_start"] for b in batch]),
        "eid": [b["eid"] for b in batch],
        "trial_index": torch.tensor([b["trial_index"] for b in batch], dtype=torch.long),
    }
