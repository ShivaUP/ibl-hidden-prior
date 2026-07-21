"""Shared smoke / training loop with split-aware full training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models.bayesian import BayesianOnlineModel
from src.models.datasets import (
    BayesTrialDataset,
    RnnBinDataset,
    bayes_collate,
    rnn_collate,
)
from src.models.interfaces import extract_latent_prior, joint_loss
from src.models.pc_rnn import PredictiveCodingRNN
from src.models.standard_rnn import StandardRNN


def build_model(name: str, input_size: int, hidden_size: int = 64) -> torch.nn.Module:
    name = name.lower()
    if name in {"standard", "standard_rnn", "rnn"}:
        return StandardRNN(input_size=input_size, hidden_size=hidden_size)
    if name in {"pc", "pc_rnn", "predictive_coding", "predictive_coding_rnn"}:
        return PredictiveCodingRNN(input_size=input_size, hidden_size=hidden_size)
    if name in {"bayes", "bayesian"}:
        return BayesianOnlineModel(input_size=input_size, hidden_size=hidden_size)
    raise ValueError(f"Unknown model: {name}")


def normalize_model_name(name: str) -> str:
    name = name.lower()
    if name in {"standard_rnn", "rnn"}:
        return "standard"
    if name in {"pc_rnn", "predictive_coding", "predictive_coding_rnn"}:
        return "pc"
    if name in {"bayesian"}:
        return "bayes"
    return name


def make_loader(
    model_name: str,
    condition: str,
    repo_root: Path,
    *,
    max_trials: int | None,
    batch_size: int,
    eid_allowlist: set[str] | list[str] | None = None,
    shuffle: bool | None = None,
) -> tuple[DataLoader, int]:
    model_name = normalize_model_name(model_name)
    allow = set(eid_allowlist) if eid_allowlist is not None else None
    if model_name == "bayes":
        path = repo_root / "data" / "processed" / "bayes_trials" / f"{condition}.parquet"
        ds = BayesTrialDataset(path, max_trials=max_trials, eid_allowlist=allow)
        # Never shuffle Bayes: online prior needs session order
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, collate_fn=bayes_collate
        )
        return loader, len(ds.feature_names)
    path = repo_root / "data" / "processed" / "rnn_bins" / f"{condition}.pkl"
    ds = RnnBinDataset(path, max_trials=max_trials, eid_allowlist=allow)
    do_shuffle = True if shuffle is None else shuffle
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=do_shuffle, collate_fn=rnn_collate
    )
    return loader, ds.input_size


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    lambda_rt: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "choice_nll": 0.0, "rt_nll": 0.0, "choice_acc": 0.0}
    n_batches = 0
    n_trials = 0
    correct = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(batch)
        loss, stats = joint_loss(
            outputs, batch["choice_right"], batch["log_rt"], lambda_rt=lambda_rt
        )
        preds = (outputs.choice_probs() >= 0.5).long()
        correct += int((preds == batch["choice_right"]).sum().item())
        n_trials += int(batch["choice_right"].numel())
        for k, v in stats.items():
            totals[k] += v
        n_batches += 1
    avg = {k: v / max(n_batches, 1) for k, v in totals.items() if k != "choice_acc"}
    avg["choice_acc"] = correct / max(n_trials, 1)
    avg["n_trials"] = float(n_trials)
    return avg


def train_epochs(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    lambda_rt: float,
    device: torch.device,
) -> list[dict[str, float]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        totals = {"loss": 0.0, "choice_nll": 0.0, "rt_nll": 0.0}
        n_batches = 0
        for batch in loader:
            batch = _move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, stats = joint_loss(
                outputs, batch["choice_right"], batch["log_rt"], lambda_rt=lambda_rt
            )
            loss.backward()
            opt.step()
            for k, v in stats.items():
                totals[k] += v
            n_batches += 1
            prior = extract_latent_prior(outputs)
            if not torch.isfinite(prior).all():
                raise RuntimeError("Non-finite latent prior")
        avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
        avg["epoch"] = float(epoch)
        history.append(avg)
    return history


def train_with_early_stopping(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    max_epochs: int,
    patience: int,
    lr: float,
    lambda_rt: float,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Train; early-stop on val choice_nll. Returns best metrics + history."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state = None
    best_metrics: dict[str, Any] = {}
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        totals = {"loss": 0.0, "choice_nll": 0.0, "rt_nll": 0.0}
        n_batches = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, stats = joint_loss(
                outputs, batch["choice_right"], batch["log_rt"], lambda_rt=lambda_rt
            )
            loss.backward()
            opt.step()
            for k, v in stats.items():
                totals[k] += v
            n_batches += 1
        train_avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
        val_avg = evaluate(model, val_loader, lambda_rt=lambda_rt, device=device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_avg["loss"],
            "train_choice_nll": train_avg["choice_nll"],
            "train_rt_nll": train_avg["rt_nll"],
            "val_loss": val_avg["loss"],
            "val_choice_nll": val_avg["choice_nll"],
            "val_rt_nll": val_avg["rt_nll"],
            "val_choice_acc": val_avg["choice_acc"],
        }
        history.append(row)
        print(
            f"  epoch {epoch}: train_choice_nll={row['train_choice_nll']:.4f} "
            f"val_choice_nll={row['val_choice_nll']:.4f} val_acc={row['val_choice_acc']:.3f}"
        )
        if val_avg["choice_nll"] < best_val - 1e-5:
            best_val = val_avg["choice_nll"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "best_epoch": epoch,
                "val_choice_nll": val_avg["choice_nll"],
                "val_rt_nll": val_avg["rt_nll"],
                "val_loss": val_avg["loss"],
                "val_choice_acc": val_avg["choice_acc"],
                "train_choice_nll": train_avg["choice_nll"],
                "train_rt_nll": train_avg["rt_nll"],
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics, history


def smoke_train(
    model_name: str,
    repo_root: Path,
    *,
    condition: str = "history_only",
    max_trials: int = 256,
    epochs: int = 3,
    batch_size: int = 32,
    hidden_size: int = 64,
    lr: float = 1e-3,
    lambda_rt: float = 0.2,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    device = torch.device("cpu")
    model_name = normalize_model_name(model_name)
    loader, input_size = make_loader(
        model_name, condition, repo_root, max_trials=max_trials, batch_size=batch_size
    )
    model = build_model(model_name, input_size=input_size, hidden_size=hidden_size)
    history = train_epochs(
        model, loader, epochs=epochs, lr=lr, lambda_rt=lambda_rt, device=device
    )

    model.eval()
    batch = _move_batch(next(iter(loader)), device)
    with torch.no_grad():
        outputs = model(batch)
        probs = outputs.choice_probs()
        prior = extract_latent_prior(outputs)

    out_dir = out_dir or (repo_root / "artifacts" / "smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{model_name}_{condition}.pt"
    torch.save(
        {
            "model_name": model_name,
            "condition": condition,
            "state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_size": hidden_size,
            "lambda_rt": lambda_rt,
            "history": history,
        },
        ckpt_path,
    )
    return {
        "checkpoint": str(ckpt_path),
        "history": history,
        "choice_prob_min": float(probs.min()),
        "choice_prob_max": float(probs.max()),
        "prior_mean": float(prior.mean()),
        "final_loss": history[-1]["loss"] if history else None,
    }


def full_train_run(
    model_name: str,
    condition: str,
    repo_root: Path,
    *,
    train_eids: list[str],
    val_eids: list[str],
    hidden_size: int = 64,
    lr: float = 1e-3,
    lambda_rt: float = 0.2,
    batch_size: int = 64,
    max_epochs: int = 12,
    patience: int = 3,
    run_name: str = "default",
) -> dict[str, Any]:
    """Train one model×condition on train eids; early-stop on val choice NLL."""
    device = torch.device("cpu")
    model_name = normalize_model_name(model_name)
    train_loader, input_size = make_loader(
        model_name,
        condition,
        repo_root,
        max_trials=None,
        batch_size=batch_size,
        eid_allowlist=train_eids,
        shuffle=(model_name != "bayes"),
    )
    val_loader, _ = make_loader(
        model_name,
        condition,
        repo_root,
        max_trials=None,
        batch_size=batch_size,
        eid_allowlist=val_eids,
        shuffle=False,
    )
    # Assert no leakage of val into train set sizes
    assert len(set(train_eids) & set(val_eids)) == 0

    model = build_model(model_name, input_size=input_size, hidden_size=hidden_size)
    print(
        f"Training {model_name}/{condition} ({run_name}): "
        f"train_batches~{len(train_loader)} val_batches~{len(val_loader)}"
    )
    best_metrics, history = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        lambda_rt=lambda_rt,
        device=device,
    )

    out_dir = repo_root / "artifacts" / "models" / model_name / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{run_name}.pt"
    meta_path = out_dir / f"{run_name}_metrics.json"
    payload = {
        "model_name": model_name,
        "condition": condition,
        "run_name": run_name,
        "state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "lr": lr,
        "lambda_rt": lambda_rt,
        "train_eids": train_eids,
        "val_eids": val_eids,
        "best_metrics": best_metrics,
        "history": history,
    }
    torch.save(payload, ckpt_path)
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "model_name": model_name,
                "condition": condition,
                "run_name": run_name,
                "best_metrics": best_metrics,
                "n_epochs_ran": len(history),
                "hyperparams": {
                    "hidden_size": hidden_size,
                    "lr": lr,
                    "lambda_rt": lambda_rt,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "patience": patience,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "checkpoint": str(ckpt_path),
        "metrics_path": str(meta_path),
        "best_metrics": best_metrics,
    }
