"""Training loops for v2 models.

Full-session exposure with truncated BPTT / PC chunks (Kyan-style), keeping
empirical phase ticks from configs/synthetic_v2.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.models_v2.bayes import ExplicitBayes
from src.models_v2.pc import PredictiveCodingTrainer
from src.models_v2.rnn_cells import Adam, GRURNN, TanhRNN
from src.synthetic.generate import build_training_tensors, generate_sessions


def _make_model(model_id: str, hidden_size: int, rng: np.random.Generator) -> Any:
    if model_id in ("tanh_bptt", "tanh_pc"):
        return TanhRNN(hidden_size=hidden_size, rng=rng)
    if model_id == "gru":
        return GRURNN(hidden_size=hidden_size, rng=rng)
    if model_id == "bayes":
        return ExplicitBayes(rng=rng)
    raise ValueError(model_id)


def exposure_summary(cfg: dict, epochs: int | None = None) -> dict:
    """Trial-exposures = epochs × sessions_per_epoch × trials_per_session."""
    train_cfg = cfg["train"]
    ep = int(epochs if epochs is not None else train_cfg["epochs"])
    n_sess = int(train_cfg["sessions_per_epoch"])
    n_trials = int(cfg["trials_per_session_default"])
    total = ep * n_sess * n_trials
    kyan_ref = 60 * 24 * 240  # BPTT defaults in kyan_Standard RNN
    return {
        "epochs": ep,
        "sessions_per_epoch": n_sess,
        "trials_per_session": n_trials,
        "bptt_trials": int(train_cfg["bptt_trials"]),
        "trial_exposures": total,
        "kyan_bptt_trial_exposures_ref": kyan_ref,
        "ratio_vs_kyan": round(total / kyan_ref, 2),
        "phase_ticks_unchanged": True,
    }


def train_model(
    model_id: str,
    cfg: dict,
    *,
    epochs: int | None = None,
    verbose: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    train_cfg = cfg["train"]
    n_sess = int(train_cfg["sessions_per_epoch"])
    bptt_trials = int(train_cfg["bptt_trials"])
    hidden = int(cfg["hidden_size"])
    seed = int(train_cfg["seed"])
    rng = np.random.default_rng(seed)
    model = _make_model(model_id, hidden, rng)

    # PC: shorter sessions (Kyan-scale). Long empirical sessions failed to learn priors.
    if model_id == "tanh_pc":
        epochs = int(
            epochs
            if epochs is not None
            else train_cfg.get("pc_epochs", train_cfg["epochs"])
        )
        n_trials_full = int(
            train_cfg.get("pc_trials_per_session", 240)
        )
        lr = float(train_cfg["pc_synaptic_learning_rate"])
        pc = PredictiveCodingTrainer(model)
        opt = Adam(model.parameters(), lr)
        infer_steps = int(train_cfg["pc_inference_steps"])
    else:
        epochs = int(epochs if epochs is not None else train_cfg["epochs"])
        n_trials_full = int(cfg["trials_per_session_default"])
        lr = float(train_cfg["learning_rate"])
        opt = Adam(model.parameters(), lr)
        pc = None
        infer_steps = 0

    # Exposure accounting uses the actual session length for this model
    cfg_exp = dict(cfg)
    cfg_exp["trials_per_session_default"] = n_trials_full
    if model_id == "tanh_pc":
        cfg_exp = dict(cfg_exp)
        cfg_exp["train"] = dict(train_cfg)
        cfg_exp["train"]["epochs"] = epochs
    exposure = exposure_summary(cfg_exp, epochs=epochs)

    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    clip = float(train_cfg.get("gradient_clip_norm", 1.0))
    history: List[Dict[str, float]] = []
    n_steps = None

    if verbose:
        print(json.dumps({"exposure": exposure}))

    for epoch in range(1, epochs + 1):
        # Full empirical-length sessions each epoch (new samples)
        batch = generate_sessions(n_sess, n_trials_full, cfg, rng)
        x, targets = build_training_tensors(batch, cfg, rng)
        n_steps = batch.phase.n_steps
        chunk_steps = bptt_trials * n_steps

        if model_id == "bayes":
            state = model.zero_state(n_sess)
        else:
            state = model.zero_state(n_sess)

        chunk_losses: List[float] = []
        gnorms: List[float] = []
        energy_reductions: List[float] = []

        # Walk the entire session in truncated-BPTT / PC chunks (Kyan-style)
        for start in range(0, x.shape[1], chunk_steps):
            stop = min(start + chunk_steps, x.shape[1])
            chunk_x = x[:, start:stop]
            chunk_y = targets[:, start:stop]
            if not np.any(chunk_y >= 0):
                # advance state through ticks with no loss targets
                if model_id == "bayes":
                    for t in range(chunk_x.shape[1]):
                        state = model.step_prior(chunk_x[:, t], state)
                else:
                    for t in range(chunk_x.shape[1]):
                        state = model.step(chunk_x[:, t], state)
                continue

            if model_id == "tanh_pc":
                assert pc is not None
                forward_values = pc.forward_values(chunk_x, state)
                forward_final = forward_values[:, -1].copy()
                inferred, energy_trace = pc.infer_values(
                    chunk_x,
                    chunk_y,
                    state,
                    inference_steps=infer_steps,
                    inference_learning_rate=float(
                        train_cfg.get("pc_inference_learning_rate", 0.15)
                    ),
                    output_precision=1.0,
                    value_clip=2.0,
                    initial_values=forward_values,
                )
                grads, energy = pc.local_synaptic_gradients(
                    chunk_x,
                    chunk_y,
                    state,
                    inferred,
                    output_precision=1.0,
                    weight_decay=weight_decay,
                )
                gnorm = opt.update(grads, clip)
                # Carry pre-update forward state only (Kyan: no label leak)
                state = forward_final
                # Log energy per timestep like Kyan
                chunk_losses.append(energy / max(chunk_x.shape[1], 1))
                energy_reductions.append(
                    float(energy_trace[0] - energy_trace[-1]) / max(chunk_x.shape[1], 1)
                )
            else:
                loss, grads, state = model.loss_and_gradients(
                    chunk_x, chunk_y, state, weight_decay
                )
                gnorm = opt.update(grads, clip)
                chunk_losses.append(loss)
            gnorms.append(gnorm)

        epoch_loss = float(np.mean(chunk_losses)) if chunk_losses else float("nan")
        hist: Dict[str, float] = {
            "epoch": float(epoch),
            "loss": epoch_loss,
            "grad_norm": float(np.mean(gnorms)) if gnorms else float("nan"),
            "n_chunks": float(len(chunk_losses)),
        }
        if energy_reductions:
            hist["energy_reduction"] = float(np.mean(energy_reductions))
        history.append(hist)
        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(json.dumps(hist))

    meta = {
        "model_id": model_id,
        "epochs": epochs,
        "history": history,
        "n_steps": n_steps,
        "hidden_size": hidden if model_id != "bayes" else None,
        "seed": seed,
        "exposure": exposure,
    }
    return model, meta


def save_checkpoint(model: Any, meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.npz"
    model.save(path, metadata={k: v for k, v in meta.items() if k != "history"})
    (out_dir / "train_history.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return path
