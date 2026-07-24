"""Training loops for v2 models.

Full-session exposure with truncated BPTT / corrected PC chunks, keeping
empirical phase ticks from configs/synthetic_v2.yaml.

Active models: tanh_bptt, tanh_pc, gru, gru_pc.
Bayes remains importable as legacy but is not trained by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.models_v2.pc import make_pc_trainer, validate_pc_inference_steps
from src.models_v2.rnn_cells import Adam, GRURNN, TanhRNN
from src.synthetic.channels import PhaseTicks
from src.synthetic.generate import build_training_tensors, generate_sessions

ACTIVE_MODELS = ("tanh_bptt", "tanh_pc", "gru", "gru_pc")
PC_MODELS = ("tanh_pc", "gru_pc")


def _make_model(model_id: str, hidden_size: int, rng: np.random.Generator) -> Any:
    if model_id in ("tanh_bptt", "tanh_pc"):
        return TanhRNN(hidden_size=hidden_size, rng=rng)
    if model_id in ("gru", "gru_pc"):
        return GRURNN(hidden_size=hidden_size, rng=rng)
    if model_id == "bayes":
        from src.models_v2.bayes import ExplicitBayes

        return ExplicitBayes(rng=rng)
    raise ValueError(model_id)


def exposure_summary(cfg: dict, epochs: int | None = None) -> dict:
    """Trial-exposures = epochs × sessions_per_epoch × trials_per_session."""
    train_cfg = cfg["train"]
    ep = int(epochs if epochs is not None else train_cfg["epochs"])
    n_sess = int(train_cfg["sessions_per_epoch"])
    n_trials = int(cfg["trials_per_session_default"])
    total = ep * n_sess * n_trials
    kyan_ref = 60 * 24 * 240
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
    phase = PhaseTicks.from_config(cfg)

    is_pc = model_id in PC_MODELS
    epochs = int(epochs if epochs is not None else train_cfg["epochs"])
    n_trials_full = int(cfg["trials_per_session_default"])

    if is_pc:
        lr = float(train_cfg["pc_synaptic_learning_rate"])
        infer_steps = int(train_cfg["pc_inference_steps"])
        validate_pc_inference_steps(infer_steps, phase)
        output_precision = float(train_cfg.get("pc_output_precision", 0.025))
        infer_lr = float(train_cfg.get("pc_inference_learning_rate", 0.15))
        infer_mom = float(train_cfg.get("pc_inference_momentum", 0.0))
        value_clip = float(train_cfg.get("pc_value_clip", 2.0))
        normalize_nudge = bool(train_cfg.get("pc_normalize_updates_by_nudge", True))
        pc = make_pc_trainer(model)
        opt = Adam(model.parameters(), lr)
    else:
        lr = float(train_cfg["learning_rate"])
        opt = Adam(model.parameters(), lr)
        pc = None
        infer_steps = 0
        output_precision = 1.0
        infer_lr = 0.0
        infer_mom = 0.0
        value_clip = 2.0
        normalize_nudge = True

    exposure = exposure_summary(cfg, epochs=epochs)
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    clip = float(train_cfg.get("gradient_clip_norm", 1.0))
    history: List[Dict[str, float]] = []
    n_steps = None

    if verbose:
        print(
            json.dumps(
                {
                    "model_id": model_id,
                    "exposure": exposure,
                    "pc": (
                        {
                            "inference_steps": infer_steps,
                            "output_precision": output_precision,
                            "normalize_updates_by_nudge": normalize_nudge,
                        }
                        if is_pc
                        else None
                    ),
                }
            )
        )

    for epoch in range(1, epochs + 1):
        batch = generate_sessions(n_sess, n_trials_full, cfg, rng)
        x, targets = build_training_tensors(batch, cfg, rng)
        n_steps = batch.phase.n_steps
        chunk_steps = bptt_trials * n_steps
        state = model.zero_state(n_sess)

        chunk_losses: List[float] = []
        gnorms: List[float] = []
        energy_reductions: List[float] = []
        forward_ces: List[float] = []

        for start in range(0, x.shape[1], chunk_steps):
            stop = min(start + chunk_steps, x.shape[1])
            chunk_x = x[:, start:stop]
            chunk_y = targets[:, start:stop]
            if not np.any(chunk_y >= 0):
                for t in range(chunk_x.shape[1]):
                    state = model.step(chunk_x[:, t], state)
                continue

            if is_pc:
                assert pc is not None
                forward_values = pc.forward_values(chunk_x, state)
                forward_final = forward_values[:, -1].copy()
                forward_ce = pc.forward_response_cross_entropy(forward_values, chunk_y)
                inferred, energy_trace = pc.infer_values(
                    chunk_x,
                    chunk_y,
                    state,
                    inference_steps=infer_steps,
                    inference_learning_rate=infer_lr,
                    output_precision=output_precision,
                    value_clip=value_clip,
                    inference_momentum=infer_mom,
                    initial_values=forward_values,
                )
                grads, energy = pc.local_synaptic_gradients(
                    chunk_x,
                    chunk_y,
                    state,
                    inferred,
                    output_precision=output_precision,
                    weight_decay=weight_decay,
                    normalize_by_nudge=normalize_nudge,
                )
                gnorm = opt.update(grads, clip)
                # Crucial: carry pre-update forward state, never inferred state.
                state = forward_final
                responses = max(float(np.count_nonzero(chunk_y >= 0)) / chunk_x.shape[0], 1.0)
                chunk_losses.append(energy / responses)
                energy_reductions.append(
                    float(energy_trace[0] - energy_trace[-1]) / responses
                )
                forward_ces.append(forward_ce)
            else:
                loss, grads, state = model.loss_and_gradients(
                    chunk_x, chunk_y, state, weight_decay
                )
                gnorm = opt.update(grads, clip)
                chunk_losses.append(loss)
            gnorms.append(gnorm)

        hist: Dict[str, float] = {
            "epoch": float(epoch),
            "loss": float(np.mean(chunk_losses)) if chunk_losses else float("nan"),
            "grad_norm": float(np.mean(gnorms)) if gnorms else float("nan"),
            "n_chunks": float(len(chunk_losses)),
        }
        if energy_reductions:
            hist["energy_reduction"] = float(np.mean(energy_reductions))
        if forward_ces:
            hist["forward_response_cross_entropy"] = float(np.mean(forward_ces))
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
        "pc_config": (
            {
                "inference_steps": infer_steps,
                "output_precision": output_precision,
                "inference_learning_rate": infer_lr,
                "normalize_updates_by_nudge": normalize_nudge,
                "trials_per_session": n_trials_full,
            }
            if is_pc
            else None
        ),
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
