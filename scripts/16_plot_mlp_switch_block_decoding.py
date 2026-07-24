#!/usr/bin/env python3
"""16 — Three-panel MLP switch-centered block decoding.

Panel A1 (history-only synth): MLP on each model's zero-current-evidence latents
(capacity / latent readability).

Panel A2 (real shared cohort): MLP on mouse subjective prior hat{p}_t and each
model's zero-evidence belief q_t (Q2 three-way scalar probe).

Panel A3 (real shared cohort): MLP on CV Ridge OOF neural prior readouts by ROI.

Window: trials relative to isolated genuine 0.2 <-> 0.8 switches, default -30..+30.

Outputs
-------
reports/v2/switch_block_decoding/mlp_switch_block_decode_metrics.json
reports/v2/figures/switch_block_decoding/mlp_rnn_vs_pc_switch_decoding.png
mlp_rnn_vs_pc_switch_decoding.png  (project-root copy)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.mouse_prior import apply_mouse_prior, fit_mouse_prior
from src.models_v2.block_decode import (
    DecoderSettings,
    extract_zero_evidence_latents,
    fit_decoder,
    make_decoding_dataset,
    make_session_split,
)
from src.models_v2.rollout import load_model
from src.neural.prior_readout import fit_prior_readout
from src.neural.regions import NEURAL_REGIONS
from src.plot.v2_style import MODEL_COLORS, PASTEL, SAVE_DPI
from src.synthetic.channels import PhaseTicks
from src.synthetic.generate import build_training_tensors, generate_sessions
from src.synthetic.schema import load_synthetic_config

MODEL_IDS = ("tanh_bptt", "tanh_pc", "gru", "gru_pc")
MODEL_LABELS = {
    "tanh_bptt": "tanh BPTT",
    "tanh_pc": "tanh PC",
    "gru": "GRU",
    "gru_pc": "GRU PC",
}
REGION_LABELS = {
    "MOs": "MOs",
    "vlOFC_orbvl": "vlOFC",
    "ACAd": "ACAd",
    "MOp": "MOp",
}
REGION_COLORS = {
    "MOs": PASTEL["blue"],
    "vlOFC_orbvl": PASTEL["orange"],
    "ACAd": PASTEL["lavender"],
    "MOp": PASTEL["teal"],
}
# Real-cohort scalar belief series (mouse + models)
BELIEF_SERIES = ("mouse",) + MODEL_IDS
BELIEF_LABELS = {
    "mouse": "mouse prior",
    **MODEL_LABELS,
}
BELIEF_COLORS = {
    "mouse": PASTEL["ink"],
    **MODEL_COLORS,
}
SHARED = ROOT / "data" / "manifests" / "shared_behavior_neural_eids.json"
REAL_HISTORY_ROLLOUT = (
    ROOT / "artifacts" / "v2" / "real" / "regimes" / "history_only"
)


def _canonical_checkpoint_path(cfg: dict, model_id: str) -> Path:
    return ROOT / cfg["paths"]["artifacts"] / "models" / model_id / "model.npz"


def _checkpoint_path(cfg: dict, model_id: str, model_seed: int) -> Path:
    seeded = (
        ROOT
        / cfg["paths"]["artifacts"]
        / "model_seed_replicates"
        / model_id
        / f"seed_{int(model_seed)}"
        / "model.npz"
    )
    if seeded.exists():
        return seeded
    if int(model_seed) == int(cfg["train"]["seed"]):
        return _canonical_checkpoint_path(cfg, model_id)
    return seeded


def _available_model_seeds(cfg: dict, requested: list[int]) -> list[int]:
    available = []
    for seed in requested:
        if all(_checkpoint_path(cfg, mid, seed).exists() for mid in MODEL_IDS):
            available.append(int(seed))
    if available:
        return available
    train_seed = int(cfg["train"]["seed"])
    if all(_canonical_checkpoint_path(cfg, mid).exists() for mid in MODEL_IDS):
        return [train_seed]
    missing = [
        str(_checkpoint_path(cfg, mid, seed))
        for mid in MODEL_IDS
        for seed in requested
        if not _checkpoint_path(cfg, mid, seed).exists()
    ]
    raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing[:8]))


def _build_matched_eval_inputs(cfg: dict, n_sessions: int, n_trials: int):
    batch_seed = int(cfg["eval"]["seed"])
    input_seed = batch_seed + 1
    batch = generate_sessions(
        n_sessions,
        n_trials,
        cfg,
        np.random.default_rng(batch_seed),
    )
    flat_inputs, _ = build_training_tensors(
        batch,
        cfg,
        np.random.default_rng(input_seed),
    )
    inputs = flat_inputs.reshape(
        n_sessions,
        n_trials,
        batch.phase.n_steps,
        flat_inputs.shape[-1],
    )
    return batch, inputs, batch_seed, input_seed


def _settings(quick: bool) -> DecoderSettings:
    if not quick:
        return DecoderSettings()
    return DecoderSettings(
        logistic_epochs=20,
        mlp_epochs=15,
        batch_size=512,
        patience=5,
        max_train_samples=2_000,
        max_validation_samples=1_000,
        mlp_hidden_size=16,
    )


def _expand_test_probabilities(
    probabilities: np.ndarray,
    p_right: np.ndarray,
    test_sessions: np.ndarray,
) -> np.ndarray:
    expanded = np.full(p_right.shape, np.nan, dtype=np.float64)
    cursor = 0
    for session in test_sessions:
        biased = np.isclose(p_right[session], 0.2) | np.isclose(p_right[session], 0.8)
        count = int(np.count_nonzero(biased))
        expanded[session, biased] = probabilities[cursor : cursor + count]
        cursor += count
    if cursor != len(probabilities):
        raise RuntimeError("flattened decoder predictions did not map back to test trials")
    return expanded


def _eligible_switches(p_right: np.ndarray, sessions: np.ndarray, before: int, after: int):
    for session in sessions:
        prior = np.asarray(p_right[int(session)], dtype=np.float64)
        finite = np.isfinite(prior)
        if not finite.any():
            continue
        # Trim trailing pad / invalid trials so NaN diffs cannot invent switches.
        last = int(np.flatnonzero(finite)[-1])
        prior = prior[: last + 1]
        changed = np.flatnonzero(np.diff(prior) != 0.0) + 1
        for switch in changed:
            previous = float(prior[switch - 1])
            current = float(prior[switch])
            if not (np.isfinite(previous) and np.isfinite(current)):
                continue
            genuine = (
                (np.isclose(previous, 0.2) and np.isclose(current, 0.8))
                or (np.isclose(previous, 0.8) and np.isclose(current, 0.2))
            )
            if not genuine:
                continue
            start = switch - before
            stop = switch + after + 1
            if start < 0 or stop > prior.shape[0]:
                continue
            window = prior[start:stop]
            if not np.all(np.isfinite(window)):
                continue
            if not np.all(np.isclose(window, 0.2) | np.isclose(window, 0.8)):
                continue
            if not np.allclose(prior[start:switch], previous):
                continue
            if not np.allclose(prior[switch:stop], current):
                continue
            direction = "low_to_high" if current > previous else "high_to_low"
            yield int(session), int(switch), direction


def _balanced_accuracy_from_binary(labels: np.ndarray, predictions: np.ndarray) -> float:
    values = []
    for label in (0, 1):
        selected = labels == label
        if np.any(selected):
            values.append(float(np.mean(predictions[selected] == label)))
    if not values:
        return float("nan")
    return float(np.mean(values))


def switch_centered_decoder_accuracy(
    decoder_probability_right_block: np.ndarray,
    p_right: np.ndarray,
    sessions: np.ndarray,
    *,
    before: int,
    after: int,
) -> dict:
    windows_probability = []
    windows_label = []
    directions = []
    for session, switch, direction in _eligible_switches(
        p_right, sessions, before, after
    ):
        start = switch - before
        stop = switch + after + 1
        probability = decoder_probability_right_block[session, start:stop]
        if not np.all(np.isfinite(probability)):
            continue
        windows_probability.append(probability)
        windows_label.append((p_right[session, start:stop] > 0.5).astype(int))
        directions.append(direction)

    offsets = np.arange(-before, after + 1)
    if not windows_probability:
        return {
            "offsets": offsets.tolist(),
            "balanced_accuracy": np.full_like(offsets, np.nan, dtype=np.float64).tolist(),
            "n_switches": 0,
            "n_low_to_high": 0,
            "n_high_to_low": 0,
        }

    probability = np.stack(windows_probability)
    labels = np.stack(windows_label)
    predictions = (probability >= 0.5).astype(int)
    accuracy = np.asarray(
        [
            _balanced_accuracy_from_binary(labels[:, index], predictions[:, index])
            for index in range(len(offsets))
        ],
        dtype=np.float64,
    )
    return {
        "offsets": offsets.tolist(),
        "balanced_accuracy": accuracy.tolist(),
        "n_switches": len(directions),
        "n_low_to_high": int(np.count_nonzero(np.asarray(directions) == "low_to_high")),
        "n_high_to_low": int(np.count_nonzero(np.asarray(directions) == "high_to_low")),
    }


def switch_centered_decoder_accuracy_1d(
    decoder_probability_right_block: np.ndarray,
    p_right: np.ndarray,
    *,
    before: int,
    after: int,
) -> dict:
    """Same as ``switch_centered_decoder_accuracy`` for a single session vector."""

    probability = np.asarray(decoder_probability_right_block, dtype=np.float64)
    prior = np.asarray(p_right, dtype=np.float64)
    # Fake a one-session batch so we can reuse eligibility rules without padding.
    return switch_centered_decoder_accuracy(
        probability[None, :],
        prior[None, :],
        np.asarray([0], dtype=int),
        before=before,
        after=after,
    )


def _aggregate_curves(per_unit: dict, value_key: str) -> dict:
    keys = sorted(per_unit, key=lambda x: str(x))
    curves = np.stack(
        [np.asarray(per_unit[k][value_key], dtype=np.float64) for k in keys],
        axis=0,
    )
    sample_sd = (
        curves.std(axis=0, ddof=1)
        if len(keys) > 1
        else np.zeros(curves.shape[1], dtype=np.float64)
    )
    first = per_unit[keys[0]]
    return {
        "offsets": first["offsets"],
        "mean": curves.mean(axis=0).tolist(),
        "sd": sample_sd.tolist(),
        "n_units": len(keys),
        "unit_ids": [str(k) for k in keys],
    }


def _plot_mean_sd(axis, offsets, mean, sd, *, label, color):
    axis.plot(offsets, mean, color=color, linewidth=2, label=label)
    axis.fill_between(offsets, mean - sd, mean + sd, color=color, alpha=0.15)


def _decorate_decoder_axis(axis, title: str) -> None:
    axis.axhline(50.0, color="0.4", linestyle="--", linewidth=1, label="chance")
    axis.axvline(0, color="black", linestyle=":", linewidth=1)
    axis.set(
        title=title,
        xlabel="Trials relative to block switch",
        ylabel="Balanced block-decoding success (%)",
        ylim=(0.0, 102.0),
    )
    axis.grid(alpha=0.2)


def _load_shared_eids() -> list[str]:
    payload = json.loads(SHARED.read_text(encoding="utf-8"))
    return [str(e) for e in payload["eids"]]


def _load_session_trials(eid: str) -> pd.DataFrame:
    paths = [
        ROOT / "data" / "processed" / "neural" / eid / "trials.parquet",
        ROOT / "data" / "processed" / "trials" / f"{eid}.parquet",
    ]
    for path in paths:
        if path.exists():
            trials = pd.read_parquet(path)
            break
    else:
        raise FileNotFoundError(f"No trials parquet for {eid}")
    if "mouse_prior_hat" not in trials.columns:
        params, _ = fit_mouse_prior(trials, train_eids=[eid])
        trials = apply_mouse_prior(trials, params)
    return trials


def _neural_belief_and_prior(eid: str, region: str) -> tuple[np.ndarray, np.ndarray] | None:
    counts_path = (
        ROOT / "data" / "processed" / "neural" / eid / f"{region}_counts.npz"
    )
    if not counts_path.exists():
        return None
    trials = _load_session_trials(eid)
    blob = np.load(counts_path, allow_pickle=True)
    counts = np.asarray(blob["counts"], dtype=float)
    mouse_prior = trials["mouse_prior_hat"].to_numpy(dtype=float)
    p_left = trials["probabilityLeft"].to_numpy(dtype=float)
    p_right = 1.0 - p_left
    n = min(counts.shape[0], len(mouse_prior), len(p_right))
    counts = counts[:n]
    mouse_prior = mouse_prior[:n]
    p_right = p_right[:n]
    readout = fit_prior_readout(counts, mouse_prior)
    belief = np.full(n, np.nan, dtype=np.float64)
    mask = readout["mask"]
    # mask indexes rows of finite mouse_prior/counts before truncation to n
    mask = np.asarray(mask, dtype=bool)[:n]
    oof = np.asarray(readout["oof_pred"], dtype=float)
    if oof.shape[0] == int(mask.sum()):
        belief[mask] = oof
    else:
        # fit_prior_readout returns oof aligned to masked length
        belief[mask] = oof[: int(mask.sum())]
    return belief, p_right


def _session_as_batch_arrays(
    beliefs: list[np.ndarray],
    priors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad variable-length sessions into (S, T, 1) features and (S, T) priors."""

    n_sessions = len(beliefs)
    max_t = max(len(b) for b in beliefs)
    features = np.full((n_sessions, max_t, 1), np.nan, dtype=np.float64)
    p_right = np.full((n_sessions, max_t), np.nan, dtype=np.float64)
    valid = np.zeros((n_sessions, max_t), dtype=bool)
    for i, (belief, prior) in enumerate(zip(beliefs, priors)):
        t = len(belief)
        features[i, :t, 0] = belief
        p_right[i, :t] = prior
        valid[i, :t] = np.isfinite(belief) & np.isfinite(prior)
    return features, p_right, valid


def _make_decoding_dataset_masked(
    features: np.ndarray,
    p_right: np.ndarray,
    valid: np.ndarray,
    sessions: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = []
    labels = []
    session_of = []
    trial_of = []
    for session in sessions:
        prior = p_right[session]
        ok = valid[session] & (np.isclose(prior, 0.2) | np.isclose(prior, 0.8))
        if not np.any(ok):
            continue
        x = features[session][ok]
        y = (prior[ok] > 0.5).astype(np.int64)
        trials = np.flatnonzero(ok)
        finite = np.isfinite(x).all(axis=1)
        x = x[finite]
        y = y[finite]
        trials = trials[finite]
        if len(y) == 0:
            continue
        rows.append(x.reshape(len(y), -1))
        labels.append(y)
        session_of.append(np.full(len(y), int(session), dtype=np.int64))
        trial_of.append(trials.astype(np.int64))
    if not rows:
        return {
            "x": np.zeros((0, features.shape[-1]), dtype=np.float64),
            "y": np.zeros(0, dtype=np.int64),
            "session": np.zeros(0, dtype=np.int64),
            "trial": np.zeros(0, dtype=np.int64),
        }
    return {
        "x": np.concatenate(rows, axis=0),
        "y": np.concatenate(labels, axis=0),
        "session": np.concatenate(session_of, axis=0),
        "trial": np.concatenate(trial_of, axis=0),
    }


def run_model_panel(
    cfg: dict,
    *,
    model_seeds: list[int],
    decoder_seeds: list[int],
    split_seed: int,
    before: int,
    after: int,
    quick: bool,
    refresh_cache: bool,
) -> dict:
    phase = PhaseTicks.from_config(cfg)
    n_sessions = 12 if quick else int(cfg["eval"]["synth_sessions"])
    n_trials = 300 if quick else int(cfg["eval"]["synth_trials"])
    settings = _settings(quick)
    split = make_session_split(n_sessions, split_seed)
    batch, matched_inputs, batch_seed, input_seed = _build_matched_eval_inputs(
        cfg, n_sessions, n_trials
    )

    report_directory = ROOT / cfg["paths"]["reports"] / "switch_block_decoding"
    cache_directory = report_directory / "checkpoint_cache_mlp4"
    report_directory.mkdir(parents=True, exist_ok=True)
    cache_directory.mkdir(parents=True, exist_ok=True)

    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    decoder_source_sha = hashlib.sha256(
        (ROOT / "src/models_v2/block_decode.py").read_bytes()
    ).hexdigest()
    config_sha = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()

    per_model_seed: dict = {mid: {} for mid in MODEL_IDS}

    for model_id in MODEL_IDS:
        for model_seed in model_seeds:
            checkpoint = _checkpoint_path(cfg, model_id, model_seed)
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            signature_payload = {
                "cache_schema": 2,
                "panel": "models_mlp4",
                "analysis_script_sha256": script_sha,
                "decoder_source_sha256": decoder_source_sha,
                "config_sha256": config_sha,
                "checkpoint_sha256": checkpoint_sha,
                "model_id": model_id,
                "model_seed": model_seed,
                "n_sessions": n_sessions,
                "n_trials": n_trials,
                "before": before,
                "after": after,
                "split_seed": split_seed,
                "batch_seed": batch_seed,
                "input_seed": input_seed,
                "decoder_seeds": [int(s) for s in decoder_seeds],
                "decoder_settings": settings.__dict__,
            }
            signature = hashlib.sha256(
                json.dumps(signature_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
            cache_path = cache_directory / f"{model_id}_seed_{model_seed}.json"
            if cache_path.exists() and not refresh_cache and not quick:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_signature") == signature:
                    print(
                        f"Loading cache: {model_id} seed {model_seed}",
                        flush=True,
                    )
                    per_model_seed[model_id][str(model_seed)] = cached["seed_result"]
                    continue

            print(f"Decoding latents: {model_id} seed {model_seed}", flush=True)
            model = load_model(model_id, checkpoint)
            latents = extract_zero_evidence_latents(model, matched_inputs, phase)
            total_features = latents.reshape(
                n_sessions,
                n_trials,
                phase.n_steps * model.hidden_size,
            )
            datasets = {
                name: make_decoding_dataset(
                    total_features,
                    batch.p_right,
                    batch.block_id,
                    sessions,
                )
                for name, sessions in split.items()
            }
            decoder_runs = [
                fit_decoder(
                    "mlp",
                    datasets["train"],
                    datasets["validation"],
                    datasets["test"],
                    seed=int(decoder_seed),
                    settings=settings,
                )
                for decoder_seed in decoder_seeds
            ]
            mean_probability = np.mean(
                np.stack([run["probabilities"] for run in decoder_runs], axis=0),
                axis=0,
            )
            expanded = _expand_test_probabilities(
                mean_probability, batch.p_right, split["test"]
            )
            curve = switch_centered_decoder_accuracy(
                expanded,
                batch.p_right,
                split["test"],
                before=before,
                after=after,
            )
            curve["best_epoch_by_decoder_seed"] = [
                int(run["best_epoch"]) for run in decoder_runs
            ]
            seed_result = {
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "mlp": curve,
            }
            per_model_seed[model_id][str(model_seed)] = seed_result
            if not quick:
                cache_path.write_text(
                    json.dumps(
                        {
                            "cache_signature": signature,
                            "signature_payload": signature_payload,
                            "seed_result": seed_result,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            del latents, total_features, datasets
            gc.collect()

    aggregate = {}
    for model_id in MODEL_IDS:
        decoder_per_seed = {
            seed: result["mlp"] for seed, result in per_model_seed[model_id].items()
        }
        aggregate[model_id] = _aggregate_curves(decoder_per_seed, "balanced_accuracy")

    return {
        "analysis": {
            "panel": "models",
            "regime": "history_only",
            "feature": (
                "all within-trial zero-current-evidence latent states concatenated"
            ),
            "before": before,
            "after": after,
            "task_model_seeds": model_seeds,
            "decoder_initialization_seeds": [int(s) for s in decoder_seeds],
            "uncertainty": "sample SD across task-model seeds",
            "session_split": {k: v.tolist() for k, v in split.items()},
            "n_sessions": n_sessions,
            "trials_per_session": n_trials,
            "batch_seed": batch_seed,
            "matched_input_seed": input_seed,
            "split_seed": split_seed,
        },
        "decoder_settings": settings.__dict__,
        "per_model_seed": per_model_seed,
        "aggregate": aggregate,
    }


def _loso_scalar_curves(
    beliefs: list[np.ndarray],
    priors: list[np.ndarray],
    unit_ids: list[str],
    *,
    decoder_seeds: list[int],
    before: int,
    after: int,
    settings: DecoderSettings,
) -> dict[str, dict]:
    """Leave-one-session-out MLP decode curves keyed by unit_id."""

    if len(unit_ids) < 2:
        return {}
    features, p_right, valid = _session_as_batch_arrays(beliefs, priors)
    features_filled = np.nan_to_num(features, nan=0.0)
    session_curves: dict[str, dict] = {}
    n_sess = len(unit_ids)
    for held_out in range(n_sess):
        train_sessions = np.asarray(
            [i for i in range(n_sess) if i != held_out], dtype=int
        )
        if len(train_sessions) >= 2:
            validation_sessions = train_sessions[-1:]
            train_fit = train_sessions[:-1]
        else:
            train_fit = train_sessions
            validation_sessions = train_sessions
        test_sessions = np.asarray([held_out], dtype=int)
        datasets = {
            "train": _make_decoding_dataset_masked(
                features_filled, p_right, valid, train_fit
            ),
            "validation": _make_decoding_dataset_masked(
                features_filled, p_right, valid, validation_sessions
            ),
            "test": _make_decoding_dataset_masked(
                features_filled, p_right, valid, test_sessions
            ),
        }
        if len(datasets["train"]["y"]) < 20 or len(datasets["test"]["y"]) < 5:
            continue
        decoder_runs = [
            fit_decoder(
                "mlp",
                datasets["train"],
                datasets["validation"],
                datasets["test"],
                seed=int(decoder_seed),
                settings=settings,
            )
            for decoder_seed in decoder_seeds
        ]
        mean_probability = np.mean(
            np.stack([run["probabilities"] for run in decoder_runs], axis=0),
            axis=0,
        )
        n_trials = int(np.sum(np.isfinite(priors[held_out])))
        expanded_1d = np.full(n_trials, np.nan, dtype=np.float64)
        for prob, trial in zip(mean_probability, datasets["test"]["trial"]):
            if 0 <= int(trial) < n_trials:
                expanded_1d[int(trial)] = float(prob)
        curve = switch_centered_decoder_accuracy_1d(
            expanded_1d,
            priors[held_out][:n_trials],
            before=before,
            after=after,
        )
        if curve["n_switches"] < 1:
            continue
        session_curves[unit_ids[held_out]] = curve
    return session_curves


def _aggregate_from_session_curves(session_curves: dict[str, dict]) -> dict | None:
    if len(session_curves) >= 2:
        return _aggregate_curves(session_curves, "balanced_accuracy")
    if len(session_curves) == 1:
        only = next(iter(session_curves.values()))
        return {
            "offsets": only["offsets"],
            "mean": only["balanced_accuracy"],
            "sd": [0.0] * len(only["offsets"]),
            "n_units": 1,
            "unit_ids": list(session_curves.keys()),
        }
    return None


def _load_real_history_beliefs(
    eids: list[str],
) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
    """Load mouse prior and model zero-evidence beliefs aligned to shared eids."""

    series: dict[str, list[np.ndarray]] = {name: [] for name in BELIEF_SERIES}
    priors: list[np.ndarray] = []
    rolls = {}
    for model_id in MODEL_IDS:
        path = REAL_HISTORY_ROLLOUT / model_id / "rollout.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing real history-only rollout for {model_id}: {path}"
            )
        rolls[model_id] = np.load(path)

    n_roll = int(rolls[MODEL_IDS[0]]["belief"].shape[0])
    if n_roll != len(eids):
        raise ValueError(
            f"real rollout session count {n_roll} != shared cohort {len(eids)}"
        )

    for session_index, eid in enumerate(eids):
        valid = np.asarray(rolls[MODEL_IDS[0]]["valid"][session_index], dtype=bool)
        n = int(valid.sum())
        true_p = np.asarray(
            rolls[MODEL_IDS[0]]["true_p_right"][session_index, :n],
            dtype=np.float64,
        )
        trials = _load_session_trials(eid)
        mouse = trials["mouse_prior_hat"].to_numpy(dtype=float)[:n]
        if len(mouse) != n:
            raise ValueError(
                f"{eid}: mouse prior length {len(mouse)} != rollout valid {n}"
            )
        series["mouse"].append(mouse)
        for model_id in MODEL_IDS:
            belief = np.asarray(
                rolls[model_id]["belief"][session_index, :n],
                dtype=np.float64,
            )
            if belief.shape[0] != n:
                raise ValueError(f"{eid}/{model_id}: belief length mismatch")
            series[model_id].append(belief)
        priors.append(true_p)

    return series, priors


def run_real_belief_panel(
    *,
    decoder_seeds: list[int],
    before: int,
    after: int,
    quick: bool,
) -> dict:
    """Panel A2: mouse prior + model zero-evidence beliefs on the shared cohort."""

    settings = _settings(quick)
    eids = _load_shared_eids()
    if quick:
        eids = eids[:4]

    series, priors = _load_real_history_beliefs(eids)

    per_series: dict = {}
    aggregate: dict = {}

    for name in BELIEF_SERIES:
        beliefs = series[name]
        print(
            f"Real belief series {name}: {len(eids)} sessions",
            flush=True,
        )
        session_curves = _loso_scalar_curves(
            beliefs,
            priors,
            eids,
            decoder_seeds=decoder_seeds,
            before=before,
            after=after,
            settings=settings,
        )
        per_series[name] = {
            "eids": eids,
            "per_session": session_curves,
        }
        agg = _aggregate_from_session_curves(session_curves)
        if agg is not None:
            aggregate[name] = agg

    return {
        "analysis": {
            "panel": "real_belief",
            "feature": (
                "mouse_prior_hat and model zero_evidence_p_right / belief "
                "(history-only real rollouts)"
            ),
            "series": list(BELIEF_SERIES),
            "before": before,
            "after": after,
            "uncertainty": "sample SD across leave-one-session-out held-out sessions",
            "decoder_initialization_seeds": [int(s) for s in decoder_seeds],
            "cohort_manifest": str(SHARED.relative_to(ROOT)),
            "n_cohort_sessions": len(eids),
            "rollout_root": str(REAL_HISTORY_ROLLOUT.relative_to(ROOT)),
        },
        "decoder_settings": settings.__dict__,
        "per_series": per_series,
        "aggregate": aggregate,
    }


def run_neural_panel(
    *,
    decoder_seeds: list[int],
    before: int,
    after: int,
    quick: bool,
) -> dict:
    settings = _settings(quick)
    eids = _load_shared_eids()
    if quick:
        eids = eids[:4]

    per_region: dict = {}
    aggregate: dict = {}

    for region in NEURAL_REGIONS:
        beliefs = []
        priors = []
        used_eids = []
        for eid in eids:
            packed = _neural_belief_and_prior(eid, region)
            if packed is None:
                continue
            belief, p_right = packed
            if np.sum(np.isfinite(belief)) < 40:
                continue
            beliefs.append(belief)
            priors.append(p_right)
            used_eids.append(eid)

        print(
            f"Neural region {region}: {len(used_eids)} sessions",
            flush=True,
        )
        if len(used_eids) < 2:
            per_region[region] = {
                "eids": used_eids,
                "per_session": {},
                "note": "fewer than 2 sessions; skipped",
            }
            continue

        session_curves = _loso_scalar_curves(
            beliefs,
            priors,
            used_eids,
            decoder_seeds=decoder_seeds,
            before=before,
            after=after,
            settings=settings,
        )
        per_region[region] = {
            "eids": used_eids,
            "per_session": session_curves,
        }
        agg = _aggregate_from_session_curves(session_curves)
        if agg is not None:
            aggregate[region] = agg

    return {
        "analysis": {
            "panel": "neural",
            "feature": "CV Ridge OOF neural prior readout (belief activity)",
            "regions": list(NEURAL_REGIONS),
            "before": before,
            "after": after,
            "uncertainty": "sample SD across leave-one-session-out held-out sessions",
            "decoder_initialization_seeds": [int(s) for s in decoder_seeds],
            "cohort_manifest": str(SHARED.relative_to(ROOT)),
            "n_cohort_sessions": len(eids),
        },
        "decoder_settings": settings.__dict__,
        "per_region": per_region,
        "aggregate": aggregate,
    }


def make_three_panel_figure(
    model_results: dict,
    real_belief_results: dict,
    neural_results: dict,
    paths: list[Path],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.2), constrained_layout=True)

    ax0 = axes[0]
    for model_id in MODEL_IDS:
        curve = model_results["aggregate"][model_id]
        offsets = np.asarray(curve["offsets"])
        mean = 100.0 * np.asarray(curve["mean"])
        sd = 100.0 * np.asarray(curve["sd"])
        _plot_mean_sd(
            ax0,
            offsets,
            mean,
            sd,
            label=MODEL_LABELS[model_id],
            color=MODEL_COLORS.get(model_id, "#444444"),
        )
    _decorate_decoder_axis(
        ax0,
        "A1 · Synth: MLP on zero-evidence latents",
    )
    ax0.legend(frameon=False, fontsize=7.5, loc="lower right")

    ax1 = axes[1]
    plotted = False
    for name in BELIEF_SERIES:
        if name not in real_belief_results.get("aggregate", {}):
            continue
        curve = real_belief_results["aggregate"][name]
        offsets = np.asarray(curve["offsets"])
        mean = 100.0 * np.asarray(curve["mean"])
        sd = 100.0 * np.asarray(curve["sd"])
        _plot_mean_sd(
            ax1,
            offsets,
            mean,
            sd,
            label=BELIEF_LABELS.get(name, name),
            color=BELIEF_COLORS.get(name, "#444444"),
        )
        plotted = True
    _decorate_decoder_axis(
        ax1,
        "A2 · Real: mouse prior + model belief q_t",
    )
    if plotted:
        ax1.legend(frameon=False, fontsize=7.5, loc="lower right")
    else:
        ax1.text(
            0.5,
            0.5,
            "No real-belief curves",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )

    ax2 = axes[2]
    plotted = False
    for region in NEURAL_REGIONS:
        if region not in neural_results.get("aggregate", {}):
            continue
        curve = neural_results["aggregate"][region]
        offsets = np.asarray(curve["offsets"])
        mean = 100.0 * np.asarray(curve["mean"])
        sd = 100.0 * np.asarray(curve["sd"])
        _plot_mean_sd(
            ax2,
            offsets,
            mean,
            sd,
            label=REGION_LABELS.get(region, region),
            color=REGION_COLORS.get(region, "#444444"),
        )
        plotted = True
    _decorate_decoder_axis(
        ax2,
        "A3 · Real: neural prior readout by ROI",
    )
    if plotted:
        ax2.legend(frameon=False, fontsize=7.5, loc="lower right")
    else:
        ax2.text(
            0.5,
            0.5,
            "No neural curves",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    fig.suptitle(
        "Switch-centered MLP block decoding (−30…+30)\n"
        "A1: mean ±1 model-seed SD · A2–A3: mean ±1 session SD (LOSO)",
        fontsize=12,
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--decoder-seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--split-seed", type=int, default=23017)
    parser.add_argument("--before", type=int, default=30)
    parser.add_argument("--after", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-real-belief", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    args = parser.parse_args()

    if args.before < 1 or args.after < 1:
        raise ValueError("--before and --after must be positive")

    cfg = load_synthetic_config()
    model_seeds = _available_model_seeds(cfg, [int(s) for s in args.model_seeds])
    if args.quick:
        model_seeds = model_seeds[:1]
    decoder_seeds = [int(s) for s in args.decoder_seeds]
    if args.quick:
        decoder_seeds = decoder_seeds[:1]

    report_directory = ROOT / cfg["paths"]["reports"] / "switch_block_decoding"
    figure_directory = (
        ROOT / cfg["paths"]["figures"] / "switch_block_decoding"
    )
    report_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = report_directory / "mlp_switch_block_decode_metrics.json"
    existing = {}
    if metrics_path.exists() and (
        args.skip_models or args.skip_real_belief or args.skip_neural
    ):
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))

    if args.skip_models:
        model_results = existing["models"]
    else:
        model_results = run_model_panel(
            cfg,
            model_seeds=model_seeds,
            decoder_seeds=decoder_seeds,
            split_seed=int(args.split_seed),
            before=int(args.before),
            after=int(args.after),
            quick=bool(args.quick),
            refresh_cache=bool(args.refresh_cache),
        )

    if args.skip_real_belief:
        if "real_belief" not in existing:
            raise KeyError(
                "metrics JSON lacks real_belief; rerun without --skip-real-belief"
            )
        real_belief_results = existing["real_belief"]
    else:
        real_belief_results = run_real_belief_panel(
            decoder_seeds=decoder_seeds,
            before=int(args.before),
            after=int(args.after),
            quick=bool(args.quick),
        )

    if args.skip_neural:
        neural_results = existing["neural"]
    else:
        neural_results = run_neural_panel(
            decoder_seeds=decoder_seeds,
            before=int(args.before),
            after=int(args.after),
            quick=bool(args.quick),
        )

    payload = {
        "models": model_results,
        "real_belief": real_belief_results,
        "neural": neural_results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    figure_paths = [
        figure_directory / "mlp_rnn_vs_pc_switch_decoding.png",
        ROOT / "mlp_rnn_vs_pc_switch_decoding.png",
    ]
    make_three_panel_figure(
        model_results, real_belief_results, neural_results, figure_paths
    )

    print(
        json.dumps(
            {
                "metrics": str(metrics_path.relative_to(ROOT)),
                "figures": [str(p.relative_to(ROOT)) for p in figure_paths],
                "model_seeds_used": model_seeds,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
