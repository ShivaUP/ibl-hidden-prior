#!/usr/bin/env python3
"""13 — Switch-centered block decoding for frozen V2 BPTT and PC RNNs.

The task models have one 48-unit recurrent layer.  Each post-hoc decoder uses
the *total zero-current-evidence latent trajectory* for a trial: all nine
48-unit recurrent states concatenated into 432 features.

For each independent task-model seed:
  1. fit logistic and one-hidden-layer MLP decoders on complete training sessions;
  2. select decoder epochs on complete validation sessions;
  3. ensemble three decoder initializations;
  4. calculate balanced percentage correctly decoded at every offset around
     genuine 0.2 <-> 0.8 block switches in held-out test sessions.

The final curves are means across independent task-model training seeds, with
shading equal to +/- one sample SD across those seeds.

Outputs
-------
reports/v2/switch_block_decoding/switch_block_decode_metrics.json
reports/v2/switch_block_decoding/switch_block_decode_curves.csv
reports/v2/figures/switch_block_decoding/
    zero_evidence_and_decoder_switches_six_panel.png
    rnn_logistic_vs_mlp_switch_decoding.png
    pc_logistic_vs_mlp_switch_decoding.png
    logistic_rnn_vs_pc_switch_decoding.png
    mlp_rnn_vs_pc_switch_decoding.png
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models_v2.block_decode import (
    DECODER_NAMES,
    DecoderSettings,
    extract_zero_evidence_latents,
    fit_decoder,
    make_decoding_dataset,
    make_session_split,
)
from src.models_v2.rollout import load_model
from src.synthetic.channels import PhaseTicks
from src.synthetic.generate import build_training_tensors, generate_sessions
from src.synthetic.schema import load_synthetic_config


MODEL_IDS = ("tanh_bptt", "tanh_pc")
MODEL_LABELS = {
    "tanh_bptt": "RNN (BPTT)",
    "tanh_pc": "PC-trained RNN",
}
MODEL_COLORS = {
    "tanh_bptt": "#3b6ea8",
    "tanh_pc": "#8e44ad",
}
DECODER_LABELS = {
    "logistic": "Logistic regression",
    "mlp": "Neural-network decoder",
}
DECODER_COLORS = {
    "logistic": "#d9822b",
    "mlp": "#2f8f6b",
}
DIRECTION_LABELS = {
    "low_to_high": "0.2 -> 0.8",
    "high_to_low": "0.8 -> 0.2",
}
DIRECTION_COLORS = {
    "low_to_high": "#0072b2",
    "high_to_low": "#d55e00",
}


def _canonical_checkpoint_path(cfg: dict, model_id: str) -> Path:
    return (
        ROOT
        / cfg["paths"]["artifacts"]
        / "models"
        / model_id
        / "model.npz"
    )


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


def _build_matched_eval_inputs(
    cfg: dict,
    n_sessions: int,
    n_trials: int,
):
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
    """Put flattened biased-trial predictions back into session x trial form."""

    expanded = np.full(p_right.shape, np.nan, dtype=np.float64)
    cursor = 0
    for session in test_sessions:
        biased = (
            np.isclose(p_right[session], 0.2)
            | np.isclose(p_right[session], 0.8)
        )
        count = int(np.count_nonzero(biased))
        expanded[session, biased] = probabilities[cursor : cursor + count]
        cursor += count
    if cursor != len(probabilities):
        raise RuntimeError(
            "flattened decoder predictions did not map back to test trials"
        )
    return expanded


def _eligible_switches(
    p_right: np.ndarray,
    sessions: np.ndarray,
    before: int,
    after: int,
):
    """Yield isolated biased-block switches with no second switch in the window."""

    for session in sessions:
        prior = p_right[session]
        changed = np.flatnonzero(np.diff(prior) != 0.0) + 1
        for switch in changed:
            previous = float(prior[switch - 1])
            current = float(prior[switch])
            genuine = (
                (np.isclose(previous, 0.2) and np.isclose(current, 0.8))
                or (
                    np.isclose(previous, 0.8)
                    and np.isclose(current, 0.2)
                )
            )
            if not genuine:
                continue
            start = switch - before
            stop = switch + after + 1
            if start < 0 or stop > prior.shape[0]:
                continue
            window = prior[start:stop]
            if not np.all(
                np.isclose(window, 0.2) | np.isclose(window, 0.8)
            ):
                continue
            # Keep the event window attributable to one switch. Otherwise a
            # short empirical block can place a second switch at (for example)
            # offset +24, making the far-right "adaptation" curve describe a
            # different transition.
            if not np.allclose(prior[start:switch], previous):
                continue
            if not np.allclose(prior[switch:stop], current):
                continue
            direction = (
                "low_to_high" if current > previous else "high_to_low"
            )
            yield int(session), int(switch), direction


def _balanced_accuracy_from_binary(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
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
    """Balanced hard-decoding success at every trial relative to a switch."""

    windows_probability = []
    windows_label = []
    directions = []
    for session, switch, direction in _eligible_switches(
        p_right,
        sessions,
        before,
        after,
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
            "balanced_accuracy": np.full_like(
                offsets, np.nan, dtype=np.float64
            ).tolist(),
            "n_switches": 0,
            "n_low_to_high": 0,
            "n_high_to_low": 0,
        }

    probability = np.stack(windows_probability)
    labels = np.stack(windows_label)
    predictions = (probability >= 0.5).astype(int)
    accuracy = np.asarray(
        [
            _balanced_accuracy_from_binary(
                labels[:, index],
                predictions[:, index],
            )
            for index in range(len(offsets))
        ],
        dtype=np.float64,
    )
    return {
        "offsets": offsets.tolist(),
        "balanced_accuracy": accuracy.tolist(),
        "n_switches": len(directions),
        "n_low_to_high": int(
            np.count_nonzero(np.asarray(directions) == "low_to_high")
        ),
        "n_high_to_low": int(
            np.count_nonzero(np.asarray(directions) == "high_to_low")
        ),
    }


def switch_centered_zero_evidence_belief(
    zero_evidence_p_right: np.ndarray,
    p_right: np.ndarray,
    sessions: np.ndarray,
    *,
    before: int,
    after: int,
) -> dict:
    """Mean P(right) around genuine 0.2 <-> 0.8 switches, by direction."""

    groups = {"low_to_high": [], "high_to_low": []}
    for session, switch, direction in _eligible_switches(
        p_right,
        sessions,
        before,
        after,
    ):
        start = switch - before
        stop = switch + after + 1
        values = zero_evidence_p_right[session, start:stop]
        if np.all(np.isfinite(values)):
            groups[direction].append(values)

    offsets = np.arange(-before, after + 1)
    result = {"offsets": offsets.tolist()}
    for direction, windows in groups.items():
        if windows:
            result[direction] = np.mean(
                np.stack(windows),
                axis=0,
            ).tolist()
        else:
            result[direction] = np.full_like(
                offsets,
                np.nan,
                dtype=np.float64,
            ).tolist()
        result[f"n_{direction}"] = len(windows)
    result["n_switches"] = sum(len(values) for values in groups.values())
    return result


def _aggregate_curves(
    per_seed: dict,
    value_key: str,
) -> dict:
    seeds = sorted(int(seed) for seed in per_seed)
    curves = np.stack(
        [
            np.asarray(per_seed[str(seed)][value_key], dtype=np.float64)
            for seed in seeds
        ],
        axis=0,
    )
    sample_sd = (
        curves.std(axis=0, ddof=1)
        if len(seeds) > 1
        else np.zeros(curves.shape[1], dtype=np.float64)
    )
    first = per_seed[str(seeds[0])]
    return {
        "offsets": first["offsets"],
        "mean": curves.mean(axis=0).tolist(),
        "model_seed_sd": sample_sd.tolist(),
        "per_model_seed": {
            str(seed): per_seed[str(seed)][value_key]
            for seed in seeds
        },
        "model_seeds": seeds,
        "n_switches_per_seed": {
            str(seed): int(per_seed[str(seed)]["n_switches"])
            for seed in seeds
        },
    }


def _plot_mean_sd(
    axis,
    offsets: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    axis.plot(
        offsets,
        mean,
        color=color,
        linestyle=linestyle,
        linewidth=2,
        label=label,
    )
    axis.fill_between(
        offsets,
        mean - sd,
        mean + sd,
        color=color,
        alpha=0.15,
    )


def _decorate_decoder_axis(axis, title: str) -> None:
    axis.axhline(
        50.0,
        color="0.4",
        linestyle="--",
        linewidth=1,
        label="chance",
    )
    axis.axvline(0, color="black", linestyle=":", linewidth=1)
    axis.set(
        title=title,
        xlabel="Trials relative to block switch",
        ylabel="Balanced block-decoding success (%)",
        ylim=(0.0, 102.0),
    )
    axis.grid(alpha=0.2)


def _plot_decoder_curves(
    axis,
    results: dict,
    series,
    title: str,
) -> None:
    for model_id, decoder_name, label, color in series:
        curve = results["aggregate"]["decoders"][model_id][decoder_name]
        offsets = np.asarray(curve["offsets"])
        mean = 100.0 * np.asarray(curve["mean"])
        sd = 100.0 * np.asarray(curve["model_seed_sd"])
        _plot_mean_sd(
            axis,
            offsets,
            mean,
            sd,
            label=label,
            color=color,
        )
    _decorate_decoder_axis(axis, title)
    axis.legend(frameon=False, fontsize=9)


def _plot_belief_axis(axis, results: dict, model_id: str) -> None:
    for direction in ("low_to_high", "high_to_low"):
        curve = results["aggregate"]["zero_evidence"][model_id][direction]
        offsets = np.asarray(curve["offsets"])
        mean = np.asarray(curve["mean"])
        sd = np.asarray(curve["model_seed_sd"])
        _plot_mean_sd(
            axis,
            offsets,
            mean,
            sd,
            label=DIRECTION_LABELS[direction],
            color=DIRECTION_COLORS[direction],
        )
        if direction == "low_to_high":
            truth = np.where(offsets < 0, 0.2, 0.8)
        else:
            truth = np.where(offsets < 0, 0.8, 0.2)
        axis.step(
            offsets,
            truth,
            where="post",
            color=DIRECTION_COLORS[direction],
            linestyle=":",
            linewidth=1,
            alpha=0.55,
        )
    axis.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    axis.axvline(0, color="black", linestyle=":", linewidth=1)
    axis.set(
        title=f"{MODEL_LABELS[model_id]}: zero-evidence belief",
        xlabel="Trials relative to block switch",
        ylabel="P(right) with current evidence removed",
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=9)


def make_figures(results: dict, output_directory: Path) -> list:
    output_directory.mkdir(parents=True, exist_ok=True)
    made = []

    # Six-panel figure: the two old-style belief panels plus four requested
    # decoder panels.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10),
        constrained_layout=True,
    )
    _plot_belief_axis(axes[0, 0], results, "tanh_bptt")
    _plot_decoder_curves(
        axes[0, 1],
        results,
        [
            (
                "tanh_bptt",
                decoder,
                DECODER_LABELS[decoder],
                DECODER_COLORS[decoder],
            )
            for decoder in DECODER_NAMES
        ],
        "RNN: logistic versus neural-network decoder",
    )
    _plot_decoder_curves(
        axes[0, 2],
        results,
        [
            (
                model_id,
                "logistic",
                MODEL_LABELS[model_id],
                MODEL_COLORS[model_id],
            )
            for model_id in MODEL_IDS
        ],
        "Logistic decoder: RNN versus PC",
    )
    _plot_belief_axis(axes[1, 0], results, "tanh_pc")
    _plot_decoder_curves(
        axes[1, 1],
        results,
        [
            (
                "tanh_pc",
                decoder,
                DECODER_LABELS[decoder],
                DECODER_COLORS[decoder],
            )
            for decoder in DECODER_NAMES
        ],
        "PC: logistic versus neural-network decoder",
    )
    _plot_decoder_curves(
        axes[1, 2],
        results,
        [
            (
                model_id,
                "mlp",
                MODEL_LABELS[model_id],
                MODEL_COLORS[model_id],
            )
            for model_id in MODEL_IDS
        ],
        "Neural-network decoder: RNN versus PC",
    )
    fig.suptitle(
        "Zero-evidence belief and latent block decoding around genuine "
        "0.2 <-> 0.8 switches\n"
        "lines = mean across task-model seeds; shading = +/-1 model-seed SD"
    )
    path = output_directory / (
        "zero_evidence_and_decoder_switches_six_panel.png"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    specifications = [
        (
            "rnn_logistic_vs_mlp_switch_decoding.png",
            [
                (
                    "tanh_bptt",
                    decoder,
                    DECODER_LABELS[decoder],
                    DECODER_COLORS[decoder],
                )
                for decoder in DECODER_NAMES
            ],
            "RNN block decoding around switches",
        ),
        (
            "pc_logistic_vs_mlp_switch_decoding.png",
            [
                (
                    "tanh_pc",
                    decoder,
                    DECODER_LABELS[decoder],
                    DECODER_COLORS[decoder],
                )
                for decoder in DECODER_NAMES
            ],
            "PC block decoding around switches",
        ),
        (
            "logistic_rnn_vs_pc_switch_decoding.png",
            [
                (
                    model_id,
                    "logistic",
                    MODEL_LABELS[model_id],
                    MODEL_COLORS[model_id],
                )
                for model_id in MODEL_IDS
            ],
            "Logistic block decoder: RNN versus PC",
        ),
        (
            "mlp_rnn_vs_pc_switch_decoding.png",
            [
                (
                    model_id,
                    "mlp",
                    MODEL_LABELS[model_id],
                    MODEL_COLORS[model_id],
                )
                for model_id in MODEL_IDS
            ],
            "Neural-network block decoder: RNN versus PC",
        ),
    ]
    for filename, series, title in specifications:
        fig, axis = plt.subplots(
            figsize=(8, 5),
            constrained_layout=True,
        )
        _plot_decoder_curves(axis, results, series, title)
        fig.suptitle(
            "Total zero-evidence latent trajectory (9 x 48 features)\n"
            "mean across task-model seeds; shading = +/-1 model-seed SD"
        )
        path = output_directory / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        made.append(path)

    return made


def write_csv(results: dict, path: Path) -> None:
    rows = []
    for model_id in MODEL_IDS:
        for direction in ("low_to_high", "high_to_low"):
            curve = results["aggregate"]["zero_evidence"][model_id][direction]
            for offset, mean, sd in zip(
                curve["offsets"],
                curve["mean"],
                curve["model_seed_sd"],
            ):
                rows.append(
                    {
                        "curve_type": "zero_evidence_belief",
                        "model_id": model_id,
                        "decoder": "",
                        "direction": direction,
                        "offset": offset,
                        "mean": mean,
                        "model_seed_sd": sd,
                        "units": "P(right)",
                    }
                )
        for decoder_name in DECODER_NAMES:
            curve = results["aggregate"]["decoders"][model_id][decoder_name]
            for offset, mean, sd in zip(
                curve["offsets"],
                curve["mean"],
                curve["model_seed_sd"],
            ):
                rows.append(
                    {
                        "curve_type": "block_decoding",
                        "model_id": model_id,
                        "decoder": decoder_name,
                        "direction": "directions_combined_balanced",
                        "offset": offset,
                        "mean": 100.0 * mean,
                        "model_seed_sd": 100.0 * sd,
                        "units": "balanced_accuracy_percent",
                    }
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=[7, 17, 27],
    )
    parser.add_argument(
        "--decoder-seeds",
        type=int,
        nargs="+",
        default=[101, 202, 303],
    )
    parser.add_argument("--split-seed", type=int, default=23017)
    parser.add_argument("--before", type=int, default=20)
    parser.add_argument("--after", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    if args.before < 1 or args.after < 1:
        raise ValueError("--before and --after must be positive")
    model_seeds = [int(seed) for seed in args.model_seeds]
    if len(set(model_seeds)) != len(model_seeds):
        raise ValueError("--model-seeds must be unique")
    if args.quick:
        model_seeds = model_seeds[:1]

    cfg = load_synthetic_config()
    phase = PhaseTicks.from_config(cfg)
    n_sessions = 12 if args.quick else int(cfg["eval"]["synth_sessions"])
    n_trials = 300 if args.quick else int(cfg["eval"]["synth_trials"])
    settings = _settings(args.quick)
    split = make_session_split(n_sessions, args.split_seed)

    missing = [
        str(_checkpoint_path(cfg, model_id, model_seed))
        for model_id in MODEL_IDS
        for model_seed in model_seeds
        if not _checkpoint_path(cfg, model_id, model_seed).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoints: " + ", ".join(missing)
        )

    batch, matched_inputs, batch_seed, input_seed = (
        _build_matched_eval_inputs(cfg, n_sessions, n_trials)
    )
    report_directory = (
        ROOT / cfg["paths"]["reports"] / "switch_block_decoding"
    )
    figure_directory = (
        ROOT / cfg["paths"]["figures"] / "switch_block_decoding"
    )
    cache_directory = report_directory / "checkpoint_cache"
    report_directory.mkdir(parents=True, exist_ok=True)
    cache_directory.mkdir(parents=True, exist_ok=True)

    results = {
        "analysis": {
            "target": "P(right)=0.2 versus P(right)=0.8 block",
            "feature": (
                "all nine 48-unit zero-current-evidence latent states "
                "concatenated (432 features)"
            ),
            "curve_metric": (
                "balanced percentage correctly decoded at each switch-relative "
                "trial; 0.5 threshold after averaging decoder probabilities "
                "across decoder initializations"
            ),
            "switches": (
                "isolated genuine 0.2 <-> 0.8 transitions only: the entire "
                "[-before, +after] window remains in the old/new block, so no "
                "second switch contaminates the curve; initial neutral-block "
                "transitions are excluded"
            ),
            "before": int(args.before),
            "after": int(args.after),
            "task_model_seeds": model_seeds,
            "decoder_initialization_seeds": [
                int(seed) for seed in args.decoder_seeds
            ],
            "seed_aggregation": (
                "decoder probabilities are first averaged across decoder "
                "initializations within each task-model seed; switch curves are "
                "then averaged across independently trained task-model seeds"
            ),
            "uncertainty": "sample SD across task-model seeds",
            "session_split": {
                name: values.tolist() for name, values in split.items()
            },
            "n_sessions": n_sessions,
            "trials_per_session": n_trials,
            "batch_seed": batch_seed,
            "matched_input_seed": input_seed,
            "split_seed": int(args.split_seed),
            "current_trial_leakage_control": (
                "visual, action, reward and non-reward inputs are zero on the "
                "decoder probe branch; only the go cue is retained"
            ),
            "decoder_training": (
                "whole-session train/validation/test split; test sessions never "
                "select decoder weights or epochs"
            ),
        },
        "decoder_settings": settings.__dict__,
        "per_model_seed": {
            model_id: {} for model_id in MODEL_IDS
        },
        "aggregate": {
            "zero_evidence": {
                model_id: {} for model_id in MODEL_IDS
            },
            "decoders": {
                model_id: {} for model_id in MODEL_IDS
            },
        },
    }

    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    decoder_source_sha = hashlib.sha256(
        (ROOT / "src/models_v2/block_decode.py").read_bytes()
    ).hexdigest()
    config_sha = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()

    for model_index, model_id in enumerate(MODEL_IDS):
        for seed_index, model_seed in enumerate(model_seeds):
            checkpoint = _checkpoint_path(cfg, model_id, model_seed)
            checkpoint_sha = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            signature_payload = {
                "cache_schema": 1,
                "analysis_script_sha256": script_sha,
                "decoder_source_sha256": decoder_source_sha,
                "config_sha256": config_sha,
                "checkpoint_sha256": checkpoint_sha,
                "model_id": model_id,
                "model_seed": model_seed,
                "n_sessions": n_sessions,
                "n_trials": n_trials,
                "before": int(args.before),
                "after": int(args.after),
                "split_seed": int(args.split_seed),
                "batch_seed": batch_seed,
                "input_seed": input_seed,
                "decoder_seeds": [
                    int(seed) for seed in args.decoder_seeds
                ],
                "decoder_settings": settings.__dict__,
            }
            signature = hashlib.sha256(
                json.dumps(
                    signature_payload,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            cache_path = (
                cache_directory / f"{model_id}_seed_{model_seed}.json"
            )
            if (
                cache_path.exists()
                and not args.refresh_cache
                and not args.quick
            ):
                cached = json.loads(
                    cache_path.read_text(encoding="utf-8")
                )
                if cached.get("cache_signature") == signature:
                    print(
                        f"Loading verified cache: {model_id}, "
                        f"task-model seed {model_seed}",
                        flush=True,
                    )
                    results["per_model_seed"][model_id][
                        str(model_seed)
                    ] = cached["seed_result"]
                    continue

            print(
                f"Extracting and decoding: {model_id}, "
                f"task-model seed {model_seed}",
                flush=True,
            )
            model = load_model(model_id, checkpoint)
            latents = extract_zero_evidence_latents(
                model,
                matched_inputs,
                phase,
            )
            total_features = latents.reshape(
                n_sessions,
                n_trials,
                phase.n_steps * model.hidden_size,
            )
            zero_evidence_p_right = model.probs(
                latents[:, :, phase.response_tick, :].reshape(
                    n_sessions * n_trials,
                    model.hidden_size,
                )
            )[:, 1].reshape(n_sessions, n_trials)

            belief_curve = switch_centered_zero_evidence_belief(
                zero_evidence_p_right,
                batch.p_right,
                split["test"],
                before=args.before,
                after=args.after,
            )
            seed_result = {
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": checkpoint_sha,
                "zero_evidence": belief_curve,
                "decoders": {},
            }

            datasets = {
                name: make_decoding_dataset(
                    total_features,
                    batch.p_right,
                    batch.block_id,
                    sessions,
                )
                for name, sessions in split.items()
            }
            for decoder_index, decoder_name in enumerate(DECODER_NAMES):
                print(f"  fitting {decoder_name}", flush=True)
                decoder_runs = [
                    fit_decoder(
                        decoder_name,
                        datasets["train"],
                        datasets["validation"],
                        datasets["test"],
                        seed=int(decoder_seed),
                        settings=settings,
                    )
                    for decoder_seed in args.decoder_seeds
                ]
                mean_probability = np.mean(
                    np.stack(
                        [
                            run["probabilities"]
                            for run in decoder_runs
                        ],
                        axis=0,
                    ),
                    axis=0,
                )
                expanded = _expand_test_probabilities(
                    mean_probability,
                    batch.p_right,
                    split["test"],
                )
                curve = switch_centered_decoder_accuracy(
                    expanded,
                    batch.p_right,
                    split["test"],
                    before=args.before,
                    after=args.after,
                )
                curve["best_epoch_by_decoder_seed"] = [
                    int(run["best_epoch"]) for run in decoder_runs
                ]
                curve["validation_cross_entropy_by_decoder_seed"] = [
                    float(run["validation_cross_entropy"])
                    for run in decoder_runs
                ]
                seed_result["decoders"][decoder_name] = curve
                print(
                    "    pre-switch -1: "
                    f"{100.0 * curve['balanced_accuracy'][args.before - 1]:.2f}%"
                    " | switch 0: "
                    f"{100.0 * curve['balanced_accuracy'][args.before]:.2f}%"
                    " | post +15: "
                    f"{100.0 * curve['balanced_accuracy'][args.before + 15]:.2f}%",
                    flush=True,
                )

            results["per_model_seed"][model_id][
                str(model_seed)
            ] = seed_result
            if not args.quick:
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

    for model_id in MODEL_IDS:
        per_seed_model = results["per_model_seed"][model_id]
        for direction in ("low_to_high", "high_to_low"):
            direction_per_seed = {
                seed: {
                    "offsets": result["zero_evidence"]["offsets"],
                    direction: result["zero_evidence"][direction],
                    "n_switches": result["zero_evidence"]["n_switches"],
                }
                for seed, result in per_seed_model.items()
            }
            results["aggregate"]["zero_evidence"][model_id][
                direction
            ] = _aggregate_curves(
                direction_per_seed,
                direction,
            )

        for decoder_name in DECODER_NAMES:
            decoder_per_seed = {
                seed: result["decoders"][decoder_name]
                for seed, result in per_seed_model.items()
            }
            results["aggregate"]["decoders"][model_id][
                decoder_name
            ] = _aggregate_curves(
                decoder_per_seed,
                "balanced_accuracy",
            )

    metrics_path = (
        report_directory / "switch_block_decode_metrics.json"
    )
    metrics_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    write_csv(
        results,
        report_directory / "switch_block_decode_curves.csv",
    )
    made = make_figures(results, figure_directory)
    print(
        json.dumps(
            {
                "metrics": str(metrics_path.relative_to(ROOT)),
                "figures": [
                    str(path.relative_to(ROOT)) for path in made
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
