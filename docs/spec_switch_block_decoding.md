# Spec: MLP switch-centered block decoding (models + neural)

Frozen decisions (2026-07-23 grilling).

## Objective

Compare how well an MLP can decode true biased-block identity (`P(right)=0.2` vs `0.8`) around genuine block switches from:

1. **History-only model latents** (all four active v2 models).
2. **Neural belief activity** (CV Ridge OOF prior readout) in the four primary ROIs.

## Scope

- Regime: **history_only** only.
- Models: `tanh_bptt`, `tanh_pc`, `gru`, `gru_pc`.
- Neural ROIs: `MOs`, `vlOFC_orbvl`, `ACAd`, `MOp` (`NEURAL_REGIONS`).
- Switch window: **−30 … +30** trials (isolated genuine 0.2↔0.8 switches only).
- Decoder: one-hidden-layer **MLP** (same family as existing switch-block decode tooling).
- Multipanel diagnostics bottom-left switch panel also uses **−30 … +30**.

## Assumptions

- Model features = concatenated zero-current-evidence latent trajectory (all within-trial ticks × hidden size); probe zeros visual/action/reward channels and keeps only the go cue.
- Neural features = scalar trial-level OOF neural prior readout (`fit_prior_readout`), one curve per ROI.
- Model uncertainty = ±1 sample SD across available task-model seeds (canonical checkpoint if no seed replicates).
- Neural uncertainty = ±1 sample SD across sessions that contain that ROI (leave-one-session-out decoder).

## Deliverables

- `scripts/16_plot_mlp_switch_block_decoding.py`
- `src/models_v2/block_decode.py`
- Figure: `reports/v2/figures/switch_block_decoding/mlp_rnn_vs_pc_switch_decoding.png`
- Copy: project-root `mlp_rnn_vs_pc_switch_decoding.png`
- Metrics under `reports/v2/switch_block_decoding/`

## Open risks

- Few neural sessions per ROI → wide session SD and sparse far-window switches.
- Scalar neural belief makes the MLP nearly a calibrated threshold; still kept for interface parity with panel 1.
- Without `model_seed_replicates/`, panel-1 shading collapses to zero width.
