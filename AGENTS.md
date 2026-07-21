# AGENTS.md

This repository is a neuroscience/AI research project on latent prior updating in the International Brain Laboratory task.

## Source of truth

`README.md` is the scientific and modeling ground truth. If this file conflicts with `README.md`, follow `README.md`.

The v1 model comparison is:
1. a standard task-trained RNN,
2. a predictive-coding RNN,
3. a Bayesian / explicit online-inference model.

## Agent behavior

Before implementing anything:
1. Read this file and the README.
2. Run a one-question-at-a-time clarification process if any core decision is unresolved.
3. Produce a compact project spec.
4. Only then scaffold code.

## Required project decisions

Clarify these before coding:
- dataset source and inclusion criteria
- behavior-only versus behavior plus neural analysis
- exact model definitions
- common input/output interface across models
- evaluation metrics
- switch-centered windows
- prior estimation method
- neural alignment method
- behavior-matching procedure
- version-1 deliverables

## Non-negotiables

- Favor clarity and reproducibility.
- Keep assumptions explicit.
- Separate behavior analysis from neural analysis.
- Do not bury design choices in code.
- Prefer small, testable modules.
- Document unresolved questions instead of guessing.

## Expected repository shape

- src/data
- src/models
- src/train
- src/eval
- src/neural
- src/plot
- scripts
- notebooks only for exploration, not as the system of record

## Modeling conventions

- Standard RNN, predictive-coding RNN, and Bayesian model must share a comparable interface where possible.
- Latent-state extraction must be an explicit function for each model.
- Keep evaluation code model-agnostic.
- Behavior-matched neural comparison must be defined explicitly.

## Output expectation

When asked to plan the project, produce:
- objective
- scope
- assumptions
- model definitions
- evaluation plan
- deliverables
- open risks
