"""Load Phase 5 checkpoints into model instances."""

from __future__ import annotations

from pathlib import Path

import torch

from src.train.loop import build_model, normalize_model_name


def load_checkpoint(
    repo_root: Path,
    model_name: str,
    condition: str = "history_only",
    run_name: str = "default",
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict]:
    model_name = normalize_model_name(model_name)
    path = repo_root / "artifacts" / "models" / model_name / condition / f"{run_name}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(
        model_name,
        input_size=int(payload["input_size"]),
        hidden_size=int(payload["hidden_size"]),
    )
    model.load_state_dict(payload["state_dict"])
    device = device or torch.device("cpu")
    model.to(device)
    model.eval()
    return model, payload
