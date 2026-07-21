"""Load frozen v1 YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "frozen_v1.yaml"


def load_frozen_config(path: Path | None = None) -> dict[str, Any]:
    """Load `configs/frozen_v1.yaml` (or an override path)."""
    cfg_path = path or DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    return data


def repo_root() -> Path:
    return REPO_ROOT
