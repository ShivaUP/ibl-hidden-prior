"""Shared constants and config loading for synthetic v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

LEFT = 0
RIGHT = 1


def load_synthetic_config(path: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    cfg_path = path or (root / "configs" / "synthetic_v2.yaml")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
