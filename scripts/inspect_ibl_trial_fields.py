#!/usr/bin/env python3
"""Thin wrapper around src.data.inspect_trials.main.

Prefer running the module directly:

    python src/data/inspect_trials.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.inspect_trials import main


if __name__ == "__main__":
    raise SystemExit(main())
