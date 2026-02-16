"""run_manifest.py

Phase-1 foundation: per-run manifest JSON.

A manifest is a single JSON file per scan run containing:
- run_id + timestamps
- all relevant knobs (env + derived config)
- universe slice summary
- cache settings
- outputs produced
- quote timestamp range and other run stats (filled at end)

The intent is that *every* run is fully reproducible and auditable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_dir(base_dir: str, run_id: str) -> str:
    run_dir = os.path.join(base_dir, run_id)
    _ensure_dir(run_dir)
    return run_dir


def write_manifest(
    *,
    run_id: str,
    payload: Dict[str, Any],
    base_dir: str = "data/runs",
    filename: str = "manifest.json",
) -> str:
    """Write a manifest file and return its path."""
    run_dir = make_run_dir(base_dir, run_id)
    path = os.path.join(run_dir, filename)
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def update_manifest(path: str, updates: Dict[str, Any]) -> str:
    """Read-modify-write manifest."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            cur = json.load(f)
    except Exception:
        cur = {}

    def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                a[k] = _deep_merge(a[k], v)
            else:
                a[k] = v
        return a

    cur = _deep_merge(cur, dict(updates or {}))

    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2, sort_keys=True)

    return path


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def env_snapshot(prefixes: Optional[list[str]] = None) -> Dict[str, str]:
    """Capture env vars for reproducibility.

    Default captures TRABOT_* plus common knobs.
    """
    prefixes = prefixes or ["TRABOT_", "INTERVAL", "LOOKBACK", "EMA_", "ADX_", "ATR_", "MIN_", "MAX_", "RISK_"]

    out: Dict[str, str] = {}
    for k, v in os.environ.items():
        for p in prefixes:
            if k == p or k.startswith(p):
                out[k] = str(v)
                break
    return dict(sorted(out.items(), key=lambda kv: kv[0]))
