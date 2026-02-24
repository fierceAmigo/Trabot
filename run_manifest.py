"""run_manifest.py

Phase-1 foundation: per-run manifest JSON.

Upgrades:
- atomic write (crash-safe)
- optional git commit capture
- env snapshot capture for reproducibility
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from io_utils import atomic_write_json, ensure_dir


def make_run_dir(base_dir: str, run_id: str) -> str:
    run_dir = os.path.join(base_dir, run_id)
    ensure_dir(run_dir)
    return run_dir


def _read_git_commit() -> str:
    """Best-effort git commit hash."""
    try:
        # If running inside a git repo
        head = os.path.join(".git", "HEAD")
        if not os.path.exists(head):
            return ""
        ref = open(head, "r", encoding="utf-8").read().strip()
        if ref.startswith("ref:"):
            ref_path = ref.split(":", 1)[1].strip()
            full = os.path.join(".git", ref_path)
            if os.path.exists(full):
                return open(full, "r", encoding="utf-8").read().strip()
        return ref
    except Exception:
        return ""


def env_snapshot(prefixes: Optional[list[str]] = None) -> Dict[str, str]:
    prefixes = prefixes or ["TRABOT_", "KITE_", "PYTHON"]
    out: Dict[str, str] = {}
    for k, v in os.environ.items():
        if any(k.startswith(p) for p in prefixes):
            out[k] = v
    return dict(sorted(out.items(), key=lambda x: x[0]))


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
    ensure_dir(os.path.dirname(path) or ".")
    atomic_write_json(path, payload, indent=2, sort_keys=True)
    return path


def build_base_manifest(*, run_id: str, mode: str, universe_count: int, slice_from: int, slice_to: int) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "created_utc": datetime.utcnow().isoformat(),
        "mode": mode,
        "universe": {"count": universe_count, "slice": [slice_from, slice_to]},
        "git_commit": _read_git_commit(),
        "env": env_snapshot(),
    }


def now_iso() -> str:
    try:
        return datetime.utcnow().isoformat()
    except Exception:
        return ""

def update_manifest(path: str, updates: Dict[str, Any]) -> None:
    """Read-modify-write manifest atomically."""
    try:
        cur = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cur = json.load(f) or {}
        cur.update(updates or {})
        atomic_write_json(path, cur, indent=2, sort_keys=True)
    except Exception:
        # best-effort
        try:
            atomic_write_json(path, updates or {}, indent=2, sort_keys=True)
        except Exception:
            pass
