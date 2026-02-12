"""
journal.py

Append-only recommendation history.

Creates:
- data/reco_history.csv  (append-only)
- helper to also write per-run snapshots like data/reco_YYYYMMDD_HHMMSS.csv

This is intentionally simple: CSV only, no DB.

Educational tool only – not financial advice.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd


DEFAULT_HISTORY_PATH = os.getenv("TRABOT_RECO_HISTORY", os.path.join("data", "reco_history.csv"))


def make_run_id(ts: Optional[datetime] = None) -> str:
    ts = ts or datetime.now()
    return ts.strftime("%Y%m%d_%H%M%S")


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _normalize(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        return " | ".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return v


def append_history(rows: List[Dict[str, Any]], path: str = DEFAULT_HISTORY_PATH) -> Tuple[str, int]:
    """
    Appends rows into an append-only CSV. If file doesn't exist, writes header once.
    Returns (path, n_rows_appended).
    """
    if not rows:
        return path, 0

    _ensure_parent(path)
    norm_rows = [{k: _normalize(v) for k, v in r.items()} for r in rows]

    df_new = pd.DataFrame(norm_rows)
    write_header = not os.path.exists(path)
    df_new.to_csv(path, mode="a", index=False, header=write_header)
    return path, len(df_new)


def save_snapshot(rows: List[Dict[str, Any]], path: str) -> str:
    """
    Writes a standalone snapshot CSV (overwrites that path).
    Returns the path.
    """
    _ensure_parent(path)
    norm_rows = [{k: _normalize(v) for k, v in r.items()} for r in rows]
    pd.DataFrame(norm_rows).to_csv(path, index=False)
    return path
