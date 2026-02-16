"""journal.py

Append-only recommendation history.

Phase-1 upgrade:
- Supports stable, versioned schemas via trabot_schema.normalize_rows().
- Default history path points to reco_history_v<schema>.csv.

Educational tool only – not financial advice.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from trabot_schema import DEFAULT_HISTORY_PATH, RECO_COLUMNS, normalize_rows


DEFAULT_HISTORY_PATH_ENV = os.getenv("TRABOT_RECO_HISTORY", DEFAULT_HISTORY_PATH)


def make_run_id(ts: Optional[datetime] = None) -> str:
    ts = ts or datetime.now()
    return ts.strftime("%Y%m%d_%H%M%S")


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_history(
    rows: List[Dict[str, Any]],
    path: str = DEFAULT_HISTORY_PATH_ENV,
    *,
    columns: Optional[List[str]] = None,
) -> Tuple[str, int]:
    """Append rows into an append-only CSV.

    If columns are provided, rows are normalized to that stable schema.
    If not provided, we default to the canonical reco schema.

    Returns (path, n_rows_appended).
    """
    if not rows:
        return path, 0

    cols = columns or RECO_COLUMNS
    norm_rows = normalize_rows(rows, columns=cols)

    _ensure_parent(path)

    df_new = pd.DataFrame(norm_rows)
    # Enforce column order (and avoid pandas auto-ordering).
    df_new = df_new.reindex(columns=cols)

    write_header = not os.path.exists(path)

    # If file exists but has a different header (corruption / old schema),
    # we *do not* append into it. Instead write to a sibling file.
    if not write_header:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first = f.readline().strip()  # header row
            existing_cols = [c.strip() for c in first.split(",")]
            if existing_cols != cols:
                alt = path.replace(".csv", "") + f"__schema_mismatch_{make_run_id()}.csv"
                path = alt
                write_header = True
        except Exception:
            # If we can't read header, safest is to write to a new file.
            alt = path.replace(".csv", "") + f"__unreadable_{make_run_id()}.csv"
            path = alt
            write_header = True

    df_new.to_csv(path, mode="a", index=False, header=write_header)
    return path, len(df_new)


def save_snapshot(
    rows: List[Dict[str, Any]],
    path: str,
    *,
    columns: Optional[List[str]] = None,
) -> str:
    """Write a standalone snapshot CSV (overwrites)."""
    _ensure_parent(path)

    cols = columns or RECO_COLUMNS
    norm_rows = normalize_rows(rows, columns=cols)

    df = pd.DataFrame(norm_rows).reindex(columns=cols)
    df.to_csv(path, index=False)
    return path
