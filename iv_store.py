"""
iv_store.py

Stores IV snapshots to data/iv_history.csv and computes IV percentile
using a rolling N trading-day window with EWMA smoothing.

Key behaviors:
- Skips invalid IV rows (<=0 or NaN) to avoid poisoning history.
- Uses last snapshot per day (per underlying) to reduce intraday noise.
- Returns percentile in [0,1].

Educational tool only – not financial advice.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd

IV_PATH = os.getenv("TRABOT_IV_PATH", "data/iv_history.csv")


def ensure_dir() -> None:
    d = os.path.dirname(IV_PATH) or "data"
    os.makedirs(d, exist_ok=True)


def _coerce_float(x) -> Optional[float]:
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def append_iv_snapshot(row: dict) -> None:
    """
    Append one IV snapshot row to IV_PATH.
    Required keys: ts, underlying, iv
    """
    if not isinstance(row, dict):
        return

    ts = row.get("ts")
    underlying = row.get("underlying")
    iv = _coerce_float(row.get("iv"))

    if not ts or not underlying:
        return
    # Skip invalid/non-positive IV to keep history clean
    if iv is None or iv <= 0:
        return

    ensure_dir()
    row = dict(row)
    row["underlying"] = str(underlying).upper().strip()
    row["iv"] = float(iv)

    df_new = pd.DataFrame([row])

    if os.path.exists(IV_PATH):
        try:
            df_old = pd.read_csv(IV_PATH)
            df = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new

    df.to_csv(IV_PATH, index=False)


def iv_percentile(underlying: str, window_days: int = 30, ewma_span: int = 10) -> Tuple[Optional[float], int, Optional[float]]:
    """
    Returns: (pct, n_samples, last_iv_smooth)

      pct in [0,1]
    Uses last snapshot per day to avoid intraday noise.
    """
    if not os.path.exists(IV_PATH):
        return None, 0, None

    try:
        df = pd.read_csv(IV_PATH)
    except Exception:
        return None, 0, None

    if df.empty:
        return None, 0, None

    u = str(underlying).upper().strip()
    df = df[df.get("underlying", "").astype(str).str.upper() == u].copy()
    if df.empty:
        return None, 0, None

    df["ts"] = pd.to_datetime(df.get("ts"), errors="coerce")
    df["iv"] = pd.to_numeric(df.get("iv"), errors="coerce")
    df = df.dropna(subset=["ts", "iv"])

    # Keep only sensible IV values
    df = df[(df["iv"] > 0.0) & (df["iv"] < 5.0)].copy()

    if df.empty:
        return None, 0, None

    df["day"] = df["ts"].dt.date
    df = df.sort_values("ts").groupby("day", as_index=False).tail(1)

    if window_days and len(df) > window_days:
        df = df.tail(int(window_days))

    iv_series = df["iv"].astype(float)
    if iv_series.empty:
        return None, 0, None

    ewma_span = int(max(2, ewma_span))
    iv_smooth = iv_series.ewm(span=ewma_span, adjust=False).mean()

    last = float(iv_smooth.iloc[-1])
    n = int(len(iv_smooth))

    pct = float((iv_smooth <= last).sum() / max(1, n))
    return pct, n, last
