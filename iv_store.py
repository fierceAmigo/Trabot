"""
iv_store.py

Stores IV snapshots to data/iv_history.csv and computes IV percentile
using a rolling 30 trading-day window with EWMA smoothing.
"""

from __future__ import annotations
import os
import pandas as pd
from datetime import datetime

IV_PATH = "data/iv_history.csv"

def ensure_dir():
    os.makedirs("data", exist_ok=True)

def append_iv_snapshot(row: dict):
    ensure_dir()
    df_new = pd.DataFrame([row])

    if os.path.exists(IV_PATH):
        df_old = pd.read_csv(IV_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(IV_PATH, index=False)

def iv_percentile(underlying: str, window_days: int = 30, ewma_span: int = 10):
    """
    Returns: (pct, n_samples, last_iv)
      pct in [0,1]
    Uses last snapshot per day to avoid intraday noise.
    """
    if not os.path.exists(IV_PATH):
        return None, 0, None

    df = pd.read_csv(IV_PATH)
    if df.empty:
        return None, 0, None

    df = df[df["underlying"].astype(str).str.upper() == underlying.upper()].copy()
    if df.empty:
        return None, 0, None

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "iv"])

    if df.empty:
        return None, 0, None

    df["day"] = df["ts"].dt.date
    df = df.sort_values("ts").groupby("day", as_index=False).tail(1)

    if len(df) > window_days:
        df = df.tail(window_days)

    iv_series = df["iv"].astype(float)
    iv_smooth = iv_series.ewm(span=ewma_span, adjust=False).mean()

    last = float(iv_smooth.iloc[-1])
    n = len(iv_smooth)
    pct = float((iv_smooth <= last).sum() / max(1, n))

    return pct, n, last
