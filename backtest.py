from __future__ import annotations

from typing import Optional, Dict, Any, List, Literal
import pandas as pd
import numpy as np


Side = Literal["LONG", "SHORT"]


def _get_col(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Returns a column case-insensitively.
    Supports 'open'/'Open', etc.
    """
    if name in df.columns:
        return df[name]
    # case-insensitive lookup
    lower_map = {c.lower(): c for c in df.columns}
    key = name.lower()
    if key in lower_map:
        return df[lower_map[key]]
    raise KeyError(f"Missing column '{name}' (or '{name.capitalize()}'). Columns={list(df.columns)}")


def _simulate_forward(
    df: pd.DataFrame,
    entry_i: int,
    side: Side,
    stop: float,
    target: float,
    max_hold_bars: int,
) -> Optional[Dict[str, Any]]:
    """
    Simple forward simulator:
      entry at next bar open
      exit when stop/target hit, else time-exit after max_hold_bars
    Returns a dict with R-multiple.
    """
    if entry_i >= len(df):
        return None

    o = _get_col(df, "open")
    h = _get_col(df, "high")
    l = _get_col(df, "low")
    c = _get_col(df, "close")

    entry = float(o.iloc[entry_i])
    entry_time = df.index[entry_i]

    # Risk per trade (denominator for R)
    if side == "LONG":
        risk = abs(entry - float(stop))
    else:
        risk = abs(float(stop) - entry)

    # avoid divide by zero / nonsense
    if risk <= 1e-9:
        return None

    end_i = min(len(df) - 1, entry_i + max_hold_bars)

    outcome = "time"
    exit_price = float(c.iloc[end_i])
    exit_time = df.index[end_i]

    for i in range(entry_i, end_i + 1):
        hi = float(h.iloc[i])
        lo = float(l.iloc[i])

        if side == "LONG":
            # Worst-case assumption: if both hit same bar, assume stop first
            if lo <= stop:
                outcome = "sl"
                exit_price = float(stop)
                exit_time = df.index[i]
                break
            if hi >= target:
                outcome = "tp"
                exit_price = float(target)
                exit_time = df.index[i]
                break
        else:  # SHORT
            if hi >= stop:
                outcome = "sl"
                exit_price = float(stop)
                exit_time = df.index[i]
                break
            if lo <= target:
                outcome = "tp"
                exit_price = float(target)
                exit_time = df.index[i]
                break

    if side == "LONG":
        r = (exit_price - entry) / risk
    else:
        r = (entry - exit_price) / risk

    return {
        "side": side,
        "entry_time": str(entry_time),
        "exit_time": str(exit_time),
        "entry": float(entry),
        "exit": float(exit_price),
        "stop": float(stop),
        "target": float(target),
        "outcome": outcome,  # tp/sl/time
        "r": float(r),
    }


def summarize(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "win_rate_%": 0.0,
            "avg_r": 0.0,
            "median_r": 0.0,
            "best_r": 0.0,
            "worst_r": 0.0,
            "tp": 0,
            "sl": 0,
            "time": 0,
        }

    rs = np.array([t["r"] for t in trades], dtype=float)
    wins = float((rs > 0).mean() * 100.0)

    tp = sum(1 for t in trades if t.get("outcome") == "tp")
    sl = sum(1 for t in trades if t.get("outcome") == "sl")
    time = sum(1 for t in trades if t.get("outcome") == "time")

    return {
        "trades": len(trades),
        "win_rate_%": round(wins, 2),
        "avg_r": round(float(rs.mean()), 3),
        "median_r": round(float(np.median(rs)), 3),
        "best_r": round(float(rs.max()), 3),
        "worst_r": round(float(rs.min()), 3),
        "tp": tp,
        "sl": sl,
        "time": time,
    }
