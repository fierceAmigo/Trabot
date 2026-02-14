"""indicators.py

Lightweight indicator library used across Trabot.

Fixes / improvements (v2.2.x):
- Works with either lowercase OHLC (open/high/low/close) or titlecase (Open/High/Low/Close).
- Adds missing zscore() used by patterns.py.

Educational tool only – not financial advice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return first matching column (case-insensitive). Raises KeyError if none."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise KeyError(f"Missing columns. Tried {candidates}. Have {list(df.columns)}")


def _ohlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (high, low, close) series using flexible column naming."""
    hi = df[_find_col(df, ['high', 'High', 'HIGH'])]
    lo = df[_find_col(df, ['low', 'Low', 'LOW'])]
    cl = df[_find_col(df, ['close', 'Close', 'CLOSE'])]
    return hi.astype(float), lo.astype(float), cl.astype(float)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    s = pd.to_numeric(series, errors='coerce')
    return s.ewm(span=int(period), adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI (simple rolling implementation)."""
    c = pd.to_numeric(close, errors='coerce')
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(int(period)).mean()
    loss = (-delta.clip(upper=0)).rolling(int(period)).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = _ohlc(df)
    pc = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - pc).abs(), (low - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(int(period)).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Welles Wilder ADX (simplified rolling)."""
    high, low, close = _ohlc(df)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat(
        [(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    atr_w = tr.rolling(int(period)).mean().replace(0, np.nan)

    plus_di = 100 * (plus_dm.rolling(int(period)).mean() / atr_w)
    minus_di = 100 * (minus_dm.rolling(int(period)).mean() / atr_w)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.rolling(int(period)).mean().fillna(0)


def zscore(series: pd.Series, window: int = 80) -> pd.Series:
    """Rolling z-score: (x - mean) / std."""
    s = pd.to_numeric(series, errors='coerce')
    w = int(max(2, window))
    m = s.rolling(w).mean()
    sd = s.rolling(w).std(ddof=0).replace(0, np.nan)
    return (s - m) / sd
