from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional
import pandas as pd
import numpy as np

Side = Literal["LONG", "SHORT", "NO_TRADE"]


@dataclass
class Signal:
    side: Side
    reason: str
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    metrics: Dict[str, float]


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts dataframes from yfinance or other sources that may have:
      Open/High/Low/Close/Adj Close/Volume/Datetime
    and normalizes to:
      open/high/low/close/volume  (lowercase)
    """
    df = df.copy()

    # If 'Datetime' exists, we keep it but don't require it.
    # Build a case-insensitive mapping.
    lower_map = {c.lower(): c for c in df.columns}

    rename = {}

    # OHLCV normalization
    for want in ["open", "high", "low", "close", "volume"]:
        if want in df.columns:
            continue
        if want in lower_map:
            rename[lower_map[want]] = want

    # Special case: sometimes only "Adj Close" exists (rare intraday)
    if "close" not in df.columns:
        if "close" in lower_map:
            rename[lower_map["close"]] = "close"
        elif "adj close" in lower_map:
            rename[lower_map["adj close"]] = "close"

    df = df.rename(columns=rename)

    return df


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    return atr


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / n, adjust=False).mean()

    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean()

    plus_di = 100 * (plus_dm_s / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm_s / atr.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx.fillna(0.0)


def compute_signal(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
    rsi_period: int = 14,
    adx_period: int = 14,
    adx_min: float = 18,
    atr_period: int = 14,
    stop_atr_mult: float = 1.5,
    target_atr_mult: float = 2.2,
) -> Signal:
    """
    Trend-follow filter:
      LONG  if EMAfast>EMAslow AND ADX>=adx_min AND RSI>=52
      SHORT if EMAfast<EMAslow AND ADX>=adx_min AND RSI<=48

    If NO_TRADE, we still attach a "watch" plan in metrics:
      watch_side, watch_trigger, watch_entry, watch_stop, watch_target
    """
    df = _normalize_ohlc(df)

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in market data. Columns={list(df.columns)}")

    close = df["close"]
    ema_f = _ema(close, ema_fast)
    ema_s = _ema(close, ema_slow)
    rsi = _rsi(close, rsi_period)
    adx = _adx(df, adx_period)
    atr = _atr(df, atr_period)

    last = df.index[-1]
    c = float(close.loc[last])
    ef = float(ema_f.loc[last])
    es = float(ema_s.loc[last])
    rv = float(rsi.loc[last])
    ax = float(adx.loc[last])
    av = float(atr.loc[last])
    atr_pct = (av / c) * 100 if c else 0.0

    trend_up = ef > es
    trend_down = ef < es
    strong = ax >= adx_min

    # Default watch info
    watch_side = "NONE"
    watch_trigger = ""
    watch_entry = None
    watch_stop = None
    watch_target = None

    buffer = 0.10 * av  # small nudge beyond current price
    if trend_up and strong:
        watch_side = "LONG"
        watch_trigger = "Trigger LONG when RSI closes >= 52 AND price closes above EMA_fast."
        watch_entry = c + buffer
        watch_stop = watch_entry - stop_atr_mult * av
        watch_target = watch_entry + target_atr_mult * av
    elif trend_down and strong:
        watch_side = "SHORT"
        watch_trigger = "Trigger SHORT when RSI closes <= 48 AND price closes below EMA_fast."
        watch_entry = c - buffer
        watch_stop = watch_entry + stop_atr_mult * av
        watch_target = watch_entry - target_atr_mult * av
    elif trend_up and not strong:
        watch_side = "LONG"
        watch_trigger = f"Trend UP but ADX<{adx_min:.0f}. Trigger LONG when ADX>={adx_min:.0f} AND RSI>=52."
        watch_entry = c + buffer
        watch_stop = watch_entry - stop_atr_mult * av
        watch_target = watch_entry + target_atr_mult * av
    elif trend_down and not strong:
        watch_side = "SHORT"
        watch_trigger = f"Trend DOWN but ADX<{adx_min:.0f}. Trigger SHORT when ADX>={adx_min:.0f} AND RSI<=48."
        watch_entry = c - buffer
        watch_stop = watch_entry + stop_atr_mult * av
        watch_target = watch_entry - target_atr_mult * av

    metrics = {
        "close": c,
        "ema_fast": ef,
        "ema_slow": es,
        "rsi": rv,
        "adx": ax,
        "atr": av,
        "atr_pct": atr_pct,

        "watch_side": watch_side,
        "watch_trigger": watch_trigger,
        "watch_entry": float(watch_entry) if watch_entry is not None else float("nan"),
        "watch_stop": float(watch_stop) if watch_stop is not None else float("nan"),
        "watch_target": float(watch_target) if watch_target is not None else float("nan"),
    }

    if trend_up and strong and rv >= 52:
        entry = c
        stop = entry - stop_atr_mult * av
        target = entry + target_atr_mult * av
        return Signal("LONG", "Uptrend (EMA) + strong (ADX) + RSI>=52", entry, stop, target, metrics)

    if trend_down and strong and rv <= 48:
        entry = c
        stop = entry + stop_atr_mult * av
        target = entry - target_atr_mult * av
        return Signal("SHORT", "Downtrend (EMA) + strong (ADX) + RSI<=48", entry, stop, target, metrics)

    reason = (
        "No setup (filters not met). | "
        + ("Trend=UP" if trend_up else "Trend=DOWN" if trend_down else "Trend=FLAT")
        + f" | ADX={ax:.1f} (min {adx_min})"
        + f" | RSI={rv:.1f} (need >=52 for LONG or <=48 for SHORT)"
    )

    return Signal("NO_TRADE", reason, None, None, None, metrics)
