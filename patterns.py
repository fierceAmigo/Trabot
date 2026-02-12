from dataclasses import dataclass
import pandas as pd
import numpy as np

from indicators import ema, rsi, atr, adx, zscore


@dataclass
class TradeIdea:
    symbol: str
    pattern: str
    side: str
    entry: float
    stop: float
    target: float
    confidence: float
    at_index: int


def _safe(v, default=0.0) -> float:
    try:
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _in_session(ts: pd.Timestamp) -> bool:
    # Simple India cash session-ish filter (demo-friendly)
    # Keep 09:30 to 15:15
    h, m = ts.hour, ts.minute
    mins = h * 60 + m
    return (9 * 60 + 30) <= mins <= (15 * 60 + 15)


def _atr_multipliers(regime: str):
    """
    Adaptive stops/targets based on volatility regime.
    In LOW_VOL, widen stops a bit and reduce targets so noise doesn't kill trades.
    """
    if regime == "LOW_VOL":
        return 1.8, 2.0   # stop_mult, target_mult
    if regime == "MID_VOL":
        return 1.4, 2.3
    return 1.3, 2.6      # HIGH_VOL


def _donchian_breakout(df: pd.DataFrame, i: int, regime: str, lookback: int = 20):
    if i - lookback < 1:
        return None

    hh = df["high"].iloc[i - lookback:i].max()
    ll = df["low"].iloc[i - lookback:i].min()
    close = _safe(df["close"].iloc[i])

    a = _safe(atr(df, 14).iloc[i], 1.0)
    stop_m, tgt_m = _atr_multipliers(regime)

    if close > hh:
        entry = close
        stop = entry - stop_m * a
        target = entry + tgt_m * a
        conf = min(1.0, max(0.2, (close - hh) / (a + 1e-9)))
        return ("breakout", "LONG", entry, stop, target, conf)

    if close < ll:
        entry = close
        stop = entry + stop_m * a
        target = entry - tgt_m * a
        conf = min(1.0, max(0.2, (ll - close) / (a + 1e-9)))
        return ("breakout", "SHORT", entry, stop, target, conf)

    return None


def _pullback_trend(df: pd.DataFrame, i: int, regime: str, adx_min: float = 18.0):
    # In LOW_VOL regime, trend pullbacks often get chopped—skip them entirely
    if regime == "LOW_VOL":
        return None

    e20 = ema(df["close"], 20)
    e50 = ema(df["close"], 50)
    r = rsi(df["close"], 14)
    a = atr(df, 14)
    ax = adx(df, 14)

    close = _safe(df["close"].iloc[i])
    e20i = _safe(e20.iloc[i], close)
    e50i = _safe(e50.iloc[i], close)
    ri = _safe(r.iloc[i], 50)
    ai = _safe(a.iloc[i], 1.0)
    adxi = _safe(ax.iloc[i], 0)

    if adxi < adx_min:
        return None

    stop_m, tgt_m = _atr_multipliers(regime)

    if e20i > e50i and close > e20i and ri > 52:
        entry = close
        stop = entry - stop_m * ai
        target = entry + tgt_m * ai
        conf = min(1.0, max(0.2, (ri - 50) / 20))
        return ("pullback_trend", "LONG", entry, stop, target, conf)

    if e20i < e50i and close < e20i and ri < 48:
        entry = close
        stop = entry + stop_m * ai
        target = entry - tgt_m * ai
        conf = min(1.0, max(0.2, (50 - ri) / 20))
        return ("pullback_trend", "SHORT", entry, stop, target, conf)

    return None


def _bollinger_mean_reversion(df: pd.DataFrame, i: int, regime: str, window: int = 20, num_std: float = 2.0):
    if i < window:
        return None

    close_series = df["close"]
    mid = close_series.rolling(window).mean()
    std = close_series.rolling(window).std().replace(0, np.nan)
    upper = mid + num_std * std
    lower = mid - num_std * std

    close = _safe(close_series.iloc[i])
    mid_i = _safe(mid.iloc[i], close)
    upper_i = _safe(upper.iloc[i], close)
    lower_i = _safe(lower.iloc[i], close)

    a = _safe(atr(df, 14).iloc[i], 1.0)

    # In LOW_VOL, mean reversion tends to work better than trend-following
    stop_m, tgt_m = _atr_multipliers("MID_VOL" if regime == "LOW_VOL" else regime)

    if close <= lower_i:
        entry = close
        stop = entry - stop_m * a
        target = mid_i  # mean reversion target stays mid
        conf = 0.60 if regime == "LOW_VOL" else 0.55
        return ("mean_reversion", "LONG", entry, stop, target, conf)

    if close >= upper_i:
        entry = close
        stop = entry + stop_m * a
        target = mid_i
        conf = 0.60 if regime == "LOW_VOL" else 0.55
        return ("mean_reversion", "SHORT", entry, stop, target, conf)

    return None


def _volatility_squeeze(df: pd.DataFrame, i: int, regime: str):
    close = df["close"]
    if i < 120:
        return None

    # In LOW_VOL, squeezes are common but direction is noisy -> require stronger squeeze
    squeeze_threshold = -1.3 if regime == "LOW_VOL" else -1.0

    mid = close.rolling(20).mean()
    std = close.rolling(20).std().replace(0, np.nan)
    upper = mid + 2 * std
    lower = mid - 2 * std
    width = (upper - lower) / close.replace(0, np.nan)

    w_z = zscore(width.fillna(0), 80)
    wz = _safe(w_z.iloc[i], 0.0)

    a = _safe(atr(df, 14).iloc[i], 1.0)
    c = _safe(close.iloc[i])
    m = _safe(mid.iloc[i], c)

    stop_m, tgt_m = _atr_multipliers(regime)

    if wz < squeeze_threshold:
        if c > m:
            entry = c
            stop = entry - stop_m * a
            target = entry + tgt_m * a
            conf = min(0.85, max(0.3, (-wz) / 3))
            return ("volatility_squeeze", "LONG", entry, stop, target, conf)
        if c < m:
            entry = c
            stop = entry + stop_m * a
            target = entry - tgt_m * a
            conf = min(0.85, max(0.3, (-wz) / 3))
            return ("volatility_squeeze", "SHORT", entry, stop, target, conf)

    return None


def generate_ideas(df: pd.DataFrame, symbol: str, scan_last_bars: int = 200, regime: str = "MID_VOL"):
    ideas = []
    n = len(df)
    start = max(0, n - scan_last_bars)

    for i in range(start, n):
        ts = df["datetime"].iloc[i]
        if not _in_session(ts):
            continue

        for fn in (_donchian_breakout, _pullback_trend, _bollinger_mean_reversion, _volatility_squeeze):
            out = fn(df, i, regime)
            if out:
                pattern, side, entry, stop, target, conf = out
                ideas.append(
                    TradeIdea(
                        symbol=symbol,
                        pattern=pattern,
                        side=side,
                        entry=float(entry),
                        stop=float(stop),
                        target=float(target),
                        confidence=float(conf),
                        at_index=i,
                    )
                )

    return ideas
