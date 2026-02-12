from __future__ import annotations

import datetime as dt
import os
import time
from typing import Tuple, List
from functools import lru_cache

import pandas as pd

from kite_client import get_kite


# Cache for NSE instruments (fallback if ltp() doesn't return instrument_token)
NSE_INSTRUMENTS_CACHE = os.path.join("data", "kite_instruments_NSE.csv")

# Candle cache for historical data
CANDLE_CACHE_DIR = os.path.join("data", "candle_cache")
DEFAULT_CACHE_TTL_MIN = int(os.getenv("TRABOT_CACHE_TTL_MIN", "5"))

# Map your interval -> Kite historical interval
INTERVAL_MAP = {
    "1m": "minute",
    "minute": "minute",
    "3m": "3minute",
    "3minute": "3minute",
    "5m": "5minute",
    "5minute": "5minute",
    "10m": "10minute",
    "10minute": "10minute",
    "15m": "15minute",
    "15minute": "15minute",
    "30m": "30minute",
    "30minute": "30minute",
    "60m": "60minute",
    "1h": "60minute",
    "60minute": "60minute",
    "1d": "day",
    "day": "day",
}

# Conservative chunk sizes (days) by interval
MAX_DAYS_PER_REQ = {
    "minute": 30,
    "3minute": 90,
    "5minute": 90,
    "10minute": 90,
    "15minute": 180,
    "30minute": 180,
    "60minute": 365,
    "day": 2000,
}


def _to_kite_interval(interval: str) -> str:
    k = INTERVAL_MAP.get(interval)
    if not k:
        raise RuntimeError(f"Unsupported interval '{interval}'. Use one of: {sorted(INTERVAL_MAP.keys())}")
    return k


def _parse_kite_symbol(symbol: str) -> tuple[str, str]:
    if ":" not in symbol:
        raise RuntimeError(f"Expected Kite symbol like 'NSE:NIFTY 50', got '{symbol}'")
    ex, ts = symbol.split(":", 1)
    return ex.strip(), ts.strip()


def _load_or_fetch_instruments(exchange: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(NSE_INSTRUMENTS_CACHE), exist_ok=True)

    if os.path.exists(NSE_INSTRUMENTS_CACHE):
        try:
            df = pd.read_csv(NSE_INSTRUMENTS_CACHE)
            if not df.empty:
                return df
        except Exception:
            pass

    kite = get_kite()
    inst = kite.instruments(exchange)  # list[dict]
    df = pd.DataFrame(inst)
    df.to_csv(NSE_INSTRUMENTS_CACHE, index=False)
    return df


def _resolve_instrument_token(symbol: str) -> int:
    """
    Best path: kite.ltp often includes instrument_token.
    Fallback: lookup in instruments master (cached).
    """
    kite = get_kite()

    # Try LTP first
    try:
        ltp = kite.ltp([symbol]).get(symbol) or {}
        tok = ltp.get("instrument_token")
        if tok is not None:
            return int(tok)
    except Exception:
        pass

    # Fallback: instruments lookup
    exchange, tradingsymbol = _parse_kite_symbol(symbol)
    inst = _load_or_fetch_instruments(exchange)

    if "tradingsymbol" not in inst.columns or "instrument_token" not in inst.columns:
        raise RuntimeError(f"Unexpected instruments format. Columns={list(inst.columns)}")

    hit = inst[inst["tradingsymbol"].astype(str) == tradingsymbol].copy()
    if hit.empty:
        raise RuntimeError(f"Could not find '{symbol}' in {exchange} instruments list/cache.")

    # Prefer INDICES if available
    if "segment" in hit.columns:
        pref = hit[hit["segment"].astype(str).str.contains("INDICES", na=False)]
        if not pref.empty:
            hit = pref

    return int(hit.iloc[0]["instrument_token"])


@lru_cache(maxsize=2048)
def resolve_nse_token_cached(symbol: str) -> int:
    """Compatibility helper used by some scanners."""
    return _resolve_instrument_token(symbol)


def _fetch_historical_chunked(token: int, start: dt.datetime, end: dt.datetime, kite_interval: str) -> List[dict]:
    kite = get_kite()
    max_days = MAX_DAYS_PER_REQ.get(kite_interval, 30)

    out: List[dict] = []
    cur = start

    while cur < end:
        cur_end = min(end, cur + dt.timedelta(days=max_days))
        rows = kite.historical_data(
            instrument_token=token,
            from_date=cur,
            to_date=cur_end,
            interval=kite_interval,
            continuous=False,
            oi=False,
        ) or []
        out.extend(rows)
        cur = cur_end + dt.timedelta(seconds=1)

    return out


def fetch_history(symbol: str, lookback_days: int = 35, interval: str = "15m") -> Tuple[pd.DataFrame, str]:
    """
    Fetch OHLCV from Kite historical candles for a Kite symbol like 'NSE:NIFTY 50'.
    Returns: (df, used_interval)
    df columns: open, high, low, close, volume with Datetime index.
    """
    kite_interval = _to_kite_interval(interval)

    token = _resolve_instrument_token(symbol)

    end = dt.datetime.now()
    start = end - dt.timedelta(days=int(lookback_days))

    rows = _fetch_historical_chunked(token, start, end, kite_interval)
    if not rows:
        raise RuntimeError("Kite historical_data returned no candles (holiday/weekend, or symbol/permission issue).")

    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        raise RuntimeError(f"Unexpected historical_data response. Columns={list(df.columns)}")

    df = df.rename(columns={"date": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").drop_duplicates(subset=["Datetime"]).set_index("Datetime")

    # Ensure standard columns exist (Kite uses lowercase)
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing '{c}' in candles. Columns={list(df.columns)}")

    if "volume" not in df.columns:
        df["volume"] = 0

    df = df[["open", "high", "low", "close", "volume"]].copy()
    return df, interval


def _cache_path(symbol: str, interval: str) -> str:
    safe = symbol.replace(":", "_").replace("/", "_").replace(" ", "_")
    return os.path.join(CANDLE_CACHE_DIR, f"{safe}__{interval}.csv")


def fetch_history_cached(
    symbol: str,
    lookback_days: int = 35,
    interval: str = "15m",
    ttl_minutes: int = DEFAULT_CACHE_TTL_MIN,
) -> Tuple[pd.DataFrame, str]:
    """
    Drop-in cached variant expected by some scripts.
    Returns the same as fetch_history: (df, used_interval).
    """
    os.makedirs(CANDLE_CACHE_DIR, exist_ok=True)
    fp = _cache_path(symbol, interval)

    if os.path.exists(fp):
        age_sec = time.time() - os.path.getmtime(fp)
        if age_sec <= ttl_minutes * 60:
            try:
                df = pd.read_csv(fp)
                if "Datetime" in df.columns:
                    df["Datetime"] = pd.to_datetime(df["Datetime"])
                    df = df.set_index("Datetime")
                else:
                    # fallback: assume first col is datetime
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    df = df.set_index(df.columns[0])

                df.columns = [c.lower() for c in df.columns]
                need = {"open", "high", "low", "close", "volume"}
                if need.issubset(set(df.columns)) and not df.empty:
                    return df[["open", "high", "low", "close", "volume"]].copy(), interval
            except Exception:
                # If cache is corrupted, fall through to fresh fetch
                pass

    df, used_interval = fetch_history(symbol, lookback_days=lookback_days, interval=interval)
    out = df.copy()
    out.index.name = "Datetime"
    out.to_csv(fp)
    return df, used_interval
