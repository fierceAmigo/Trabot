from __future__ import annotations

"""market_data.py

Kite historical candle fetch with disk cache + backoff.

Phase-1 upgrade
- Enable disk caching/TTL to reduce "Too many requests" errors.
- Add retry/backoff around historical_data calls.

This module is intentionally minimal and used by scan_options_v22.py.
"""

import datetime as dt
import os
import time
from functools import lru_cache
from typing import List, Tuple

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
    """Load instruments master for an exchange with safe refresh/backoff."""

    os.makedirs(os.path.dirname(NSE_INSTRUMENTS_CACHE), exist_ok=True)

    refresh = (
        os.getenv("TRABOT_REFRESH_INSTRUMENTS", "0").strip().lower() in ("1", "true", "yes")
        or os.getenv("TRABOT_REFRESH_NSE_INSTRUMENTS", "0").strip().lower() in ("1", "true", "yes")
    )
    max_age_hours = int(os.getenv("TRABOT_INSTRUMENTS_MAX_AGE_HOURS", "24"))

    def _is_stale(path: str) -> bool:
        try:
            mtime = os.path.getmtime(path)
            age_hours = (time.time() - mtime) / 3600.0
            return age_hours > max_age_hours
        except Exception:
            return True

    if (not refresh) and os.path.exists(NSE_INSTRUMENTS_CACHE) and (not _is_stale(NSE_INSTRUMENTS_CACHE)):
        try:
            df = pd.read_csv(NSE_INSTRUMENTS_CACHE)
            if not df.empty:
                return df
        except Exception:
            pass

    kite = get_kite()
    last_err: Exception | None = None
    for i in range(6):
        try:
            inst = kite.instruments(exchange)
            df = pd.DataFrame(inst)
            if df.empty:
                raise RuntimeError(f"Kite instruments('{exchange}') returned empty")
            try:
                df.to_csv(NSE_INSTRUMENTS_CACHE, index=False)
            except Exception:
                pass
            return df
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "too many requests" in msg or "429" in msg:
                time.sleep(1.0 * (2**i))
                continue
            raise

    # If still rate-limited, fall back to cache if present
    if os.path.exists(NSE_INSTRUMENTS_CACHE):
        try:
            df = pd.read_csv(NSE_INSTRUMENTS_CACHE)
            if not df.empty:
                return df
        except Exception:
            pass

    raise RuntimeError(f"Failed to fetch instruments('{exchange}') (rate limited). Last error: {last_err}")


def _resolve_instrument_token(symbol: str) -> int:
    """Resolve instrument token for Kite symbol like 'NSE:NIFTY 50'."""

    kite = get_kite()

    # Try LTP first (often returns instrument_token)
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

        rows: List[dict] = []
        last_err: Exception | None = None
        for i in range(6):
            try:
                rows = kite.historical_data(
                    instrument_token=token,
                    from_date=cur,
                    to_date=cur_end,
                    interval=kite_interval,
                    continuous=False,
                    oi=False,
                ) or []
                last_err = None
                break
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "too many requests" in msg or "429" in msg:
                    time.sleep(0.7 * (2**i))
                    continue
                if "timeout" in msg or "tempor" in msg or "connection" in msg:
                    time.sleep(0.5 * (2**i))
                    continue
                raise

        if last_err is not None and not rows:
            raise RuntimeError(f"Kite historical_data failed after retries: {last_err}")

        out.extend(rows)
        cur = cur_end + dt.timedelta(seconds=1)

    return out


def fetch_history(symbol: str, lookback_days: int = 35, interval: str = "15m") -> Tuple[pd.DataFrame, str]:
    """Fetch OHLCV from Kite historical candles."""

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


def _read_cache(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")
    # Ensure standard shape
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[["open", "high", "low", "close", "volume"]].copy()


def fetch_history_cached(
    symbol: str,
    lookback_days: int = 35,
    interval: str = "15m",
    ttl_minutes: int = DEFAULT_CACHE_TTL_MIN,
) -> Tuple[pd.DataFrame, str]:
    """Fetch candles with a disk cache.

    Cache semantics:
    - Cache key: (symbol, interval)
    - Cache is considered valid if:
        - mtime <= ttl_minutes, and
        - covers at least the requested lookback window.
    """

    os.makedirs(CANDLE_CACHE_DIR, exist_ok=True)
    fp = _cache_path(symbol, interval)

    # Try cache
    try:
        if os.path.exists(fp):
            age_min = (time.time() - os.path.getmtime(fp)) / 60.0
            if ttl_minutes is None or age_min <= float(ttl_minutes):
                cached = _read_cache(fp)
                if not cached.empty:
                    end = dt.datetime.now()
                    start_need = end - dt.timedelta(days=int(lookback_days))
                    try:
                        idx_min = pd.to_datetime(cached.index.min()).to_pydatetime()
                    except Exception:
                        idx_min = None
                    if idx_min is None or idx_min <= start_need:
                        return cached, interval
    except Exception:
        # Cache read issues should not break runtime.
        pass

    # Fetch fresh
    df, used_interval = fetch_history(symbol, lookback_days=lookback_days, interval=interval)

    try:
        out = df.copy()
        out.index.name = "Datetime"
        out.to_csv(fp)
    except Exception:
        pass

    return df, used_interval
