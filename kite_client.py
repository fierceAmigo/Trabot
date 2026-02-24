from __future__ import annotations

import os
import time
import random
from typing import Any, Dict, List

from dotenv import load_dotenv
from kiteconnect import KiteConnect

from throttle import TokenBucket
from stats import observe_api, inc


# Global limiter (best-effort)
_KITE_RPS = float(os.getenv("TRABOT_KITE_RPS", "3.0"))
_KITE_BURST = int(os.getenv("TRABOT_KITE_BURST", "6"))
_LIMITER = TokenBucket(rate_per_sec=_KITE_RPS, burst=_KITE_BURST)


def get_kite() -> KiteConnect:
    """Creates an authenticated KiteConnect client using .env values.

    Requires:
      KITE_API_KEY
      KITE_ACCESS_TOKEN
    """
    load_dotenv()

    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not api_key:
        raise RuntimeError("Missing KITE_API_KEY in .env")
    if not access_token:
        raise RuntimeError("Missing KITE_ACCESS_TOKEN in .env. Run: python kite_login.py")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def _is_429(err: Exception) -> bool:
    s = str(err).lower()
    return ("too many requests" in s) or ("429" in s)


def kite_call(fn, *args, **kwargs):
    """Throttle + retry/backoff wrapper for Kite API calls."""
    max_tries = int(os.getenv("TRABOT_KITE_MAX_TRIES", "6"))
    base_sleep = float(os.getenv("TRABOT_KITE_BACKOFF_BASE", "0.8"))
    kite = get_kite()
    last = None
    retries = 0
    for i in range(max_tries):
        _LIMITER.acquire(1.0)
        t0 = time.time()
        try:
            out = fn(kite, *args, **kwargs)
            ms = (time.time() - t0) * 1000.0
            observe_api(ms, is_429=False, retries=retries)
            return out
        except Exception as e:
            last = e
            ms = (time.time() - t0) * 1000.0
            is429 = _is_429(e)
            observe_api(ms, is_429=is429, retries=0)
            if is429:
                inc("api_429", 1)
            # backoff
            retries += 1
            sleep = base_sleep * (2 ** i)
            sleep = min(20.0, sleep) + random.uniform(0, 0.25)
            time.sleep(sleep)
            continue
    raise last if last else RuntimeError("kite_call failed")


def kite_quote_safe(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Batch quote (best-effort). Returns dict keyed by symbol.

    Kite sometimes limits the number of symbols per request; we chunk safely.
    """
    if not symbols:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    def _q(kite, syms):
        return kite.quote(syms)
    try:
        for i in range(0, len(symbols), 150):
            chunk = symbols[i:i+150]
            part = kite_call(_q, chunk) or {}
            out.update(part)
        return out
    except Exception:
        return out


def kite_ltp_safe(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    if not symbols:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    def _q(kite, syms):
        return kite.ltp(syms)
    try:
        for i in range(0, len(symbols), 200):
            chunk = symbols[i:i+200]
            part = kite_call(_q, chunk) or {}
            out.update(part)
        return out
    except Exception:
        return out


def kite_historical_safe(instrument_token: int, from_dt, to_dt, interval: str, continuous: bool = False, oi: bool = False):
    def _h(kite, tok, f, t, iv, cont, oi):
        return kite.historical_data(tok, f, t, iv, continuous=cont, oi=oi)
    return kite_call(_h, instrument_token, from_dt, to_dt, interval, continuous, oi)


def kite_positions_safe() -> Dict[str, Any]:
    def _p(kite):
        return kite.positions()
    try:
        return kite_call(_p) or {}
    except Exception:
        return {}
