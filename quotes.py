"""quotes.py

Phase-5: quote caching + throttled batch quote fetch.

Kite's quote/ltp calls can rate-limit quickly when scanning large universes.
We:
- cache quotes for a small TTL (seconds)
- fetch in batches and reuse within a scan loop
- expose bid/ask/mid/ltp/spread_pct + quote_ts (if present)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from kite_client import kite_quote_safe
from stats import inc

TTL_SEC = int(os.getenv("TRABOT_QUOTE_TTL_SEC", "3"))


@dataclass
class Quote:
    ts: float
    data: Dict[str, Any]


_CACHE: Dict[str, Quote] = {}


def _fresh(q: Quote) -> bool:
    return (time.time() - float(q.ts)) <= float(TTL_SEC)


def get_quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for s in symbols:
        q = _CACHE.get(s)
        if q and _fresh(q):
            inc("quote_cache_hits", 1)
            out[s] = q.data
        else:
            inc("quote_cache_misses", 1)
            missing.append(s)

    if missing:
        fetched = kite_quote_safe(missing)
        now = time.time()
        for s in missing:
            d = fetched.get(s)
            if d is None:
                continue
            _CACHE[s] = Quote(ts=now, data=d)
            out[s] = d
    return out


def quote_fields(q: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Kite quote dict to our stable fields."""
    bid = None
    ask = None
    ltp = None
    try:
        depth = q.get("depth") or {}
        buy = (depth.get("buy") or [])
        sell = (depth.get("sell") or [])
        if buy:
            bid = float(buy[0].get("price"))
        if sell:
            ask = float(sell[0].get("price"))
    except Exception:
        pass
    try:
        ltp = float(q.get("last_price"))
    except Exception:
        pass
    mid = None
    spread_pct = None
    try:
        if bid is not None and ask is not None:
            mid = (float(bid) + float(ask)) / 2.0
            if mid > 0:
                spread_pct = (float(ask) - float(bid)) / float(mid)
    except Exception:
        pass

    # quote timestamp (if present)
    qt = q.get("timestamp") or q.get("last_trade_time") or None
    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "ltp": ltp,
        "spread_pct": spread_pct,
        "quote_ts": str(qt) if qt is not None else "",
    }
