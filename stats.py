"""stats.py

Lightweight stats collector shared across modules.
Used to populate run manifests and to debug rate-limits.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any
import time

_STATS: Dict[str, Any] = {
    "api_calls": 0,
    "api_retries": 0,
    "api_429": 0,
    "api_latency_ms_sum": 0.0,
    "api_latency_ms_count": 0,
    "candle_cache_hits": 0,
    "candle_cache_misses": 0,
    "quote_cache_hits": 0,
    "quote_cache_misses": 0,
}

def inc(key: str, n: int = 1) -> None:
    _STATS[key] = int(_STATS.get(key, 0)) + int(n)

def observe_ms(key_prefix: str, ms: float) -> None:
    inc(f"{key_prefix}_count", 1)
    _STATS[f"{key_prefix}_sum"] = float(_STATS.get(f"{key_prefix}_sum", 0.0)) + float(ms)

def observe_api(ms: float, *, is_429: bool = False, retries: int = 0) -> None:
    inc("api_calls", 1)
    if retries:
        inc("api_retries", retries)
    if is_429:
        inc("api_429", 1)
    _STATS["api_latency_ms_sum"] = float(_STATS.get("api_latency_ms_sum", 0.0)) + float(ms)
    _STATS["api_latency_ms_count"] = int(_STATS.get("api_latency_ms_count", 0)) + 1

def snapshot() -> Dict[str, Any]:
    # derive averages
    out = dict(_STATS)
    c = int(out.get("api_latency_ms_count", 0))
    out["api_latency_ms_avg"] = (float(out.get("api_latency_ms_sum", 0.0)) / c) if c else None
    return out
