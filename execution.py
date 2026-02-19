"""execution.py

Phase-5: execution realism primitives.

We do not have per-candle bid/ask series; we approximate with spread_pct.

Fill models supported:
- mid:      no execution penalty (optimistic)
- mid_k:    k*spread penalty
- bid/ask:  half-spread penalty (good default when you only know mid + spread)
- realistic:alias of mid_k
- pessimistic: full-spread penalty
- optimistic: alias of mid

Model:
- entry_fill(value) = value + k_eff*spread*abs(value)   (worse for trader, debit higher / credit less)
- exit_fill(value)  = value - k_eff*spread*abs(value)   (worse for trader, reduces proceeds / increases cost)
- for stop/target detection, use executable extrema:
    hi_exec = hi - k_eff*spread*abs(hi)
    lo_exec = lo - k_eff*spread*abs(lo)

This moves trigger detection closer to bid/ask reality without needing bid/ask candles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class FillConfig:
    model: str = "realistic"   # mid|mid_k|bid|ask|optimistic|realistic|pessimistic
    k: float = 0.25            # used for mid_k/realistic
    spread_pct: float = 0.0    # 0..0.80


def _k_eff(model: str, k: float) -> float:
    """Map a fill model name to an effective k in [0..1]."""
    m = (model or "realistic").strip().lower()
    if m in ("mid", "optimistic"):
        return 0.0
    if m in ("pessimistic",):
        return 1.0
    if m in ("bid", "ask"):
        return 0.5
    # realistic / mid_k
    try:
        return max(0.0, min(float(k), 1.0))
    except Exception:
        return 0.25


def clamp_spread(spread_pct: float) -> float:
    try:
        sp = float(spread_pct or 0.0)
    except Exception:
        sp = 0.0
    return max(0.0, min(sp, 0.80))


def entry_fill(value: float, cfg: FillConfig) -> float:
    sp = clamp_spread(cfg.spread_pct)
    k = _k_eff(cfg.model, cfg.k)
    return float(value) + (k * sp * abs(float(value)))


def exit_fill(value: float, cfg: FillConfig) -> float:
    sp = clamp_spread(cfg.spread_pct)
    k = _k_eff(cfg.model, cfg.k)
    return float(value) - (k * sp * abs(float(value)))


def exec_extrema(hi: float, lo: float, cfg: FillConfig) -> Tuple[float, float]:
    """Return (hi_exec, lo_exec) after execution penalty."""
    sp = clamp_spread(cfg.spread_pct)
    k = _k_eff(cfg.model, cfg.k)
    hi_exec = float(hi) - (k * sp * abs(float(hi)))
    lo_exec = float(lo) - (k * sp * abs(float(lo)))
    return hi_exec, lo_exec
