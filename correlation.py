"""correlation.py

Phase-4: correlation-aware portfolio caps.

We approximate correlation using spot (cash) daily returns of underlyings.
This is a *risk control*, not a trading signal.

Implementation:
- fetch daily candles via market_data
- compute rolling correlation on close-to-close returns

Feature flag:
  TRABOT_PORTFOLIO_CORR_ENABLE=1
Env:
  TRABOT_PORTFOLIO_MAX_CORR=0.85
  TRABOT_PORTFOLIO_CORR_LOOKBACK_DAYS=120
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd

from market_data import fetch_candles_for_symbol


LOOKBACK_DAYS = int(os.getenv("TRABOT_PORTFOLIO_CORR_LOOKBACK_DAYS", "120"))
MAX_CORR = float(os.getenv("TRABOT_PORTFOLIO_MAX_CORR", "0.85"))


@lru_cache(maxsize=1024)
def _returns_series(kite_symbol: str) -> pd.Series:
    df = fetch_candles_for_symbol(kite_symbol, interval="day", lookback_days=LOOKBACK_DAYS)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df["close"], errors="coerce").dropna()
    r = s.pct_change().dropna()
    r.name = kite_symbol
    return r


def max_corr_with_portfolio(candidate_kite_symbol: str, portfolio_kite_symbols: List[str]) -> Tuple[Optional[float], Optional[str]]:
    if not portfolio_kite_symbols:
        return None, None
    r0 = _returns_series(candidate_kite_symbol)
    if r0.empty:
        return None, None
    best = None
    best_sym = None
    for sym in portfolio_kite_symbols:
        r1 = _returns_series(sym)
        if r1.empty:
            continue
        x = pd.concat([r0, r1], axis=1, join="inner").dropna()
        if len(x) < 20:
            continue
        c = float(x.corr().iloc[0, 1])
        if best is None or abs(c) > abs(best):
            best = c
            best_sym = sym
    return best, best_sym
