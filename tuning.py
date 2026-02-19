"""tuning.py

Phase-6: metrics + walk-forward friendly scoring on *evaluated* reco rows.

This operates on the analyzer output CSV (evaluated recos),
so it does not refetch candles and does not hit rate limits.

We tune *filters* and *risk knobs* that can later be applied at scan time:
- min_score
- max_spread_pct
- min_ivp / max_ivp
- max_quote_age_s
- require_regime (optional)

This is intentionally simple to reduce overfitting: evaluate on rolling windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import math

import pandas as pd
import numpy as np


@dataclass
class Metrics:
    n: int
    win_rate: float
    avg_pnl_pct: float
    profit_factor: float
    sharpe: float
    sortino: float
    max_drawdown: float


def compute_metrics(df: pd.DataFrame) -> Metrics:
    if df is None or df.empty:
        return Metrics(n=0, win_rate=0.0, avg_pnl_pct=0.0, profit_factor=float("nan"),
                       sharpe=float("nan"), sortino=float("nan"), max_drawdown=0.0)

    rets = pd.to_numeric(df.get("pnl_pct"), errors="coerce").fillna(0.0).astype(float).values
    n = int(len(rets))
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate = float((rets > 0).mean()) if n else 0.0
    avg = float(np.mean(rets)) if n else 0.0

    pf = (wins.sum() / abs(losses.sum())) if losses.size and abs(losses.sum()) > 1e-12 else float("inf")

    sd = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    sharpe = (avg / sd * math.sqrt(n)) if sd > 1e-12 else float("nan")
    neg = rets[rets < 0]
    sd_down = float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0
    sortino = (avg / sd_down * math.sqrt(n)) if sd_down > 1e-12 else float("nan")

    eq = np.cumprod(1.0 + rets) if n else np.array([1.0])
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    max_dd = float(dd.min()) if dd.size else 0.0

    return Metrics(n=n, win_rate=win_rate, avg_pnl_pct=avg, profit_factor=float(pf),
                   sharpe=float(sharpe), sortino=float(sortino), max_drawdown=max_dd)


def apply_filters(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    x = df.copy()
    if x.empty:
        return x

    def _flt(col, op):
        if col not in x.columns:
            return
        v = params.get(col, None)
        if v is None:
            return
        try:
            val = float(v)
        except Exception:
            return
        s = pd.to_numeric(x[col], errors="coerce")
        if op == "ge":
            x.drop(x.index[s < val], inplace=True)
        elif op == "le":
            x.drop(x.index[s > val], inplace=True)

    _flt("score", "ge")
    if params.get("max_spread_pct") is not None and "spread_pct" in x.columns:
        try:
            mx = float(params["max_spread_pct"])
            s = pd.to_numeric(x["spread_pct"], errors="coerce")
            x.drop(x.index[s > mx], inplace=True)
        except Exception:
            pass

    if params.get("min_ivp") is not None and "iv_pct" in x.columns:
        try:
            mn = float(params["min_ivp"])
            s = pd.to_numeric(x["iv_pct"], errors="coerce")
            x.drop(x.index[s < mn], inplace=True)
        except Exception:
            pass

    if params.get("max_ivp") is not None and "iv_pct" in x.columns:
        try:
            mx = float(params["max_ivp"])
            s = pd.to_numeric(x["iv_pct"], errors="coerce")
            x.drop(x.index[s > mx], inplace=True)
        except Exception:
            pass

    if params.get("max_quote_age_s") is not None and "quote_age_s_max" in x.columns:
        try:
            mx = float(params["max_quote_age_s"])
            s = pd.to_numeric(x["quote_age_s_max"], errors="coerce")
            x.drop(x.index[s > mx], inplace=True)
        except Exception:
            pass

    req_reg = params.get("require_regime", None)
    if req_reg and "regime" in x.columns:
        x = x[x["regime"].astype(str).str.upper() == str(req_reg).upper()]

    return x
