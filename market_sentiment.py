"""
market_sentiment.py

Market sentiment (India / NSE) from *only* Kite-available data.

Why this exists:
  Your scanners already score underlyings using price action + IV regime + Greeks.
  But options performance is heavily regime/sentiment dependent.

This module builds a light "market context" snapshot using:
  - INDIA VIX level + EWMA-smoothed 30D percentile
  - Index option-chain aggregates (PCR from OI/Volume)
  - Simple skew proxy (OTM Put IV - OTM Call IV)
  - OI "walls" (largest OI strikes as support/resistance)
  - Index trend direction (using your existing strategy.compute_signal)

It appends snapshots to: data/market_sentiment_history.csv

Notes / limitations:
  - No external news feed. We treat VIX spikes + skew + PCR shifts as a proxy.
  - OI is point-in-time; for *change* we rely on snapshot history.
  - Skew here is a *proxy* using strikes ~4 steps away from ATM.

Educational tool only. Not financial advice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from kite_chain import get_kite_chain_slice
from market_data import fetch_history
from strategy import compute_signal
from iv_greeks import implied_volatility, time_to_expiry_years


SENTIMENT_HISTORY_PATH = Path(os.environ.get("TRABOT_SENTIMENT_HISTORY", "data/market_sentiment_history.csv"))

# These are the only global symbols we assume exist in Kite.
INDIA_VIX_SYMBOL = os.environ.get("TRABOT_VIX_SYMBOL", "NSE:INDIA VIX")


def _pct_rank_last(values: pd.Series) -> Optional[float]:
    """Percentile rank of the *last* value inside the series, in [0,1]."""
    if values is None:
        return None
    s = pd.to_numeric(values, errors="coerce").dropna()
    if len(s) < 5:
        return None
    last = float(s.iloc[-1])
    pct = float((s < last).sum()) / float(len(s))  # strictly-less avoids duplicate->always 1.0
    return max(0.0, min(1.0, pct))


def _mid_price(bid: Any, ask: Any, ltp: Any) -> Optional[float]:
    b = None
    a = None
    try:
        if bid is not None:
            b = float(bid)
        if ask is not None:
            a = float(ask)
    except Exception:
        b = None
        a = None

    if b is not None and a is not None and b > 0 and a > 0:
        return (a + b) / 2.0

    try:
        x = float(ltp)
        if x == x and x > 0:
            return x
    except Exception:
        return None

    return None


def vix_snapshot(lookback_days: int = 90, ewma_span: int = 10, pct_window_days: int = 30) -> Dict[str, Any]:
    """Fetch INDIA VIX daily candles and compute an EWMA-smoothed percentile."""
    df, _ = fetch_history(INDIA_VIX_SYMBOL, lookback_days=lookback_days, interval="1d")
    df = df.copy()
    if df.empty:
        raise RuntimeError("No candles for INDIA VIX from Kite")

    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close.empty:
        raise RuntimeError("INDIA VIX candles missing close")

    ewma = close.ewm(span=int(ewma_span), adjust=False).mean()
    tail = ewma.tail(int(pct_window_days))
    pct = _pct_rank_last(tail)

    vix = float(close.iloc[-1])
    vix_ewma = float(ewma.iloc[-1])

    vix_chg_1d = None
    vix_chg_5d = None
    if len(close) >= 2:
        prev = float(close.iloc[-2])
        if prev > 0:
            vix_chg_1d = (vix - prev) / prev
    if len(close) >= 6:
        prev5 = float(close.iloc[-6])
        if prev5 > 0:
            vix_chg_5d = (vix - prev5) / prev5

    return {
        "vix": vix,
        "vix_ewma": vix_ewma,
        "vix_pct": pct,
        "vix_chg_1d": vix_chg_1d,
        "vix_chg_5d": vix_chg_5d,
    }


def _pick_strike_near(df_side: pd.DataFrame, strike_target: int) -> Optional[Dict[str, Any]]:
    if df_side is None or df_side.empty:
        return None
    tmp = df_side.copy()
    tmp["dist"] = (pd.to_numeric(tmp["strike"], errors="coerce") - float(strike_target)).abs()
    tmp = tmp.dropna(subset=["dist"]).sort_values("dist")
    if tmp.empty:
        return None
    return tmp.iloc[0].to_dict()


def chain_sentiment(
    underlying: str,
    spot_symbol: str,
    instruments_cache_path: str,
    strikes_around_atm: int = 12,
    skew_steps: int = 4,
) -> Dict[str, Any]:
    """Compute PCR + skew proxy + OI walls from a Kite chain slice."""
    chain = get_kite_chain_slice(
        underlying=underlying,
        kite_spot_symbol=spot_symbol,
        strike_step=0,
        strikes_around_atm=int(strikes_around_atm),
        cache_path=instruments_cache_path,
    )

    calls = chain.calls.copy() if chain.calls is not None else pd.DataFrame()
    puts = chain.puts.copy() if chain.puts is not None else pd.DataFrame()

    for df_ in (calls, puts):
        if not df_.empty:
            df_["oi"] = pd.to_numeric(df_.get("oi"), errors="coerce").fillna(0.0)
            df_["volume"] = pd.to_numeric(df_.get("volume"), errors="coerce").fillna(0.0)

    call_oi = float(calls["oi"].sum()) if not calls.empty else 0.0
    put_oi = float(puts["oi"].sum()) if not puts.empty else 0.0
    call_vol = float(calls["volume"].sum()) if not calls.empty else 0.0
    put_vol = float(puts["volume"].sum()) if not puts.empty else 0.0

    pcr_oi = (put_oi / call_oi) if call_oi > 0 else None
    pcr_vol = (put_vol / call_vol) if call_vol > 0 else None

    spot = float(chain.spot)
    atm = int(chain.atm)

    # Estimate step from available strikes
    strikes_all = sorted(
        set(
            pd.to_numeric(
                pd.concat([calls.get("strike"), puts.get("strike")]), errors="coerce"
            ).dropna().astype(int).tolist()
        )
    )
    step = 1
    if len(strikes_all) >= 3:
        diffs = pd.Series(strikes_all).diff().dropna()
        step = int(diffs.median()) if not diffs.empty else 1
        step = max(1, step)

    # Walls: max OI strikes (prefer above spot for calls, below spot for puts)
    call_wall = None
    call_wall_oi = None
    call_wall_dist_pct = None
    if not calls.empty:
        above = calls[calls["strike"].astype(int) >= int(spot)].copy()
        use = above if not above.empty else calls
        row = use.sort_values("oi", ascending=False).iloc[0]
        call_wall = int(row["strike"])
        call_wall_oi = float(row["oi"])
        call_wall_dist_pct = abs(call_wall - spot) / spot * 100.0 if spot > 0 else None

    put_wall = None
    put_wall_oi = None
    put_wall_dist_pct = None
    if not puts.empty:
        below = puts[puts["strike"].astype(int) <= int(spot)].copy()
        use = below if not below.empty else puts
        row = use.sort_values("oi", ascending=False).iloc[0]
        put_wall = int(row["strike"])
        put_wall_oi = float(row["oi"])
        put_wall_dist_pct = abs(spot - put_wall) / spot * 100.0 if spot > 0 else None

    # Skew proxy: IV(OTM put) - IV(OTM call) at ~skew_steps from ATM
    skew = None
    iv_put = None
    iv_call = None
    try:
        T = time_to_expiry_years(str(chain.expiry))
        put_row = _pick_strike_near(puts, atm - skew_steps * step)
        call_row = _pick_strike_near(calls, atm + skew_steps * step)
        if put_row and call_row and T is not None and T > 0:
            put_px = _mid_price(put_row.get("bid"), put_row.get("ask"), put_row.get("last_price"))
            call_px = _mid_price(call_row.get("bid"), call_row.get("ask"), call_row.get("last_price"))
            if put_px and call_px and put_px > 0 and call_px > 0:
                iv_put_val, okp = implied_volatility(put_px, spot, int(put_row["strike"]), T, 0.06, "PE")
                iv_call_val, okc = implied_volatility(call_px, spot, int(call_row["strike"]), T, 0.06, "CE")
                if okp and okc and iv_put_val and iv_call_val:
                    iv_put = float(iv_put_val)
                    iv_call = float(iv_call_val)
                    skew = float(iv_put - iv_call)
    except Exception:
        skew = None

    return {
        "spot": spot,
        "atm": atm,
        "pcr_oi": pcr_oi,
        "pcr_vol": pcr_vol,
        "skew": skew,
        "iv_put_otm": iv_put,
        "iv_call_otm": iv_call,
        "call_wall": call_wall,
        "call_wall_oi": call_wall_oi,
        "call_wall_dist_pct": call_wall_dist_pct,
        "put_wall": put_wall,
        "put_wall_oi": put_wall_oi,
        "put_wall_dist_pct": put_wall_dist_pct,
        "expiry": chain.expiry,
    }


def market_trend_snapshot(
    spot_symbol: str,
    lookback_days: int,
    interval: str,
    ema_fast: int,
    ema_slow: int,
    rsi_period: int,
    adx_period: int,
    adx_min: float,
    atr_period: int,
    stop_atr_mult: float,
    target_atr_mult: float,
) -> Dict[str, Any]:
    df, _ = fetch_history(spot_symbol, lookback_days=lookback_days, interval=interval)
    if df is None or df.empty:
        raise RuntimeError(f"No candles for {spot_symbol}")

    sig = compute_signal(
        df=df,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_period,
        adx_period=adx_period,
        adx_min=adx_min,
        atr_period=atr_period,
        stop_atr_mult=stop_atr_mult,
        target_atr_mult=target_atr_mult,
    )
    side = str(sig.side)
    adx = float(sig.metrics.get("adx", 0.0) or 0.0)
    atr_pct = float(sig.metrics.get("atr_pct", 0.0) or 0.0)
    return {
        "trend_side": side,
        "trend_adx": adx,
        "trend_atr_pct": atr_pct,
    }


@dataclass
class MarketContext:
    ts: str
    bias: str
    strength: float
    risk_off: bool
    vix: Optional[float] = None
    vix_pct: Optional[float] = None
    vix_chg_1d: Optional[float] = None
    vix_chg_5d: Optional[float] = None
    pcr_oi: Optional[float] = None
    pcr_vol: Optional[float] = None
    skew: Optional[float] = None
    call_wall_dist_pct: Optional[float] = None
    put_wall_dist_pct: Optional[float] = None
    trend_side: Optional[str] = None
    trend_adx: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "bias": self.bias,
            "strength": self.strength,
            "risk_off": self.risk_off,
            "vix": self.vix,
            "vix_pct": self.vix_pct,
            "vix_chg_1d": self.vix_chg_1d,
            "vix_chg_5d": self.vix_chg_5d,
            "pcr_oi": self.pcr_oi,
            "pcr_vol": self.pcr_vol,
            "skew": self.skew,
            "call_wall_dist_pct": self.call_wall_dist_pct,
            "put_wall_dist_pct": self.put_wall_dist_pct,
            "trend_side": self.trend_side,
            "trend_adx": self.trend_adx,
        }


def compute_market_context(
    instruments_cache_path: str,
    index_underlying: str = "NIFTY",
    index_spot_symbol: str = "NSE:NIFTY 50",
    strikes_around_atm: int = 12,
    skew_steps: int = 4,
    lookback_days: int = 30,
    interval: str = "60m",
    ema_fast: int = 20,
    ema_slow: int = 50,
    rsi_period: int = 14,
    adx_period: int = 14,
    adx_min: float = 18,
    atr_period: int = 14,
    stop_atr_mult: float = 1.5,
    target_atr_mult: float = 2.2,
) -> MarketContext:
    """Build one market context snapshot. Default uses NIFTY + INDIA VIX."""
    ts = datetime.now().isoformat(timespec="seconds")

    vix = vix_snapshot()
    ch = chain_sentiment(
        underlying=index_underlying,
        spot_symbol=index_spot_symbol,
        instruments_cache_path=instruments_cache_path,
        strikes_around_atm=strikes_around_atm,
        skew_steps=skew_steps,
    )
    tr = market_trend_snapshot(
        spot_symbol=index_spot_symbol,
        lookback_days=lookback_days,
        interval=interval,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_period,
        adx_period=adx_period,
        adx_min=adx_min,
        atr_period=atr_period,
        stop_atr_mult=stop_atr_mult,
        target_atr_mult=target_atr_mult,
    )

    score = 0.0
    tside = str(tr.get("trend_side") or "NO_TRADE")
    if tside == "LONG":
        score += 1.0
    elif tside == "SHORT":
        score -= 1.0

    pcr_oi = ch.get("pcr_oi")
    if pcr_oi is not None:
        if pcr_oi < 0.90:
            score += 0.5
        elif pcr_oi > 1.10:
            score -= 0.5

    skew = ch.get("skew")
    if skew is not None:
        if skew > 0.05:
            score -= 0.5
        elif skew < -0.02:
            score += 0.3

    vix_pct = vix.get("vix_pct")
    if vix_pct is not None:
        if vix_pct >= 0.70:
            score -= 0.6
        elif vix_pct <= 0.30:
            score += 0.3

    bias = "NEUTRAL"
    if score >= 0.8:
        bias = "BULLISH"
    elif score <= -0.8:
        bias = "BEARISH"

    strength = min(1.0, abs(score) / 1.6)

    vix_chg_1d = vix.get("vix_chg_1d")
    risk_off = False
    if vix_pct is not None and vix_pct >= 0.75:
        risk_off = True
    if vix_chg_1d is not None and vix_chg_1d >= 0.08:
        risk_off = True
    if skew is not None and skew >= 0.08:
        risk_off = True

    return MarketContext(
        ts=ts,
        bias=bias,
        strength=float(strength),
        risk_off=bool(risk_off),
        vix=float(vix.get("vix")) if vix.get("vix") is not None else None,
        vix_pct=float(vix_pct) if vix_pct is not None else None,
        vix_chg_1d=float(vix_chg_1d) if vix_chg_1d is not None else None,
        vix_chg_5d=float(vix.get("vix_chg_5d")) if vix.get("vix_chg_5d") is not None else None,
        pcr_oi=float(pcr_oi) if pcr_oi is not None else None,
        pcr_vol=float(ch.get("pcr_vol")) if ch.get("pcr_vol") is not None else None,
        skew=float(skew) if skew is not None else None,
        call_wall_dist_pct=float(ch.get("call_wall_dist_pct")) if ch.get("call_wall_dist_pct") is not None else None,
        put_wall_dist_pct=float(ch.get("put_wall_dist_pct")) if ch.get("put_wall_dist_pct") is not None else None,
        trend_side=tside,
        trend_adx=float(tr.get("trend_adx")) if tr.get("trend_adx") is not None else None,
    )


def append_market_context(ctx: MarketContext, path: Path = SENTIMENT_HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = ctx.to_dict()
    df = pd.DataFrame([row])
    if path.exists():
        try:
            df.to_csv(path, mode="a", header=False, index=False)
            return
        except Exception:
            pass
    df.to_csv(path, index=False)
