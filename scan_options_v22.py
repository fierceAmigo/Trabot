# ===========================
# scan_options_v22.py
# SENTIMENT ENABLED VERSION
# ===========================

"""scan_options.py

Main full-universe options scanner (Kite-only) with V2.1 strategy upgrades
while keeping your existing core strengths:

Keeps:
  - Full NFO-OPT universe scan (≈200+ underlyings)
  - Underlying signal + watch-plan from strategy.compute_signal
  - HTF (60m) alignment penalty
  - Session factor
  - BOS detection
  - Strike selection via liquidity+spread+moneyness (strike_score)
  - Learned edge feature from backtest._simulate_forward

Adds (V2.1):
  - Daily IV snapshot -> EWMA smoothed rolling 30D percentile (iv_store)
  - BS Greeks for the picked contract (iv_greeks)
  - Expiry-week penalty (DTE<=3)
  - Replace approx-delta with BS delta in stop/target mapping
  - Regime label (TREND/CHOP/VOLATILE) driven primarily by IV percentile
  - Dynamic position sizing (max lots) via risk_caps.compute_max_lots

NEW (Sentiment overlay):
  - INDIA VIX percentile (fear gauge)
  - NIFTY 1H trend bias
  - NIFTY option-chain PCR + IV skew + OI walls
  - Score multiplier based on market bias/risk-off state
  - Append market sentiment history (data/market_sentiment_history.csv)

Outputs:
  data/options_scan_results.csv
  data/options_top{TOP_OVERALL}.csv

NOTE: Educational tool only. Not financial advice.
"""

import os
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    LOOKBACK_DAYS, INTERVAL,
    EMA_FAST, EMA_SLOW, RSI_PERIOD, ADX_PERIOD, ADX_MIN,
    ATR_PERIOD, STOP_ATR_MULT, TARGET_ATR_MULT,
    INSTRUMENTS_CACHE_PATH,
)

from kite_client import get_kite
from market_data import fetch_history
from strategy import compute_signal
from kite_chain import get_kite_chain_slice

from iv_greeks import implied_volatility, greeks, time_to_expiry_years
from iv_store import append_iv_snapshot, iv_percentile
from risk_caps import compute_max_lots


# =========================
# Runtime knobs (env vars)
# =========================
TOP_N = int(os.environ.get("TRABOT_TOP_N", "5"))
TOP_OVERALL = int(os.environ.get("TRABOT_TOP_OVERALL", "20"))
STRIKES_AROUND_ATM = int(os.environ.get("TRABOT_STRIKES_AROUND_ATM", "6"))

UNIVERSE_START = int(os.environ.get("TRABOT_UNIVERSE_START", "0"))
UNIVERSE_COUNT = os.environ.get("TRABOT_UNIVERSE_COUNT", "")
UNIVERSE_COUNT = int(UNIVERSE_COUNT) if UNIVERSE_COUNT.strip() else None

# ✅ Default TTL lowered to 5 minutes (realtime-friendly)
CACHE_TTL_MINUTES = int(os.environ.get("TRABOT_CACHE_TTL_MIN", "5"))
CANDLE_CACHE_DIR = Path("data/candle_cache")

SLEEP_BETWEEN_SYMBOLS = float(os.environ.get("TRABOT_SLEEP", "0.05"))

TRABOT_CAPITAL = float(os.environ.get("TRABOT_CAPITAL", "20000"))
TRABOT_RISK_PROFILE = os.environ.get("TRABOT_RISK_PROFILE", "high").strip().lower()

# Optional: lookup one tradingsymbol and show its rank
LOOKUP_SYMBOL = os.environ.get("TRABOT_LOOKUP", "").strip().upper()


# Index spot symbol mapping (Kite uses special names for indices)
INDEX_SPOT_MAP = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
    "NIFTYNXT50": "NSE:NIFTY NEXT 50",
}

# =========================
# Market sentiment (Kite-only)
# =========================
# We use *proxy* sentiment derived from:
#   - INDIA VIX (fear / uncertainty)
#   - Index (NIFTY) trend on 60m candles (directional bias)
#   - Index option-chain: OI PCR + OTM IV skew + OI "walls"
#
# This is intentionally simple and robust: if any component fails, we degrade gracefully.

SENTIMENT_INDEX_UNDERLYING = os.environ.get("TRABOT_SENTIMENT_INDEX", "NIFTY").strip().upper()
SENTIMENT_SKEW_STEPS = int(os.environ.get("TRABOT_SENTIMENT_SKEW_STEPS", "4"))  # OTM distance in strike-steps
SENTIMENT_WALL_NEAR_PCT = float(os.environ.get("TRABOT_SENTIMENT_WALL_NEAR_PCT", "0.006"))  # 0.6% from spot
SENTIMENT_USE = os.environ.get("TRABOT_SENTIMENT", "1").strip() not in ("0", "false", "False", "no", "NO")

MARKET_SENTIMENT_HISTORY_CSV = os.path.join("data", "market_sentiment_history.csv")


def _safe_float(x, default: float | None = None) -> float | None:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _pct_rank_last(series: pd.Series, window: int = 30) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < max(10, window):
        return None
    w = s.iloc[-window:]
    last = float(w.iloc[-1])
    return float((w <= last).mean())


def _infer_strike_step(strikes: list[int]) -> int:
    """Infer strike increment from a list of strikes."""
    try:
        xs = sorted({int(x) for x in strikes if x is not None})
        if len(xs) < 6:
            return 50
        diffs = [xs[i + 1] - xs[i] for i in range(min(len(xs) - 1, 25)) if xs[i + 1] - xs[i] > 0]
        if not diffs:
            return 50
        diffs = sorted(diffs)
        mid = diffs[len(diffs) // 2]
        return int(mid) if mid > 0 else 50
    except Exception:
        return 50


def _mid_from_row(row: dict) -> float | None:
    b = _safe_float(row.get("bid_num", row.get("bid")))
    a = _safe_float(row.get("ask_num", row.get("ask")))
    ltp = _safe_float(row.get("last_price", row.get("ltp", row.get("close"))))
    if b is not None and a is not None and a > 0:
        return (b + a) / 2.0
    if ltp is not None and ltp > 0:
        return ltp
    return None


def _compute_chain_metrics(chain) -> dict:
    """OI/volume PCR + OI walls + simple OTM skew (iv_put - iv_call)."""
    out = {
        "pcr_oi": None,
        "pcr_vol": None,
        "skew": None,
        "call_wall_strike": None,
        "put_wall_strike": None,
        "call_wall_up_pct": None,
        "put_wall_dn_pct": None,
    }

    calls = getattr(chain, "calls", None)
    puts = getattr(chain, "puts", None)
    if calls is None or puts is None:
        return out
    try:
        calls = calls.copy()
        puts = puts.copy()
    except Exception:
        return out
    if calls.empty or puts.empty:
        return out

    calls["oi_num"] = pd.to_numeric(calls.get("oi"), errors="coerce").fillna(0.0)
    puts["oi_num"] = pd.to_numeric(puts.get("oi"), errors="coerce").fillna(0.0)
    calls["vol_num"] = pd.to_numeric(calls.get("volume"), errors="coerce").fillna(0.0)
    puts["vol_num"] = pd.to_numeric(puts.get("volume"), errors="coerce").fillna(0.0)

    sum_call_oi = float(calls["oi_num"].sum())
    sum_put_oi = float(puts["oi_num"].sum())
    sum_call_vol = float(calls["vol_num"].sum())
    sum_put_vol = float(puts["vol_num"].sum())

    if sum_call_oi > 0:
        out["pcr_oi"] = float(sum_put_oi / sum_call_oi)
    if sum_call_vol > 0:
        out["pcr_vol"] = float(sum_put_vol / sum_call_vol)

    # OI walls
    try:
        call_wall = calls.sort_values("oi_num", ascending=False).iloc[0]
        put_wall = puts.sort_values("oi_num", ascending=False).iloc[0]
        out["call_wall_strike"] = int(call_wall.get("strike"))
        out["put_wall_strike"] = int(put_wall.get("strike"))
        spot = float(chain.spot)
        if spot > 0:
            if out["call_wall_strike"] >= spot:
                out["call_wall_up_pct"] = float((out["call_wall_strike"] - spot) / spot)
            if out["put_wall_strike"] <= spot:
                out["put_wall_dn_pct"] = float((spot - out["put_wall_strike"]) / spot)
    except Exception:
        pass

    # OTM skew (iv_put - iv_call), using same expiry as slice
    try:
        strikes = list(pd.to_numeric(calls.get("strike"), errors="coerce").dropna().astype(int).tolist())
        step = _infer_strike_step(strikes)
        atm = int(chain.atm)
        c_strike = atm + int(SENTIMENT_SKEW_STEPS) * step
        p_strike = atm - int(SENTIMENT_SKEW_STEPS) * step

        calls_strike = pd.to_numeric(calls.get("strike"), errors="coerce").astype("Int64")
        puts_strike = pd.to_numeric(puts.get("strike"), errors="coerce").astype("Int64")
        c_df = calls[calls_strike == int(c_strike)]
        p_df = puts[puts_strike == int(p_strike)]
        if (not c_df.empty) and (not p_df.empty):
            c_row = c_df.iloc[0].to_dict()
            p_row = p_df.iloc[0].to_dict()
            c_px = _mid_from_row(c_row)
            p_px = _mid_from_row(p_row)
            if c_px and p_px and c_px > 0 and p_px > 0:
                T = time_to_expiry_years(chain.expiry)
                ivc, okc = implied_volatility(float(c_px), float(chain.spot), int(c_strike), T, 0.06, "CE")
                ivp, okp = implied_volatility(float(p_px), float(chain.spot), int(p_strike), T, 0.06, "PE")
                if okc and okp and ivc and ivp:
                    out["skew"] = float(ivp - ivc)
    except Exception:
        pass

    return out


def compute_market_context() -> dict:
    """Compute a lightweight market context dict (bias/risk-off) using Kite data."""
    ctx = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "bias": "NEUTRAL",          # BULLISH / BEARISH / NEUTRAL
        "strength": 0.0,            # 0..1
        "risk_off": False,
        "vix": None,
        "vix_pct": None,
        "pcr_oi": None,
        "skew": None,
        "call_wall_up_pct": None,
        "put_wall_dn_pct": None,
        "notes": [],
    }

    if not SENTIMENT_USE:
        return ctx

    # VIX percentile
    try:
        df_vix, _ = fetch_history_cached("NSE:INDIA VIX", lookback_days=120, interval="1d")
        if not df_vix.empty:
            vix = float(df_vix["close"].iloc[-1])
            ctx["vix"] = vix
            ew = pd.to_numeric(df_vix["close"], errors="coerce").ewm(span=10, adjust=False).mean()
            ctx["vix_pct"] = _pct_rank_last(ew, window=30)
    except Exception as e:
        ctx["notes"].append(f"VIX fail: {e}")

    # Index trend + chain
    index_spot = INDEX_SPOT_MAP.get(SENTIMENT_INDEX_UNDERLYING, f"NSE:{SENTIMENT_INDEX_UNDERLYING}")
    trend_side = "NO_TRADE"
    adx_1h = None
    try:
        df_1h, _ = fetch_history_cached(index_spot, lookback_days=LOOKBACK_DAYS, interval="60m")
        if not df_1h.empty:
            sig_1h = compute_signal(
                df=df_1h,
                ema_fast=EMA_FAST, ema_slow=EMA_SLOW,
                rsi_period=RSI_PERIOD, adx_period=ADX_PERIOD, adx_min=ADX_MIN,
                atr_period=ATR_PERIOD,
                stop_atr_mult=STOP_ATR_MULT, target_atr_mult=TARGET_ATR_MULT,
            )
            trend_side = str(sig_1h.side)
            adx_1h = _safe_float(sig_1h.metrics.get("adx", None))
    except Exception as e:
        ctx["notes"].append(f"Index trend fail: {e}")

    try:
        chain = get_kite_chain_slice(
            underlying=SENTIMENT_INDEX_UNDERLYING,
            kite_spot_symbol=index_spot,
            strike_step=0,
            strikes_around_atm=max(10, STRIKES_AROUND_ATM * 2),
            cache_path=INSTRUMENTS_CACHE_PATH,
        )
        m = _compute_chain_metrics(chain)
        ctx.update({
            "pcr_oi": m.get("pcr_oi"),
            "skew": m.get("skew"),
            "call_wall_up_pct": m.get("call_wall_up_pct"),
            "put_wall_dn_pct": m.get("put_wall_dn_pct"),
        })
    except Exception as e:
        ctx["notes"].append(f"Index chain fail: {e}")

    # combine
    s = 0.0
    if trend_side == "LONG":
        s += 0.80
    elif trend_side == "SHORT":
        s -= 0.80

    pcr = ctx.get("pcr_oi")
    if pcr is not None:
        if pcr <= 0.85:
            s += 0.25
        elif pcr >= 1.15:
            s -= 0.25

    skew = ctx.get("skew")
    if skew is not None:
        if skew >= 0.08:
            s -= 0.35
        elif skew <= 0.00:
            s += 0.10

    vix_pct = ctx.get("vix_pct")
    if vix_pct is not None:
        if vix_pct >= 0.75:
            s -= 0.35
        elif vix_pct <= 0.30:
            s += 0.10

    ctx["risk_off"] = bool((vix_pct is not None and vix_pct >= 0.80) or (skew is not None and skew >= 0.10))

    if s >= 0.35:
        ctx["bias"] = "BULLISH"
    elif s <= -0.35:
        ctx["bias"] = "BEARISH"
    else:
        ctx["bias"] = "NEUTRAL"

    ctx["strength"] = float(min(1.0, abs(s) / 1.2))

    # append history
    try:
        os.makedirs("data", exist_ok=True)
        row = {
            "ts": ctx["ts"],
            "bias": ctx["bias"],
            "strength": ctx["strength"],
            "risk_off": int(ctx["risk_off"]),
            "vix": ctx.get("vix"),
            "vix_pct": ctx.get("vix_pct"),
            "pcr_oi": ctx.get("pcr_oi"),
            "skew": ctx.get("skew"),
            "call_wall_up_pct": ctx.get("call_wall_up_pct"),
            "put_wall_dn_pct": ctx.get("put_wall_dn_pct"),
            "index": SENTIMENT_INDEX_UNDERLYING,
        }
        hist_exists = os.path.exists(MARKET_SENTIMENT_HISTORY_CSV)
        pd.DataFrame([row]).to_csv(MARKET_SENTIMENT_HISTORY_CSV, mode="a", header=not hist_exists, index=False)
    except Exception:
        pass

    if adx_1h is not None:
        ctx["notes"].append(f"{SENTIMENT_INDEX_UNDERLYING} 1H trend={trend_side} ADX≈{adx_1h:.0f}")

    return ctx


def _market_multiplier(ctx: dict, side: str, htf_align: bool) -> tuple[float, list[str]]:
    """Return (multiplier, reasons) to adjust raw score."""
    if not ctx or (not SENTIMENT_USE):
        return 1.0, []

    mult = 1.0
    rs: list[str] = []

    bias = ctx.get("bias", "NEUTRAL")
    strength = float(ctx.get("strength", 0.0) or 0.0)
    risk_off = bool(ctx.get("risk_off", False))

    if risk_off:
        mult *= (0.92 - 0.04 * strength)
        rs.append("Market risk-off (VIX/skew) → reduce aggressiveness")

    if bias == "BULLISH":
        if side == "LONG":
            mult *= (1.04 + 0.04 * strength)
            rs.append("Market bias bullish → boost LONGs")
        elif side == "SHORT":
            mult *= (0.92 - 0.04 * strength) if not htf_align else 0.95
            rs.append("Market bias bullish → penalize SHORTs")
    elif bias == "BEARISH":
        if side == "SHORT":
            mult *= (1.04 + 0.04 * strength)
            rs.append("Market bias bearish → boost SHORTs")
        elif side == "LONG":
            mult *= (0.92 - 0.04 * strength) if not htf_align else 0.95
            rs.append("Market bias bearish → penalize LONGs")

    wall_near = float(ctx.get("call_wall_up_pct") or 0.0) if side == "LONG" else float(ctx.get("put_wall_dn_pct") or 0.0)
    if wall_near and wall_near < SENTIMENT_WALL_NEAR_PCT:
        mult *= 0.94
        rs.append("Index OI wall is close → reduce conviction")

    return mult, rs

# ---------------------------------------------------------------------
# IMPORTANT:
# The rest of the file below is identical to your original scan_options,
# except where it:
#   - computes market_ctx once in main()
#   - passes market_ctx to _build_candidate()
#   - applies multiplier inside _build_candidate()
#   - adds sentiment columns into output rows
#
# Due to message size limit, I cannot paste the remaining ~700 lines here.
# ---------------------------------------------------------------------
