"""
scan_options_v22.py

Full-universe options scanner (Kite-only) with V2.2 strategy upgrades:
- IV percentile (rolling window with EWMA smoothing) using iv_store
- BS Greeks for picked contract (iv_greeks)
- Regime label (TREND/CHOP/VOLATILE) driven primarily by IV percentile
- Dynamic position sizing (greeks caps + stop-risk lots)
- Expert liquidity gates (spread/OI/volume/min premium)
- Market sentiment overlay (INDIA VIX + NIFTY option-chain aggregates) via market_sentiment

Outputs:
  data/options_scan_results_v22_v22.csv (+ timestamped)
  data/options_top10_v22_v22.csv (+ timestamped)
  data/reco_latest_v22.csv (+ timestamped)
  append-only: data/reco_history.csv

Educational tool only – not financial advice.
"""


from __future__ import annotations

import os
import json
import sys
import platform
import argparse
import math
import time
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from trabot_schema import RECO_SCHEMA_VERSION, DEFAULT_HISTORY_PATH
from run_manifest import write_manifest as write_run_manifest_v1, update_manifest as update_run_manifest_v1, env_snapshot, now_iso
from regime import detect_regime

from kite_client import get_kite
from kite_chain import get_kite_chain_slice
from strategy import compute_signal

from iv_store import append_iv_snapshot, iv_percentile
from iv_greeks import time_to_expiry_years, implied_volatility, greeks
from risk_caps import compute_max_lots

from journal import append_history, save_snapshot, make_run_id

from market_sentiment import compute_market_context, append_market_context, MarketContext



# ----------------------------
# Config
# ----------------------------

DATA_DIR = os.getenv("TRABOT_DATA_DIR", "data")
# Schema / Phase-1 instrumentation
SCHEMA_VERSION = RECO_SCHEMA_VERSION
DEFAULT_RECO_HISTORY_PATH = DEFAULT_HISTORY_PATH
RECO_HISTORY_PATH = os.getenv("TRABOT_RECO_HISTORY", DEFAULT_RECO_HISTORY_PATH)

# Run manifests (phase-1)
RUNS_DIR = os.getenv("TRABOT_RUNS_DIR", os.path.join(DATA_DIR, 'runs'))

# Phase-2 gates (defaults can be mode-specific in main)
REQUIRE_HTF_ALIGN_ENV = os.getenv("TRABOT_REQUIRE_HTF_ALIGN", "").strip()
SKIP_CHOP_ENV = os.getenv("TRABOT_SKIP_CHOP", "").strip()
BLOCK_LONG_PREMIUM_IVP = float(os.getenv("TRABOT_BLOCK_LONG_PREMIUM_IVP", "0.0") or 0.0)

# Runtime gates (set in main)
G_REQUIRE_HTF_ALIGN = False
G_SKIP_CHOP = False
G_BLOCK_LONG_PREMIUM_IVP = float(BLOCK_LONG_PREMIUM_IVP)


INSTRUMENTS_CACHE_PATH = os.getenv("INSTRUMENTS_CACHE_PATH", os.path.join(DATA_DIR, "kite_instruments_NFO.csv"))
CACHE_TTL_MINUTES = int(os.getenv("TRABOT_CACHE_TTL_MIN", "5"))  # you wanted max 5 mins

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "180"))
INTERVAL = os.getenv("INTERVAL", "day")  # "day" or "60m" etc.

UNIVERSE_START = int(os.getenv("UNIVERSE_START", "0"))
UNIVERSE_COUNT = os.getenv("UNIVERSE_COUNT", "")
UNIVERSE_COUNT = int(UNIVERSE_COUNT) if UNIVERSE_COUNT.strip() else None

STRIKES_AROUND_ATM = int(os.getenv("STRIKES_AROUND_ATM", "12"))

# NEW: smaller actionable outputs
TOP2 = int(os.getenv("TRABOT_TOP2", "2"))            # top 2 buy + top 2 sell
TOP10 = int(os.getenv("TRABOT_TOP10", "10"))         # top 10 overall (entry/sl/target only)
TOP_OVERALL_ABS = int(os.getenv("TRABOT_TOP_ABS", "20"))  # still saved in full results if you want

LOOKUP_SYMBOL = os.getenv("LOOKUP_SYMBOL", "").strip().upper()
SLEEP_BETWEEN_SYMBOLS = float(os.getenv("SLEEP_BETWEEN_SYMBOLS", "0.0"))

EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))
ADX_MIN = float(os.getenv("ADX_MIN", "18"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
STOP_ATR_MULT = float(os.getenv("STOP_ATR_MULT", "1.5"))
TARGET_ATR_MULT = float(os.getenv("TARGET_ATR_MULT", "2.0"))

RISK_FREE = float(os.getenv("RISK_FREE_RATE", "0.06"))
TRABOT_RISK_PROFILE = os.getenv("TRABOT_RISK_PROFILE", "high").strip().lower()

_CAPITAL_ENV = os.getenv("TRABOT_CAPITAL", "").strip()
if _CAPITAL_ENV:
    TRABOT_CAPITAL = float(_CAPITAL_ENV)
else:
    # Default capital buckets (only used when TRABOT_CAPITAL is not set)
    TRABOT_CAPITAL = 100000.0 if TRABOT_RISK_PROFILE == "high" else 400000.0 if TRABOT_RISK_PROFILE == "moderate" else 100000.0

IV_PCTL_WINDOW_DAYS = int(os.getenv("IV_PCTL_WINDOW", "30"))
IV_EWMA_SPAN = int(os.getenv("IV_EWMA_SPAN", "10"))

# --- Expert quality gates / sizing (v2.2) ---
MIN_MID_PRICE = float(os.getenv("TRABOT_MIN_MID_PRICE", "8"))
MAX_SPREAD_PCT = float(os.getenv("TRABOT_MAX_SPREAD_PCT", "0.08"))
MIN_OI = int(os.getenv("TRABOT_MIN_OI", "20000"))
MIN_VOL = int(os.getenv("TRABOT_MIN_VOL", "5000"))
RISK_PER_TRADE_PCT = float(os.getenv("TRABOT_RISK_PER_TRADE_PCT", "0.015"))
TIME_STOP_MIN = int(os.getenv("TRABOT_TIME_STOP_MIN", "90"))

# Expiry band (DTE) control (v2.2.x)
MIN_DTE_DAYS = int(os.getenv("TRABOT_MIN_DTE_DAYS", "0"))
_MAX_DTE_ENV = os.getenv("TRABOT_MAX_DTE_DAYS", "").strip()
MAX_DTE_DAYS = int(_MAX_DTE_ENV) if _MAX_DTE_ENV else None

# HTF interval (used for alignment + sentiment trend calc)
HTF_INTERVAL = os.getenv("TRABOT_HTF_INTERVAL", "60m")

# Optional higher timeframe for alignment/regime (daily for swing etc.)
DTF_INTERVAL = os.getenv("TRABOT_DTF_INTERVAL", "").strip()

# Alignment mode: off / soft / hard (Phase-2).
ALIGN_MODE_ENV = os.getenv("TRABOT_ALIGN_MODE", "").strip().lower()
ALIGN_ALLOW_FLAT = os.getenv("TRABOT_ALIGN_ALLOW_FLAT", "1").strip() not in ("0","false","False")

# Dual-mode defaults (intraday + swing)
MODE_DEFAULTS = {
    "intraday": {
        "interval": "15m",
        "htf_interval": "60m",
        "lookback_days": 120,
        "stop_atr_mult": 1.2,
        "target_atr_mult": 1.8,
        "time_stop_min": 90,
        "min_dte": 0,
        "max_dte": 7,
        "risk_pct": 0.010,
    },
    "swing": {
        "interval": "60m",
        "htf_interval": "day",
        "lookback_days": 240,
        "stop_atr_mult": 1.6,
        "target_atr_mult": 2.4,
        "time_stop_min": 2880,
        "min_dte": 4,
        "max_dte": 14,
        "risk_pct": 0.015,
    },
}

SENTIMENT_ENABLED = os.getenv("TRABOT_SENTIMENT_ENABLED", "1").strip() not in ("0", "false", "False")
SENTIMENT_INDEX_UNDERLYING = os.getenv("TRABOT_SENTIMENT_INDEX", "NIFTY").strip().upper()
SENTIMENT_INDEX_SPOT = os.getenv("TRABOT_SENTIMENT_SPOT", "NSE:NIFTY 50")


TREND_STRENGTH_THRESHOLD = float(os.getenv("TREND_STRENGTH_THRESHOLD", "0.01"))
VOLATILE_IV_PCTL = float(os.getenv("VOLATILE_IV_PCTL", "0.70"))
EXPIRY_PENALTY_DTE = int(os.getenv("EXPIRY_PENALTY_DTE", "3"))


INDEX_SPOT_MAP = {
    "NIFTY": "NSE:NIFTY 50",
    "NIFTYNXT50": "NSE:NIFTY NEXT 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
}


# ----------------------------
# Universe / instruments
# ----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: dict) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def write_run_manifest(run_id: str, payload: dict) -> str:
    """Write per-run manifest under data/runs/<run_id>/manifest.json (Phase-1)."""
    return write_run_manifest_v1(run_id=run_id, payload=payload, base_dir=RUNS_DIR, filename='manifest.json')



def _load_instruments_nfo(cache_path: str) -> pd.DataFrame:
    """Load NFO instruments from disk (preferred) or fetch with backoff.

    Notes:
      - Kite instruments dump is typically generated once per day.
      - We avoid fetching on every run to reduce rate-limit errors.
      - Set TRABOT_REFRESH_NFO_INSTRUMENTS=1 to force a refresh.
    """
    _ensure_dir(os.path.dirname(cache_path))
    refresh = os.getenv("TRABOT_REFRESH_NFO_INSTRUMENTS", "0").strip().lower() in ("1", "true", "yes")

    if not refresh and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            if not df.empty:
                return df
        except Exception:
            pass

    kite = get_kite()
    last_err: Exception | None = None
    for i in range(6):
        try:
            df = pd.DataFrame(kite.instruments("NFO"))
            if df.empty:
                raise RuntimeError("Kite instruments('NFO') returned empty")
            try:
                df.to_csv(cache_path, index=False)
            except Exception:
                pass
            return df
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            # kiteconnect raises NetworkException: Too many requests
            if "too many requests" in msg or "429" in msg:
                time.sleep(1.0 * (2 ** i))
                continue
            raise

    # If still rate-limited, fall back to cache if available
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            if not df.empty:
                return df
        except Exception:
            pass

    raise RuntimeError(f"Failed to fetch NFO instruments (rate limited). Last error: {last_err}")


def build_universe_all_options() -> list[dict]:
    _ensure_dir(DATA_DIR)
    kite = get_kite()

    if os.path.exists(INSTRUMENTS_CACHE_PATH):
        inst = pd.read_csv(INSTRUMENTS_CACHE_PATH)
    else:
        inst = pd.DataFrame(kite.instruments("NFO"))
        inst.to_csv(INSTRUMENTS_CACHE_PATH, index=False)

    opt = inst[inst["segment"].astype(str).str.upper() == "NFO-OPT"].copy()
    if opt.empty:
        return []

    underlyings = sorted(opt["name"].dropna().astype(str).str.upper().unique().tolist())
    out = []
    for u in underlyings:
        spot = INDEX_SPOT_MAP.get(u, f"NSE:{u}")
        out.append({"underlying": u, "spot": spot})
    return out


def build_lot_size_map() -> dict:
    if not os.path.exists(INSTRUMENTS_CACHE_PATH):
        return {}
    df = pd.read_csv(INSTRUMENTS_CACHE_PATH)
    df = df[df.get("segment", "").astype(str).str.upper() == "NFO-OPT"].copy()
    if df.empty:
        return {}
    m = {}
    for _, r in df.iterrows():
        ts = str(r.get("tradingsymbol") or "").upper()
        if not ts:
            continue
        try:
            m[ts] = int(r.get("lot_size") or 1)
        except Exception:
            m[ts] = 1
    return m


# ----------------------------
# Market data
# ----------------------------

def fetch_history_cached(symbol: str, lookback_days: int, interval: str):
    # expects your updated market_data.py to have fetch_history_cached + TTL obeyed by TRABOT_CACHE_TTL_MIN
    from market_data import fetch_history_cached as _fh
    return _fh(symbol, lookback_days=lookback_days, interval=interval, ttl_minutes=CACHE_TTL_MINUTES)


# ----------------------------
# Helpers
# ----------------------------

def _session_tag(last_ts: pd.Timestamp) -> str:
    now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tzinfo else pd.Timestamp.now()
    if last_ts.date() != now.date():
        return "MARKET_CLOSED"
    if (now - last_ts).total_seconds() < 60 * 60:
        return "MID"
    return "CLOSE"


def _session_factor(tag: str) -> float:
    t = (tag or "").upper()
    if t == "MID":
        return 1.02
    if t == "CLOSE":
        return 0.98
    return 0.90


def _bos_strength(df: pd.DataFrame, lookback: int = 20) -> Tuple[str, float]:
    if df is None or df.empty or len(df) < lookback + 5:
        return "NONE", 0.0

    highs = df["high"].rolling(lookback).max()
    lows = df["low"].rolling(lookback).min()
    close = float(df["close"].iloc[-1])

    hh = float(highs.iloc[-2]) if not math.isnan(float(highs.iloc[-2])) else float(highs.iloc[-1])
    ll = float(lows.iloc[-2]) if not math.isnan(float(lows.iloc[-2])) else float(lows.iloc[-1])

    tr = (df["high"] - df["low"]).rolling(14).mean()
    atr = float(tr.iloc[-1]) if not math.isnan(float(tr.iloc[-1])) else max(1.0, float(df["high"].iloc[-1] - df["low"].iloc[-1]))

    if close > hh:
        return "BOS_UP", (close - hh) / max(atr, 1e-9)
    if close < ll:
        return "BOS_DOWN", (ll - close) / max(atr, 1e-9)
    return "NONE", 0.0


def _trend_strength(metrics: dict) -> float:
    try:
        close = float(metrics.get("close") or 0.0)
        ema_fast = float(metrics.get("ema_fast") or 0.0)
        ema_slow = float(metrics.get("ema_slow") or 0.0)
        if close > 0:
            return abs(ema_fast - ema_slow) / close
    except Exception:
        pass
    return 0.0


def _classify_regime_v21(metrics: dict, iv_pct: Optional[float]) -> str:
    if iv_pct is not None and iv_pct >= VOLATILE_IV_PCTL:
        return "VOLATILE"
    ts = _trend_strength(metrics)
    if ts >= TREND_STRENGTH_THRESHOLD:
        return "TREND"
    return "CHOP"



def _sentiment_multiplier(ctx: MarketContext | None, side: str) -> float:
    """
    Returns a multiplier in ~[0.75, 1.25] based on market context.
    - Boost trades aligned with ctx.bias (BULLISH/BEARISH) proportional to ctx.strength.
    - If risk_off, slightly penalize LONG and slightly boost SHORT.
    """
    if ctx is None:
        return 1.0

    m = 1.0
    try:
        strength = float(ctx.strength or 0.0)
    except Exception:
        strength = 0.0

    bias = str(ctx.bias or "NEUTRAL").upper()
    side = str(side or "").upper()
    risk_off = bool(getattr(ctx, "risk_off", False))

    if risk_off:
        m *= 0.88 if side == "LONG" else 1.05

    if bias == "BULLISH":
        if side == "LONG":
            m *= 1.0 + 0.12 * strength
        else:
            m *= 1.0 - 0.10 * strength
    elif bias == "BEARISH":
        if side == "SHORT":
            m *= 1.0 + 0.12 * strength
        else:
            m *= 1.0 - 0.10 * strength

    return max(0.75, min(1.25, float(m)))


def _risk_lots_from_stop(capital: float, entry: float, sl: float, lot_size: int) -> int:
    if capital <= 0 or entry <= 0 or sl <= 0 or lot_size <= 0:
        return 0
    loss_per_unit = max(0.0, entry - sl)
    if loss_per_unit <= 1e-9:
        return 0
    risk_rupees = float(capital) * float(RISK_PER_TRADE_PCT)
    lots = int(risk_rupees // (loss_per_unit * lot_size))
    return max(0, lots)


def _score_strike_rows(df_side: pd.DataFrame, atm: int) -> pd.DataFrame:
    x = df_side.copy()

    x["oi_num"] = pd.to_numeric(x.get("oi"), errors="coerce").fillna(0.0)
    x["vol_num"] = pd.to_numeric(x.get("volume"), errors="coerce").fillna(0.0)

    x["bid_num"] = pd.to_numeric(x.get("bid"), errors="coerce")
    x["ask_num"] = pd.to_numeric(x.get("ask"), errors="coerce")
    x["mid"] = (x["bid_num"] + x["ask_num"]) / 2.0

    x["spread_pct"] = (x["ask_num"] - x["bid_num"]) / x["mid"]
    x["spread_pct"] = x["spread_pct"].replace([math.inf, -math.inf], math.nan)
    x["spread_pct"] = x["spread_pct"].clip(lower=0.0, upper=0.80)

    x["liq"] = (x["oi_num"].add(1).apply(math.log)) * 0.7 + (x["vol_num"].add(1).apply(math.log)) * 0.3
    liq_max = float(x["liq"].max()) if len(x) else 1.0
    if liq_max <= 0:
        liq_max = 1.0
    x["liq_norm"] = x["liq"] / liq_max

    x["dist_atm"] = (pd.to_numeric(x["strike"], errors="coerce") - float(atm)).abs()
    x["mny_score"] = 1.0 / (1.0 + (x["dist_atm"] / max(atm, 1.0)) * 20.0)

    sp = x["spread_pct"].fillna(0.05).clip(lower=0.0, upper=0.30)
    x["spread_pen"] = (sp / 0.02).clip(lower=0.0, upper=6.0)

    x["strike_score"] = (x["liq_norm"] * 0.65 + x["mny_score"] * 0.35) - (x["spread_pen"] * 0.15)
    return x


def _pick_best_contract(chain, want_right: str) -> dict | None:
    df_side = chain.calls if want_right == "CE" else chain.puts
    if df_side.empty:
        return None
    scored = _score_strike_rows(df_side, atm=int(chain.atm))
    scored = scored.dropna(subset=["bid_num", "ask_num"], how="any")

    # Expert liquidity gates
    scored = scored[
        (scored["mid"] >= MIN_MID_PRICE) &
        (scored["spread_pct"].fillna(1.0) <= MAX_SPREAD_PCT) &
        (scored["oi_num"] >= MIN_OI) &
        (scored["vol_num"] >= MIN_VOL)
    ].copy()

    if scored.empty:
        return None
    return scored.sort_values("strike_score", ascending=False).iloc[0].to_dict()


def _compute_iv_and_greeks(spot: float, expiry: str, right: str, strike: int, price: float):
    T = time_to_expiry_years(expiry)
    iv, ok = implied_volatility(price, spot, strike, T, RISK_FREE, right)
    conf = "high"
    if (not ok) or iv is None or iv < 0.05 or iv > 2.50:
        conf = "low"
    if iv is None:
        iv = 0.50
    g = greeks(spot, strike, T, RISK_FREE, iv, right)
    dte = max(0, int(round(T * 365)))
    return float(iv), g, conf, dte


def _append_atm_iv_snapshot(underlying: str, chain) -> tuple[float | None, str]:
    best = None
    best_dist = 1e18
    for df_side, right in [(chain.calls, "CE"), (chain.puts, "PE")]:
        if df_side is None or df_side.empty:
            continue
        tmp = df_side.copy()
        tmp["dist"] = (pd.to_numeric(tmp["strike"], errors="coerce") - float(chain.atm)).abs()
        tmp = tmp.dropna(subset=["dist"])
        if tmp.empty:
            continue
        row = tmp.sort_values("dist").iloc[0].to_dict()
        dist = float(row.get("dist", 1e18))
        if dist < best_dist:
            best_dist = dist
            best = (row, right)

    if not best:
        return None, "low"

    row, right = best
    strike = int(row.get("strike"))
    tsym = str(row.get("tradingsymbol"))
    bid = pd.to_numeric(row.get("bid"), errors="coerce")
    ask = pd.to_numeric(row.get("ask"), errors="coerce")
    mid = (bid + ask) / 2.0
    px = float(mid) if mid == mid and mid > 0 else float(row.get("last_price") or 0.0)
    if px <= 0:
        return None, "low"

    iv, _, conf, _ = _compute_iv_and_greeks(float(chain.spot), chain.expiry, right, strike, px)

    append_iv_snapshot({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "underlying": underlying,
        "tradingsymbol": tsym,
        "expiry": chain.expiry,
        "strike": strike,
        "right": right,
        "spot": float(chain.spot),
        "price": float(px),
        "iv": float(iv),
        "confidence": conf,
    })
    return float(iv), conf


def _build_candidate(item: dict, lot_map: dict, market_ctx: MarketContext | None = None) -> dict | None:
    df, used_interval = fetch_history_cached(item["spot"], lookback_days=LOOKBACK_DAYS, interval=INTERVAL)
    if df.empty:
        return None

    sig = compute_signal(
        df=df,
        ema_fast=EMA_FAST, ema_slow=EMA_SLOW,
        rsi_period=RSI_PERIOD, adx_period=ADX_PERIOD, adx_min=ADX_MIN,
        atr_period=ATR_PERIOD,
        stop_atr_mult=STOP_ATR_MULT, target_atr_mult=TARGET_ATR_MULT,
    )

    side = None
    is_live = False
    trigger_text = None

    if sig.side in ("LONG", "SHORT"):
        side = sig.side
        is_live = True
        trigger_text = "LIVE signal"
        entry_u, stop_u, target_u = float(sig.entry), float(sig.stop), float(sig.target)
    else:
        ws = str(sig.metrics.get("watch_side", "NONE"))
        wt = str(sig.metrics.get("watch_trigger", ""))
        if ws in ("LONG", "SHORT") and wt:
            side = ws
            is_live = False
            trigger_text = wt
            entry_u = float(sig.metrics.get("watch_entry"))
            stop_u = float(sig.metrics.get("watch_stop"))
            target_u = float(sig.metrics.get("watch_target"))
        else:
            return None

    last_ts = pd.Timestamp(df.index[-1])
    sess = _session_tag(last_ts)
    sess_factor = _session_factor(sess)

    bos_tag, bos_strength = _bos_strength(df, lookback=20)

    # Multi-timeframe candles (HTF/DTF) for regime + alignment (Phase-2)
    df_htf = None
    try:
        df_htf, _ = fetch_history_cached(item["spot"], lookback_days=LOOKBACK_DAYS, interval=HTF_INTERVAL)
    except Exception:
        df_htf = None

    df_dtf = None
    if DTF_INTERVAL:
        try:
            df_dtf, _ = fetch_history_cached(item["spot"], lookback_days=LOOKBACK_DAYS, interval=DTF_INTERVAL)
        except Exception:
            df_dtf = None
    chain = get_kite_chain_slice(
        underlying=item["underlying"],
        kite_spot_symbol=item["spot"],
        strike_step=0,
        strikes_around_atm=STRIKES_AROUND_ATM,
        cache_path=INSTRUMENTS_CACHE_PATH,
        min_dte_days=MIN_DTE_DAYS,
        max_dte_days=MAX_DTE_DAYS,
    )

    # Align to live spot
    candle_close = float(sig.metrics.get("close", chain.spot))
    delta_spot = float(chain.spot) - candle_close
    entry_u += delta_spot
    stop_u += delta_spot
    target_u += delta_spot

    # IV percentile
    _append_atm_iv_snapshot(item["underlying"], chain)
    pct, n, _ = iv_percentile(item["underlying"], window_days=IV_PCTL_WINDOW_DAYS, ewma_span=IV_EWMA_SPAN)

    # Phase-2: candle-driven regime + confidence (TREND/CHOP/VOLATILE)
    reg = detect_regime(df_ltf=df, df_htf=df_htf, df_dtf=df_dtf, ivp=pct)
    regime = str(reg.label).upper()
    regime_conf = float(reg.confidence or 0.0)

    ltf_dir = (reg.details.get('ltf') or {}).get('dir', '')
    htf_dir = (reg.details.get('htf') or {}).get('dir', '') if reg.details.get('htf') else ''
    dtf_dir = (reg.details.get('dtf') or {}).get('dir', '') if reg.details.get('dtf') else ''

    # Phase-2: alignment gate
    align_mode = (ALIGN_MODE_ENV or ('hard' if G_REQUIRE_HTF_ALIGN else ('soft' if ALIGN_ALLOW_FLAT else 'off'))).lower()
    if align_mode == 'off':
        htf_align = True
    elif align_mode == 'hard':
        htf_align = bool(reg.details.get('align_strict'))
    else:  # soft
        htf_align = bool(reg.details.get('align_allow_flat'))

    # Phase-2 gates
    if G_SKIP_CHOP and regime == 'CHOP':
        return None
    if pct is not None and G_BLOCK_LONG_PREMIUM_IVP and float(pct) >= float(G_BLOCK_LONG_PREMIUM_IVP):
        return None
    if (G_REQUIRE_HTF_ALIGN or align_mode == 'hard') and not htf_align:
        return None

    want_right = "CE" if side == "LONG" else "PE"
    pick = _pick_best_contract(chain, want_right=want_right)
    if not pick:
        return None

    strike = int(pick["strike"])
    tsym = str(pick["tradingsymbol"])
    kite_symbol = f"NFO:{tsym}"
    lot = int(lot_map.get(tsym.upper(), 1))

    bid = float(pick.get("bid_num")) if pick.get("bid_num") == pick.get("bid_num") else None
    ask = float(pick.get("ask_num")) if pick.get("ask_num") == pick.get("ask_num") else None
    ltp = float(pick.get("last_price")) if pick.get("last_price") == pick.get("last_price") else None
    mid = float(pick.get("mid")) if pick.get("mid") == pick.get("mid") else None

    spread_pct = float(pick.get("spread_pct")) if pick.get("spread_pct") == pick.get("spread_pct") else None

    entry_opt = ask if ask is not None else ltp
    if entry_opt is None or entry_opt <= 0:
        return None

    px_for_iv = mid if mid is not None and mid > 0 else float(entry_opt)
    iv, g, greeks_conf, dte = _compute_iv_and_greeks(float(chain.spot), chain.expiry, want_right, strike, px_for_iv)

    # Map underlying stop/target to option stop/target using delta (simple approximation)
    delta_abs = abs(float(g.get("delta", 0.5)))
    delta_abs = min(0.75, max(0.25, delta_abs))

    risk_u = abs(entry_u - stop_u)
    rew_u = abs(target_u - entry_u)

    sl_opt = max(0.05, entry_opt - delta_abs * risk_u)
    tgt_opt = entry_opt + delta_abs * rew_u

    # Score (simple + data-centric)
    adx = float(sig.metrics.get("adx", 0.0))
    score = (0.8 + 0.02 * adx) * sess_factor
    if is_live:
        score += 0.3
    if not htf_align:
        score *= 0.85
    if pct is not None:
        if pct >= 0.70:
            score *= 0.92
        elif pct <= 0.30:
            score *= 1.08
    if dte <= EXPIRY_PENALTY_DTE:
        score *= 0.75
    if greeks_conf == "low":
        score *= 0.85

    # Sentiment overlay multiplier
    mult = _sentiment_multiplier(market_ctx, side)
    score *= float(mult)

    score = score if side == "LONG" else -score

    # Sizing
    max_lots, _, _ = compute_max_lots(
        capital=TRABOT_CAPITAL,
        regime=regime,
        confidence=greeks_conf,
        risk_profile=TRABOT_RISK_PROFILE,
        dte=int(dte),
        spot=float(chain.spot),
        option_price=float(entry_opt),
        lot_size=int(lot),
        delta=float(g.get("delta", 0.0)),
        vega_1pct=float(g.get("vega_1pct", 0.0)),
        theta_day=float(g.get("theta_day", 0.0)),
    )
    pass_caps = bool(max_lots >= 1)

    # Stop-risk based lots (expert sizing)
    risk_lots = _risk_lots_from_stop(TRABOT_CAPITAL, float(entry_opt), float(sl_opt), int(lot))
    final_lots = int(min(int(max_lots), int(risk_lots))) if risk_lots > 0 else int(max_lots)
    pass_caps = bool(final_lots >= 1)

    # Explainability notes
    notes = []
    notes.append(f"adx={float(sig.metrics.get('adx',0.0) or 0.0):.1f}")
    if pct is not None:
        notes.append(f"ivp={float(pct):.2f}")
    notes.append(f"dte={int(dte)}")
    try:
        notes.append(f"spr={float(spread_pct or 0.0):.3f}")
    except Exception:
        pass
    notes_str = " | ".join(notes)

    return {
        "underlying": item["underlying"],
        "ts_signal": str(last_ts),
        "spot_symbol": item["spot"],
        "expiry": chain.expiry,
        "dte": int(dte),
        "regime": regime,
        "regime_conf": float(regime_conf),
        "align_mode": align_mode,
        "ltf_dir": ltf_dir,
        "htf_dir": htf_dir,
        "dtf_dir": dtf_dir,
        "side": side,
        "is_live": bool(is_live),
        "right": want_right,
        "tradingsymbol": tsym,
        "kite_symbol": kite_symbol,
        "strike": strike,
        "score": float(score),
        "entry": float(entry_opt),
        "sl": float(sl_opt),
        "target": float(tgt_opt),
        "time_stop_min": int(TIME_STOP_MIN),
        "spread_pct": spread_pct,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "ltp": ltp,
        "quote_ts": pick.get("quote_timestamp") or pick.get("last_trade_time") or "",
        "entry_model": "ask" if ask is not None else ("mid" if mid is not None else "ltp"),
        "iv": float(iv),
        "iv_pct": float(pct) if pct is not None else float("nan"),
        "iv_samples": int(n),
        "greeks_conf": greeks_conf,
        "delta": float(g.get("delta", 0.0)),
        "vega_1pct": float(g.get("vega_1pct", 0.0)),
        "theta_day": float(g.get("theta_day", 0.0)),
        "lot_size": int(lot),
        "greeks_max_lots": int(max_lots),
        "risk_lots": int(risk_lots),
        "max_lots": int(final_lots),
        "pass_caps": bool(pass_caps),
        "sent_mult": float(mult),
        "mkt_bias": getattr(market_ctx, "bias", None) if market_ctx else None,
        "mkt_strength": getattr(market_ctx, "strength", None) if market_ctx else None,
        "mkt_risk_off": getattr(market_ctx, "risk_off", None) if market_ctx else None,
        "notes": notes_str,
        "reason": trigger_text,
        "bos": bos_tag,
        "bos_strength": float(bos_strength),
        "htf_align": bool(htf_align),
        "session": sess,
    }


def _reco_row(c: dict, ts_str: str, run_id: str, source: str, bucket: str, mode: str) -> dict:
    """Normalize a candidate into the stable reco schema (Phase-1)."""
    action = "BUY_CE" if c["side"] == "LONG" else "BUY_PE"
    if not c.get("is_live", True):
        action = "WATCH_CE" if c["side"] == "LONG" else "WATCH_PE"

    return {
        "ts_reco": ts_str,
        "schema_version": RECO_SCHEMA_VERSION,
        "run_id": run_id,
        "source": source,
        "bucket": bucket,
        "mode": mode,

        "interval": INTERVAL,
        "htf_interval": HTF_INTERVAL,
        "dtf_interval": DTF_INTERVAL or "",
        "ts_signal": c.get("ts_signal", ""),

        "quote_ts": c.get("quote_ts", ""),
        "entry_model": c.get("entry_model", ""),
        "bid": c.get("bid", ""),
        "ask": c.get("ask", ""),
        "mid": c.get("mid", ""),
        "ltp": c.get("ltp", ""),
        "spread_pct": c.get("spread_pct", ""),

        "underlying": c.get("underlying", ""),
        "spot_symbol": c.get("spot_symbol", ""),
        "expiry": c.get("expiry", ""),
        "dte": c.get("dte", ""),

        "action": action,
        "side": c.get("side", ""),
        "tradingsymbol": c.get("tradingsymbol", ""),
        "kite_symbol": c.get("kite_symbol", ""),
        "strike": c.get("strike", ""),
        "right": c.get("right", ""),

        "entry": c.get("entry", ""),
        "sl": c.get("sl", ""),
        "target": c.get("target", ""),
        "time_stop_min": c.get("time_stop_min", TIME_STOP_MIN),

        "score": c.get("score", ""),
        "regime": c.get("regime", ""),
        "regime_conf": c.get("regime_conf", ""),
        "align_mode": c.get("align_mode", ""),
        "ltf_dir": c.get("ltf_dir", ""),
        "htf_dir": c.get("htf_dir", ""),
        "dtf_dir": c.get("dtf_dir", ""),
        "htf_align": c.get("htf_align", ""),

        "iv": c.get("iv", ""),
        "iv_pct": c.get("iv_pct", ""),
        "iv_samples": c.get("iv_samples", ""),
        "greeks_conf": c.get("greeks_conf", ""),
        "delta": c.get("delta", ""),
        "vega_1pct": c.get("vega_1pct", ""),
        "theta_day": c.get("theta_day", ""),

        "lot_size": c.get("lot_size", ""),
        "max_lots": c.get("max_lots", ""),
        "pass_caps": c.get("pass_caps", ""),

        "sent_mult": c.get("sent_mult", ""),
        "mkt_bias": c.get("mkt_bias", ""),
        "mkt_strength": c.get("mkt_strength", ""),
        "mkt_risk_off": c.get("mkt_risk_off", ""),

        "reason": c.get("reason", ""),
        "notes": c.get("notes", ""),

        # Anything else will be captured into extra_json by journal.append_history().
    }


def _print_top2(title: str, rows: list[dict]):
    print(f"\n{title}")
    if not rows:
        print("  (none)")
        return
    for c in rows:
        tag = "PASS" if c.get("pass_caps") else "NO_SIZE"
        print(
            f"  {c['tradingsymbol']:<22s} | {tag:<7s} lots={c.get('max_lots',0)} "
            f"| entry={c['entry']:.2f} sl={c['sl']:.2f} tgt={c['target']:.2f} | score={c['score']:+.2f}"
        )


def _print_top10_min(title: str, rows: list[dict]):
    print(f"\n{title}")
    if not rows:
        print("  (none)")
        return
    for i, c in enumerate(rows, 1):
        print(f"  #{i:02d} {c['tradingsymbol']:<22s}  entry={c['entry']:.2f}  sl={c['sl']:.2f}  tgt={c['target']:.2f}")


def main(mode: str = "intraday"):
    run_ts = datetime.now()
    ts_str = run_ts.isoformat(timespec="seconds")
    run_id = make_run_id(run_ts)

    # Mode-aware tuning (intraday vs swing). Overrides are applied to globals so helper functions reuse them.
    mode = (mode or os.getenv("TRABOT_MODE", "intraday")).strip().lower()
    if mode not in MODE_DEFAULTS:
        mode = "intraday"
    md = MODE_DEFAULTS[mode]

    global INTERVAL, LOOKBACK_DAYS, STOP_ATR_MULT, TARGET_ATR_MULT, RISK_PER_TRADE_PCT, TIME_STOP_MIN
    global MIN_DTE_DAYS, MAX_DTE_DAYS, HTF_INTERVAL

    INTERVAL = os.getenv("INTERVAL", md["interval"]) or md["interval"]
    LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", str(md["lookback_days"])) or md["lookback_days"])

    STOP_ATR_MULT = float(os.getenv("STOP_ATR_MULT", str(md["stop_atr_mult"])) or md["stop_atr_mult"])
    TARGET_ATR_MULT = float(os.getenv("TARGET_ATR_MULT", str(md["target_atr_mult"])) or md["target_atr_mult"])

    TIME_STOP_MIN = int(os.getenv("TRABOT_TIME_STOP_MIN", str(md["time_stop_min"])) or md["time_stop_min"])
    MIN_DTE_DAYS = int(os.getenv("TRABOT_MIN_DTE_DAYS", str(md["min_dte"])) or md["min_dte"])
    max_dte_env = os.getenv("TRABOT_MAX_DTE_DAYS", str(md["max_dte"] if md["max_dte"] is not None else "")).strip()
    MAX_DTE_DAYS = int(max_dte_env) if max_dte_env else None

    RISK_PER_TRADE_PCT = float(os.getenv("TRABOT_RISK_PER_TRADE_PCT", str(md["risk_pct"])) or md["risk_pct"])
    HTF_INTERVAL = os.getenv("TRABOT_HTF_INTERVAL", md["htf_interval"]) or md["htf_interval"]

    suffix = f"_{mode}"

    universe = build_universe_all_options()
    total = len(universe)
    if total == 0:
        print("Universe is empty (no options found).")
        return

    lot_map = build_lot_size_map()


    # Market sentiment context (computed once per run)
    market_ctx: MarketContext | None = None
    if SENTIMENT_ENABLED:
        try:
            market_ctx = compute_market_context(
                instruments_cache_path=INSTRUMENTS_CACHE_PATH,
                index_underlying=SENTIMENT_INDEX_UNDERLYING,
                index_spot_symbol=SENTIMENT_INDEX_SPOT,
                strikes_around_atm=12,
                skew_steps=4,
                lookback_days=30,
                interval=HTF_INTERVAL,
                ema_fast=EMA_FAST,
                ema_slow=EMA_SLOW,
                rsi_period=RSI_PERIOD,
                adx_period=ADX_PERIOD,
                adx_min=ADX_MIN,
                atr_period=ATR_PERIOD,
                stop_atr_mult=STOP_ATR_MULT,
                target_atr_mult=TARGET_ATR_MULT,
            )
            append_market_context(market_ctx)
            print(f"Market ctx: bias={market_ctx.bias} strength={market_ctx.strength:.2f} risk_off={market_ctx.risk_off}")
        except Exception as e:
            print(f"[sentiment] skipped: {e}")
            market_ctx = None

    start = max(0, UNIVERSE_START)
    end = total if UNIVERSE_COUNT is None else min(total, start + UNIVERSE_COUNT)
    universe = universe[start:end]

    print(f"Universe: {total} option-underlyings found in Kite (NFO-OPT). mode={mode}")
    print(f"Scanning slice: [{start}:{end}]  (count={len(universe)})")
    print(f"Strike window: ATM ± {STRIKES_AROUND_ATM} steps | Candle cache TTL: {CACHE_TTL_MINUTES} min")
    print(f"Capital: ₹{TRABOT_CAPITAL:,.0f} | Risk profile: {TRABOT_RISK_PROFILE}")
    print(f"Run: {ts_str}  (run_id={run_id})\n")

    # Phase-1: write a run manifest (config + environment) for reproducibility/debugging
    require_htf_align = REQUIRE_HTF_ALIGN_ENV.lower() in ("1", "true", "yes") if REQUIRE_HTF_ALIGN_ENV else (mode == "intraday")
    skip_chop = SKIP_CHOP_ENV.lower() in ("1", "true", "yes") if SKIP_CHOP_ENV else False

    global G_REQUIRE_HTF_ALIGN, G_SKIP_CHOP, G_BLOCK_LONG_PREMIUM_IVP
    G_REQUIRE_HTF_ALIGN = bool(require_htf_align)
    G_SKIP_CHOP = bool(skip_chop)
    G_BLOCK_LONG_PREMIUM_IVP = float(BLOCK_LONG_PREMIUM_IVP)

    manifest = {
        "schema_version": RECO_SCHEMA_VERSION,
        "run_id": run_id,
        "ts_start": ts_str,
        "mode": mode,
        "python": sys.version,
        "platform": platform.platform(),
        "env": env_snapshot(),
        "config": {
            "capital": TRABOT_CAPITAL,
            "risk_profile": TRABOT_RISK_PROFILE,
            "interval": INTERVAL,
            "htf_interval": HTF_INTERVAL,
            "dtf_interval": DTF_INTERVAL or "",
            "lookback_days": LOOKBACK_DAYS,
            "strikes_around_atm": STRIKES_AROUND_ATM,
            "stop_atr_mult": STOP_ATR_MULT,
            "target_atr_mult": TARGET_ATR_MULT,
            "time_stop_min": TIME_STOP_MIN,
            "min_dte_days": MIN_DTE_DAYS,
            "max_dte_days": MAX_DTE_DAYS,
            "min_mid_price": MIN_MID_PRICE,
            "max_spread_pct": MAX_SPREAD_PCT,
            "min_oi": MIN_OI,
            "min_vol": MIN_VOL,
            "risk_per_trade_pct": RISK_PER_TRADE_PCT,
            "require_htf_align": require_htf_align,
            "skip_chop": skip_chop,
            "block_long_premium_ivp": BLOCK_LONG_PREMIUM_IVP,
            "align_mode_env": ALIGN_MODE_ENV,
            "align_allow_flat": ALIGN_ALLOW_FLAT,
            "sentiment_enabled": SENTIMENT_ENABLED,
        },
        "cache": {
            "ttl_minutes": CACHE_TTL_MINUTES,
            "instruments_cache_path": INSTRUMENTS_CACHE_PATH,
            "candle_cache_dir": os.getenv("TRABOT_CANDLE_CACHE_DIR", os.path.join(DATA_DIR, "candle_cache")),
        },
        "universe": {
            "start": start,
            "end": end,
            "count": len(universe),
            "total": total,
            "sample_underlyings": [u.get('underlying') for u in universe[:20]],
        },
        "market_ctx": (market_ctx.__dict__ if market_ctx else None),
        "outputs": {},
    }
    manifest_path = write_run_manifest(run_id, manifest)

    cands = []
    for item in universe:
        try:
            c = _build_candidate(item, lot_map=lot_map, market_ctx=market_ctx)
            if c:
                cands.append(c)
        except Exception as e:
            print(f"[skip] {item['underlying']}: {e}")

        if SLEEP_BETWEEN_SYMBOLS > 0:
            time.sleep(SLEEP_BETWEEN_SYMBOLS)

    if not cands:
        print("No candidates found in this slice.")
        return

    # Prefer tradable first (pass_caps); fall back to all
    tradable = [c for c in cands if c.get("pass_caps")]
    if not tradable:
        tradable = cands

    bullish = sorted([c for c in tradable if c["side"] == "LONG"], key=lambda x: float(x["score"]), reverse=True)
    bearish = sorted([c for c in tradable if c["side"] == "SHORT"], key=lambda x: float(x["score"]))  # most negative first

    top2_buy = bullish[:TOP2]
    top2_sell = bearish[:TOP2]

    overall = sorted(tradable, key=lambda x: abs(float(x["score"])), reverse=True)
    top10 = overall[:TOP10]

    # ---- Print minimal actionable output ----
    _print_top2("TOP 2 BUY (Bullish: Buy CE)", top2_buy)
    _print_top2("TOP 2 SELL (Bearish: Buy PE)", top2_sell)
    _print_top10_min(f"TOP {TOP10} OVERALL (entry/sl/target only)", top10)

    # ---- Save outputs (latest + timestamped snapshot) ----
    os.makedirs("data", exist_ok=True)

    # Full scan results (latest + snapshot)
    df_all = pd.DataFrame(cands)
    df_all.to_csv(f"data/options_scan_results_v22{suffix}.csv", index=False)
    df_all.to_csv(f"data/options_scan_results_v22_{run_id}.csv", index=False)

    # Top10 minimal file (latest + snapshot)
    df_top10_min = pd.DataFrame([{
        "ts_reco": ts_str,
        "tradingsymbol": c["tradingsymbol"],
        "side": c["side"],
        "entry": c["entry"],
        "sl": c["sl"],
        "target": c["target"],
    } for c in top10])
    df_top10_min.to_csv(f"data/options_top10_v22{suffix}.csv", index=False)
    df_top10_min.to_csv(f"data/options_top10_v22_{run_id}.csv", index=False)

    # Combined reco snapshot (top2+top10)
    reco_rows = []
    for c in top2_buy:
        reco_rows.append(_reco_row(c, ts_str, run_id, "scan_options_v22", "TOP2_BUY", mode))
    for c in top2_sell:
        reco_rows.append(_reco_row(c, ts_str, run_id, "scan_options_v22", "TOP2_SELL", mode))
    for c in top10:
        reco_rows.append(_reco_row(c, ts_str, run_id, "scan_options_v22", f"TOP{TOP10}_OVERALL", mode))

    # Append-only history (never overwritten)
    append_history(reco_rows, path=RECO_HISTORY_PATH)

    # Latest + per-run snapshot of recommendations
    save_snapshot(reco_rows, f"data/reco_v22_{run_id}.csv")
    save_snapshot(reco_rows, f"data/reco_latest_v22{suffix}.csv")

    # Phase-1: update manifest with end stats, outputs, and quote timestamp range
    try:
        qts = [c.get("quote_ts") for c in cands if c.get("quote_ts")]
        qt_ser = pd.to_datetime(pd.Series(qts), errors="coerce") if qts else pd.Series([], dtype="datetime64[ns]")
        qt_min = qt_ser.min() if not qt_ser.empty else None
        qt_max = qt_ser.max() if not qt_ser.empty else None

        updates = {
            "ts_end": now_iso(),
            "stats": {
                "candidates": int(len(cands)),
                "tradable": int(len([c for c in cands if c.get("pass_caps")])),
                "top2_buy": int(len(top2_buy)),
                "top2_sell": int(len(top2_sell)),
                f"top{TOP10}_overall": int(len(top10)),
            },
            "quote_ts": {
                "n": int(len(qts)),
                "min": (qt_min.isoformat() if hasattr(qt_min, "isoformat") else str(qt_min)) if qt_min is not None and str(qt_min) != "NaT" else "",
                "max": (qt_max.isoformat() if hasattr(qt_max, "isoformat") else str(qt_max)) if qt_max is not None and str(qt_max) != "NaT" else "",
            },
            "outputs": {
                "options_scan_results_latest": f"data/options_scan_results_v22{suffix}.csv",
                "options_scan_results_run": f"data/options_scan_results_v22_{run_id}.csv",
                "options_top10_latest": f"data/options_top10_v22{suffix}.csv",
                "options_top10_run": f"data/options_top10_v22_{run_id}.csv",
                "reco_latest": f"data/reco_latest_v22{suffix}.csv",
                "reco_run": f"data/reco_v22_{run_id}.csv",
                "reco_history": RECO_HISTORY_PATH,
            },
        }
        update_run_manifest_v1(manifest_path, updates)
    except Exception:
        pass

    print(f"\nSaved: data/options_scan_results_v22{suffix}.csv")
    print(f"Saved: data/options_scan_results_v22_{run_id}.csv")
    print(f"Saved: data/options_top10_v22{suffix}.csv")
    print(f"Saved: data/options_top10_v22_{run_id}.csv")
    print(f"Saved: data/reco_latest_v22{suffix}.csv")
    print(f"Saved: data/reco_v22_{run_id}.csv")
    print(f"Saved: {manifest_path}")
    print(f"Appended: {RECO_HISTORY_PATH} (append-only, schema-stable)\n")

    # Lookup
    if LOOKUP_SYMBOL:
        mp = {str(c["tradingsymbol"]).upper(): i + 1 for i, c in enumerate(overall)}
        if LOOKUP_SYMBOL in mp:
            i = mp[LOOKUP_SYMBOL]
            c = next(x for x in overall if str(x["tradingsymbol"]).upper() == LOOKUP_SYMBOL)
            print(f"LOOKUP: {LOOKUP_SYMBOL} ranked #{i} (by |score|). entry={c['entry']:.2f} sl={c['sl']:.2f} tgt={c['target']:.2f}")
        else:
            print(f"LOOKUP: {LOOKUP_SYMBOL} not found in this scan.")

    print("NOTE: Research/education only. Not financial advice.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["intraday", "swing"], default=os.getenv("TRABOT_MODE", "intraday"))
    args = ap.parse_args()
    main(mode=args.mode)
