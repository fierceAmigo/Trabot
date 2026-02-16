"""
scan_options_global_v22.py

Global scan V2.2 (Kite-only).
Lean version of scan_options_v22.py:
- Still uses directional confidence gate + high IV gate
- Delta-target strike selection
- Option ATR SL/Target + time stop
- Output: Top2 Buy/Sell + Top10 minimal
- Writes: data/options_scan_global_v22_results{suffix}.csv + snapshots + reco_latest_global_v22{suffix}.csv
- Appends to shared data/reco_history.csv

Educational tool only – not financial advice.
"""

from __future__ import annotations

import os
import argparse
import time
import math
import datetime as dt
from functools import lru_cache

import pandas as pd

from kite_client import get_kite
from kite_chain import get_kite_chain_slice
from strategy import compute_signal

from iv_store import append_iv_snapshot, iv_percentile
from iv_greeks import time_to_expiry_years, implied_volatility, greeks
from risk_caps import compute_max_lots

from journal import append_history, save_snapshot, make_run_id


DATA_DIR = os.getenv("TRABOT_DATA_DIR", "data")

# Schema-stable reco history (prevents CSV field-count corruption)
SCHEMA_VERSION = os.getenv("TRABOT_SCHEMA_VERSION", "v22_p1").strip()
DEFAULT_RECO_HISTORY_PATH = os.path.join(DATA_DIR, f"reco_history_{SCHEMA_VERSION}.csv")
RECO_HISTORY_PATH = os.getenv("TRABOT_RECO_HISTORY", DEFAULT_RECO_HISTORY_PATH)

INSTRUMENTS_CACHE_PATH = os.getenv("INSTRUMENTS_CACHE_PATH", os.path.join(DATA_DIR, "kite_instruments_NFO.csv"))

CACHE_TTL_MINUTES = int(os.getenv("TRABOT_CACHE_TTL_MIN", "5"))

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "120"))
INTERVAL = os.getenv("INTERVAL", "60m")  # global scan tends to do 60m

STRIKES_AROUND_ATM = int(os.getenv("STRIKES_AROUND_ATM", "6"))

TOP2 = int(os.getenv("TRABOT_TOP2", "2"))
TOP10 = int(os.getenv("TRABOT_TOP10", "10"))
REFINE_TOPK = int(os.getenv("TRABOT_REFINE_TOPK", "25"))

SLEEP_BETWEEN_SYMBOLS = float(os.getenv("SLEEP_BETWEEN_SYMBOLS", "0.05"))

EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))
ADX_MIN = float(os.getenv("ADX_MIN", "18"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

ADX_GATE = float(os.getenv("TRABOT_ADX_GATE", "22"))
EMA_SEP_GATE = float(os.getenv("TRABOT_EMA_SEP_GATE", "0.0015"))
BOS_STRENGTH_GATE = float(os.getenv("TRABOT_BOS_STRENGTH_GATE", "0.60"))
MOVE_ATR_GATE = float(os.getenv("TRABOT_MOVE_ATR_GATE", "1.0"))

HIGH_IV_PCTL = float(os.getenv("TRABOT_HIGH_IV_PCTL", "0.70"))
HIGH_IV_ADX = float(os.getenv("TRABOT_HIGH_IV_ADX", "25"))
HIGH_IV_BOS = float(os.getenv("TRABOT_HIGH_IV_BOS", "0.80"))

OPT_ATR_INTERVAL = os.getenv("TRABOT_OPT_ATR_INTERVAL", "5minute")
OPT_ATR_PERIOD = int(os.getenv("TRABOT_OPT_ATR_PERIOD", "14"))
OPT_ATR_BARS = int(os.getenv("TRABOT_OPT_ATR_BARS", "80"))
SL_ATR_MULT_OPT = float(os.getenv("TRABOT_SL_ATR_MULT_OPT", "1.2"))
TGT_ATR_MULT_OPT = float(os.getenv("TRABOT_TGT_ATR_MULT_OPT", "1.8"))
TIME_STOP_MIN = int(os.getenv("TRABOT_TIME_STOP_MIN", "90"))

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

# --- Expert quality gates / sizing (v2.2.1) ---
MIN_MID_PRICE = float(os.getenv("TRABOT_MIN_MID_PRICE", "8"))          # ignore tiny premiums (tick noise)
MAX_SPREAD_PCT = float(os.getenv("TRABOT_MAX_SPREAD_PCT", "0.08"))     # max (ask-bid)/mid
MIN_OI = int(os.getenv("TRABOT_MIN_OI", "20000"))                      # per strike
MIN_VOL = int(os.getenv("TRABOT_MIN_VOL", "5000"))                     # per strike
RISK_PER_TRADE_PCT = float(os.getenv("TRABOT_RISK_PER_TRADE_PCT", "0.015"))  # stop-risk based sizing

# Expiry band (DTE) control (v2.2.x)
MIN_DTE_DAYS = int(os.getenv("TRABOT_MIN_DTE_DAYS", "0"))
_MAX_DTE_ENV = os.getenv("TRABOT_MAX_DTE_DAYS", "").strip()
MAX_DTE_DAYS = int(_MAX_DTE_ENV) if _MAX_DTE_ENV else None

# Dual-mode defaults (intraday + swing)
MODE_DEFAULTS = {
    # Spot interval uses market_data interval keys (e.g., 15m, 60m).
    "intraday": {
        "interval": "15m",
        "time_stop_min": 90,
        "min_dte": 0,
        "max_dte": 7,
        "opt_atr_interval": "5minute",
        "sl_mult_opt": 1.2,
        "tgt_mult_opt": 1.8,
        "risk_pct": 0.010,
    },
    "swing": {
        "interval": "60m",
        "time_stop_min": 2880,  # 2 days
        "min_dte": 4,
        "max_dte": 14,
        "opt_atr_interval": "15minute",
        "sl_mult_opt": 1.5,
        "tgt_mult_opt": 2.2,
        "risk_pct": 0.015,
    },
}


INDEX_SPOT_MAP = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
    "NIFTYNXT50": "NSE:NIFTY NEXT 50",
}

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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def fetch_history_cached(symbol: str, lookback_days: int, interval: str):
    from market_data import fetch_history_cached as _fh
    return _fh(symbol, lookback_days=lookback_days, interval=interval, ttl_minutes=CACHE_TTL_MINUTES)


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
    return [{"underlying": u, "spot": INDEX_SPOT_MAP.get(u, f"NSE:{u}")} for u in underlyings]


def build_lot_size_map() -> dict:
    if not os.path.exists(INSTRUMENTS_CACHE_PATH):
        return {}
    df = pd.read_csv(INSTRUMENTS_CACHE_PATH)
    df = df[df.get("segment", "").astype(str).str.upper() == "NFO-OPT"].copy()
    m = {}
    for _, r in df.iterrows():
        ts = str(r.get("tradingsymbol") or "").upper().strip()
        if not ts:
            continue
        try:
            m[ts] = int(r.get("lot_size") or 1)
        except Exception:
            m[ts] = 1
    return m


def _ema_sep(metrics: dict) -> float:
    try:
        close = float(metrics.get("close") or 0.0)
        ema_fast = float(metrics.get("ema_fast") or 0.0)
        ema_slow = float(metrics.get("ema_slow") or 0.0)
        if close > 0:
            return abs(ema_fast - ema_slow) / close
    except Exception:
        pass
    return 0.0


def _get_atr(metrics: dict, df: pd.DataFrame) -> float:
    if "atr" in metrics:
        try:
            v = float(metrics.get("atr"))
            if v > 0:
                return v
        except Exception:
            pass
    if df is None or df.empty:
        return 0.0
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0


def _bos_strength(df: pd.DataFrame, lookback: int = 20):
    if df is None or df.empty or len(df) < lookback + 5:
        return "NONE", 0.0
    highs = df["high"].rolling(lookback).max()
    lows = df["low"].rolling(lookback).min()
    close = float(df["close"].iloc[-1])
    hh = float(highs.iloc[-2]) if pd.notna(highs.iloc[-2]) else float(highs.iloc[-1])
    ll = float(lows.iloc[-2]) if pd.notna(lows.iloc[-2]) else float(lows.iloc[-1])
    tr = (df["high"] - df["low"]).rolling(14).mean()
    atr = float(tr.iloc[-1]) if pd.notna(tr.iloc[-1]) else max(1.0, float(df["high"].iloc[-1] - df["low"].iloc[-1]))
    if close > hh:
        return "BOS_UP", (close - hh) / max(atr, 1e-9)
    if close < ll:
        return "BOS_DOWN", (ll - close) / max(atr, 1e-9)
    return "NONE", 0.0


def directional_confidence(side: str, metrics: dict, bos_tag: str, bos_strength: float, htf_align: bool) -> int:
    adx = float(metrics.get("adx", 0.0) or 0.0)
    sep = _ema_sep(metrics)
    A = (adx >= ADX_GATE and sep >= EMA_SEP_GATE)
    B = (bos_tag == ("BOS_UP" if side == "LONG" else "BOS_DOWN") and bos_strength >= BOS_STRENGTH_GATE)
    C = bool(htf_align)
    return int(A) + int(B) + int(C)


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
    liq_max = liq_max if liq_max > 0 else 1.0
    x["liq_norm"] = x["liq"] / liq_max
    x["dist_atm"] = (pd.to_numeric(x["strike"], errors="coerce") - float(atm)).abs()
    x["mny_score"] = 1.0 / (1.0 + (x["dist_atm"] / max(atm, 1.0)) * 20.0)
    sp = x["spread_pct"].fillna(0.05).clip(lower=0.0, upper=0.30)
    x["spread_pen"] = (sp / 0.02).clip(lower=0.0, upper=6.0)
    x["strike_score"] = (x["liq_norm"] * 0.65 + x["mny_score"] * 0.35) - (x["spread_pen"] * 0.15)
    return x



def _safe_mid(bid: float | None, ask: float | None, ltp: float | None) -> float | None:
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if ltp is not None and ltp > 0:
        return float(ltp)
    return None


def _atm_iv_from_chain(chain) -> tuple[float | None, str]:
    """
    Estimate an ATM IV for the *underlying* using ATM CE/PE mid prices.
    Returns (atm_iv, confidence).
    """
    try:
        atm = int(chain.atm)
        ivs: list[float] = []
        confs: list[str] = []
        for df_side, right in [(chain.calls, "CE"), (chain.puts, "PE")]:
            if df_side is None or df_side.empty:
                continue
            row = df_side[df_side["strike"].astype(int) == atm].head(1)
            if row.empty:
                continue
            r = row.iloc[0].to_dict()
            bid = float(pd.to_numeric(r.get("bid"), errors="coerce") or 0) or None
            ask = float(pd.to_numeric(r.get("ask"), errors="coerce") or 0) or None
            ltp = float(pd.to_numeric(r.get("last_price"), errors="coerce") or 0) or None
            px = _safe_mid(bid, ask, ltp)
            if px is None or px <= 0:
                continue
            iv, _, conf, _ = _compute_iv_and_greeks(float(chain.spot), chain.expiry, right, atm, float(px))
            if iv and iv > 0:
                ivs.append(float(iv))
                confs.append(conf)

        if not ivs:
            return None, "low"
        atm_iv = float(sum(ivs) / len(ivs))
        confidence = "high" if all(c == "high" for c in confs) else "mid"
        return atm_iv, confidence
    except Exception:
        return None, "low"


def _risk_lots_from_stop(capital: float, entry: float, sl: float, lot_size: int) -> int:
    """
    Stop-risk based sizing (₹). Returns lots.
    """
    if capital <= 0 or entry <= 0 or sl <= 0 or lot_size <= 0:
        return 0
    loss_per_unit = max(0.0, entry - sl)
    if loss_per_unit <= 1e-9:
        return 0
    risk_rupees = float(capital) * float(RISK_PER_TRADE_PCT)
    lots = int(risk_rupees // (loss_per_unit * lot_size))
    return max(0, lots)


def pick_contract_delta_target(chain, want_right: str, delta_target: float = 0.50) -> dict | None:
    df_side = chain.calls if want_right == "CE" else chain.puts
    if df_side is None or df_side.empty:
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
    top = scored.sort_values("strike_score", ascending=False).head(6).copy()

    best = None
    best_key = None
    for _, r in top.iterrows():
        strike = int(r.get("strike"))
        mid = float(r.get("mid")) if pd.notna(r.get("mid")) else None
        ask = float(r.get("ask_num")) if pd.notna(r.get("ask_num")) else None
        ltp = float(r.get("last_price")) if pd.notna(r.get("last_price")) else None

        px = mid if mid and mid > 0 else (ask if ask and ask > 0 else ltp)
        if px is None or px <= 0:
            continue
        iv, g, conf, dte = _compute_iv_and_greeks(float(chain.spot), chain.expiry, want_right, strike, float(px))
        delta_abs = abs(float(g.get("delta", 0.0)))
        delta_pen = abs(delta_abs - delta_target)
        band_pen = 0.0 if (0.40 <= delta_abs <= 0.65) else 0.35
        spread_pct = float(r.get("spread_pct")) if pd.notna(r.get("spread_pct")) else 0.10
        key = (band_pen + delta_pen) + 0.3 * min(0.30, max(0.0, spread_pct)) - 0.1 * float(r.get("strike_score", 0.0))
        if best is None or key < best_key:
            best_key = key
            d = r.to_dict()
            d["_iv"] = iv
            d["_greeks"] = g
            d["_greeks_conf"] = conf
            d["_dte"] = int(dte)
            d["_px_ref"] = float(px)
            best = d
    return best


@lru_cache(maxsize=20000)
def _resolve_nfo_token(tradingsymbol: str) -> int:
    if not os.path.exists(INSTRUMENTS_CACHE_PATH):
        raise RuntimeError("Missing instruments cache.")
    df = pd.read_csv(INSTRUMENTS_CACHE_PATH)
    ts = str(tradingsymbol).upper().strip()
    hit = df[df["tradingsymbol"].astype(str).str.upper() == ts]
    if hit.empty:
        raise RuntimeError(f"Cannot resolve token for {tradingsymbol}")
    return int(hit.iloc[0]["instrument_token"])


def _chunked_historical(token: int, start: dt.datetime, end: dt.datetime, interval: str) -> list[dict]:
    kite = get_kite()
    max_days = MAX_DAYS_PER_REQ.get(interval, 30)
    cur = start
    out: list[dict] = []
    while cur < end:
        cur_end = min(end, cur + dt.timedelta(days=max_days))
        rows = kite.historical_data(token, cur, cur_end, interval, continuous=False, oi=False) or []
        out.extend(rows)
        cur = cur_end + dt.timedelta(seconds=1)
    return out


@lru_cache(maxsize=5000)
def fetch_option_candles_recent(tradingsymbol: str, interval: str, bars: int) -> pd.DataFrame:
    token = _resolve_nfo_token(tradingsymbol)
    end = dt.datetime.now()
    start = end - dt.timedelta(days=5)
    rows = _chunked_historical(token, start, end, interval)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"date": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").drop_duplicates(subset=["Datetime"]).set_index("Datetime")
    df = df[["open", "high", "low", "close", "volume"]].copy()
    if len(df) > bars:
        df = df.iloc[-bars:].copy()
    return df


def option_atr(df: pd.DataFrame, period: int) -> float:
    if df is None or df.empty or len(df) < period + 2:
        return 0.0
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0


def refine_levels(c: dict, opt_interval: str, opt_period: int, opt_bars: int, sl_mult: float, tgt_mult: float) -> dict:
    tsym = str(c.get("tradingsymbol"))
    entry = float(c.get("entry") or 0.0)
    if not tsym or entry <= 0:
        return c
    try:
        df_opt = fetch_option_candles_recent(tsym, opt_interval, opt_bars)
        atr = option_atr(df_opt, opt_period)
        if atr <= 0:
            return c
        c = c.copy()
        c["opt_atr"] = float(atr)
        c["sl"] = max(0.05, entry - sl_mult * atr)
        c["target"] = entry + tgt_mult * atr
        return c
    except Exception:
        return c


def reco_row(c: dict, ts_str: str, run_id: str, bucket: str, mode: str) -> dict:
    if c.get("trade_ok"):
        action = "BUY_CE" if c["side"] == "LONG" else "BUY_PE"
    else:
        action = "WATCH_CE" if c["side"] == "LONG" else "WATCH_PE"

    return {
        "ts_reco": ts_str,
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "run_id": run_id,
        "source": "scan_global_v22",
        "bucket": bucket,
        "action": action,
        "underlying": c.get("underlying"),
        "tradingsymbol": c.get("tradingsymbol"),
        "kite_symbol": c.get("kite_symbol"),
        "expiry": c.get("expiry"),
        "dte": c.get("dte"),
        "side": c.get("side"),
        "entry": c.get("entry"),
        "sl": c.get("sl"),
        "target": c.get("target"),
        "time_stop_min": c.get("time_stop_min", TIME_STOP_MIN),
        "score": c.get("score"),
        "trade_ok": c.get("trade_ok"),
        "conf_score": c.get("conf_score"),
        "bos": c.get("bos"),
        "bos_strength": c.get("bos_strength"),
        "htf_align": c.get("htf_align"),
        "move_atr_ratio": c.get("move_atr_ratio"),
        "high_iv_block": c.get("high_iv_block"),
        "regime": c.get("regime"),
        "iv": c.get("iv"),
        "iv_pct": c.get("iv_pct"),
        "greeks_conf": c.get("greeks_conf"),
        "delta": c.get("delta"),
        "vega_1pct": c.get("vega_1pct"),
        "theta_day": c.get("theta_day"),
        "lot_size": c.get("lot_size"),
        "greeks_max_lots": c.get("max_lots"),
        "risk_lots": c.get("risk_lots"),
        "max_lots": c.get("final_lots", c.get("max_lots")),
        "pass_caps": c.get("pass_caps"),
        "reason": c.get("reason"),
        "notes": c.get("notes"),
        "strike_score": c.get("strike_score"),
        "spread_pct": c.get("spread_pct"),
    }


def main(mode: str = "intraday"):
    run_ts = dt.datetime.now()
    ts_str = run_ts.isoformat(timespec="seconds")
    run_id = make_run_id(run_ts)

    # Mode-aware parameters (intraday vs swing)
    mode = (mode or os.getenv("TRABOT_MODE", "intraday")).strip().lower()
    if mode not in MODE_DEFAULTS:
        mode = "intraday"
    md = MODE_DEFAULTS[mode]

    interval = os.getenv("INTERVAL", md["interval"]) or md["interval"]
    time_stop_min = int(os.getenv("TRABOT_TIME_STOP_MIN", str(md["time_stop_min"])) or md["time_stop_min"])

    min_dte_days = int(os.getenv("TRABOT_MIN_DTE_DAYS", str(md["min_dte"])) or md["min_dte"])
    max_dte_days_env = os.getenv("TRABOT_MAX_DTE_DAYS", str(md["max_dte"] if md["max_dte"] is not None else "")).strip()
    max_dte_days = int(max_dte_days_env) if max_dte_days_env else None

    opt_atr_interval = os.getenv("TRABOT_OPT_ATR_INTERVAL", md["opt_atr_interval"]) or md["opt_atr_interval"]
    sl_mult_opt = float(os.getenv("TRABOT_SL_ATR_MULT_OPT", str(md["sl_mult_opt"])) or md["sl_mult_opt"])
    tgt_mult_opt = float(os.getenv("TRABOT_TGT_ATR_MULT_OPT", str(md["tgt_mult_opt"])) or md["tgt_mult_opt"])

    risk_pct = float(os.getenv("TRABOT_RISK_PER_TRADE_PCT", str(md["risk_pct"])) or md["risk_pct"])

    suffix = f"_{mode}"

    universe = build_universe_all_options()
    lot_map = build_lot_size_map()

    print(f"\n=== GLOBAL OPTIONS SCAN V2.2 ===")
    print(f"Universe: {len(universe)} underlyings | interval={interval}")
    print(f"Capital: ₹{TRABOT_CAPITAL:,.0f} | risk={TRABOT_RISK_PROFILE} | TTL={CACHE_TTL_MINUTES} min")
    print(f"Run: {ts_str} (run_id={run_id})\n")

    cands = []
    for item in universe:
        try:
            df, _ = fetch_history_cached(item["spot"], lookback_days=LOOKBACK_DAYS, interval=interval)
            if df is None or df.empty:
                continue

            sig = compute_signal(
                df=df,
                ema_fast=EMA_FAST, ema_slow=EMA_SLOW,
                rsi_period=RSI_PERIOD, adx_period=ADX_PERIOD, adx_min=ADX_MIN,
                atr_period=ATR_PERIOD,
                stop_atr_mult=1.5, target_atr_mult=2.2,
            )

            # side selection (live or watch)
            side = None
            is_live = False
            reason = None
            if sig.side in ("LONG", "SHORT"):
                side = sig.side
                is_live = True
                entry_u, stop_u, target_u = float(sig.entry), float(sig.stop), float(sig.target)
                reason = "LIVE"
            else:
                ws = str(sig.metrics.get("watch_side", "NONE"))
                wt = str(sig.metrics.get("watch_trigger", ""))
                if ws in ("LONG", "SHORT") and wt:
                    side = ws
                    is_live = False
                    entry_u = float(sig.metrics.get("watch_entry"))
                    stop_u = float(sig.metrics.get("watch_stop"))
                    target_u = float(sig.metrics.get("watch_target"))
                    reason = wt
                else:
                    continue

            bos_tag, bos_strength = _bos_strength(df, 20)

            # quick HTF align with 1D (since this is global scan)
            try:
                df_htf, _ = fetch_history_cached(item["spot"], lookback_days=LOOKBACK_DAYS, interval="day")
                sig_htf = compute_signal(
                    df=df_htf,
                    ema_fast=EMA_FAST, ema_slow=EMA_SLOW,
                    rsi_period=RSI_PERIOD, adx_period=ADX_PERIOD, adx_min=ADX_MIN,
                    atr_period=ATR_PERIOD,
                    stop_atr_mult=1.5, target_atr_mult=2.2,
                )
                htf_align = (sig_htf.side == "NO_TRADE") or (sig_htf.side == side)
            except Exception:
                htf_align = True

            atr_u = _get_atr(sig.metrics, df)
            move_u = abs(target_u - entry_u)
            move_atr_ratio = (move_u / atr_u) if atr_u > 1e-9 else 0.0

            chain = get_kite_chain_slice(
                underlying=item["underlying"],
                kite_spot_symbol=item["spot"],
                strike_step=0,
                strikes_around_atm=STRIKES_AROUND_ATM,
                cache_path=INSTRUMENTS_CACHE_PATH,
                min_dte_days=min_dte_days,
                max_dte_days=max_dte_days,
            )

            # IV percentile (ATM IV snapshot; skips invalid rows)
            atm_iv, atm_conf = _atm_iv_from_chain(chain)
            if atm_iv is not None:
                append_iv_snapshot({
                    "ts": dt.datetime.now().isoformat(timespec="seconds"),
                    "underlying": item["underlying"],
                    "tradingsymbol": "",
                    "expiry": chain.expiry,
                    "strike": int(chain.atm),
                    "right": "ATM",
                    "spot": float(chain.spot),
                    "price": 0.0,
                    "iv": float(atm_iv),
                    "confidence": atm_conf,
                })
            pct, n, _ = iv_percentile(item["underlying"], window_days=IV_PCTL_WINDOW_DAYS, ewma_span=IV_EWMA_SPAN)

            regime = "VOLATILE" if (pct is not None and pct >= HIGH_IV_PCTL) else ("TREND" if float(sig.metrics.get("adx", 0)) >= ADX_GATE else "CHOP")

            conf_score = directional_confidence(side, sig.metrics, bos_tag, float(bos_strength), bool(htf_align))

            want_right = "CE" if side == "LONG" else "PE"
            pick = pick_contract_delta_target(chain, want_right=want_right, delta_target=0.50)
            if not pick:
                continue

            tsym = str(pick.get("tradingsymbol"))
            strike = int(pick.get("strike"))
            entry_opt = float(pick.get("_px_ref", 0.0))
            if entry_opt <= 0:
                continue

            iv = float(pick.get("_iv", 0.5))
            g = pick.get("_greeks") or {}
            greeks_conf = str(pick.get("_greeks_conf", "low"))
            dte = int(pick.get("_dte", 0))
            lot = int(lot_map.get(tsym.upper(), 1))

            # caps
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

            # High IV gate
            high_iv_block = False
            if pct is not None and pct >= HIGH_IV_PCTL:
                adx = float(sig.metrics.get("adx", 0.0) or 0.0)
                if not (regime == "TREND" and adx >= HIGH_IV_ADX and bos_strength >= HIGH_IV_BOS and htf_align):
                    high_iv_block = True

            trade_ok = bool(is_live and pass_caps and conf_score >= 2 and move_atr_ratio >= MOVE_ATR_GATE and (not high_iv_block) and greeks_conf != "low")

            adx = float(sig.metrics.get("adx", 0.0) or 0.0)
            base = (0.85 + 0.02 * adx)
            base *= (1.0 + 0.05 * (conf_score - 1))
            if pct is not None and pct >= HIGH_IV_PCTL:
                base *= 0.90
            if dte <= 3:
                base *= 0.78
            if move_atr_ratio < 1.0:
                base *= 0.85
            if greeks_conf == "low":
                base *= 0.85
            score = base if side == "LONG" else -base

            # Explainability notes (expert-style)
            notes = []
            notes.append(f"conf={conf_score}/3")
            notes.append(f"adx={float(sig.metrics.get('adx',0.0) or 0.0):.1f}")
            if pct is not None:
                notes.append(f"ivp={float(pct):.2f}")
            notes.append(f"dte={dte}")
            try:
                notes.append(f"spr={float(pick.get('spread_pct', 0.0)):.3f}")
            except Exception:
                pass
            try:
                notes.append(f"liq={float(pick.get('strike_score', 0.0)):.2f}")
            except Exception:
                pass
            if high_iv_block:
                notes.append("HIGH_IV_BLOCK")
            if greeks_conf == "low":
                notes.append("GREEKS_LOW")
            if move_atr_ratio < MOVE_ATR_GATE:
                notes.append("MOVE_WEAK")
            notes_str = " | ".join(notes)

            cands.append({
                "underlying": item["underlying"],
                "spot_symbol": item["spot"],
                "expiry": chain.expiry,
                "dte": int(dte),
                "regime": regime,
                "side": side,
                "is_live": bool(is_live),
                "trade_ok": bool(trade_ok),
                "conf_score": int(conf_score),
                "bos": bos_tag,
                "bos_strength": float(bos_strength),
                "htf_align": bool(htf_align),
                "move_atr_ratio": float(move_atr_ratio),
                "high_iv_block": bool(high_iv_block),

                "strike_score": float(pick.get("strike_score", 0.0)) if pick.get("strike_score") == pick.get("strike_score") else float("nan"),
                "spread_pct": float(pick.get("spread_pct", 0.0)) if pick.get("spread_pct") == pick.get("spread_pct") else float("nan"),
                "notes": notes_str,

                "tradingsymbol": tsym,
                "kite_symbol": f"NFO:{tsym}",
                "strike": int(strike),

                "entry": float(entry_opt),
                "sl": float("nan"),
                "target": float("nan"),
                "time_stop_min": int(time_stop_min),

                "iv": float(iv),
                "iv_pct": float(pct) if pct is not None else float("nan"),
                "iv_samples": int(n),

                "greeks_conf": greeks_conf,
                "delta": float(g.get("delta", 0.0)),
                "vega_1pct": float(g.get("vega_1pct", 0.0)),
                "theta_day": float(g.get("theta_day", 0.0)),

                "lot_size": int(lot),
                "max_lots": int(max_lots),
                "pass_caps": bool(pass_caps),
                "reason": reason,
                "score": float(score),
            })

        except Exception as e:
            print(f"[skip] {item['underlying']}: {e}")

        time.sleep(SLEEP_BETWEEN_SYMBOLS)

    if not cands:
        print("No candidates found.")
        return

    ranked = sorted(cands, key=lambda x: abs(float(x["score"])), reverse=True)

    # refine topK with option ATR
    refined = [refine_levels(c, opt_atr_interval, OPT_ATR_PERIOD, OPT_ATR_BARS, sl_mult_opt, tgt_mult_opt) for c in ranked[:max(REFINE_TOPK, TOP10, TOP2 * 2)]]
    ref_map = {r["tradingsymbol"]: r for r in refined}
    ranked = [ref_map.get(c["tradingsymbol"], c) for c in ranked]

    for c in ranked:
        if not (isinstance(c.get("sl"), (int, float)) and pd.notna(c.get("sl"))):
            c["sl"] = max(0.05, float(c["entry"]) * 0.70)
        if not (isinstance(c.get("target"), (int, float)) and pd.notna(c.get("target"))):
            c["target"] = float(c["entry"]) * 1.35

    # Final sizing: stop-risk lots + greeks lots
    for c in ranked:
        try:
            rl = _risk_lots_from_stop(TRABOT_CAPITAL, float(c["entry"]), float(c["sl"]), int(c.get("lot_size", 1)))
        except Exception:
            rl = 0
        c["risk_lots"] = int(rl)
        if rl > 0:
            c["final_lots"] = int(min(int(c.get("max_lots", 0)), rl))
        else:
            c["final_lots"] = int(c.get("max_lots", 0))
        c["pass_caps"] = bool(int(c.get("final_lots", 0)) >= 1)

        # If previously trade_ok, re-check lots
        if c.get("trade_ok"):
            c["trade_ok"] = bool(int(c.get("final_lots", 0)) >= 1)

    trade = [c for c in ranked if c.get("trade_ok")]
    watch = [c for c in ranked if not c.get("trade_ok")]

    buy = [c for c in trade if c["side"] == "LONG"][:TOP2] + [c for c in watch if c["side"] == "LONG"][:max(0, TOP2 - len([c for c in trade if c["side"] == "LONG"][:TOP2]))]
    sell = [c for c in trade if c["side"] == "SHORT"][:TOP2] + [c for c in watch if c["side"] == "SHORT"][:max(0, TOP2 - len([c for c in trade if c["side"] == "SHORT"][:TOP2]))]
    top10 = ranked[:TOP10]

    print("\nTOP 2 BUY (CE) [TRADE preferred]")
    for c in buy:
        print(f"  {c['tradingsymbol']:<22s} {'TRADE' if c.get('trade_ok') else 'WATCH':<5s} entry={c['entry']:.2f} sl={c['sl']:.2f} tgt={c['target']:.2f} score={c['score']:+.2f}")

    print("\nTOP 2 SELL (PE) [TRADE preferred]")
    for c in sell:
        print(f"  {c['tradingsymbol']:<22s} {'TRADE' if c.get('trade_ok') else 'WATCH':<5s} entry={c['entry']:.2f} sl={c['sl']:.2f} tgt={c['target']:.2f} score={c['score']:+.2f}")

    print(f"\nTOP {TOP10} OVERALL (entry/sl/target only)")
    for i, c in enumerate(top10, 1):
        print(f"  #{i:02d} {c['tradingsymbol']:<22s} entry={c['entry']:.2f} sl={c['sl']:.2f} tgt={c['target']:.2f}")

    _ensure_dir(DATA_DIR)

    df_all = pd.DataFrame(ranked)
    df_all.to_csv(os.path.join(DATA_DIR, f"options_scan_global_v22_results{suffix}.csv"), index=False)
    df_all.to_csv(os.path.join(DATA_DIR, f"options_scan_global_v22_results{suffix}_{run_id}.csv"), index=False)

    df_top10 = pd.DataFrame([{
        "ts_reco": ts_str,
        "tradingsymbol": c["tradingsymbol"],
        "side": c["side"],
        "entry": c["entry"],
        "sl": c["sl"],
        "target": c["target"],
        "time_stop_min": c.get("time_stop_min", TIME_STOP_MIN),
    } for c in top10])
    df_top10.to_csv(os.path.join(DATA_DIR, f"options_global_top10_v22{suffix}.csv"), index=False)
    df_top10.to_csv(os.path.join(DATA_DIR, f"options_global_top10_v22{suffix}_{run_id}.csv"), index=False)

    reco_rows = []
    for c in buy:
        reco_rows.append(reco_row(c, ts_str, run_id, "TOP2_BUY", mode))
    for c in sell:
        reco_rows.append(reco_row(c, ts_str, run_id, "TOP2_SELL", mode))
    for c in top10:
        reco_rows.append(reco_row(c, ts_str, run_id, f"TOP{TOP10}_OVERALL", mode))

    append_history(reco_rows, path=RECO_HISTORY_PATH)
    save_snapshot(reco_rows, os.path.join(DATA_DIR, f"reco_latest_global_v22{suffix}.csv"))
    save_snapshot(reco_rows, os.path.join(DATA_DIR, f"reco_global_v22{suffix}_{run_id}.csv"))

    print(f"\nSaved: {os.path.join(DATA_DIR, 'options_scan_global_v22_results{suffix}.csv')}")
    print(f"Saved: {os.path.join(DATA_DIR, 'options_global_top10_v22{suffix}.csv')}")
    print(f"Saved: {os.path.join(DATA_DIR, 'reco_latest_global_v22{suffix}.csv')}")
    print(f"Appended: {os.path.join(DATA_DIR, 'reco_history.csv')}")
    print("\nNOTE: Research/education only. Not financial advice.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["intraday", "swing"], default=os.getenv("TRABOT_MODE", "intraday"))
    args = ap.parse_args()
    main(mode=args.mode)
