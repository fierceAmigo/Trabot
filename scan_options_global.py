import os
import math
import time
from pathlib import Path
from dataclasses import dataclass
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

# backtest helper (for learned edge)
from backtest import _simulate_forward


# ============================================================
# Runtime knobs (env vars)
# ============================================================
TOP_BEST = int(os.environ.get("TRABOT_TOP_BEST", "5"))
TOP_RANK = int(os.environ.get("TRABOT_TOP_RANK", "20"))
STRIKES_AROUND_ATM = int(os.environ.get("TRABOT_STRIKES_AROUND_ATM", "6"))

TRABOT_CAPITAL = float(os.environ.get("TRABOT_CAPITAL", "20000"))  # INR
TRABOT_RISK_PROFILE = os.environ.get("TRABOT_RISK_PROFILE", "high").strip().lower()  # high | moderate

# V1 risk caps (defaults you asked for)
PER_TRADE_RISK_PCT = float(os.environ.get("TRABOT_PER_TRADE_RISK_PCT", "0.06"))   # 6%
TOTAL_RISK_PCT = float(os.environ.get("TRABOT_TOTAL_RISK_PCT", "0.18"))           # 18%

# Liquidity thresholds (index vs stock)
MAX_SPREAD_PCT_INDEX = float(os.environ.get("TRABOT_MAX_SPREAD_PCT_INDEX", "0.003"))  # 0.30%
MAX_SPREAD_PCT_STOCK = float(os.environ.get("TRABOT_MAX_SPREAD_PCT_STOCK", "0.03"))   # 3.00%

MIN_OI_INDEX = int(os.environ.get("TRABOT_MIN_OI_INDEX", "20000"))
MIN_VOL_INDEX = int(os.environ.get("TRABOT_MIN_VOL_INDEX", "5000"))

MIN_OI_STOCK = int(os.environ.get("TRABOT_MIN_OI_STOCK", "2000"))
MIN_VOL_STOCK = int(os.environ.get("TRABOT_MIN_VOL_STOCK", "500"))

# Slippage model
TICK_SIZE = float(os.environ.get("TRABOT_TICK_SIZE", "0.05"))
SLIPPAGE_TICKS = int(os.environ.get("TRABOT_SLIPPAGE_TICKS", "1"))

# Cache & scan slice
CACHE_TTL_MINUTES = int(os.environ.get("TRABOT_CACHE_TTL_MIN", "60"))
CANDLE_CACHE_DIR = Path("data/candle_cache")
SLEEP_BETWEEN_SYMBOLS = float(os.environ.get("TRABOT_SLEEP", "0.05"))

UNIVERSE_START = int(os.environ.get("TRABOT_UNIVERSE_START", "0"))
UNIVERSE_COUNT = os.environ.get("TRABOT_UNIVERSE_COUNT", "")
UNIVERSE_COUNT = int(UNIVERSE_COUNT) if UNIVERSE_COUNT.strip() else None

LOOKUP_SYMBOL = os.environ.get("TRABOT_LOOKUP", "").strip().upper()


# ============================================================
# Index mapping (Kite)
# ============================================================
INDEX_SPOT_MAP = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
    "NIFTYNXT50": "NSE:NIFTY NEXT 50",
}
INDEX_UNDERLYINGS = set(INDEX_SPOT_MAP.keys())


# ============================================================
# Optional global context (US cues via yfinance)
# ============================================================
def _try_import_yf():
    try:
        import yfinance as yf  # type: ignore
        return yf
    except Exception:
        return None


def _pct(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b * 100.0


def fetch_global_context():
    """
    global_bias in [-1,+1]
      +1 risk-on, -1 risk-off
    """
    yf = _try_import_yf()
    ctx = {"available": False, "global_bias": 0.0, "notes": [], "metrics": {}}
    if yf is None:
        ctx["notes"].append("yfinance not installed -> global bias disabled. (pip install yfinance)")
        return ctx

    tickers = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
        "CRUDE": "CL=F",
        "US10Y": "^TNX",
    }

    def last_two_closes(tkr):
        try:
            df = yf.download(tkr, period="10d", interval="1d", progress=False)
            if df is None or df.empty:
                return None, None, None
            closes = df["Close"].dropna()
            if len(closes) < 2:
                return None, None, None
            prev = float(closes.iloc[-2])
            last = float(closes.iloc[-1])
            dt = closes.index[-1].to_pydatetime()
            return prev, last, dt
        except Exception:
            return None, None, None

    data = {}
    for k, tkr in tickers.items():
        prev, last, dt = last_two_closes(tkr)
        data[k] = {"prev": prev, "last": last, "date": dt, "ret_pct": _pct(last, prev)}

    # Risk-on: SPY up, QQQ up, VIX down, DXY down, crude not spiking
    score = 0.0
    w = {"SPY": 0.35, "QQQ": 0.25, "VIX": 0.25, "DXY": 0.10, "CRUDE": 0.05}

    if data["SPY"]["ret_pct"] is not None:
        score += w["SPY"] * (data["SPY"]["ret_pct"] / 1.0)
    if data["QQQ"]["ret_pct"] is not None:
        score += w["QQQ"] * (data["QQQ"]["ret_pct"] / 1.0)
    if data["VIX"]["ret_pct"] is not None:
        score += w["VIX"] * (-(data["VIX"]["ret_pct"] / 2.0))
    if data["DXY"]["ret_pct"] is not None:
        score += w["DXY"] * (-(data["DXY"]["ret_pct"] / 0.5))
    if data["CRUDE"]["ret_pct"] is not None:
        score += w["CRUDE"] * (-(data["CRUDE"]["ret_pct"] / 1.0))

    bias = math.tanh(score / 1.5)
    ctx["available"] = True
    ctx["global_bias"] = float(max(-1.0, min(1.0, bias)))
    ctx["metrics"] = data

    if ctx["global_bias"] > 0.25:
        ctx["notes"].append("Global cue: RISK-ON tilt (supports bullish follow-through).")
    elif ctx["global_bias"] < -0.25:
        ctx["notes"].append("Global cue: RISK-OFF tilt (supports defensive / bearish setups).")
    else:
        ctx["notes"].append("Global cue: MIXED/NEUTRAL (prefer defined-risk / volatility-aware setups).")
    return ctx


# ============================================================
# Utilities
# ============================================================
def _slip(price: float, action: str) -> float:
    """
    Simple slippage model:
      BUY pays a bit more, SELL receives a bit less.
    """
    if price is None or not (price == price):
        return price
    delta = SLIPPAGE_TICKS * TICK_SIZE
    if action.upper() == "BUY":
        return float(price + delta)
    if action.upper() == "SELL":
        return float(max(0.0, price - delta))
    return float(price)


def _session_tag(ts: pd.Timestamp) -> str:
    mins = ts.hour * 60 + ts.minute
    if 9 * 60 + 15 <= mins < 10 * 60 + 30:
        return "OPEN"
    if 10 * 60 + 30 <= mins < 14 * 60 + 30:
        return "MID"
    if 14 * 60 + 30 <= mins <= 15 * 60 + 30:
        return "CLOSE"
    return "MARKET_CLOSED"


def _session_factor(tag: str) -> float:
    if tag == "OPEN":
        return 1.00
    if tag == "MID":
        return 1.05
    if tag == "CLOSE":
        return 0.95
    return 0.85  # slightly less aggressive than before


def _cache_key(symbol: str, interval: str) -> Path:
    safe = symbol.replace(":", "_").replace("/", "_").replace(" ", "_")
    return CANDLE_CACHE_DIR / f"{safe}__{interval}.csv"


def fetch_history_cached(symbol: str, lookback_days: int, interval: str):
    CANDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _cache_key(symbol, interval)

    if fp.exists():
        age_sec = time.time() - fp.stat().st_mtime
        if age_sec <= CACHE_TTL_MINUTES * 60:
            try:
                df = pd.read_csv(fp)
                if "Datetime" in df.columns:
                    df["Datetime"] = pd.to_datetime(df["Datetime"])
                    df = df.set_index("Datetime")
                else:
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    df = df.set_index(df.columns[0])
                df.columns = [c.lower() for c in df.columns]
                needed = {"open", "high", "low", "close", "volume"}
                if needed.issubset(set(df.columns)):
                    return df, interval
            except Exception:
                pass

    df, used_interval = fetch_history(symbol, lookback_days=lookback_days, interval=interval)
    out = df.copy()
    out.index.name = "Datetime"
    out.to_csv(fp)
    return df, used_interval


def _load_or_fetch_nfo_instruments(cache_path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            if not df.empty:
                return df
        except Exception:
            pass
    kite = get_kite()
    inst = kite.instruments("NFO")
    df = pd.DataFrame(inst)
    df.to_csv(cache_path, index=False)
    return df


def build_universe_all_options() -> list[dict]:
    df = _load_or_fetch_nfo_instruments(INSTRUMENTS_CACHE_PATH)
    df_opt = df[df["segment"].astype(str) == "NFO-OPT"].copy()
    names = sorted(set(x for x in df_opt["name"].dropna().astype(str).tolist() if x.strip()))
    universe = []
    for name in names:
        spot = INDEX_SPOT_MAP.get(name, f"NSE:{name}")
        universe.append({"underlying": name, "spot": spot})
    return universe


def build_lot_size_map() -> dict:
    df = _load_or_fetch_nfo_instruments(INSTRUMENTS_CACHE_PATH)
    mp = {}
    if "tradingsymbol" in df.columns and "lot_size" in df.columns:
        for _, r in df.iterrows():
            ts = str(r.get("tradingsymbol", "")).strip().upper()
            ls = r.get("lot_size", None)
            if ts and ls is not None and not (isinstance(ls, float) and math.isnan(ls)):
                try:
                    mp[ts] = int(ls)
                except Exception:
                    pass
    return mp


# ============================================================
# Regime classification (V1)
# ============================================================
def classify_regime(metrics: dict) -> str:
    """
    Very practical regime labels:
      - TREND: ADX strong and EMA separation meaningful
      - CHOP: weak ADX and low/medium vol
      - VOLATILE: ATR% high
    """
    close = float(metrics.get("close", 0.0))
    adx = float(metrics.get("adx", 0.0))
    atr_pct = float(metrics.get("atr_pct", 0.0))
    ema_fast = float(metrics.get("ema_fast", 0.0))
    ema_slow = float(metrics.get("ema_slow", 0.0))

    if close <= 0:
        return "UNKNOWN"

    ema_sep = abs(ema_fast - ema_slow) / close  # normalized separation

    # Volatile overrides
    if atr_pct >= 0.012:  # ~1.2%
        return "VOLATILE"

    # Trend if ADX + EMA separation
    if adx >= max(20.0, float(ADX_MIN)) and ema_sep >= 0.0008:
        return "TREND"

    # Otherwise chop / range-ish
    return "CHOP"


# ============================================================
# Liquidity filter (V1)
# ============================================================
def _liq_thresholds(underlying: str):
    if underlying in INDEX_UNDERLYINGS:
        return MIN_OI_INDEX, MIN_VOL_INDEX, MAX_SPREAD_PCT_INDEX
    return MIN_OI_STOCK, MIN_VOL_STOCK, MAX_SPREAD_PCT_STOCK


def _passes_liquidity(underlying: str, oi: float, vol: float, spread_pct: float) -> tuple[bool, str]:
    min_oi, min_vol, max_spread = _liq_thresholds(underlying)
    if oi is None or vol is None or spread_pct is None:
        return False, "missing liquidity fields"
    if oi < min_oi:
        return False, f"OI<{min_oi}"
    if vol < min_vol:
        return False, f"VOL<{min_vol}"
    if spread_pct > max_spread:
        return False, f"spread>{max_spread*100:.2f}%"
    return True, "ok"


# ============================================================
# Strike scoring and selection
# ============================================================
def _score_strike_rows(df_side: pd.DataFrame, atm: int) -> pd.DataFrame:
    x = df_side.copy()
    x["oi_num"] = pd.to_numeric(x.get("oi"), errors="coerce").fillna(0.0)
    x["vol_num"] = pd.to_numeric(x.get("volume"), errors="coerce").fillna(0.0)
    x["bid_num"] = pd.to_numeric(x.get("bid"), errors="coerce")
    x["ask_num"] = pd.to_numeric(x.get("ask"), errors="coerce")
    x["mid"] = (x["bid_num"] + x["ask_num"]) / 2.0
    x["spread_pct"] = (x["ask_num"] - x["bid_num"]) / x["mid"]
    x["spread_pct"] = x["spread_pct"].replace([math.inf, -math.inf], math.nan)

    x["liq"] = (x["oi_num"].add(1).apply(math.log)) * 0.7 + (x["vol_num"].add(1).apply(math.log)) * 0.3
    liq_max = float(x["liq"].max()) if len(x) else 1.0
    liq_max = liq_max if liq_max > 0 else 1.0
    x["liq_norm"] = x["liq"] / liq_max

    x["dist_atm"] = (pd.to_numeric(x["strike"], errors="coerce") - float(atm)).abs()
    x["mny_score"] = 1.0 / (1.0 + (x["dist_atm"] / max(atm, 1.0)) * 20.0)

    sp = x["spread_pct"].fillna(0.05).clip(lower=0.0, upper=0.20)
    x["spread_pen"] = (sp / 0.02).clip(lower=0.0, upper=5.0)

    x["strike_score"] = (x["liq_norm"] * 0.65 + x["mny_score"] * 0.35) - (x["spread_pen"] * 0.15)
    return x


def _pick_best_contract(underlying: str, chain, want_right: str) -> dict | None:
    df_side = chain.calls if want_right == "CE" else chain.puts
    if df_side is None or df_side.empty:
        return None

    scored = _score_strike_rows(df_side, atm=int(chain.atm))
    scored = scored.dropna(subset=["bid_num", "ask_num", "spread_pct"], how="any")
    if scored.empty:
        return None

    scored = scored.sort_values("strike_score", ascending=False)

    # Liquidity hard filter: pick best that passes
    for _, r in scored.iterrows():
        ok, _why = _passes_liquidity(
            underlying=underlying,
            oi=float(r.get("oi_num", 0.0)),
            vol=float(r.get("vol_num", 0.0)),
            spread_pct=float(r.get("spread_pct", 999.0)),
        )
        if ok:
            return r.to_dict()

    return None


def _approx_delta(right: str, spot: float, strike: int) -> float:
    if spot <= 0:
        return 0.5
    m = (spot - strike) / spot
    if right == "CE":
        if m > 0.01:
            return 0.60
        if abs(m) < 0.005:
            return 0.50
        return 0.40
    else:
        if m < -0.01:
            return 0.60
        if abs(m) < 0.005:
            return 0.50
        return 0.40


# ============================================================
# Learned edge (cached) — still from last ~month
# ============================================================
_EDGE_CACHE = {}  # key: (symbol, interval) -> edge dict


def _learn_edge(df: pd.DataFrame, used_interval: str) -> dict:
    """
    Walk-forward-ish summary for LONG/SHORT.
    Cached per underlying to avoid recompute spam.
    """
    max_hold_bars = 24 if used_interval != "1d" else 5
    min_bars = max(EMA_SLOW, RSI_PERIOD, ADX_PERIOD, ATR_PERIOD) + 5

    stats = {"LONG": {"trades": 0, "wins": 0, "sum_r": 0.0},
             "SHORT": {"trades": 0, "wins": 0, "sum_r": 0.0}}

    for i in range(min_bars, len(df) - 2):
        hist = df.iloc[: i + 1].copy()
        sig = compute_signal(
            df=hist,
            ema_fast=EMA_FAST, ema_slow=EMA_SLOW,
            rsi_period=RSI_PERIOD, adx_period=ADX_PERIOD, adx_min=ADX_MIN,
            atr_period=ATR_PERIOD,
            stop_atr_mult=STOP_ATR_MULT, target_atr_mult=TARGET_ATR_MULT,
        )
        if sig.side == "NO_TRADE":
            continue

        entry_i = i + 1
        entry_open = float(df["open"].iloc[entry_i])

        # align to forward bar open
        delta = entry_open - float(sig.entry)
        stop = float(sig.stop) + delta
        target = float(sig.target) + delta

        t = _simulate_forward(df, entry_i, sig.side, stop, target, max_hold_bars)
        if not t:
            continue

        side = t["side"]
        stats[side]["trades"] += 1
        stats[side]["sum_r"] += float(t["r"])
        if float(t["r"]) > 0:
            stats[side]["wins"] += 1

    out = {}
    for side in ["LONG", "SHORT"]:
        tr = stats[side]["trades"]
        win_rate = (stats[side]["wins"] / tr * 100.0) if tr else 0.0
        avg_r = (stats[side]["sum_r"] / tr) if tr else 0.0
        out[side] = {"trades": tr, "win_rate": win_rate, "avg_r": avg_r}
    return out


# ============================================================
# Candidate + ranking (with regime + global + slippage penalties)
# ============================================================
def _global_adjust(rank_score: float, side: str, global_bias: float) -> float:
    # Weighting rule you asked: reduce bullish setups under risk-off by ~30% (smoothly)
    k = 0.30
    if side == "LONG":
        mult = 1.0 + k * global_bias
    else:
        mult = 1.0 - k * global_bias
    return float(max(0.0, rank_score * mult))


@dataclass
class Candidate:
    underlying: str
    spot_symbol: str
    expiry: str
    side: str  # LONG/SHORT
    right: str  # CE/PE
    action: str  # BUY_CE/BUY_PE/WATCH_*
    is_live: bool
    regime: str

    tradingsymbol: str
    strike: int
    lot_size: int

    spot: float
    atm: int

    entry: float
    sl: float
    target: float

    entry_u: float
    sl_u: float
    target_u: float

    score: float
    rank_score: float

    spread_pct: float
    oi: float
    volume: float

    reason: str
    reasons: list


def build_candidate(item: dict, lot_map: dict, global_ctx: dict) -> Candidate | None:
    df, used_interval = fetch_history_cached(item["spot"], lookback_days=LOOKBACK_DAYS, interval=INTERVAL)
    if df is None or df.empty:
        return None

    sig = compute_signal(
        df=df,
        ema_fast=EMA_FAST, ema_slow=EMA_SLOW,
        rsi_period=RSI_PERIOD, adx_period=ADX_PERIOD, adx_min=ADX_MIN,
        atr_period=ATR_PERIOD,
        stop_atr_mult=STOP_ATR_MULT, target_atr_mult=TARGET_ATR_MULT,
    )

    # pick side: live or watch
    is_live = False
    side = None
    reason = None
    entry_u = stop_u = target_u = None

    if sig.side in ("LONG", "SHORT"):
        is_live = True
        side = sig.side
        reason = "LIVE signal (filters met)"
        entry_u, stop_u, target_u = float(sig.entry), float(sig.stop), float(sig.target)
    else:
        ws = str(sig.metrics.get("watch_side", "NONE"))
        wt = str(sig.metrics.get("watch_trigger", ""))
        if ws in ("LONG", "SHORT") and wt:
            side = ws
            reason = wt
            entry_u = float(sig.metrics.get("watch_entry"))
            stop_u = float(sig.metrics.get("watch_stop"))
            target_u = float(sig.metrics.get("watch_target"))

    if side is None:
        return None

    regime = classify_regime(sig.metrics)

    # Option chain
    chain = get_kite_chain_slice(
        underlying=item["underlying"],
        kite_spot_symbol=item["spot"],
        strike_step=0,
        strikes_around_atm=STRIKES_AROUND_ATM,
        cache_path=INSTRUMENTS_CACHE_PATH,
    )

    # align underlying levels to live spot
    candle_close = float(sig.metrics.get("close", chain.spot))
    delta_spot = float(chain.spot) - candle_close
    entry_u += delta_spot
    stop_u += delta_spot
    target_u += delta_spot

    want_right = "CE" if side == "LONG" else "PE"
    pick = _pick_best_contract(item["underlying"], chain, want_right)
    if not pick:
        return None

    tsym = str(pick.get("tradingsymbol", ""))
    strike = int(pick.get("strike", 0))

    lot = int(lot_map.get(tsym.upper(), 1))
    lot = lot if lot > 0 else 1

    bid = float(pick.get("bid_num", float("nan")))
    ask = float(pick.get("ask_num", float("nan")))
    spread_pct = float(pick.get("spread_pct", float("nan")))
    oi = float(pick.get("oi_num", 0.0))
    vol = float(pick.get("vol_num", 0.0))

    ok, why = _passes_liquidity(item["underlying"], oi, vol, spread_pct)
    if not ok:
        return None

    # Use BUY at ask (with slippage)
    entry_opt = _slip(float(ask), "BUY")
    if entry_opt is None or entry_opt <= 0:
        return None

    # Premium SL/Target (delta-proxy, with a bit of slippage realism)
    d_abs = _approx_delta(want_right, float(chain.spot), strike)
    risk_u = abs(entry_u - stop_u)
    rew_u = abs(target_u - entry_u)

    stop_opt = max(0.05, entry_opt - d_abs * risk_u)
    target_opt = entry_opt + d_abs * rew_u

    # slippage on exits (conservative)
    stop_opt_eff = _slip(stop_opt, "SELL")   # assume we sell to exit long option
    target_opt_eff = _slip(target_opt, "SELL")

    # Learned edge (cached)
    edge_key = (item["spot"], used_interval)
    if edge_key not in _EDGE_CACHE:
        _EDGE_CACHE[edge_key] = _learn_edge(df, used_interval)
    edge = _EDGE_CACHE[edge_key]
    edge_side = edge[side]

    # Score breakdown (similar to your scan_options.py style)
    adx = float(sig.metrics.get("adx", 0.0))
    sess = _session_tag(pd.Timestamp(df.index[-1]))
    sess_factor = _session_factor(sess)

    edge_component = (5.0 * float(edge_side["avg_r"])) + (0.03 * float(edge_side["win_rate"]))
    trend_component = (0.01 * adx)

    # Regime multiplier: avoid overconfidence in CHOP for directional ideas
    regime_mult = 1.0
    if regime == "CHOP":
        regime_mult = 0.88
    elif regime == "TREND":
        regime_mult = 1.08
    elif regime == "VOLATILE":
        regime_mult = 1.00

    raw = (edge_component + trend_component) * sess_factor * regime_mult

    # Spread penalty (slippage-aware)
    spread_pen = 0.0
    if spread_pct > 0.02:
        spread_pen = 1.2
    elif spread_pct > 0.01:
        spread_pen = 0.6
    raw -= spread_pen

    # Sample factor
    n = float(edge_side["trades"])
    sample_factor = min(1.0, n / 150.0)
    base_conf = max(0.0, raw) * sample_factor

    # Signed score
    score = base_conf if side == "LONG" else -base_conf

    # Live bonus
    rank_score = abs(score) + (0.35 if is_live else 0.0)

    # Global adjustment (your 30% weight rule)
    rank_score = _global_adjust(rank_score, side, float(global_ctx.get("global_bias", 0.0)))

    # Exposure feasibility (V1): if one-lot premium already breaks cap, reduce rank
    per_trade_cap = TRABOT_CAPITAL * PER_TRADE_RISK_PCT
    one_lot_premium = entry_opt * lot
    feasibility_mult = 1.0
    if one_lot_premium > per_trade_cap:
        # still might fit via spreads later, so don't kill it
        feasibility_mult = 0.80 if TRABOT_RISK_PROFILE == "moderate" else 0.65
    if one_lot_premium > TRABOT_CAPITAL * 0.50:
        # too expensive for this capital bucket
        return None

    rank_score *= feasibility_mult

    if rank_score < 0.10:
        return None

    action = ("BUY_CE" if side == "LONG" else "BUY_PE") if is_live else ("WATCH_CE" if side == "LONG" else "WATCH_PE")

    reasons = []
    reasons.append(f"Session factor applied ({sess})")
    reasons.append(f"Regime={regime}")
    reasons.append(f"Spread≈{spread_pct*100:.2f}% | OI≈{int(oi)} | Vol≈{int(vol)} (liquidity filter: {why})")
    reasons.append(f"Strike picked by strike_score (slice ATM±{STRIKES_AROUND_ATM})")
    reasons.append(f"Learned edge {side}: win_rate={edge_side['win_rate']:.1f}% avg_R={edge_side['avg_r']:.2f} trades={edge_side['trades']}")
    reasons.append(f"Exposure check: 1-lot premium≈₹{one_lot_premium:,.0f} vs cap≈₹{per_trade_cap:,.0f} (per-trade {PER_TRADE_RISK_PCT*100:.1f}%)")

    return Candidate(
        underlying=item["underlying"],
        spot_symbol=item["spot"],
        expiry=str(chain.expiry),
        side=side,
        right=want_right,
        action=action,
        is_live=is_live,
        regime=regime,

        tradingsymbol=tsym,
        strike=strike,
        lot_size=lot,

        spot=float(chain.spot),
        atm=int(chain.atm),

        entry=float(entry_opt),
        sl=float(stop_opt_eff),
        target=float(target_opt_eff),

        entry_u=float(entry_u),
        sl_u=float(stop_u),
        target_u=float(target_u),

        score=float(score),
        rank_score=float(rank_score),

        spread_pct=float(spread_pct),
        oi=float(oi),
        volume=float(vol),

        reason=reason,
        reasons=reasons,
    )


# ============================================================
# Strategy selection (V1) + exposure caps + slippage-aware costs
# ============================================================
def _find_row(df_side: pd.DataFrame, strike: int):
    if df_side is None or df_side.empty:
        return None
    x = df_side[df_side["strike"].astype(int) == int(strike)]
    if x.empty:
        return None
    return x.iloc[0].to_dict()


def _price_leg(row: dict, action: str) -> float | None:
    # BUY pays ask, SELL receives bid (then slippage applied)
    bid = row.get("bid")
    ask = row.get("ask")
    ltp = row.get("last_price", None)
    if action == "BUY":
        px = float(ask) if ask is not None else float(ltp) if ltp is not None else None
    else:
        px = float(bid) if bid is not None else float(ltp) if ltp is not None else None
    if px is None:
        return None
    return _slip(px, action)


def _strike_step_from_chain(chain) -> int:
    strikes = sorted(set([int(s) for s in list(chain.calls["strike"].dropna().astype(int).values)] +
                         [int(s) for s in list(chain.puts["strike"].dropna().astype(int).values)]))
    if len(strikes) < 2:
        return 50
    diffs = [strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1)]
    diffs = [d for d in diffs if d > 0]
    return int(min(diffs)) if diffs else 50


def _estimate_risk_for_strategy(strategy_name: str, legs: list[dict], width: int, lot: int) -> float | None:
    """
    Returns a conservative risk estimate in INR for 1 lot.
    For debit/long options: cost paid.
    For credit spreads/condors: max_loss approx (width - credit)*lot.
    """
    if not legs:
        return None

    # debit-like: sum BUY - sum SELL (should be positive)
    buy = sum(l["price"] * l["qty"] for l in legs if l["action"] == "BUY")
    sell = sum(l["price"] * l["qty"] for l in legs if l["action"] == "SELL")
    net = buy - sell  # >0 for debit

    if "CREDIT" in strategy_name or "CONDOR" in strategy_name:
        # credit = -net
        credit = max(0.0, -net)
        return float(max(0.0, (width - credit / lot) * lot))
    else:
        return float(max(0.0, net))


def plan_strategy(c: Candidate, global_ctx: dict) -> tuple[str, list[dict], float | None, list[str]]:
    """
    V1: Uses regime + risk profile.
    Enforces per-trade cap. (total cap printed in summary)
    """
    capital = float(TRABOT_CAPITAL)
    per_trade_cap = capital * PER_TRADE_RISK_PCT

    chain = get_kite_chain_slice(
        underlying=c.underlying,
        kite_spot_symbol=c.spot_symbol,
        strike_step=0,
        strikes_around_atm=max(STRIKES_AROUND_ATM, 10),
        cache_path=INSTRUMENTS_CACHE_PATH,
    )
    step = _strike_step_from_chain(chain)
    atm = int(chain.atm)
    lot = int(c.lot_size)

    notes = []
    notes.append(f"Regime={c.regime} | GlobalBias={float(global_ctx.get('global_bias', 0.0)):+.2f}")
    notes.append(f"Risk cap: per-trade ≤ ₹{per_trade_cap:,.0f} ({PER_TRADE_RISK_PCT*100:.1f}% of capital)")

    def ce(k): return _find_row(chain.calls, k)
    def pe(k): return _find_row(chain.puts, k)

    # -------------------------
    # HIGH risk profile
    # -------------------------
    if TRABOT_RISK_PROFILE == "high":
        # VOLATILE + mixed global -> Straddle (if affordable)
        if c.regime == "VOLATILE" and abs(float(global_ctx.get("global_bias", 0.0))) <= 0.25:
            r_ce = ce(atm); r_pe = pe(atm)
            if r_ce and r_pe:
                p_ce = _price_leg(r_ce, "BUY")
                p_pe = _price_leg(r_pe, "BUY")
                if p_ce and p_pe:
                    cost = (p_ce + p_pe) * lot
                    if cost <= per_trade_cap:
                        legs = [
                            {"action": "BUY", "tradingsymbol": r_ce["tradingsymbol"], "qty": lot, "price": p_ce},
                            {"action": "BUY", "tradingsymbol": r_pe["tradingsymbol"], "qty": lot, "price": p_pe},
                        ]
                        return "LONG_STRADDLE_ATM", legs, cost, notes + ["Picked because VOLATILE regime + mixed global mood."]

        # TREND -> directional naked buy (only if fits cap), else tight debit spread
        if c.side == "LONG":
            base = ce(c.strike) or ce(atm)
        else:
            base = pe(c.strike) or pe(atm)

        if base:
            p = _price_leg(base, "BUY")
            if p:
                cost = p * lot
                if cost <= per_trade_cap:
                    legs = [{"action": "BUY", "tradingsymbol": base["tradingsymbol"], "qty": lot, "price": p}]
                    return "NAKED_BUY_OPTION", legs, cost, notes + ["Picked naked buy because high risk profile and it fits cap."]

        # If naked too expensive -> debit spread (defined risk)
        if c.side == "LONG":
            r_buy = ce(atm)
            r_sell = ce(atm + step)
            name = "BULL_CALL_DEBIT_SPREAD"
        else:
            r_buy = pe(atm)
            r_sell = pe(atm - step)
            name = "BEAR_PUT_DEBIT_SPREAD"

        if r_buy and r_sell:
            buy_px = _price_leg(r_buy, "BUY")
            sell_px = _price_leg(r_sell, "SELL")
            if buy_px and sell_px:
                debit = max(0.0, buy_px - sell_px)
                cost = debit * lot
                if cost <= per_trade_cap and cost > 0:
                    legs = [
                        {"action": "BUY", "tradingsymbol": r_buy["tradingsymbol"], "qty": lot, "price": buy_px},
                        {"action": "SELL", "tradingsymbol": r_sell["tradingsymbol"], "qty": lot, "price": sell_px},
                    ]
                    return name, legs, cost, notes + ["Naked was too expensive -> using defined-risk debit spread."]

        return "NO_STRATEGY_UNDER_CAP", [], None, notes + ["No strategy could fit the per-trade risk cap."]

    # -------------------------
    # MODERATE risk profile
    # -------------------------
    else:
        # CHOP -> Iron Condor (defined risk)
        if c.regime == "CHOP":
            k_put_sell = atm - 3 * step
            k_put_buy = atm - 4 * step
            k_call_sell = atm + 3 * step
            k_call_buy = atm + 4 * step

            rp_sell = pe(k_put_sell); rp_buy = pe(k_put_buy)
            rc_sell = ce(k_call_sell); rc_buy = ce(k_call_buy)

            if rp_sell and rp_buy and rc_sell and rc_buy:
                sp = _price_leg(rp_sell, "SELL"); bp = _price_leg(rp_buy, "BUY")
                sc = _price_leg(rc_sell, "SELL"); bc = _price_leg(rc_buy, "BUY")
                if sp and bp and sc and bc:
                    legs = [
                        {"action": "SELL", "tradingsymbol": rp_sell["tradingsymbol"], "qty": lot, "price": sp},
                        {"action": "BUY", "tradingsymbol": rp_buy["tradingsymbol"], "qty": lot, "price": bp},
                        {"action": "SELL", "tradingsymbol": rc_sell["tradingsymbol"], "qty": lot, "price": sc},
                        {"action": "BUY", "tradingsymbol": rc_buy["tradingsymbol"], "qty": lot, "price": bc},
                    ]
                    width = step
                    risk = _estimate_risk_for_strategy("IRON_CONDOR_CREDIT", legs, width=width, lot=lot)
                    if risk is not None and risk <= per_trade_cap:
                        return "IRON_CONDOR_DEFINED_RISK", legs, risk, notes + ["Picked because CHOP regime (range better)."]

        # TREND -> debit spread
        if c.side == "LONG":
            r_buy = ce(atm)
            r_sell = ce(atm + step)
            name = "BULL_CALL_DEBIT_SPREAD"
        else:
            r_buy = pe(atm)
            r_sell = pe(atm - step)
            name = "BEAR_PUT_DEBIT_SPREAD"

        if r_buy and r_sell:
            buy_px = _price_leg(r_buy, "BUY")
            sell_px = _price_leg(r_sell, "SELL")
            if buy_px and sell_px:
                debit = max(0.0, buy_px - sell_px)
                cost = debit * lot
                if cost <= per_trade_cap and cost > 0:
                    legs = [
                        {"action": "BUY", "tradingsymbol": r_buy["tradingsymbol"], "qty": lot, "price": buy_px},
                        {"action": "SELL", "tradingsymbol": r_sell["tradingsymbol"], "qty": lot, "price": sell_px},
                    ]
                    return name, legs, cost, notes + ["Picked debit spread (defined risk) aligned with trend."]

        # fallback: butterfly (cheap defined risk)
        if c.side == "LONG":
            k1, k2, k3 = atm - step, atm, atm + step
            r1 = ce(k1); r2 = ce(k2); r3 = ce(k3)
            name = "CALL_BUTTERFLY"
        else:
            k1, k2, k3 = atm + step, atm, atm - step
            r1 = pe(k1); r2 = pe(k2); r3 = pe(k3)
            name = "PUT_BUTTERFLY"

        if r1 and r2 and r3:
            p1 = _price_leg(r1, "BUY")
            p2 = _price_leg(r2, "SELL")
            p3 = _price_leg(r3, "BUY")
            if p1 and p2 and p3:
                debit = max(0.0, (p1 - 2 * p2 + p3))
                cost = debit * lot
                if cost <= per_trade_cap and cost > 0:
                    legs = [
                        {"action": "BUY", "tradingsymbol": r1["tradingsymbol"], "qty": lot, "price": p1},
                        {"action": "SELL", "tradingsymbol": r2["tradingsymbol"], "qty": 2 * lot, "price": p2},
                        {"action": "BUY", "tradingsymbol": r3["tradingsymbol"], "qty": lot, "price": p3},
                    ]
                    return name, legs, cost, notes + ["Fallback butterfly to keep defined risk under cap."]

        return "NO_STRATEGY_UNDER_CAP", [], None, notes + ["No defined-risk strategy fit the cap from this chain slice."]


# ============================================================
# Output helpers
# ============================================================
def _print_global(global_ctx: dict):
    print("\n=== GLOBAL CONTEXT (US cues) ===")
    if not global_ctx["available"]:
        for n in global_ctx["notes"]:
            print(f"- {n}")
        return

    def fmt(k):
        r = global_ctx["metrics"][k]["ret_pct"]
        dt = global_ctx["metrics"][k]["date"]
        ds = dt.strftime("%Y-%m-%d") if dt else "?"
        return f"{k}:{r:+.2f}%({ds})" if r is not None else f"{k}:n/a"

    print("  " + " | ".join([fmt(k) for k in ["SPY", "QQQ", "VIX", "DXY", "CRUDE", "US10Y"]]))
    print(f"  global_bias = {global_ctx['global_bias']:+.2f}")
    for n in global_ctx["notes"]:
        print(f"  - {n}")


def _print_rank_table(ranked: list[Candidate], top_k: int):
    print(f"\n=== TOP {top_k} RANKING (V1: regime+liquidity+caps+slippage) ===")
    for i, c in enumerate(ranked[:top_k], 1):
        print(
            f"#{i:02d} {c.action:<8s} | {c.tradingsymbol:<22s} | rank={c.rank_score:.2f} | "
            f"AI={c.score:+.2f} | {c.regime:<8s} | spread={c.spread_pct*100:.2f}% | {c.underlying}"
        )


def main():
    global_ctx = fetch_global_context()
    _print_global(global_ctx)

    print("\n=== CAPITAL / CAPS (V1) ===")
    print(f"  Capital: ₹{int(TRABOT_CAPITAL)} | RiskProfile: {TRABOT_RISK_PROFILE}")
    print(f"  Per-trade risk cap: {PER_TRADE_RISK_PCT*100:.1f}% (₹{TRABOT_CAPITAL*PER_TRADE_RISK_PCT:,.0f})")
    print(f"  Total risk cap (if taking multiple positions): {TOTAL_RISK_PCT*100:.1f}% (₹{TRABOT_CAPITAL*TOTAL_RISK_PCT:,.0f})")
    print(f"  Slippage: {SLIPPAGE_TICKS} tick(s) @ tick_size={TICK_SIZE}")
    print("  Liquidity filters:")
    print(f"    INDEX: spread≤{MAX_SPREAD_PCT_INDEX*100:.2f}%, OI≥{MIN_OI_INDEX}, vol≥{MIN_VOL_INDEX}")
    print(f"    STOCK: spread≤{MAX_SPREAD_PCT_STOCK*100:.2f}%, OI≥{MIN_OI_STOCK}, vol≥{MIN_VOL_STOCK}")

    universe = build_universe_all_options()
    total = len(universe)
    if total == 0:
        print("\nUniverse empty (no options found).")
        return

    start = max(0, UNIVERSE_START)
    end = total if UNIVERSE_COUNT is None else min(total, start + UNIVERSE_COUNT)
    scan = universe[start:end]

    print("\n=== SCAN (India options via Kite) ===")
    print(f"Universe: {total} option-underlyings found in Kite (NFO-OPT).")
    print(f"Scanning slice: [{start}:{end}] (count={len(scan)})")
    print(f"Strike window: ATM ± {STRIKES_AROUND_ATM} | Candle cache TTL: {CACHE_TTL_MINUTES} min\n")

    lot_map = build_lot_size_map()

    cands: list[Candidate] = []
    for item in scan:
        try:
            c = build_candidate(item, lot_map, global_ctx)
            if c:
                cands.append(c)
        except Exception as e:
            print(f"[skip] {item['underlying']}: {e}")
        if SLEEP_BETWEEN_SYMBOLS > 0:
            time.sleep(SLEEP_BETWEEN_SYMBOLS)

    if not cands:
        print("No candidates found after V1 filters.")
        return

    ranked = sorted(cands, key=lambda x: float(x.rank_score), reverse=True)
    _print_rank_table(ranked, TOP_RANK)

    # Best 5 with strategy legs + cap enforcement
    print(f"\n=== BEST {TOP_BEST} CHAINS TO TRADE (with V1 strategy + caps) ===")
    used_risk_total = 0.0
    picks_out = []
    for i, c in enumerate(ranked[:TOP_BEST], 1):
        strat, legs, est_risk, notes = plan_strategy(c, global_ctx)

        print(f"\n#{i} {'BULLISH' if c.side=='LONG' else 'BEARISH'} | {c.tradingsymbol} | Expiry={c.expiry}")
        print(f"  Regime: {c.regime} | Action: {c.action}")
        print(f"  Underlying plan: enter≈{c.entry_u:.2f}  stop≈{c.sl_u:.2f}  target≈{c.target_u:.2f}")
        print(f"  Suggested strategy: {strat}")
        if est_risk is not None:
            print(f"  Est. risk (cap-checked): ₹{est_risk:,.0f}")
        if legs:
            print("  Legs (slippage-adjusted prices):")
            for lg in legs:
                print(f"    {lg['action']:<4s} {lg['tradingsymbol']}  qty={lg['qty']}  px≈{lg['price']:.2f}")
        else:
            print("  Legs: (none)")

        print("  Notes:")
        for n in notes:
            print(f"   - {n}")
        print("  Scanner reasons:")
        for r in c.reasons:
            print(f"   - {r}")

        if est_risk is not None:
            used_risk_total += float(est_risk)

        picks_out.append({
            **c.__dict__,
            "strategy": strat,
            "est_risk": est_risk,
            "legs": legs,
            "notes": " | ".join(notes),
        })

    total_cap_amt = TRABOT_CAPITAL * TOTAL_RISK_PCT
    print("\n=== TOTAL RISK CHECK (if you took ALL top picks) ===")
    print(f"  Sum of est_risk across best picks: ₹{used_risk_total:,.0f}")
    print(f"  Total risk cap:                 ₹{total_cap_amt:,.0f}")
    if used_risk_total > total_cap_amt:
        print("  ⚠️ Above total cap -> take fewer positions or reduce size.")
    else:
        print("  ✅ Within total cap (still not a recommendation; research only).")

    if LOOKUP_SYMBOL:
        print("\n=== LOOKUP ===")
        found = None
        for idx, c in enumerate(ranked, 1):
            if c.tradingsymbol.upper() == LOOKUP_SYMBOL:
                found = (idx, c)
                break
        if found:
            idx, c = found
            print(f"{LOOKUP_SYMBOL} rank #{idx} | rank_score={c.rank_score:.2f} | {c.underlying} | {c.regime}")
        else:
            print(f"{LOOKUP_SYMBOL} not found in this scan.")

    # Save
    os.makedirs("data", exist_ok=True)
    df_all = pd.DataFrame([c.__dict__ for c in cands])
    df_all.to_csv("data/options_scan_global_v1_results.csv", index=False)

    df_top = pd.DataFrame(picks_out)
    df_top.to_csv(f"data/options_scan_global_v1_top{TOP_BEST}.csv", index=False)

    print("\nSaved: data/options_scan_global_v1_results.csv")
    print(f"Saved: data/options_scan_global_v1_top{TOP_BEST}.csv")
    print("\nNOTE: Research/education only. Not financial advice.")


if __name__ == "__main__":
    main()
