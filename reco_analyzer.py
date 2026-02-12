"""
reco_analyzer.py

Evaluate recommendation history (append-only reco_history.csv) using Kite market data.

Timezone fix:
- Force ts_reco to Asia/Kolkata
- Force candle indexes to Asia/Kolkata
This eliminates tz-naive vs tz-aware comparison errors.

Assumptions:
- You "entered" immediately when recommendation was printed.
- Entry price = first candle OPEN at/after ts_reco (option candles).
- SL hit if candle LOW <= SL (long option)
- Target hit if candle HIGH >= Target (long option)
- If both SL and Target hit in the same candle => mark BOTH_SAME_BAR (ambiguous).
- If neither hit by evaluation horizon => TIME_EXIT at last candle CLOSE.

Outputs:
- data/reco_evaluated_latest.csv
- data/reco_eval_summary_latest.txt
- timestamped versions too

Usage:
  python3 reco_analyzer.py
  python3 reco_analyzer.py --last_n 100 --interval 5minute --max_hours 24
  python3 reco_analyzer.py --from_date 2026-02-01 --interval minute --max_hours 6

Educational tool only – not financial advice.
"""

from __future__ import annotations

import argparse
import os
import re
import time
import datetime as dt
from functools import lru_cache
from typing import Dict, Optional, Tuple, List

import pandas as pd

from kite_client import get_kite
from iv_greeks import implied_volatility  # (price, S, K, T, r, right) -> (iv, ok)


# ----------------------------
# Timezone helpers (IST)
# ----------------------------

IST = "Asia/Kolkata"


def to_ist(ts) -> pd.Timestamp:
    """
    Convert a datetime-like to tz-aware Asia/Kolkata pandas Timestamp.
    - If ts is naive -> localize to IST
    - If ts is aware -> convert to IST
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(IST)
    return t.tz_convert(IST)


def ensure_index_ist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df.index is tz-aware and converted to IST.
    """
    if df is None or df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)
    df = df.copy()
    df.index = idx
    return df


def naive_ist_datetime(t: pd.Timestamp) -> dt.datetime:
    """
    Kite accepts naive datetimes (treated as local exchange time).
    Convert tz-aware IST timestamp into naive datetime.
    """
    t = to_ist(t)
    return t.tz_localize(None).to_pydatetime()


# ----------------------------
# Defaults / config
# ----------------------------

DATA_DIR = os.getenv("TRABOT_DATA_DIR", "data")
HISTORY_PATH = os.getenv("TRABOT_RECO_HISTORY", os.path.join(DATA_DIR, "reco_history.csv"))

NFO_INSTRUMENTS_CACHE = os.getenv("INSTRUMENTS_CACHE_PATH", os.path.join(DATA_DIR, "kite_instruments_NFO.csv"))

CACHE_DIR = os.path.join(DATA_DIR, "reco_eval_cache")

RISK_FREE = float(os.getenv("RISK_FREE_RATE", "0.06"))

INDEX_SPOT_MAP = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
    "NIFTYNXT50": "NSE:NIFTY NEXT 50",
}

# conservative chunk limits
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


# ----------------------------
# Token resolution (NFO)
# ----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_or_fetch_nfo_instruments() -> pd.DataFrame:
    _ensure_dir(os.path.dirname(NFO_INSTRUMENTS_CACHE))
    if os.path.exists(NFO_INSTRUMENTS_CACHE):
        try:
            df = pd.read_csv(NFO_INSTRUMENTS_CACHE)
            if not df.empty:
                return df
        except Exception:
            pass

    kite = get_kite()
    df = pd.DataFrame(kite.instruments("NFO"))
    df.to_csv(NFO_INSTRUMENTS_CACHE, index=False)
    return df


@lru_cache(maxsize=20000)
def resolve_nfo_token(tradingsymbol: str) -> int:
    df = _load_or_fetch_nfo_instruments()
    ts = str(tradingsymbol).strip().upper()
    hit = df[df["tradingsymbol"].astype(str).str.upper() == ts]
    if hit.empty:
        raise RuntimeError(f"Could not resolve NFO token for {tradingsymbol}")
    return int(hit.iloc[0]["instrument_token"])


# ----------------------------
# Candle fetch (NFO options + NSE spot)
# ----------------------------

def _cache_path(prefix: str, key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", key)
    return os.path.join(CACHE_DIR, f"{prefix}__{safe}.csv")


def _read_cache(path: str, max_age_minutes: Optional[int]) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    if max_age_minutes is not None:
        age_sec = time.time() - os.path.getmtime(path)
        if age_sec > max_age_minutes * 60:
            return None
    try:
        df = pd.read_csv(path)
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")
        df = ensure_index_ist(df)
        return df
    except Exception:
        return None


def _write_cache(path: str, df: pd.DataFrame) -> None:
    _ensure_dir(os.path.dirname(path))
    out = df.copy()
    out = ensure_index_ist(out)
    out.index.name = "Datetime"
    out.to_csv(path, index=True)


def _chunked_historical(kite, token: int, start: dt.datetime, end: dt.datetime, interval: str) -> List[dict]:
    max_days = MAX_DAYS_PER_REQ.get(interval, 30)
    cur = start
    out: List[dict] = []
    while cur < end:
        cur_end = min(end, cur + dt.timedelta(days=max_days))
        rows = kite.historical_data(
            instrument_token=token,
            from_date=cur,
            to_date=cur_end,
            interval=interval,
            continuous=False,
            oi=False,
        ) or []
        out.extend(rows)
        cur = cur_end + dt.timedelta(seconds=1)
    return out


def fetch_option_candles(
    tradingsymbol: str,
    start_ts_ist: pd.Timestamp,
    end_ts_ist: pd.Timestamp,
    interval: str,
    cache_age_min: Optional[int] = 7 * 24 * 60,
) -> pd.DataFrame:
    """
    Fetch NFO option OHLC candles for tradingsymbol.
    Inputs are tz-aware IST timestamps; converted to naive for Kite API.
    """
    _ensure_dir(CACHE_DIR)
    start_ts_ist = to_ist(start_ts_ist)
    end_ts_ist = to_ist(end_ts_ist)

    key = f"{tradingsymbol}__{interval}__{start_ts_ist.strftime('%Y%m%d%H%M')}__{end_ts_ist.strftime('%Y%m%d%H%M')}"
    cp = _cache_path("opt", key)
    cached = _read_cache(cp, max_age_minutes=cache_age_min)
    if cached is not None and not cached.empty:
        return cached

    kite = get_kite()
    token = resolve_nfo_token(tradingsymbol)

    start = naive_ist_datetime(start_ts_ist)
    end = naive_ist_datetime(end_ts_ist)

    rows = _chunked_historical(kite, token, start, end, interval)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).rename(columns={"date": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").drop_duplicates(subset=["Datetime"]).set_index("Datetime")
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = ensure_index_ist(df)

    _write_cache(cp, df)
    return df


def spot_symbol_from_underlying(underlying: str) -> str:
    u = str(underlying).strip().upper()
    return INDEX_SPOT_MAP.get(u, f"NSE:{u}")


def fetch_spot_candles(
    underlying: str,
    start_ts_ist: pd.Timestamp,
    end_ts_ist: pd.Timestamp,
    interval: str,
    cache_age_min: Optional[int] = 7 * 24 * 60,
) -> pd.DataFrame:
    """
    Fetch underlying spot candles using market_data.fetch_history_cached,
    then slice in IST time.
    """
    from market_data import fetch_history_cached as fh

    start_ts_ist = to_ist(start_ts_ist)
    end_ts_ist = to_ist(end_ts_ist)

    sym = spot_symbol_from_underlying(underlying)
    lookback_days = max(5, int((end_ts_ist - start_ts_ist).days) + 2)

    key = f"{sym}__{interval}__{start_ts_ist.strftime('%Y%m%d')}__{end_ts_ist.strftime('%Y%m%d')}"
    cp = _cache_path("spot", key)
    cached = _read_cache(cp, max_age_minutes=cache_age_min)
    if cached is not None and not cached.empty:
        return cached.loc[(cached.index >= start_ts_ist) & (cached.index <= end_ts_ist)].copy()

    df, _ = fh(sym, lookback_days=lookback_days, interval=interval, ttl_minutes=5)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = ensure_index_ist(df)

    _write_cache(cp, df)
    return df.loc[(df.index >= start_ts_ist) & (df.index <= end_ts_ist)].copy()


# ----------------------------
# Parsing helpers
# ----------------------------

_STRIKE_RE = re.compile(r"(\d+)(CE|PE)$", re.IGNORECASE)


def parse_strike_and_right(tradingsymbol: str) -> Tuple[Optional[int], Optional[str]]:
    ts = str(tradingsymbol).strip().upper()
    m = _STRIKE_RE.search(ts)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2).upper()


def expiry_dt_end(expiry_str: str) -> Optional[pd.Timestamp]:
    if not expiry_str or str(expiry_str).strip() == "":
        return None
    try:
        d = pd.to_datetime(expiry_str).date()
        # approximate expiry time as 15:25 IST
        return pd.Timestamp(dt.datetime(d.year, d.month, d.day, 15, 25, 0)).tz_localize(IST)
    except Exception:
        return None


def t_years_at(expiry_end_ist: pd.Timestamp, at_ist: pd.Timestamp) -> float:
    expiry_end_ist = to_ist(expiry_end_ist)
    at_ist = to_ist(at_ist)
    sec = (expiry_end_ist - at_ist).total_seconds()
    if sec <= 0:
        return 0.0
    return sec / (365.0 * 24 * 3600)


# ----------------------------
# Evaluation logic
# ----------------------------

def evaluate_trade_path(
    opt_df: pd.DataFrame,
    ts_reco_ist: pd.Timestamp,
    reco_entry: float,
    sl: float,
    target: float,
) -> Dict:
    """
    Decide first touch: SL / TARGET / BOTH / TIME_EXIT.
    Entry price = first candle OPEN at/after ts_reco.
    """
    if opt_df is None or opt_df.empty:
        return {"status": "NO_CANDLES"}

    opt_df = ensure_index_ist(opt_df).sort_index()
    ts_reco_ist = to_ist(ts_reco_ist)

    after = opt_df.loc[opt_df.index >= ts_reco_ist]
    if after.empty:
        return {"status": "NO_CANDLES_AFTER_RECO"}

    entry_ts = after.index[0]
    entry_px = float(after.iloc[0]["open"])

    if entry_px <= 0 or sl <= 0 or target <= 0:
        return {"status": "BAD_LEVELS"}

    bad_levels = not (sl < entry_px < target)

    hit = None
    exit_ts = None
    exit_px = None
    both_same_bar = False
    bars = 0

    for t, row in after.iterrows():
        bars += 1
        lo = float(row["low"])
        hi = float(row["high"])

        sl_hit = (lo <= sl)
        tgt_hit = (hi >= target)

        if sl_hit and tgt_hit:
            hit = "BOTH_SAME_BAR"
            both_same_bar = True
            exit_ts = t
            exit_px = float(sl)  # conservative
            break
        if tgt_hit:
            hit = "TARGET"
            exit_ts = t
            exit_px = float(target)
            break
        if sl_hit:
            hit = "SL"
            exit_ts = t
            exit_px = float(sl)
            break

    if hit is None:
        hit = "TIME_EXIT"
        exit_ts = after.index[-1]
        exit_px = float(after.iloc[-1]["close"])

    pnl = exit_px - entry_px
    pnl_pct = pnl / entry_px if entry_px > 0 else 0.0
    risk = entry_px - sl
    r_mult = (pnl / risk) if risk > 0 else None

    return {
        "status": "OK",
        "entry_ts": entry_ts,
        "entry_px_mkt": entry_px,
        "entry_px_reco": float(reco_entry) if reco_entry else None,
        "hit": hit,
        "exit_ts": exit_ts,
        "exit_px": exit_px,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "r_mult": r_mult,
        "bars_to_exit": bars,
        "bad_levels": bad_levels,
        "both_same_bar": both_same_bar,
    }


def loss_reason_tags(
    *,
    hit: str,
    pnl_pct: float,
    dte: Optional[int],
    iv_pct: Optional[float],
    greeks_conf: Optional[str],
    delta: Optional[float],
    spot_move_pct: Optional[float],
    entry_iv: Optional[float],
    exit_iv: Optional[float],
    minutes_to_exit: Optional[float],
) -> str:
    tags = []

    if dte is not None and dte <= 3:
        tags.append("EXPIRY_WEEK")
    if iv_pct is not None and iv_pct >= 0.70:
        tags.append("HIGH_IV_PCTL")
    if greeks_conf and str(greeks_conf).lower() == "low":
        tags.append("LOW_GREEKS_CONF")
    if delta is not None and abs(float(delta)) < 0.35:
        tags.append("LOW_DELTA")

    if pnl_pct < 0:
        if spot_move_pct is not None:
            if spot_move_pct < -0.25 / 100:
                tags.append("WRONG_DIRECTION")
            if abs(spot_move_pct) < 0.15 / 100 and (minutes_to_exit or 0) >= 60:
                tags.append("THETA_DECAY_FLAT_SPOT")

        if entry_iv is not None and exit_iv is not None:
            if (entry_iv - exit_iv) >= 0.05:
                tags.append("IV_CRUSH")

    if hit == "BOTH_SAME_BAR":
        tags.append("AMBIGUOUS_BAR")

    if not tags:
        tags.append("OK_OR_UNKNOWN")

    return " | ".join(tags)


# ----------------------------
# Summary / suggestions
# ----------------------------

def winrate(df: pd.DataFrame, pnl_col: str = "pnl_pct") -> float:
    if df.empty:
        return 0.0
    return float((df[pnl_col] > 0).mean())


def summarize_and_suggest(df: pd.DataFrame, min_count: int = 10) -> str:
    lines = []

    total = len(df)
    closed = df[df["eval_status"] == "OK"]
    lines.append(f"Total rows read: {total}")
    lines.append(f"Rows evaluated: {len(closed)}")

    if closed.empty:
        return "\n".join(lines)

    overall_wr = winrate(closed)
    avg_pnl = float(closed["pnl_pct"].mean())
    lines.append(f"Overall win-rate: {overall_wr*100:.1f}%")
    lines.append(f"Average PnL%: {avg_pnl*100:.2f}%")
    lines.append("")

    lines.append("Outcome distribution:")
    oc = closed["hit"].value_counts(dropna=False)
    for k, v in oc.items():
        lines.append(f"  {k}: {v} ({v/len(closed)*100:.1f}%)")
    lines.append("")

    def add_bins():
        nonlocal df
        if "iv_pct" in df.columns:
            df["iv_bin"] = pd.cut(df["iv_pct"].astype(float), [-1, 0.30, 0.70, 10], labels=["LOW_IV", "MID_IV", "HIGH_IV"])
        if "dte" in df.columns:
            df["dte_bin"] = pd.cut(df["dte"].astype(float), [-1, 3, 7, 3650], labels=["DTE_0_3", "DTE_4_7", "DTE_8_PLUS"])
        if "delta" in df.columns:
            df["delta_abs"] = df["delta"].astype(float).abs()
            df["delta_bin"] = pd.cut(df["delta_abs"], [-1, 0.35, 0.55, 10], labels=["LOW_DELTA", "MID_DELTA", "HIGH_DELTA"])

    add_bins()
    closed = df[df["eval_status"] == "OK"].copy()

    def group_report(col: str, title: str):
        nonlocal lines
        if col not in closed.columns:
            return
        g = closed.groupby(col).agg(
            n=("pnl_pct", "size"),
            win=("pnl_pct", lambda x: float((x > 0).mean())),
            avg_pnl=("pnl_pct", "mean"),
        ).reset_index().sort_values(["n"], ascending=False)

        lines.append(title + ":")
        for _, r in g.iterrows():
            lines.append(f"  {r[col]}: n={int(r['n'])} win={r['win']*100:.1f}% avgPnL={r['avg_pnl']*100:.2f}%")
        lines.append("")

        bad = g[(g["n"] >= min_count) & (g["win"] <= overall_wr - 0.10)]
        for _, r in bad.iterrows():
            lines.append(f"Suggestion: {title}='{r[col]}' has low win-rate ({r['win']*100:.1f}%). Consider penalizing or filtering this case.")
        if not bad.empty:
            lines.append("")

    group_report("source", "By source")
    group_report("bucket", "By bucket")
    group_report("regime", "By regime")
    group_report("iv_bin", "By IV bin")
    group_report("dte_bin", "By DTE bin")
    group_report("delta_bin", "By delta bin")
    group_report("greeks_conf", "By greeks confidence")

    if "loss_tags" in closed.columns:
        losses = closed[closed["pnl_pct"] < 0]
        if not losses.empty:
            lines.append("Top loss tags (frequency among losing trades):")
            tags = losses["loss_tags"].astype(str).str.split(r"\s*\|\s*", regex=True).explode()
            vc = tags.value_counts()
            for k, v in vc.head(10).items():
                lines.append(f"  {k}: {v} ({v/len(losses)*100:.1f}%)")
            lines.append("")

            mapping = {
                "EXPIRY_WEEK": "Avoid DTE<=3 for long premium OR reduce size drastically OR shift to spreads.",
                "HIGH_IV_PCTL": "High IV tends to IV-crush; require stronger trend confirmation, smaller size, or prefer spreads.",
                "LOW_DELTA": "Low delta options decay fast; bias strike selection to delta ~0.35–0.55 (closer to ATM).",
                "WRONG_DIRECTION": "Increase directional confidence: stronger ADX/BOS + 1H alignment, or reduce trading in CHOP regime.",
                "THETA_DECAY_FLAT_SPOT": "Add do-not-trade filter for CHOP/low-trend; shorter holding window; higher delta.",
                "IV_CRUSH": "If IV high at entry, consider exiting faster, or avoid entries right after event spikes.",
            }
            for k, v in vc.items():
                if v / len(losses) >= 0.30 and k in mapping:
                    lines.append(f"High-impact fix: '{k}' appears in {v/len(losses)*100:.1f}% of losses. {mapping[k]}")
            lines.append("")

    return "\n".join(lines)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=HISTORY_PATH, help="Path to reco_history.csv")
    ap.add_argument("--interval", default="5minute", help="Option candle interval: minute/5minute/15minute/60minute")
    ap.add_argument("--max_hours", type=float, default=24.0, help="Evaluation horizon from ts_reco")
    ap.add_argument("--last_n", type=int, default=0, help="Evaluate only last N recommendations (0 = all)")
    ap.add_argument("--from_date", default="", help="Only evaluate recommendations on/after YYYY-MM-DD")
    ap.add_argument("--include_watch", action="store_true", help="Include WATCH_* actions (default: only BUY_*)")
    args = ap.parse_args()

    if not os.path.exists(args.history):
        raise SystemExit(f"Missing history file: {args.history}")

    df = pd.read_csv(args.history)
    if df.empty:
        raise SystemExit("History file is empty.")

    # Parse ts_reco robustly + force IST tz
    df["ts_reco"] = pd.to_datetime(df.get("ts_reco"), errors="coerce")
    df = df.dropna(subset=["ts_reco"]).copy()
    df["ts_reco"] = df["ts_reco"].apply(to_ist)

    if args.from_date.strip():
        start_d = to_ist(pd.to_datetime(args.from_date.strip()))
        df = df[df["ts_reco"] >= start_d].copy()

    df["action"] = df.get("action", "").astype(str)
    if args.include_watch:
        df = df[df["action"].str.contains("BUY_|WATCH_", regex=True)].copy()
    else:
        df = df[df["action"].str.startswith("BUY_")].copy()

    df = df.sort_values("ts_reco")
    if args.last_n and args.last_n > 0:
        df = df.tail(args.last_n).copy()

    if df.empty:
        raise SystemExit("No rows to evaluate after filters.")

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _ensure_dir(DATA_DIR)

    out_rows = []

    for _, r in df.iterrows():
        ts_reco_ist: pd.Timestamp = to_ist(r["ts_reco"])

        tradingsymbol = str(r.get("tradingsymbol") or "").strip()
        underlying = str(r.get("underlying") or "").strip().upper()
        if not tradingsymbol or not underlying:
            continue

        try:
            reco_entry = float(r.get("entry") or 0.0)
            sl = float(r.get("sl") or 0.0)
            target = float(r.get("target") or 0.0)
        except Exception:
            continue

        expiry_str = str(r.get("expiry") or "").strip()
        exp_end_ist = expiry_dt_end(expiry_str)

        end_ist = ts_reco_ist + pd.Timedelta(hours=float(args.max_hours))
        if exp_end_ist is not None:
            end_ist = min(end_ist, exp_end_ist)
        if end_ist <= ts_reco_ist:
            out_rows.append({**r.to_dict(), "eval_status": "SKIP_EXPIRED"})
            continue

        # fetch option candles
        try:
            opt_df = fetch_option_candles(tradingsymbol, ts_reco_ist, end_ist, args.interval)
        except Exception as e:
            out_rows.append({**r.to_dict(), "eval_status": f"OPT_FETCH_FAIL: {e}"})
            continue

        ev = evaluate_trade_path(opt_df, ts_reco_ist, reco_entry, sl, target)
        if ev.get("status") != "OK":
            out_rows.append({**r.to_dict(), "eval_status": ev.get("status")})
            continue

        # spot move in favorable direction frame
        spot_move_pct_signed = None
        spot_exit = None
        try:
            spot_df = fetch_spot_candles(underlying, ev["entry_ts"], ev["exit_ts"], "5minute")
            if not spot_df.empty:
                spot_entry = float(spot_df.iloc[0]["open"])
                spot_exit = float(spot_df.iloc[-1]["close"])
                move = (spot_exit - spot_entry) / spot_entry if spot_entry > 0 else 0.0
                action = str(r.get("action") or "")
                if action.endswith("_PE"):
                    move = -move
                spot_move_pct_signed = move
        except Exception:
            pass

        # entry_iv from reco row (if present)
        try:
            entry_iv = float(r.get("iv")) if str(r.get("iv", "")).strip() != "" else None
        except Exception:
            entry_iv = None

        # exit iv estimate (needs strike/right + spot_exit)
        exit_iv = None
        try:
            strike, right = parse_strike_and_right(tradingsymbol)
            if strike is not None and right is not None and exp_end_ist is not None and spot_exit is not None:
                T = t_years_at(exp_end_ist, ev["exit_ts"])
                if T > 0:
                    iv2, ok2 = implied_volatility(float(ev["exit_px"]), float(spot_exit), float(strike), float(T), RISK_FREE, right)
                    if ok2 and iv2 is not None:
                        exit_iv = float(iv2)
        except Exception:
            pass

        minutes_to_exit = (to_ist(ev["exit_ts"]) - to_ist(ev["entry_ts"])).total_seconds() / 60.0

        # meta fields for tagging
        try:
            dte_val = int(r.get("dte")) if str(r.get("dte", "")).strip() != "" else None
        except Exception:
            dte_val = None
        try:
            iv_pct_val = float(r.get("iv_pct")) if str(r.get("iv_pct", "")).strip() != "" else None
        except Exception:
            iv_pct_val = None
        greeks_conf = str(r.get("greeks_conf") or "").strip() or None
        try:
            delta_val = float(r.get("delta")) if str(r.get("delta", "")).strip() != "" else None
        except Exception:
            delta_val = None

        tags = loss_reason_tags(
            hit=str(ev["hit"]),
            pnl_pct=float(ev["pnl_pct"]),
            dte=dte_val,
            iv_pct=iv_pct_val,
            greeks_conf=greeks_conf,
            delta=delta_val,
            spot_move_pct=spot_move_pct_signed,
            entry_iv=entry_iv,
            exit_iv=exit_iv,
            minutes_to_exit=minutes_to_exit,
        )

        out = {**r.to_dict()}
        out.update({
            "eval_status": "OK",
            "entry_ts_mkt": to_ist(ev["entry_ts"]),
            "entry_px_mkt": ev["entry_px_mkt"],
            "hit": ev["hit"],
            "exit_ts": to_ist(ev["exit_ts"]),
            "exit_px": ev["exit_px"],
            "pnl": ev["pnl"],
            "pnl_pct": ev["pnl_pct"],
            "r_mult": ev["r_mult"],
            "bars_to_exit": ev["bars_to_exit"],
            "minutes_to_exit": minutes_to_exit,
            "bad_levels": ev["bad_levels"],
            "spot_move_pct_favorable": spot_move_pct_signed,
            "entry_iv": entry_iv,
            "exit_iv_est": exit_iv,
            "loss_tags": tags,
        })
        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)

    latest_csv = os.path.join(DATA_DIR, "reco_evaluated_latest.csv")
    snap_csv = os.path.join(DATA_DIR, f"reco_evaluated_{run_id}.csv")
    out_df.to_csv(latest_csv, index=False)
    out_df.to_csv(snap_csv, index=False)

    summary = summarize_and_suggest(out_df, min_count=10)
    latest_txt = os.path.join(DATA_DIR, "reco_eval_summary_latest.txt")
    snap_txt = os.path.join(DATA_DIR, f"reco_eval_summary_{run_id}.txt")
    with open(latest_txt, "w", encoding="utf-8") as f:
        f.write(summary)
    with open(snap_txt, "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n=== RECO EVAL DONE ===")
    print(f"Saved: {latest_csv}")
    print(f"Saved: {snap_csv}")
    print(f"Saved: {latest_txt}")
    print(f"Saved: {snap_txt}")
    print("\n--- Summary (top) ---")
    print("\n".join(summary.splitlines()[:30]))
    print("\n(See full summary file for suggestions.)")


if __name__ == "__main__":
    main()
