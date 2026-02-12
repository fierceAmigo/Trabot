"""
reco_analyzer_v22.py

V2.2 analyzer:
- IST timezone-safe
- SL/Target evaluation on option candles
- MFE/MAE + target reach fraction
- Uses per-row time_stop_min if present (else args.max_hours)
- Produces:
    data/reco_evaluated_v22_latest.csv
    data/reco_eval_v22_summary_latest.txt
    timestamped snapshots

Run:
  python3 reco_analyzer_v22.py --last_n 50 --interval minute --max_hours 6
"""

from __future__ import annotations

import argparse
import os
import re
import time
import datetime as dt
from functools import lru_cache
from typing import Optional, Tuple, List, Dict

import pandas as pd

from kite_client import get_kite
from iv_greeks import implied_volatility

IST = "Asia/Kolkata"

DATA_DIR = os.getenv("TRABOT_DATA_DIR", "data")
HISTORY_PATH = os.getenv("TRABOT_RECO_HISTORY", os.path.join(DATA_DIR, "reco_history.csv"))
INSTRUMENTS_CACHE_PATH = os.getenv("INSTRUMENTS_CACHE_PATH", os.path.join(DATA_DIR, "kite_instruments_NFO.csv"))
CACHE_DIR = os.path.join(DATA_DIR, "reco_eval_cache_v22")

RISK_FREE = float(os.getenv("RISK_FREE_RATE", "0.06"))

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

_STRIKE_RE = re.compile(r"(\d+)(CE|PE)$", re.IGNORECASE)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_ist(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(IST)
    return t.tz_convert(IST)


def ensure_index_ist(df: pd.DataFrame) -> pd.DataFrame:
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
    t = to_ist(t)
    return t.tz_localize(None).to_pydatetime()


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
        return ensure_index_ist(df)
    except Exception:
        return None


def _write_cache(path: str, df: pd.DataFrame) -> None:
    _ensure_dir(os.path.dirname(path))
    out = ensure_index_ist(df.copy())
    out.index.name = "Datetime"
    out.to_csv(path, index=True)


def _load_or_fetch_nfo_instruments() -> pd.DataFrame:
    _ensure_dir(os.path.dirname(INSTRUMENTS_CACHE_PATH))
    if os.path.exists(INSTRUMENTS_CACHE_PATH):
        df = pd.read_csv(INSTRUMENTS_CACHE_PATH)
        if not df.empty:
            return df
    kite = get_kite()
    df = pd.DataFrame(kite.instruments("NFO"))
    df.to_csv(INSTRUMENTS_CACHE_PATH, index=False)
    return df


@lru_cache(maxsize=20000)
def resolve_nfo_token(tradingsymbol: str) -> int:
    df = _load_or_fetch_nfo_instruments()
    ts = str(tradingsymbol).strip().upper()
    hit = df[df["tradingsymbol"].astype(str).str.upper() == ts]
    if hit.empty:
        raise RuntimeError(f"Could not resolve NFO token for {tradingsymbol}")
    return int(hit.iloc[0]["instrument_token"])


def _chunked_historical(kite, token: int, start: dt.datetime, end: dt.datetime, interval: str) -> List[dict]:
    max_days = MAX_DAYS_PER_REQ.get(interval, 30)
    cur = start
    out: List[dict] = []
    while cur < end:
        cur_end = min(end, cur + dt.timedelta(days=max_days))
        rows = kite.historical_data(token, cur, cur_end, interval, continuous=False, oi=False) or []
        out.extend(rows)
        cur = cur_end + dt.timedelta(seconds=1)
    return out


def fetch_option_candles(tradingsymbol: str, start_ist: pd.Timestamp, end_ist: pd.Timestamp, interval: str) -> pd.DataFrame:
    _ensure_dir(CACHE_DIR)
    start_ist = to_ist(start_ist)
    end_ist = to_ist(end_ist)

    key = f"{tradingsymbol}__{interval}__{start_ist.strftime('%Y%m%d%H%M')}__{end_ist.strftime('%Y%m%d%H%M')}"
    cp = _cache_path("opt", key)
    cached = _read_cache(cp, max_age_minutes=7 * 24 * 60)
    if cached is not None and not cached.empty:
        return cached

    kite = get_kite()
    token = resolve_nfo_token(tradingsymbol)
    rows = _chunked_historical(kite, token, naive_ist_datetime(start_ist), naive_ist_datetime(end_ist), interval)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).rename(columns={"date": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").drop_duplicates(subset=["Datetime"]).set_index("Datetime")
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = ensure_index_ist(df)
    _write_cache(cp, df)
    return df


def parse_strike_and_right(tradingsymbol: str) -> Tuple[Optional[int], Optional[str]]:
    ts = str(tradingsymbol).strip().upper()
    m = _STRIKE_RE.search(ts)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2).upper()


def expiry_end_ist(expiry_str: str) -> Optional[pd.Timestamp]:
    if not expiry_str or str(expiry_str).strip() == "":
        return None
    try:
        d = pd.to_datetime(expiry_str).date()
        return pd.Timestamp(dt.datetime(d.year, d.month, d.day, 15, 25, 0)).tz_localize(IST)
    except Exception:
        return None


def t_years(exp_end: pd.Timestamp, at: pd.Timestamp) -> float:
    exp_end = to_ist(exp_end)
    at = to_ist(at)
    sec = (exp_end - at).total_seconds()
    if sec <= 0:
        return 0.0
    return sec / (365.0 * 24 * 3600)


def evaluate_path_with_mfe_mae(opt_df: pd.DataFrame, ts_reco: pd.Timestamp, sl: float, target: float):
    if opt_df is None or opt_df.empty:
        return {"status": "NO_CANDLES"}

    opt_df = ensure_index_ist(opt_df).sort_index()
    ts_reco = to_ist(ts_reco)

    after = opt_df.loc[opt_df.index >= ts_reco]
    if after.empty:
        return {"status": "NO_CANDLES_AFTER_RECO"}

    entry_ts = after.index[0]
    entry_px = float(after.iloc[0]["open"])
    if entry_px <= 0 or sl <= 0 or target <= 0:
        return {"status": "BAD_LEVELS"}

    # MFE/MAE measured from entry until exit
    max_high = entry_px
    min_low = entry_px

    hit = None
    exit_ts = None
    exit_px = None
    bars = 0

    for t, row in after.iterrows():
        bars += 1
        lo = float(row["low"])
        hi = float(row["high"])

        max_high = max(max_high, hi)
        min_low = min(min_low, lo)

        sl_hit = lo <= sl
        tgt_hit = hi >= target

        if sl_hit and tgt_hit:
            hit = "BOTH_SAME_BAR"
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

    mfe = max_high - entry_px
    mae = min_low - entry_px  # negative if adverse

    target_dist = target - entry_px
    sl_dist = entry_px - sl

    mfe_frac = (mfe / target_dist) if target_dist > 1e-9 else None
    mae_frac = (abs(mae) / sl_dist) if sl_dist > 1e-9 else None

    return {
        "status": "OK",
        "entry_ts": entry_ts,
        "entry_px": entry_px,
        "hit": hit,
        "exit_ts": exit_ts,
        "exit_px": exit_px,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "bars_to_exit": bars,
        "mfe": mfe,
        "mae": mae,
        "mfe_frac_to_target": mfe_frac,
        "mae_frac_to_sl": mae_frac,
        "max_high": max_high,
        "min_low": min_low,
    }


def loss_tags(row: dict) -> str:
    tags = []
    pnl_pct = float(row.get("pnl_pct", 0.0))
    hit = str(row.get("hit", ""))
    dte = row.get("dte", None)
    iv_pct = row.get("iv_pct", None)
    conf = str(row.get("greeks_conf", "")).lower()
    delta = row.get("delta", None)
    move_fav = row.get("spot_move_pct_favorable", None)
    mfe_frac = row.get("mfe_frac_to_target", None)

    if dte is not None:
        try:
            if int(dte) <= 3:
                tags.append("EXPIRY_WEEK")
        except Exception:
            pass

    if iv_pct is not None:
        try:
            if float(iv_pct) >= 0.70:
                tags.append("HIGH_IV_PCTL")
        except Exception:
            pass

    if conf == "low":
        tags.append("LOW_GREEKS_CONF")

    if delta is not None:
        try:
            if abs(float(delta)) < 0.35:
                tags.append("LOW_DELTA")
        except Exception:
            pass

    if pnl_pct < 0:
        if move_fav is not None:
            try:
                if float(move_fav) < -0.25 / 100:
                    tags.append("WRONG_DIRECTION")
                if abs(float(move_fav)) < 0.15 / 100:
                    tags.append("THETA_BLEED_FLAT_SPOT")
            except Exception:
                pass

        # "Target too far" heuristic: never reached even 60% of target distance
        if hit == "TIME_EXIT" and mfe_frac is not None:
            try:
                if float(mfe_frac) < 0.60:
                    tags.append("TARGET_TOO_FAR_OR_NO_MOMENTUM")
            except Exception:
                pass

    if hit == "BOTH_SAME_BAR":
        tags.append("AMBIGUOUS_BAR")

    return " | ".join(tags) if tags else "OK_OR_UNKNOWN"


def winrate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float((df["pnl_pct"] > 0).mean())


def summarize(df: pd.DataFrame) -> str:
    lines = []
    ok = df[df["eval_status"] == "OK"].copy()
    lines.append(f"Total rows read: {len(df)}")
    lines.append(f"Rows evaluated: {len(ok)}")
    if ok.empty:
        return "\n".join(lines)

    lines.append(f"Overall win-rate: {winrate(ok) * 100:.1f}%")
    lines.append(f"Average PnL%: {ok['pnl_pct'].mean() * 100:.2f}%\n")

    lines.append("Outcome distribution:")
    for k, v in ok["hit"].value_counts().items():
        lines.append(f"  {k}: {v} ({v/len(ok)*100:.1f}%)")
    lines.append("")

    # show how often targets were unrealistic (MFE frac low)
    if "mfe_frac_to_target" in ok.columns:
        x = ok["mfe_frac_to_target"].dropna()
        if not x.empty:
            lines.append(f"Avg MFE fraction to target: {float(x.mean()):.2f}")
            lines.append(f"% trades reaching >=0.60 of target distance: {(x >= 0.60).mean()*100:.1f}%\n")

    # loss tags
    losses = ok[ok["pnl_pct"] < 0]
    if not losses.empty and "loss_tags" in losses.columns:
        lines.append("Top loss tags:")
        tags = losses["loss_tags"].astype(str).str.split(r"\s*\|\s*", regex=True).explode()
        vc = tags.value_counts()
        for k, v in vc.head(12).items():
            lines.append(f"  {k}: {v} ({v/len(losses)*100:.1f}%)")
        lines.append("")

    # group winrates by regime/iv_bin/dte
    def grp(col: str, title: str):
        nonlocal lines
        if col not in ok.columns:
            return
        g = ok.groupby(col).agg(n=("pnl_pct", "size"), win=("pnl_pct", lambda s: float((s > 0).mean())), avg=("pnl_pct", "mean")).reset_index()
        g = g.sort_values("n", ascending=False)
        lines.append(title + ":")
        for _, r in g.iterrows():
            lines.append(f"  {r[col]}: n={int(r['n'])} win={r['win']*100:.1f}% avgPnL={r['avg']*100:.2f}%")
        lines.append("")

    # bins
    if "iv_pct" in ok.columns:
        ok["iv_bin"] = pd.cut(ok["iv_pct"].astype(float), [-1, 0.30, 0.70, 10], labels=["LOW_IV", "MID_IV", "HIGH_IV"])
    if "dte" in ok.columns:
        ok["dte_bin"] = pd.cut(ok["dte"].astype(float), [-1, 3, 7, 3650], labels=["DTE_0_3", "DTE_4_7", "DTE_8_PLUS"])
    if "delta" in ok.columns:
        ok["delta_bin"] = pd.cut(ok["delta"].astype(float).abs(), [-1, 0.35, 0.55, 10], labels=["LOW_DELTA", "MID_DELTA", "HIGH_DELTA"])

    grp("source", "By source")
    grp("bucket", "By bucket")
    grp("regime", "By regime")
    grp("iv_bin", "By IV bin")
    grp("dte_bin", "By DTE bin")
    grp("delta_bin", "By delta bin")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=HISTORY_PATH)
    ap.add_argument("--interval", default="5minute")
    ap.add_argument("--max_hours", type=float, default=24.0)
    ap.add_argument("--last_n", type=int, default=0)
    ap.add_argument("--from_date", default="")
    ap.add_argument("--include_watch", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.history):
        raise SystemExit(f"Missing: {args.history}")

    df = pd.read_csv(args.history)
    if df.empty:
        raise SystemExit("History file empty.")

    df["ts_reco"] = pd.to_datetime(df["ts_reco"], errors="coerce")
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
        ts_reco = to_ist(r["ts_reco"])
        tsym = str(r.get("tradingsymbol") or "").strip()
        if not tsym:
            continue

        try:
            entry = float(r.get("entry") or 0.0)
            sl = float(r.get("sl") or 0.0)
            tgt = float(r.get("target") or 0.0)
        except Exception:
            out_rows.append({**r.to_dict(), "eval_status": "BAD_LEVELS"})
            continue

        # horizon uses per-row time_stop_min if present, else max_hours
        time_stop_min = None
        try:
            if str(r.get("time_stop_min", "")).strip() != "":
                time_stop_min = int(float(r.get("time_stop_min")))
        except Exception:
            time_stop_min = None

        horizon_hours = args.max_hours
        if time_stop_min is not None and time_stop_min > 0:
            horizon_hours = min(horizon_hours, time_stop_min / 60.0)

        exp_end = expiry_end_ist(str(r.get("expiry") or "").strip())
        end_ist = ts_reco + pd.Timedelta(hours=float(horizon_hours))
        if exp_end is not None:
            end_ist = min(end_ist, exp_end)
        if end_ist <= ts_reco:
            out_rows.append({**r.to_dict(), "eval_status": "SKIP_EXPIRED"})
            continue

        try:
            opt_df = fetch_option_candles(tsym, ts_reco, end_ist, args.interval)
        except Exception as e:
            out_rows.append({**r.to_dict(), "eval_status": f"OPT_FETCH_FAIL: {e}"})
            continue

        ev = evaluate_path_with_mfe_mae(opt_df, ts_reco, sl, tgt)
        if ev.get("status") != "OK":
            out_rows.append({**r.to_dict(), "eval_status": ev.get("status")})
            continue

        # signed favorable spot move heuristic (best-effort, optional)
        spot_move_pct_fav = None
        try:
            # uses market_data for spot; ok if missing
            from market_data import fetch_history_cached as fh
            underlying = str(r.get("underlying") or "").strip().upper()
            if underlying:
                spot_symbol = f"NSE:{underlying}"
                # indices mapping not needed for heuristic; ok
                df_spot, _ = fh(spot_symbol, lookback_days=10, interval="5minute", ttl_minutes=5)
                df_spot.index = pd.to_datetime(df_spot.index)
                df_spot = ensure_index_ist(df_spot)
                seg = df_spot.loc[(df_spot.index >= ev["entry_ts"]) & (df_spot.index <= ev["exit_ts"])]
                if not seg.empty:
                    s0 = float(seg.iloc[0]["open"])
                    s1 = float(seg.iloc[-1]["close"])
                    mv = (s1 - s0) / s0 if s0 > 0 else 0.0
                    if str(r.get("action") or "").endswith("_PE"):
                        mv = -mv
                    spot_move_pct_fav = mv
        except Exception:
            pass

        out = {**r.to_dict()}
        out.update({
            "eval_status": "OK",
            "entry_ts_mkt": ev["entry_ts"],
            "entry_px_mkt": ev["entry_px"],
            "hit": ev["hit"],
            "exit_ts": ev["exit_ts"],
            "exit_px": ev["exit_px"],
            "pnl": ev["pnl"],
            "pnl_pct": ev["pnl_pct"],
            "bars_to_exit": ev["bars_to_exit"],
            "minutes_to_exit": (to_ist(ev["exit_ts"]) - to_ist(ev["entry_ts"])).total_seconds() / 60.0,
            "mfe": ev["mfe"],
            "mae": ev["mae"],
            "mfe_frac_to_target": ev["mfe_frac_to_target"],
            "mae_frac_to_sl": ev["mae_frac_to_sl"],
            "spot_move_pct_favorable": spot_move_pct_fav,
        })
        out["loss_tags"] = loss_tags(out)
        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)

    latest_csv = os.path.join(DATA_DIR, "reco_evaluated_v22_latest.csv")
    snap_csv = os.path.join(DATA_DIR, f"reco_evaluated_v22_{run_id}.csv")
    out_df.to_csv(latest_csv, index=False)
    out_df.to_csv(snap_csv, index=False)

    summary = summarize(out_df)
    latest_txt = os.path.join(DATA_DIR, "reco_eval_v22_summary_latest.txt")
    snap_txt = os.path.join(DATA_DIR, f"reco_eval_v22_summary_{run_id}.txt")
    with open(latest_txt, "w", encoding="utf-8") as f:
        f.write(summary)
    with open(snap_txt, "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n=== RECO ANALYZER V2.2 DONE ===")
    print(f"Saved: {latest_csv}")
    print(f"Saved: {snap_csv}")
    print(f"Saved: {latest_txt}")
    print(f"Saved: {snap_txt}\n")
    print("\n".join(summary.splitlines()[:35]))


if __name__ == "__main__":
    main()
