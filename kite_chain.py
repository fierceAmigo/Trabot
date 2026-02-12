from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import os
import datetime as dt

import pandas as pd

from kite_client import get_kite


@dataclass
class ChainSlice:
    expiry: Optional[str]
    spot: float
    atm: int
    strike_col: str
    calls: pd.DataFrame
    puts: pd.DataFrame
    source: str  # "kite"


def _today_ist_date() -> dt.date:
    return dt.date.today()


def _load_or_fetch_instruments(cache_path: str) -> pd.DataFrame:
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


def _pick_nearest_expiry(df_opt: pd.DataFrame, min_dte_days: int = 1) -> Optional[dt.date]:
    if df_opt.empty:
        return None

    exp = pd.to_datetime(df_opt["expiry"], errors="coerce").dt.date
    tmp = df_opt.copy()
    tmp["expiry_norm"] = exp

    today = _today_ist_date()
    tmp["dte"] = tmp["expiry_norm"].apply(lambda d: (d - today).days if pd.notna(d) else -999)

    candidates = tmp[tmp["dte"] >= min_dte_days]["expiry_norm"].dropna().unique()
    if len(candidates) == 0:
        candidates = tmp[tmp["dte"] >= 0]["expiry_norm"].dropna().unique()
        if len(candidates) == 0:
            return None

    return sorted(candidates)[0]


def _batch(lst: List[str], n: int = 150):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _infer_strike_step(strikes: List[int]) -> int:
    strikes = sorted(set(int(s) for s in strikes if s is not None))
    if len(strikes) < 3:
        return 1
    diffs = pd.Series(strikes).diff().dropna()
    step = int(diffs.median())
    if step <= 0:
        step = int(diffs.mode().iloc[0]) if not diffs.mode().empty else 1
    return max(step, 1)


def get_kite_chain_slice(
    underlying: str,
    kite_spot_symbol: str,
    strike_step: int,
    strikes_around_atm: int,
    cache_path: str,
) -> ChainSlice:
    """
    strike_step:
      - pass a real step (NIFTY=50, BANKNIFTY=100) OR
      - pass 0 to auto-infer from available strikes (works for stocks)
    """
    kite = get_kite()
    instruments = _load_or_fetch_instruments(cache_path)

    needed_cols = ["segment", "name", "instrument_type", "expiry", "strike", "tradingsymbol", "instrument_token"]
    for c in needed_cols:
        if c not in instruments.columns:
            raise RuntimeError(f"Missing '{c}' in instruments dump. Columns={list(instruments.columns)}")

    df_opt = instruments[
        (instruments["segment"].astype(str) == "NFO-OPT") &
        (instruments["name"].astype(str) == underlying) &
        (instruments["instrument_type"].astype(str).isin(["CE", "PE"]))
    ].copy()

    if df_opt.empty:
        raise RuntimeError(f"No options found for {underlying} in Kite instruments dump.")

    expiry_date = _pick_nearest_expiry(df_opt, min_dte_days=1)
    if not expiry_date:
        raise RuntimeError("Could not determine a valid expiry from instruments dump.")

    df_opt["expiry_norm"] = pd.to_datetime(df_opt["expiry"], errors="coerce").dt.date
    df_opt = df_opt[df_opt["expiry_norm"] == expiry_date].copy()

    # Spot from Kite
    spot_payload = kite.ltp([kite_spot_symbol])
    spot = float(spot_payload[kite_spot_symbol]["last_price"])

    # Infer strike step if requested
    df_opt["strike_int"] = pd.to_numeric(df_opt["strike"], errors="coerce").round(0).astype("Int64")
    available_strikes = df_opt["strike_int"].dropna().astype(int).unique().tolist()

    if strike_step is None or int(strike_step) <= 0:
        strike_step = _infer_strike_step(available_strikes)

    # Compute ATM from inferred/known step
    atm = int(round(spot / strike_step) * strike_step)
    lo = atm - strikes_around_atm * strike_step
    hi = atm + strikes_around_atm * strike_step

    df_opt = df_opt[(df_opt["strike_int"] >= lo) & (df_opt["strike_int"] <= hi)].copy()
    if df_opt.empty:
        # fallback: keep nearest strikes if window filtering too strict
        df_opt = instruments[
            (instruments["segment"].astype(str) == "NFO-OPT") &
            (instruments["name"].astype(str) == underlying) &
            (instruments["instrument_type"].astype(str).isin(["CE", "PE"])) &
            (pd.to_datetime(instruments["expiry"], errors="coerce").dt.date == expiry_date)
        ].copy()
        df_opt["strike_int"] = pd.to_numeric(df_opt["strike"], errors="coerce").round(0).astype("Int64")
        # pick nearest 2*strikes_around_atm+1 strikes around spot
        strikes_all = sorted(df_opt["strike_int"].dropna().astype(int).unique().tolist())
        if not strikes_all:
            raise RuntimeError("No strikes available after fallback.")
        nearest = min(strikes_all, key=lambda s: abs(s - spot))
        atm = int(nearest)
        lo = atm - strikes_around_atm * strike_step
        hi = atm + strikes_around_atm * strike_step
        df_opt = df_opt[(df_opt["strike_int"] >= lo) & (df_opt["strike_int"] <= hi)].copy()

    df_opt["strike"] = df_opt["strike_int"].astype(int)
    df_opt["kite_symbol"] = "NFO:" + df_opt["tradingsymbol"].astype(str)
    symbols = df_opt["kite_symbol"].tolist()

    # Quote options (batched)
    quotes: dict = {}
    for chunk in _batch(symbols, n=150):
        q = kite.quote(chunk) or {}
        quotes.update(q)

    rows = []
    for sym in symbols:
        d = quotes.get(sym) or {}
        depth = d.get("depth") or {}
        buy0 = (depth.get("buy") or [{}])[0]
        sell0 = (depth.get("sell") or [{}])[0]

        bid = buy0.get("price")
        ask = sell0.get("price")

        rows.append({
            "kite_symbol": sym,
            "last_price": d.get("last_price", None),
            "oi": d.get("oi"),
            "volume": d.get("volume"),
            "bid": bid,
            "ask": ask,
        })

    extra = pd.DataFrame(rows, columns=["kite_symbol", "last_price", "oi", "volume", "bid", "ask"])
    merged = df_opt.merge(extra, on="kite_symbol", how="left")

    # Numeric + LTP fallback = mid(bid,ask)
    merged["last_price"] = pd.to_numeric(merged.get("last_price"), errors="coerce")
    merged["bid"] = pd.to_numeric(merged.get("bid"), errors="coerce")
    merged["ask"] = pd.to_numeric(merged.get("ask"), errors="coerce")

    mid = (merged["bid"] + merged["ask"]) / 2.0
    merged["last_price"] = merged["last_price"].fillna(mid)

    expiry_str = expiry_date.isoformat()

    calls = merged[merged["instrument_type"] == "CE"].copy()
    puts = merged[merged["instrument_type"] == "PE"].copy()

    keep_cols = ["strike", "tradingsymbol", "instrument_token", "last_price", "oi", "volume", "bid", "ask"]
    calls = calls[keep_cols].sort_values("strike").reset_index(drop=True)
    puts = puts[keep_cols].sort_values("strike").reset_index(drop=True)

    return ChainSlice(
        expiry=expiry_str,
        spot=spot,
        atm=atm,
        strike_col="strike",
        calls=calls,
        puts=puts,
        source="kite",
    )
