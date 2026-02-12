from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, List
import math
import pandas as pd

from option_chain import ChainSlice

Action = Literal[
    "BUY_CE", "BUY_PE",
    "BULL_CALL_SPREAD", "BEAR_PUT_SPREAD",
    "WATCH_CE", "WATCH_PE",
    "NO_TRADE"
]


@dataclass
class OptionLeg:
    side: Literal["BUY", "SELL"]
    right: Literal["CE", "PE"]
    strike: int
    tradingsymbol: Optional[str] = None
    instrument_token: Optional[int] = None
    ltp: Optional[float] = None
    oi: Optional[float] = None


@dataclass
class Recommendation:
    underlying: str
    action: Action
    expiry: Optional[str]
    entry_underlying: Optional[float]
    stop_underlying: Optional[float]
    target_underlying: Optional[float]
    reason: str
    metrics: Dict[str, float]
    legs: List[OptionLeg]
    chain_preview: pd.DataFrame
    trigger: Optional[str] = None


def _pick_expected_width(entry: float, target: float, step: int) -> int:
    move = abs(target - entry)
    raw_steps = max(2, min(10, int(round(move / step))))
    return raw_steps * step


def _best_strike_by_oi(df_side: pd.DataFrame, candidates: List[int], strike_col: str = "strike") -> int:
    sub = df_side[df_side[strike_col].isin(candidates)].copy()
    if sub.empty:
        return candidates[0]
    if "oi" not in sub.columns:
        return int(sub[strike_col].iloc[0])
    sub["oi_num"] = pd.to_numeric(sub["oi"], errors="coerce").fillna(-1)
    best = sub.sort_values("oi_num", ascending=False).iloc[0]
    return int(best[strike_col])


def _lookup_leg(df_side: pd.DataFrame, strike_col: str, strike: int) -> dict | None:
    hit = df_side[df_side[strike_col] == strike]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def build_recommendation(underlying: str, signal, chain: ChainSlice, high_vol_atr_pct: float) -> Recommendation:
    strike_col = chain.strike_col

    # Preview table: merge CE & PE rows by strike
    ce = chain.calls.rename(columns={
        "tradingsymbol": "CE_tradingsymbol",
        "last_price": "CE_ltp",
        "oi": "CE_oi",
        "bid": "CE_bid",
        "ask": "CE_ask",
        "volume": "CE_volume",
    })
    pe = chain.puts.rename(columns={
        "tradingsymbol": "PE_tradingsymbol",
        "last_price": "PE_ltp",
        "oi": "PE_oi",
        "bid": "PE_bid",
        "ask": "PE_ask",
        "volume": "PE_volume",
    })

    preview = pd.merge(
        ce[[strike_col, "CE_tradingsymbol", "CE_ltp", "CE_oi", "CE_bid", "CE_ask", "CE_volume"]],
        pe[[strike_col, "PE_tradingsymbol", "PE_ltp", "PE_oi", "PE_bid", "PE_ask", "PE_volume"]],
        on=strike_col,
        how="outer",
    ).sort_values(strike_col)

    atr_pct = float(signal.metrics.get("atr_pct", 0.0))
    use_spread = atr_pct >= high_vol_atr_pct

    # If NO_TRADE, try watch plan
    trigger = None
    if signal.side == "NO_TRADE":
        ws = str(signal.metrics.get("watch_side", "NONE"))
        trig = str(signal.metrics.get("watch_trigger", ""))
        we = float(signal.metrics.get("watch_entry", float("nan")))
        wst = float(signal.metrics.get("watch_stop", float("nan")))
        wt = float(signal.metrics.get("watch_target", float("nan")))

        if ws in ("LONG", "SHORT") and trig:
            trigger = trig
            # We'll output a WATCH recommendation using chain
            side = ws
            entry = None if math.isnan(we) else we
            stop = None if math.isnan(wst) else wst
            target = None if math.isnan(wt) else wt
            reason = f"Watch mode: {ws}. Filters not met yet. " + trig
        else:
            return Recommendation(
                underlying=underlying,
                action="NO_TRADE",
                expiry=chain.expiry,
                entry_underlying=None,
                stop_underlying=None,
                target_underlying=None,
                reason=signal.reason,
                metrics=signal.metrics,
                legs=[],
                chain_preview=preview,
                trigger=None,
            )
    else:
        side = signal.side
        entry = signal.entry
        stop = signal.stop
        target = signal.target
        reason = signal.reason

    atm = int(chain.atm)
    strikes = sorted(preview[strike_col].dropna().astype(int).unique().tolist())
    step = int(pd.Series(strikes).diff().median()) if len(strikes) >= 3 else 50
    if step <= 0:
        step = 50

    legs: List[OptionLeg] = []

    # Strike choice (OI-aware around ATM)
    if side == "LONG":
        candidates = [atm, atm - step, atm + step]
        chosen = _best_strike_by_oi(chain.calls, candidates, strike_col=strike_col)

        if signal.side == "NO_TRADE":
            action: Action = "WATCH_CE"
        else:
            action = "BULL_CALL_SPREAD" if use_spread else "BUY_CE"

        if action == "BULL_CALL_SPREAD":
            width = _pick_expected_width(entry, target, step) if entry and target else 2 * step
            buy_strike = chosen
            sell_strike = buy_strike + width

            buy_row = _lookup_leg(chain.calls, strike_col, buy_strike)
            sell_row = _lookup_leg(chain.calls, strike_col, sell_strike)

            legs = [
                OptionLeg("BUY", "CE", buy_strike,
                          tradingsymbol=buy_row.get("tradingsymbol") if buy_row else None,
                          instrument_token=int(buy_row.get("instrument_token")) if buy_row and buy_row.get("instrument_token") else None,
                          ltp=float(buy_row.get("last_price")) if buy_row and buy_row.get("last_price") is not None else None,
                          oi=float(buy_row.get("oi")) if buy_row and buy_row.get("oi") is not None else None),
                OptionLeg("SELL", "CE", sell_strike,
                          tradingsymbol=sell_row.get("tradingsymbol") if sell_row else None,
                          instrument_token=int(sell_row.get("instrument_token")) if sell_row and sell_row.get("instrument_token") else None,
                          ltp=float(sell_row.get("last_price")) if sell_row and sell_row.get("last_price") is not None else None,
                          oi=float(sell_row.get("oi")) if sell_row and sell_row.get("oi") is not None else None),
            ]
        else:
            row = _lookup_leg(chain.calls, strike_col, chosen)
            legs = [
                OptionLeg("BUY", "CE", chosen,
                          tradingsymbol=row.get("tradingsymbol") if row else None,
                          instrument_token=int(row.get("instrument_token")) if row and row.get("instrument_token") else None,
                          ltp=float(row.get("last_price")) if row and row.get("last_price") is not None else None,
                          oi=float(row.get("oi")) if row and row.get("oi") is not None else None)
            ]

    elif side == "SHORT":
        candidates = [atm, atm + step, atm - step]
        chosen = _best_strike_by_oi(chain.puts, candidates, strike_col=strike_col)

        if signal.side == "NO_TRADE":
            action = "WATCH_PE"
        else:
            action = "BEAR_PUT_SPREAD" if use_spread else "BUY_PE"

        if action == "BEAR_PUT_SPREAD":
            width = _pick_expected_width(entry, target, step) if entry and target else 2 * step
            buy_strike = chosen
            sell_strike = buy_strike - width

            buy_row = _lookup_leg(chain.puts, strike_col, buy_strike)
            sell_row = _lookup_leg(chain.puts, strike_col, sell_strike)

            legs = [
                OptionLeg("BUY", "PE", buy_strike,
                          tradingsymbol=buy_row.get("tradingsymbol") if buy_row else None,
                          instrument_token=int(buy_row.get("instrument_token")) if buy_row and buy_row.get("instrument_token") else None,
                          ltp=float(buy_row.get("last_price")) if buy_row and buy_row.get("last_price") is not None else None,
                          oi=float(buy_row.get("oi")) if buy_row and buy_row.get("oi") is not None else None),
                OptionLeg("SELL", "PE", sell_strike,
                          tradingsymbol=sell_row.get("tradingsymbol") if sell_row else None,
                          instrument_token=int(sell_row.get("instrument_token")) if sell_row and sell_row.get("instrument_token") else None,
                          ltp=float(sell_row.get("last_price")) if sell_row and sell_row.get("last_price") is not None else None,
                          oi=float(sell_row.get("oi")) if sell_row and sell_row.get("oi") is not None else None),
            ]
        else:
            row = _lookup_leg(chain.puts, strike_col, chosen)
            legs = [
                OptionLeg("BUY", "PE", chosen,
                          tradingsymbol=row.get("tradingsymbol") if row else None,
                          instrument_token=int(row.get("instrument_token")) if row and row.get("instrument_token") else None,
                          ltp=float(row.get("last_price")) if row and row.get("last_price") is not None else None,
                          oi=float(row.get("oi")) if row and row.get("oi") is not None else None)
            ]
    else:
        action = "NO_TRADE"
        reason = signal.reason

    return Recommendation(
        underlying=underlying,
        action=action,
        expiry=chain.expiry,
        entry_underlying=entry,
        stop_underlying=stop,
        target_underlying=target,
        reason=reason,
        metrics=signal.metrics,
        legs=legs,
        chain_preview=preview,
        trigger=trigger,
    )
