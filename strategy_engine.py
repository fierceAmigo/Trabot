"""strategy_engine.py

Phase-3: multi-leg strategy engine (blueprint-level).

This module is intentionally *quote-source agnostic*:
- It does NOT fetch instruments/quotes.
- It only decides *what structure* to trade and the *desired strikes* relative to ATM.

Scanner/recommender is responsible for:
- mapping strikes -> tradingsymbol/instrument_token
- applying liquidity gates
- computing IV/Greeks
- sizing & risk caps

Strategy set (Phase-3):
- Debit spreads: BULL_CALL_SPREAD, BEAR_PUT_SPREAD
- Credit spreads: BULL_PUT_CREDIT, BEAR_CALL_CREDIT
- Selective long straddle/strangle: LONG_STRADDLE, LONG_STRANGLE
- Optional defined-risk neutral: IRON_CONDOR (two credit spreads)

Educational tool only – not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, List, Optional, Dict

Side = Literal["LONG", "SHORT"]
Regime = Literal["TREND", "CHOP", "VOLATILE", "UNKNOWN"]

Action = Literal[
    "BUY_CE", "BUY_PE",
    "BULL_CALL_SPREAD", "BEAR_PUT_SPREAD",
    "BULL_PUT_CREDIT", "BEAR_CALL_CREDIT",
    "LONG_STRADDLE", "LONG_STRANGLE",
    "IRON_CONDOR",
]

LegSide = Literal["BUY", "SELL"]
Right = Literal["CE", "PE"]


@dataclass
class LegSpec:
    side: LegSide
    right: Right
    strike: int
    tag: str = ""  # e.g., "long", "short", "hedge"


@dataclass
class Blueprint:
    action: Action
    legs: List[LegSpec]
    notes: str = ""
    anchor_tag: str = "long"  # preferred leg tag to anchor compatibility fields
    max_legs: int = 4


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def decide_blueprint(
    *,
    side: Side,
    regime: str,
    ivp: Optional[float],
    signal_strength: float,
    atm: int,
    step: int,
    width: int,
    is_live: bool = True,
    allow_neutral: bool = True,
    ivp_high: float = 0.65,
    ivp_low: float = 0.35,
    strength_high: float = 1.35,
    strength_med: float = 1.05,
) -> Blueprint:
    """Return an action + desired legs (strikes) as a blueprint.

    Notes:
    - ivp is expected in [0..1] (percentile fraction). None -> treated neutral.
    - width should be a multiple of step (recommended).
    - signal_strength is unitless (abs(score) works).
    """
    r = (regime or "UNKNOWN").upper()
    if r not in ("TREND", "CHOP", "VOLATILE"):
        r = "UNKNOWN"

    ivp_v = None if ivp is None else float(_clamp(float(ivp), 0.0, 1.0))
    s = float(signal_strength or 0.0)

    # Watch plans remain single-leg for now (Phase-5/6 will make multi-leg triggers realistic)
    if not is_live:
        if side == "LONG":
            return Blueprint("BUY_CE", [LegSpec("BUY", "CE", atm, "long")], notes="watch-mode single-leg")
        return Blueprint("BUY_PE", [LegSpec("BUY", "PE", atm, "long")], notes="watch-mode single-leg")

    # --- Regime-driven selection ---
    # TREND: prefer directionals (debit when IVP low/med, credit when IVP high)
    if r == "TREND":
        if ivp_v is not None and ivp_v >= ivp_high:
            if side == "LONG":
                # Bull Put Credit: sell put just OTM, buy lower strike hedge
                return Blueprint(
                    "BULL_PUT_CREDIT",
                    [
                        LegSpec("SELL", "PE", atm - step, "short"),
                        LegSpec("BUY", "PE", atm - step - width, "hedge"),
                    ],
                    notes="trend + high IVP -> defined-risk credit spread",
                    anchor_tag="hedge",
                )
            return Blueprint(
                "BEAR_CALL_CREDIT",
                [
                    LegSpec("SELL", "CE", atm + step, "short"),
                    LegSpec("BUY", "CE", atm + step + width, "hedge"),
                ],
                notes="trend + high IVP -> defined-risk credit spread",
                anchor_tag="hedge",
            )
        # Otherwise debit spread when strong, else single-leg
        if s >= strength_med:
            if side == "LONG":
                return Blueprint(
                    "BULL_CALL_SPREAD",
                    [
                        LegSpec("BUY", "CE", atm, "long"),
                        LegSpec("SELL", "CE", atm + width, "short"),
                    ],
                    notes="trend -> debit spread",
                    anchor_tag="long",
                )
            return Blueprint(
                "BEAR_PUT_SPREAD",
                [
                    LegSpec("BUY", "PE", atm, "long"),
                    LegSpec("SELL", "PE", atm - width, "short"),
                ],
                notes="trend -> debit spread",
                anchor_tag="long",
            )
        # weak trend: single-leg
        if side == "LONG":
            return Blueprint("BUY_CE", [LegSpec("BUY", "CE", atm, "long")], notes="trend weak -> single-leg")
        return Blueprint("BUY_PE", [LegSpec("BUY", "PE", atm, "long")], notes="trend weak -> single-leg")

    # CHOP: if IVP high and allow_neutral -> iron condor; else lean to credit spreads lightly
    if r == "CHOP":
        if allow_neutral and ivp_v is not None and ivp_v >= (ivp_high + 0.05):
            # Defined-risk neutral: two credit spreads around ATM
            return Blueprint(
                "IRON_CONDOR",
                [
                    LegSpec("SELL", "PE", atm - step, "short_put"),
                    LegSpec("BUY",  "PE", atm - step - width, "hedge_put"),
                    LegSpec("SELL", "CE", atm + step, "short_call"),
                    LegSpec("BUY",  "CE", atm + step + width, "hedge_call"),
                ],
                notes="chop + very high IVP -> defined-risk neutral (iron condor)",
                anchor_tag="hedge_put",
                max_legs=4,
            )

        if ivp_v is not None and ivp_v >= ivp_high:
            # Prefer credit spread in the direction of signal, but conservative
            if side == "LONG":
                return Blueprint(
                    "BULL_PUT_CREDIT",
                    [
                        LegSpec("SELL", "PE", atm - step, "short"),
                        LegSpec("BUY", "PE", atm - step - width, "hedge"),
                    ],
                    notes="chop + high IVP -> collect premium with defined risk",
                    anchor_tag="hedge",
                )
            return Blueprint(
                "BEAR_CALL_CREDIT",
                [
                    LegSpec("SELL", "CE", atm + step, "short"),
                    LegSpec("BUY", "CE", atm + step + width, "hedge"),
                ],
                notes="chop + high IVP -> collect premium with defined risk",
                anchor_tag="hedge",
            )

        # Low IVP chop: generally avoid; if very strong signal, allow debit spread
        if s >= strength_high:
            if side == "LONG":
                return Blueprint(
                    "BULL_CALL_SPREAD",
                    [LegSpec("BUY", "CE", atm, "long"), LegSpec("SELL", "CE", atm + width, "short")],
                    notes="chop + very strong signal -> allow debit spread",
                    anchor_tag="long",
                )
            return Blueprint(
                "BEAR_PUT_SPREAD",
                [LegSpec("BUY", "PE", atm, "long"), LegSpec("SELL", "PE", atm - width, "short")],
                notes="chop + very strong signal -> allow debit spread",
                anchor_tag="long",
            )

        # Default: keep directional but single-leg (scanner may still skip CHOP via env)
        if side == "LONG":
            return Blueprint("BUY_CE", [LegSpec("BUY", "CE", atm, "long")], notes="chop -> single-leg")
        return Blueprint("BUY_PE", [LegSpec("BUY", "PE", atm, "long")], notes="chop -> single-leg")

    # VOLATILE: if IVP low, prefer long straddle/strangle; else trend-direction with spreads
    if r == "VOLATILE":
        if ivp_v is not None and ivp_v <= ivp_low and s < strength_high:
            # Prefer straddle when ATM is available; strangle otherwise can be switched by scanner
            return Blueprint(
                "LONG_STRADDLE",
                [LegSpec("BUY", "CE", atm, "call"), LegSpec("BUY", "PE", atm, "put")],
                notes="volatile + low IVP -> long straddle",
                anchor_tag="call",
                max_legs=2,
            )
        # If IVP not low, use debit spread when signal is decent; else single-leg
        if s >= strength_med:
            if side == "LONG":
                return Blueprint(
                    "BULL_CALL_SPREAD",
                    [LegSpec("BUY", "CE", atm, "long"), LegSpec("SELL", "CE", atm + width, "short")],
                    notes="volatile -> direction via debit spread",
                    anchor_tag="long",
                )
            return Blueprint(
                "BEAR_PUT_SPREAD",
                [LegSpec("BUY", "PE", atm, "long"), LegSpec("SELL", "PE", atm - width, "short")],
                notes="volatile -> direction via debit spread",
                anchor_tag="long",
            )
        if side == "LONG":
            return Blueprint("BUY_CE", [LegSpec("BUY", "CE", atm, "long")], notes="volatile weak -> single-leg")
        return Blueprint("BUY_PE", [LegSpec("BUY", "PE", atm, "long")], notes="volatile weak -> single-leg")

    # UNKNOWN fallback
    if side == "LONG":
        return Blueprint("BUY_CE", [LegSpec("BUY", "CE", atm, "long")], notes="unknown regime -> single-leg")
    return Blueprint("BUY_PE", [LegSpec("BUY", "PE", atm, "long")], notes="unknown regime -> single-leg")
