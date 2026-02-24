"""expected_move.py

P1: Expected move calculation for spread width selection.

The expected move tells us how much the underlying is likely to move
by expiry, based on implied volatility. This is crucial for:
- Setting realistic spread widths
- Selecting appropriate strikes for credit spreads
- Calculating probability of profit

Formula:
    Expected Move = Spot × IV × sqrt(DTE/365) × Z-score

Z-scores for confidence levels:
    68% (1 std): 1.00
    80%: 1.28
    90%: 1.64
    95% (2 std): 1.96

Usage:
    from expected_move import calculate_expected_move, suggest_spread_width

    em = calculate_expected_move(spot=20000, iv=0.15, dte=7)
    width = suggest_spread_width(em, step=50, strategy="BULL_CALL_SPREAD")
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# Z-scores for different confidence levels
Z_SCORES = {
    0.50: 0.67,
    0.68: 1.00,  # 1 standard deviation
    0.80: 1.28,
    0.90: 1.64,
    0.95: 1.96,  # 2 standard deviations
    0.99: 2.58,
}

# Environment configuration
MAX_SPREAD_WIDTH_STEPS = int(os.getenv("TRABOT_MAX_SPREAD_WIDTH_STEPS", "10"))
MIN_SPREAD_WIDTH_STEPS = int(os.getenv("TRABOT_MIN_SPREAD_WIDTH_STEPS", "2"))


@dataclass
class ExpectedMoveResult:
    """Result of expected move calculation."""
    spot: float
    iv: float
    dte: int
    confidence: float
    expected_move: float
    expected_move_pct: float
    upper_bound: float
    lower_bound: float


def calculate_expected_move(
    spot: float,
    iv: float,
    dte: int,
    confidence: float = 0.68,
) -> ExpectedMoveResult:
    """Calculate expected move based on implied volatility.

    Args:
        spot: Current spot price
        iv: Implied volatility (annualized, e.g., 0.20 for 20%)
        dte: Days to expiry
        confidence: Confidence level (0.50 to 0.99)

    Returns:
        ExpectedMoveResult with move details
    """
    # Validate inputs
    spot = float(spot)
    iv = float(iv)
    dte = max(1, int(dte))  # Minimum 1 day
    confidence = float(confidence)

    # Get z-score for confidence level
    z = Z_SCORES.get(confidence)
    if z is None:
        # Interpolate
        sorted_confs = sorted(Z_SCORES.keys())
        for i, c in enumerate(sorted_confs):
            if c >= confidence:
                if i == 0:
                    z = Z_SCORES[c]
                else:
                    # Linear interpolation
                    c_low, c_high = sorted_confs[i-1], c
                    z_low, z_high = Z_SCORES[c_low], Z_SCORES[c_high]
                    z = z_low + (z_high - z_low) * (confidence - c_low) / (c_high - c_low)
                break
        else:
            z = Z_SCORES[0.68]  # Default

    # Calculate expected move
    # EM = Spot × IV × sqrt(DTE/365) × Z
    time_factor = math.sqrt(dte / 365.0)
    expected_move = spot * iv * time_factor * z

    expected_move_pct = (expected_move / spot) * 100 if spot > 0 else 0

    return ExpectedMoveResult(
        spot=spot,
        iv=iv,
        dte=dte,
        confidence=confidence,
        expected_move=expected_move,
        expected_move_pct=expected_move_pct,
        upper_bound=spot + expected_move,
        lower_bound=spot - expected_move,
    )


def suggest_spread_width(
    expected_move: float,
    step: int,
    strategy_type: str,
    regime: str = "TREND",
) -> int:
    """Suggest spread width based on expected move.

    Args:
        expected_move: Expected move in price points
        step: Strike step size
        strategy_type: Type of strategy
        regime: Market regime (TREND/CHOP/VOLATILE)

    Returns:
        Suggested width in price points (multiple of step)
    """
    step = int(step)
    if step <= 0:
        step = 50  # Default

    # Strategy-specific multipliers
    # Debit spreads: Capture most of move, but limit risk
    # Credit spreads: Stay outside expected move
    strategy_mults = {
        # Debit spreads - width should be 60-80% of expected move
        "BULL_CALL_SPREAD": 0.70,
        "BEAR_PUT_SPREAD": 0.70,

        # Credit spreads - width should be 30-50% of expected move
        "BULL_PUT_CREDIT": 0.40,
        "BEAR_CALL_CREDIT": 0.40,

        # Iron condor - narrower wings
        "IRON_CONDOR": 0.35,

        # Straddles/strangles - use for strike distance
        "LONG_STRADDLE": 0.0,  # ATM
        "LONG_STRANGLE": 0.50,  # Distance from ATM

        # Default
        "BUY_CE": 0.0,
        "BUY_PE": 0.0,
    }

    mult = strategy_mults.get(strategy_type.upper(), 0.50)

    # Regime adjustments
    regime_adj = {
        "TREND": 1.1,  # Wider in trends
        "VOLATILE": 1.2,  # Wider in volatile
        "CHOP": 0.8,  # Tighter in chop
    }
    regime_mult = regime_adj.get(regime.upper(), 1.0)

    # Calculate width
    raw_width = expected_move * mult * regime_mult

    # Round to step
    steps = int(round(raw_width / step))

    # Apply limits
    steps = max(MIN_SPREAD_WIDTH_STEPS, min(MAX_SPREAD_WIDTH_STEPS, steps))

    return steps * step


def get_credit_spread_strikes(
    spot: float,
    iv: float,
    dte: int,
    side: str,  # "LONG" (bull put) or "SHORT" (bear call)
    step: int,
    target_pop: float = 0.65,  # Target probability of profit
) -> Tuple[int, int]:
    """Calculate strikes for a credit spread.

    For credit spreads, we want the short strike to be outside
    the expected move range, giving us a probability of profit.

    Args:
        spot: Current spot price
        iv: Implied volatility
        dte: Days to expiry
        side: "LONG" for bull put credit, "SHORT" for bear call credit
        step: Strike step size
        target_pop: Target probability of profit (0.5 to 0.9)

    Returns:
        (short_strike, long_strike) where short is closer to ATM
    """
    atm = int(round(spot / step) * step)

    # Calculate expected move at (1 - target_pop) confidence
    # Higher POP = further from ATM
    em = calculate_expected_move(spot, iv, dte, confidence=1.0 - target_pop)

    # Distance from ATM to short strike
    distance = em.expected_move

    # Round to step
    distance_steps = int(round(distance / step))
    distance_steps = max(1, distance_steps)

    # Calculate width
    width = suggest_spread_width(em.expected_move, step,
                                  "BULL_PUT_CREDIT" if side == "LONG" else "BEAR_CALL_CREDIT")
    width_steps = max(2, int(width / step))

    if side == "LONG":
        # Bull put credit: sell put below ATM, buy lower put
        short_strike = atm - (distance_steps * step)
        long_strike = short_strike - (width_steps * step)
    else:
        # Bear call credit: sell call above ATM, buy higher call
        short_strike = atm + (distance_steps * step)
        long_strike = short_strike + (width_steps * step)

    return short_strike, long_strike


def get_iron_condor_strikes(
    spot: float,
    iv: float,
    dte: int,
    step: int,
    target_pop: float = 0.50,  # Iron condors typically 50% POP
) -> Dict[str, int]:
    """Calculate strikes for an iron condor.

    Iron condor has:
    - Bull put credit spread (below ATM)
    - Bear call credit spread (above ATM)

    Args:
        spot: Current spot price
        iv: Implied volatility
        dte: Days to expiry
        step: Strike step size
        target_pop: Target probability of profit

    Returns:
        Dict with 'short_put', 'long_put', 'short_call', 'long_call'
    """
    # Get each spread
    short_put, long_put = get_credit_spread_strikes(
        spot, iv, dte, "LONG", step, target_pop
    )
    short_call, long_call = get_credit_spread_strikes(
        spot, iv, dte, "SHORT", step, target_pop
    )

    return {
        "short_put": short_put,
        "long_put": long_put,
        "short_call": short_call,
        "long_call": long_call,
    }


def expected_move_multiple(
    actual_move: float,
    spot: float,
    iv: float,
    dte: int,
) -> float:
    """Calculate how many expected moves the actual move represents.

    Useful for analyzing if a move was unusual.

    Args:
        actual_move: Actual price change
        spot: Original spot price
        iv: Implied volatility at entry
        dte: Days held

    Returns:
        Multiple of expected move (1.0 = exactly expected)
    """
    em = calculate_expected_move(spot, iv, dte, confidence=0.68)
    if em.expected_move == 0:
        return 0.0
    return abs(actual_move) / em.expected_move


def iv_from_expected_move(
    spot: float,
    expected_move: float,
    dte: int,
    confidence: float = 0.68,
) -> float:
    """Reverse calculation: Get implied IV from expected move.

    Useful when you have a price target and want to know what IV that implies.

    Args:
        spot: Current spot price
        expected_move: Desired expected move
        dte: Days to expiry
        confidence: Confidence level

    Returns:
        Implied volatility (annualized)
    """
    z = Z_SCORES.get(confidence, 1.0)
    time_factor = math.sqrt(dte / 365.0)

    if spot <= 0 or time_factor <= 0 or z <= 0:
        return 0.0

    return expected_move / (spot * time_factor * z)


def breakeven_distance_vs_expected(
    breakeven: float,
    spot: float,
    iv: float,
    dte: int,
) -> float:
    """Calculate breakeven distance as fraction of expected move.

    Values < 1.0 mean breakeven is within expected move (favorable)
    Values > 1.0 mean breakeven is outside expected move (risky)

    Args:
        breakeven: Breakeven price
        spot: Current spot price
        iv: Implied volatility
        dte: Days to expiry

    Returns:
        Fraction of expected move to breakeven
    """
    em = calculate_expected_move(spot, iv, dte, confidence=0.68)
    be_distance = abs(breakeven - spot)

    if em.expected_move == 0:
        return float('inf')

    return be_distance / em.expected_move


if __name__ == "__main__":
    # Demo
    print("Expected Move Calculator")
    print("=" * 50)

    spot = 20000
    iv = 0.15  # 15% IV
    dte = 7

    print(f"\nInputs:")
    print(f"  Spot: ₹{spot:,}")
    print(f"  IV: {iv*100:.1f}%")
    print(f"  DTE: {dte}")

    # Calculate expected moves at different confidence levels
    print(f"\nExpected Moves:")
    for conf in [0.50, 0.68, 0.80, 0.90, 0.95]:
        em = calculate_expected_move(spot, iv, dte, conf)
        print(f"  {conf*100:.0f}% confidence: ±₹{em.expected_move:.0f} ({em.expected_move_pct:.2f}%)")
        print(f"    Range: ₹{em.lower_bound:.0f} - ₹{em.upper_bound:.0f}")

    # Suggest spread widths
    em = calculate_expected_move(spot, iv, dte, 0.68)
    print(f"\nSuggested Spread Widths (step=50):")
    for strat in ["BULL_CALL_SPREAD", "BULL_PUT_CREDIT", "IRON_CONDOR"]:
        width = suggest_spread_width(em.expected_move, 50, strat)
        print(f"  {strat}: {width} points")

    # Credit spread strikes
    print(f"\nBull Put Credit Spread (65% POP target):")
    short, long = get_credit_spread_strikes(spot, iv, dte, "LONG", 50, 0.65)
    print(f"  Sell {short} Put, Buy {long} Put")

    # Iron condor
    print(f"\nIron Condor (50% POP target):")
    ic = get_iron_condor_strikes(spot, iv, dte, 50, 0.50)
    print(f"  Sell {ic['short_put']} Put, Buy {ic['long_put']} Put")
    print(f"  Sell {ic['short_call']} Call, Buy {ic['long_call']} Call")
