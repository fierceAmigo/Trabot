"""iv_term_structure.py

P1: IV Term Structure Analysis for options strategy selection.

The IV term structure (contango/backwardation) affects strategy selection:
- Contango (near IV < far IV): Calendar spreads favorable, front-month selling
- Backwardation (near IV > far IV): Front-month buying, avoid calendars
- Flat: Neutral, standard strategies apply

This module computes term structure metrics and adjusts strategy recommendations.

Usage:
    from iv_term_structure import analyze_term_structure, get_term_structure_bias

    ts = analyze_term_structure(underlying, chain_data)
    bias = get_term_structure_bias(ts)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

try:
    from dateutil import tz
    IST = tz.gettz("Asia/Kolkata")
except ImportError:
    IST = None


class TermStructure(Enum):
    CONTANGO = "CONTANGO"       # Near IV < Far IV (normal)
    BACKWARDATION = "BACKWARDATION"  # Near IV > Far IV (inverted)
    FLAT = "FLAT"              # Near IV ≈ Far IV


@dataclass
class TermStructureResult:
    """Result of term structure analysis."""
    underlying: str
    structure: TermStructure
    near_expiry: str           # Near-term expiry date
    far_expiry: str            # Far-term expiry date
    near_atm_iv: float         # ATM IV for near expiry
    far_atm_iv: float          # ATM IV for far expiry
    iv_diff: float             # far_iv - near_iv
    iv_diff_pct: float         # Percentage difference
    confidence: float          # 0-1 confidence in classification
    slope: float               # IV per day of DTE
    recommendation: str        # Strategy recommendation


# Environment configuration
TERM_STRUCTURE_ENABLE = os.getenv("TRABOT_TERM_STRUCTURE_ENABLE", "1").strip() == "1"
CONTANGO_THRESHOLD = float(os.getenv("TRABOT_CONTANGO_THRESHOLD", "0.02"))  # 2% difference
BACKWARDATION_THRESHOLD = float(os.getenv("TRABOT_BACKWARDATION_THRESHOLD", "-0.02"))


def _now_ist() -> datetime:
    """Get current time in IST."""
    if IST:
        return datetime.now(IST)
    return datetime.now()


def _get_atm_iv(
    options_data: List[Dict],
    spot: float,
    expiry: str,
) -> Optional[float]:
    """Extract ATM IV from options data for a specific expiry.

    Args:
        options_data: List of option records with strike, expiry, iv, right
        spot: Current spot price
        expiry: Expiry date string (YYYY-MM-DD or similar)

    Returns:
        ATM IV or None if not found
    """
    # Filter to this expiry
    expiry_options = [o for o in options_data if o.get("expiry") == expiry]
    if not expiry_options:
        return None

    # Find ATM strike (closest to spot)
    atm_strike = min(
        set(o.get("strike", 0) for o in expiry_options),
        key=lambda s: abs(s - spot),
        default=None
    )
    if atm_strike is None:
        return None

    # Get ATM call and put IVs
    atm_options = [o for o in expiry_options if o.get("strike") == atm_strike]

    ivs = []
    for opt in atm_options:
        iv = opt.get("iv")
        if iv and iv > 0:
            ivs.append(iv)

    if not ivs:
        return None

    # Average of call and put ATM IVs
    return sum(ivs) / len(ivs)


def _classify_structure(
    near_iv: float,
    far_iv: float,
    near_dte: int,
    far_dte: int,
) -> Tuple[TermStructure, float, str]:
    """Classify term structure and generate recommendation.

    Returns:
        (structure, confidence, recommendation)
    """
    if near_iv <= 0 or far_iv <= 0:
        return TermStructure.FLAT, 0.0, "Insufficient IV data"

    iv_diff_pct = (far_iv - near_iv) / near_iv

    # Calculate confidence based on magnitude of difference
    abs_diff = abs(iv_diff_pct)
    if abs_diff >= 0.10:  # 10%+ difference
        confidence = 0.95
    elif abs_diff >= 0.05:
        confidence = 0.80
    elif abs_diff >= 0.02:
        confidence = 0.60
    else:
        confidence = 0.30

    # Classify
    if iv_diff_pct >= CONTANGO_THRESHOLD:
        structure = TermStructure.CONTANGO
        recommendation = (
            "CONTANGO: Far IV higher. Favor front-month selling, "
            "calendar spreads (sell near, buy far), diagonal spreads."
        )
    elif iv_diff_pct <= BACKWARDATION_THRESHOLD:
        structure = TermStructure.BACKWARDATION
        recommendation = (
            "BACKWARDATION: Near IV higher. Avoid calendar spreads, "
            "favor near-term directional plays, consider ratio spreads."
        )
    else:
        structure = TermStructure.FLAT
        recommendation = (
            "FLAT: IV term structure neutral. Standard strategy selection applies."
        )

    return structure, confidence, recommendation


def analyze_term_structure(
    underlying: str,
    options_data: List[Dict],
    spot: float,
    near_dte_target: int = 7,
    far_dte_target: int = 30,
) -> Optional[TermStructureResult]:
    """Analyze IV term structure for an underlying.

    Args:
        underlying: Underlying symbol
        options_data: List of option records with strike, expiry, iv, right, dte
        spot: Current spot price
        near_dte_target: Target DTE for near-term (default 7)
        far_dte_target: Target DTE for far-term (default 30)

    Returns:
        TermStructureResult or None if insufficient data
    """
    if not TERM_STRUCTURE_ENABLE:
        return None

    if not options_data or spot <= 0:
        return None

    underlying = underlying.upper()

    # Get unique expiries with their DTEs
    expiries_with_dte = {}
    for opt in options_data:
        exp = opt.get("expiry")
        dte = opt.get("dte")
        if exp and dte is not None:
            if exp not in expiries_with_dte:
                expiries_with_dte[exp] = dte

    if len(expiries_with_dte) < 2:
        return None  # Need at least 2 expiries

    # Sort by DTE
    sorted_expiries = sorted(expiries_with_dte.items(), key=lambda x: x[1])

    # Find near and far expiries closest to targets
    near_expiry = None
    near_dte = None
    far_expiry = None
    far_dte = None

    # Find nearest expiry to near_dte_target (but at least 1 DTE)
    for exp, dte in sorted_expiries:
        if dte >= 1:
            if near_expiry is None or abs(dte - near_dte_target) < abs(near_dte - near_dte_target):
                near_expiry = exp
                near_dte = dte

    # Find nearest expiry to far_dte_target (must be > near_dte)
    for exp, dte in sorted_expiries:
        if near_dte is not None and dte > near_dte + 7:  # At least 7 days apart
            if far_expiry is None or abs(dte - far_dte_target) < abs(far_dte - far_dte_target):
                far_expiry = exp
                far_dte = dte

    if near_expiry is None or far_expiry is None:
        return None

    # Get ATM IVs for each expiry
    near_atm_iv = _get_atm_iv(options_data, spot, near_expiry)
    far_atm_iv = _get_atm_iv(options_data, spot, far_expiry)

    if near_atm_iv is None or far_atm_iv is None:
        return None

    # Classify and generate recommendation
    structure, confidence, recommendation = _classify_structure(
        near_atm_iv, far_atm_iv, near_dte, far_dte
    )

    # Calculate slope (IV change per day)
    dte_diff = far_dte - near_dte
    iv_diff = far_atm_iv - near_atm_iv
    slope = iv_diff / dte_diff if dte_diff > 0 else 0.0

    iv_diff_pct = (iv_diff / near_atm_iv * 100) if near_atm_iv > 0 else 0.0

    return TermStructureResult(
        underlying=underlying,
        structure=structure,
        near_expiry=near_expiry,
        far_expiry=far_expiry,
        near_atm_iv=near_atm_iv,
        far_atm_iv=far_atm_iv,
        iv_diff=iv_diff,
        iv_diff_pct=iv_diff_pct,
        confidence=confidence,
        slope=slope,
        recommendation=recommendation,
    )


def get_term_structure_bias(
    ts_result: Optional[TermStructureResult],
) -> Tuple[str, float]:
    """Get strategy bias based on term structure.

    Returns:
        (bias_direction, adjustment_factor)

    Bias directions:
        "SELL_NEAR": Favor selling near-term premium
        "BUY_NEAR": Favor buying near-term options
        "NEUTRAL": No strong bias
    """
    if ts_result is None:
        return "NEUTRAL", 1.0

    if ts_result.confidence < 0.5:
        return "NEUTRAL", 1.0

    if ts_result.structure == TermStructure.CONTANGO:
        # In contango, near-term IV is relatively cheap
        # Favor selling near-term, buying far-term
        factor = 1.0 + (ts_result.iv_diff_pct / 100) * 0.5
        return "SELL_NEAR", min(1.3, max(0.8, factor))

    elif ts_result.structure == TermStructure.BACKWARDATION:
        # In backwardation, near-term IV is elevated
        # Favor buying near-term for volatility plays
        factor = 1.0 - (ts_result.iv_diff_pct / 100) * 0.5
        return "BUY_NEAR", min(1.3, max(0.8, factor))

    return "NEUTRAL", 1.0


def adjust_strategy_for_term_structure(
    strategy_type: str,
    ts_result: Optional[TermStructureResult],
) -> Tuple[str, str]:
    """Adjust strategy selection based on term structure.

    Args:
        strategy_type: Proposed strategy type
        ts_result: Term structure analysis result

    Returns:
        (adjusted_strategy, reason)
    """
    if ts_result is None or ts_result.confidence < 0.5:
        return strategy_type, ""

    # Calendar-type strategies
    calendar_strategies = {"CALENDAR_SPREAD", "DIAGONAL_SPREAD"}

    if ts_result.structure == TermStructure.BACKWARDATION:
        if strategy_type in calendar_strategies:
            return "SINGLE_LEG", "Avoided calendar in backwardation term structure"

    elif ts_result.structure == TermStructure.CONTANGO:
        # Contango is favorable for calendar spreads
        # Could upgrade single-leg to calendar if supported
        pass

    return strategy_type, ""


def get_optimal_expiry(
    ts_result: Optional[TermStructureResult],
    side: str,  # "LONG" or "SHORT"
    default_dte: int = 7,
) -> Tuple[int, str]:
    """Get optimal DTE based on term structure and trade side.

    Args:
        ts_result: Term structure analysis
        side: Trade side (LONG for buying premium, SHORT for selling)
        default_dte: Default DTE if no strong preference

    Returns:
        (recommended_dte, reason)
    """
    if ts_result is None or ts_result.confidence < 0.5:
        return default_dte, ""

    if ts_result.structure == TermStructure.CONTANGO:
        if side == "SHORT":
            # Sell near-term (cheaper IV, faster decay)
            return min(default_dte, 7), "Near-term sell in contango"
        else:
            # Buy far-term (higher IV priced in, but more time)
            return max(default_dte, 21), "Far-term buy in contango"

    elif ts_result.structure == TermStructure.BACKWARDATION:
        if side == "LONG":
            # Buy near-term (elevated IV = expecting vol)
            return min(default_dte, 7), "Near-term buy in backwardation"
        else:
            # Sell far-term if selling (avoid near-term elevated IV)
            return max(default_dte, 21), "Far-term sell in backwardation"

    return default_dte, ""


def compute_calendar_edge(
    near_iv: float,
    far_iv: float,
    near_dte: int,
    far_dte: int,
) -> Dict:
    """Compute edge metrics for calendar spread.

    Calendar spread edge exists when:
    - Near IV > Far IV (sell expensive, buy cheap)
    - Or when term structure is steep in contango

    Returns dict with edge metrics.
    """
    if near_iv <= 0 or far_iv <= 0:
        return {"has_edge": False, "edge_pct": 0.0}

    # IV differential edge
    iv_edge = (near_iv - far_iv) / near_iv

    # Time decay edge (near decays faster)
    dte_ratio = near_dte / far_dte if far_dte > 0 else 0

    # Calendar has edge when selling higher IV near-term
    has_edge = iv_edge > 0.02 and dte_ratio < 0.5

    return {
        "has_edge": has_edge,
        "edge_pct": iv_edge * 100,
        "iv_differential": near_iv - far_iv,
        "dte_ratio": dte_ratio,
        "recommendation": "Consider calendar spread" if has_edge else "No calendar edge",
    }


def get_iv_surface_metrics(
    options_data: List[Dict],
    spot: float,
) -> Dict:
    """Compute IV surface metrics across strikes and expiries.

    Returns metrics useful for understanding the full IV landscape.
    """
    if not options_data or spot <= 0:
        return {}

    # Group by expiry
    by_expiry = {}
    for opt in options_data:
        exp = opt.get("expiry")
        if exp:
            if exp not in by_expiry:
                by_expiry[exp] = []
            by_expiry[exp].append(opt)

    metrics = {
        "expiry_count": len(by_expiry),
        "expiries": {},
    }

    for expiry, opts in by_expiry.items():
        ivs = [o.get("iv", 0) for o in opts if o.get("iv", 0) > 0]
        if ivs:
            metrics["expiries"][expiry] = {
                "mean_iv": sum(ivs) / len(ivs),
                "min_iv": min(ivs),
                "max_iv": max(ivs),
                "iv_range": max(ivs) - min(ivs),
                "option_count": len(opts),
            }

    return metrics


if __name__ == "__main__":
    # Demo with synthetic data
    print("IV Term Structure Analyzer")
    print("=" * 50)

    # Synthetic options data
    spot = 20000
    sample_data = [
        # Near-term (7 DTE) - lower IV (contango example)
        {"strike": 19900, "expiry": "2024-01-11", "dte": 7, "iv": 0.14, "right": "PE"},
        {"strike": 20000, "expiry": "2024-01-11", "dte": 7, "iv": 0.13, "right": "CE"},
        {"strike": 20000, "expiry": "2024-01-11", "dte": 7, "iv": 0.13, "right": "PE"},
        {"strike": 20100, "expiry": "2024-01-11", "dte": 7, "iv": 0.14, "right": "CE"},
        # Far-term (30 DTE) - higher IV
        {"strike": 19900, "expiry": "2024-02-01", "dte": 28, "iv": 0.18, "right": "PE"},
        {"strike": 20000, "expiry": "2024-02-01", "dte": 28, "iv": 0.17, "right": "CE"},
        {"strike": 20000, "expiry": "2024-02-01", "dte": 28, "iv": 0.17, "right": "PE"},
        {"strike": 20100, "expiry": "2024-02-01", "dte": 28, "iv": 0.18, "right": "CE"},
    ]

    result = analyze_term_structure("NIFTY", sample_data, spot)

    if result:
        print(f"\nUnderlying: {result.underlying}")
        print(f"Structure: {result.structure.value}")
        print(f"Near Expiry: {result.near_expiry} (IV: {result.near_atm_iv:.2%})")
        print(f"Far Expiry: {result.far_expiry} (IV: {result.far_atm_iv:.2%})")
        print(f"IV Difference: {result.iv_diff:.4f} ({result.iv_diff_pct:.1f}%)")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Slope: {result.slope:.6f} IV/day")
        print(f"\nRecommendation: {result.recommendation}")

        bias, factor = get_term_structure_bias(result)
        print(f"\nBias: {bias}, Factor: {factor:.2f}")
