"""liquidity_model.py

P3: Advanced liquidity scoring for options selection.

Poor liquidity leads to:
- Wide bid-ask spreads (execution slippage)
- Difficulty exiting positions
- Price manipulation risk
- Inaccurate Greeks calculations

This module provides:
- Multi-factor liquidity scoring
- Strike-level liquidity assessment
- Expiry-level liquidity ranking
- Slippage estimation

Usage:
    from liquidity_model import calculate_liquidity_score, estimate_slippage

    score = calculate_liquidity_score(option_data)
    slippage = estimate_slippage(contracts, option_data)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class LiquidityGrade(Enum):
    EXCELLENT = "A"    # Highly liquid, minimal slippage
    GOOD = "B"         # Good liquidity, low slippage
    FAIR = "C"         # Moderate liquidity, some slippage
    POOR = "D"         # Low liquidity, significant slippage
    AVOID = "F"        # Very illiquid, avoid trading


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for an option."""
    bid_ask_spread: float         # Spread in rupees
    spread_pct: float             # Spread as % of mid price
    bid_size: int                 # Bid quantity
    ask_size: int                 # Ask quantity
    volume: int                   # Day volume
    oi: int                       # Open interest
    oi_change: int                # OI change from yesterday
    last_trade_time: str          # Time of last trade


@dataclass
class LiquidityScore:
    """Comprehensive liquidity score."""
    score: float                  # 0-100 score
    grade: LiquidityGrade
    factors: Dict[str, float]     # Individual factor scores
    estimated_slippage_pct: float
    recommendation: str


# Environment configuration
LIQUIDITY_CHECK_ENABLE = os.getenv("TRABOT_LIQUIDITY_CHECK_ENABLE", "1").strip() == "1"
MIN_LIQUIDITY_SCORE = float(os.getenv("TRABOT_MIN_LIQUIDITY_SCORE", "40"))
MAX_SPREAD_PCT = float(os.getenv("TRABOT_MAX_SPREAD_PCT", "0.05"))  # 5%
MIN_OI = int(os.getenv("TRABOT_MIN_OI", "500"))
MIN_VOLUME = int(os.getenv("TRABOT_MIN_VOLUME", "100"))

# Weights for liquidity factors
FACTOR_WEIGHTS = {
    "spread": 0.30,        # Bid-ask spread weight
    "depth": 0.20,         # Order book depth weight
    "volume": 0.20,        # Trading volume weight
    "oi": 0.20,            # Open interest weight
    "oi_trend": 0.10,      # OI change trend weight
}


def _normalize_score(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to 0-100 scale."""
    if max_val <= min_val:
        return 50.0
    normalized = (value - min_val) / (max_val - min_val) * 100
    return max(0.0, min(100.0, normalized))


def score_spread(spread_pct: float) -> float:
    """Score based on bid-ask spread percentage.

    Lower spread = higher score.
    """
    if spread_pct <= 0:
        return 100.0  # Zero spread is best (unlikely)
    if spread_pct >= 0.10:  # 10%+ spread
        return 0.0

    # Inverse scoring: 0% -> 100, 5% -> 50, 10% -> 0
    return max(0, 100 - spread_pct * 1000)


def score_depth(bid_size: int, ask_size: int) -> float:
    """Score based on order book depth.

    Higher depth = higher score.
    """
    total_depth = bid_size + ask_size

    if total_depth >= 10000:
        return 100.0
    elif total_depth >= 5000:
        return 85.0
    elif total_depth >= 1000:
        return 70.0
    elif total_depth >= 500:
        return 55.0
    elif total_depth >= 100:
        return 40.0
    elif total_depth >= 50:
        return 25.0
    else:
        return 10.0


def score_volume(volume: int) -> float:
    """Score based on daily trading volume.

    Higher volume = higher score.
    """
    if volume >= 50000:
        return 100.0
    elif volume >= 20000:
        return 90.0
    elif volume >= 10000:
        return 80.0
    elif volume >= 5000:
        return 70.0
    elif volume >= 1000:
        return 55.0
    elif volume >= 500:
        return 40.0
    elif volume >= 100:
        return 25.0
    else:
        return 10.0


def score_oi(oi: int) -> float:
    """Score based on open interest.

    Higher OI = higher score.
    """
    if oi >= 100000:
        return 100.0
    elif oi >= 50000:
        return 90.0
    elif oi >= 20000:
        return 80.0
    elif oi >= 10000:
        return 70.0
    elif oi >= 5000:
        return 60.0
    elif oi >= 1000:
        return 45.0
    elif oi >= 500:
        return 30.0
    else:
        return 15.0


def score_oi_trend(oi: int, oi_change: int) -> float:
    """Score based on OI trend.

    Rising OI = higher score (indicates interest).
    """
    if oi <= 0:
        return 50.0  # Neutral

    change_pct = oi_change / oi if oi > 0 else 0

    if change_pct >= 0.20:  # 20%+ increase
        return 100.0
    elif change_pct >= 0.10:
        return 85.0
    elif change_pct >= 0.05:
        return 70.0
    elif change_pct >= 0:
        return 55.0  # Flat to slightly positive
    elif change_pct >= -0.10:
        return 40.0  # Slight decrease
    else:
        return 20.0  # Significant decrease


def calculate_liquidity_score(
    metrics: LiquidityMetrics,
    weights: Dict[str, float] = None,
) -> LiquidityScore:
    """Calculate comprehensive liquidity score.

    Args:
        metrics: Raw liquidity metrics
        weights: Optional custom weights

    Returns:
        LiquidityScore with score, grade, and details
    """
    if weights is None:
        weights = FACTOR_WEIGHTS

    # Calculate individual factor scores
    factors = {
        "spread": score_spread(metrics.spread_pct),
        "depth": score_depth(metrics.bid_size, metrics.ask_size),
        "volume": score_volume(metrics.volume),
        "oi": score_oi(metrics.oi),
        "oi_trend": score_oi_trend(metrics.oi, metrics.oi_change),
    }

    # Calculate weighted score
    total_weight = sum(weights.get(k, 0) for k in factors.keys())
    if total_weight <= 0:
        total_weight = 1

    weighted_score = sum(
        factors[k] * weights.get(k, 0)
        for k in factors.keys()
    ) / total_weight

    # Determine grade
    if weighted_score >= 80:
        grade = LiquidityGrade.EXCELLENT
    elif weighted_score >= 65:
        grade = LiquidityGrade.GOOD
    elif weighted_score >= 50:
        grade = LiquidityGrade.FAIR
    elif weighted_score >= 35:
        grade = LiquidityGrade.POOR
    else:
        grade = LiquidityGrade.AVOID

    # Estimate slippage
    estimated_slippage = estimate_slippage_pct(metrics)

    # Generate recommendation
    if grade == LiquidityGrade.EXCELLENT:
        recommendation = "Highly liquid - trade with confidence"
    elif grade == LiquidityGrade.GOOD:
        recommendation = "Good liquidity - use limit orders"
    elif grade == LiquidityGrade.FAIR:
        recommendation = "Moderate liquidity - expect some slippage"
    elif grade == LiquidityGrade.POOR:
        recommendation = "Poor liquidity - reduce size or avoid"
    else:
        recommendation = "Very illiquid - AVOID trading"

    return LiquidityScore(
        score=weighted_score,
        grade=grade,
        factors=factors,
        estimated_slippage_pct=estimated_slippage,
        recommendation=recommendation,
    )


def estimate_slippage_pct(metrics: LiquidityMetrics) -> float:
    """Estimate expected slippage as percentage.

    Based on spread and depth analysis.
    """
    # Base slippage is half the spread (market order)
    base_slippage = metrics.spread_pct / 2

    # Adjust for depth
    total_depth = metrics.bid_size + metrics.ask_size
    if total_depth < 100:
        depth_multiplier = 2.0
    elif total_depth < 500:
        depth_multiplier = 1.5
    elif total_depth < 1000:
        depth_multiplier = 1.2
    else:
        depth_multiplier = 1.0

    # Adjust for OI (low OI = harder to exit)
    if metrics.oi < 500:
        oi_multiplier = 1.5
    elif metrics.oi < 1000:
        oi_multiplier = 1.2
    else:
        oi_multiplier = 1.0

    estimated = base_slippage * depth_multiplier * oi_multiplier
    return min(estimated, 0.10)  # Cap at 10%


def estimate_slippage(
    contracts: int,
    metrics: LiquidityMetrics,
    is_entry: bool = True,
) -> float:
    """Estimate slippage for a specific trade size.

    Args:
        contracts: Number of contracts to trade
        metrics: Liquidity metrics
        is_entry: True for entry, False for exit

    Returns:
        Estimated slippage in rupees per contract
    """
    base_slippage_pct = estimate_slippage_pct(metrics)

    # Impact multiplier based on size vs available depth
    relevant_depth = metrics.bid_size if not is_entry else metrics.ask_size

    if relevant_depth <= 0:
        size_impact = 2.0
    elif contracts > relevant_depth:
        size_impact = 1.5 + (contracts / relevant_depth - 1) * 0.5
    else:
        size_impact = 1.0

    # Calculate mid price
    mid_price = metrics.bid_ask_spread / 2 if metrics.spread_pct > 0 else 100

    slippage_per_contract = mid_price * base_slippage_pct * size_impact
    return slippage_per_contract


def rank_strikes_by_liquidity(
    options: List[Dict],
) -> List[Tuple[int, LiquidityScore]]:
    """Rank strikes by liquidity score.

    Args:
        options: List of option dicts with liquidity data

    Returns:
        List of (strike, score) tuples sorted by score descending
    """
    results = []

    for opt in options:
        metrics = LiquidityMetrics(
            bid_ask_spread=opt.get("ask", 0) - opt.get("bid", 0),
            spread_pct=opt.get("spread_pct", 0),
            bid_size=opt.get("bid_qty", 0),
            ask_size=opt.get("ask_qty", 0),
            volume=opt.get("volume", 0),
            oi=opt.get("oi", 0),
            oi_change=opt.get("oi_change", 0),
            last_trade_time=opt.get("last_trade_time", ""),
        )

        score = calculate_liquidity_score(metrics)
        results.append((opt.get("strike", 0), score))

    # Sort by score descending
    results.sort(key=lambda x: x[1].score, reverse=True)
    return results


def filter_by_liquidity(
    options: List[Dict],
    min_score: float = None,
    min_oi: int = None,
    max_spread_pct: float = None,
) -> List[Dict]:
    """Filter options by liquidity criteria.

    Args:
        options: List of option dicts
        min_score: Minimum liquidity score
        min_oi: Minimum open interest
        max_spread_pct: Maximum bid-ask spread %

    Returns:
        Filtered list of options
    """
    if min_score is None:
        min_score = MIN_LIQUIDITY_SCORE
    if min_oi is None:
        min_oi = MIN_OI
    if max_spread_pct is None:
        max_spread_pct = MAX_SPREAD_PCT

    filtered = []

    for opt in options:
        # Basic filters
        if opt.get("oi", 0) < min_oi:
            continue
        if opt.get("spread_pct", 1) > max_spread_pct:
            continue

        # Score filter
        metrics = LiquidityMetrics(
            bid_ask_spread=opt.get("ask", 0) - opt.get("bid", 0),
            spread_pct=opt.get("spread_pct", 0),
            bid_size=opt.get("bid_qty", 0),
            ask_size=opt.get("ask_qty", 0),
            volume=opt.get("volume", 0),
            oi=opt.get("oi", 0),
            oi_change=opt.get("oi_change", 0),
            last_trade_time=opt.get("last_trade_time", ""),
        )
        score = calculate_liquidity_score(metrics)

        if score.score < min_score:
            continue

        # Add score to option dict
        opt_with_score = opt.copy()
        opt_with_score["liquidity_score"] = score.score
        opt_with_score["liquidity_grade"] = score.grade.value

        filtered.append(opt_with_score)

    return filtered


def get_best_liquid_strike(
    options: List[Dict],
    near_strike: float,
    max_distance: int = 3,
) -> Optional[Dict]:
    """Find most liquid strike near a target.

    Args:
        options: List of option dicts
        near_strike: Target strike
        max_distance: Max strikes away from target

    Returns:
        Best liquid option near target or None
    """
    # Filter to strikes within distance
    step = 50  # Assume 50-point steps
    min_strike = near_strike - (max_distance * step)
    max_strike = near_strike + (max_distance * step)

    nearby = [
        o for o in options
        if min_strike <= o.get("strike", 0) <= max_strike
    ]

    if not nearby:
        return None

    # Rank by liquidity
    ranked = rank_strikes_by_liquidity(nearby)

    if ranked:
        best_strike, best_score = ranked[0]
        # Find the original option dict
        for o in nearby:
            if o.get("strike") == best_strike:
                o["liquidity_score"] = best_score.score
                o["liquidity_grade"] = best_score.grade.value
                return o

    return None


def calculate_effective_spread(
    options: List[Dict],
    strategy: str,
) -> float:
    """Calculate effective spread for a multi-leg strategy.

    Args:
        options: List of leg options
        strategy: Strategy type

    Returns:
        Total effective spread in rupees
    """
    total_spread = 0

    for opt in options:
        spread = opt.get("ask", 0) - opt.get("bid", 0)
        total_spread += spread

    # Multi-leg strategies have compounded slippage
    if len(options) > 1:
        total_spread *= 1.1  # 10% penalty for multi-leg

    return total_spread


def liquidity_adjusted_price(
    option: Dict,
    side: str,  # "BUY" or "SELL"
    contracts: int = 1,
) -> float:
    """Get liquidity-adjusted price (expected fill).

    Args:
        option: Option dict with bid/ask
        side: BUY or SELL
        contracts: Number of contracts

    Returns:
        Expected fill price including slippage
    """
    bid = option.get("bid", 0)
    ask = option.get("ask", 0)
    mid = (bid + ask) / 2 if bid and ask else option.get("ltp", 0)

    # Calculate slippage
    metrics = LiquidityMetrics(
        bid_ask_spread=ask - bid,
        spread_pct=(ask - bid) / mid if mid > 0 else 0,
        bid_size=option.get("bid_qty", 100),
        ask_size=option.get("ask_qty", 100),
        volume=option.get("volume", 500),
        oi=option.get("oi", 1000),
        oi_change=0,
        last_trade_time="",
    )

    slippage = estimate_slippage(contracts, metrics, is_entry=(side == "BUY"))

    if side == "BUY":
        return ask + slippage  # Pay more than ask
    else:
        return bid - slippage  # Receive less than bid


def get_liquidity_summary(score: LiquidityScore) -> Dict:
    """Get summary dict for logging."""
    return {
        "score": f"{score.score:.0f}/100",
        "grade": score.grade.value,
        "estimated_slippage": f"{score.estimated_slippage_pct:.2%}",
        "recommendation": score.recommendation,
        "factors": {k: f"{v:.0f}" for k, v in score.factors.items()},
    }


if __name__ == "__main__":
    # Demo
    print("Liquidity Model Demo")
    print("=" * 50)

    # Sample option data
    sample_metrics = LiquidityMetrics(
        bid_ask_spread=3.0,
        spread_pct=0.015,  # 1.5%
        bid_size=500,
        ask_size=600,
        volume=5000,
        oi=15000,
        oi_change=1200,
        last_trade_time="14:30:00",
    )

    print(f"\nSample Option Metrics:")
    print(f"  Spread: Rs {sample_metrics.bid_ask_spread} ({sample_metrics.spread_pct:.1%})")
    print(f"  Depth: {sample_metrics.bid_size} / {sample_metrics.ask_size}")
    print(f"  Volume: {sample_metrics.volume}")
    print(f"  OI: {sample_metrics.oi} (+{sample_metrics.oi_change})")

    # Calculate score
    score = calculate_liquidity_score(sample_metrics)

    print(f"\n{'='*50}")
    print(f"Liquidity Score: {score.score:.0f}/100")
    print(f"Grade: {score.grade.value} ({score.grade.name})")
    print(f"Estimated Slippage: {score.estimated_slippage_pct:.2%}")
    print(f"Recommendation: {score.recommendation}")

    print(f"\nFactor Breakdown:")
    for factor, value in score.factors.items():
        print(f"  {factor}: {value:.0f}")

    # Slippage estimate for 5 contracts
    slippage = estimate_slippage(5, sample_metrics)
    print(f"\nEstimated slippage for 5 contracts: Rs {slippage:.2f}/contract")
