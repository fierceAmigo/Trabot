"""probability_analysis.py

P2: Probability of Profit (POP) and Expected Value calculations.

This module computes option trade probabilities using:
- Delta-approximated probability (quick)
- Log-normal distribution (more accurate)
- Monte Carlo simulation (most accurate, slower)

Key Metrics:
- POP: Probability of profit at expiry
- POS: Probability of touching stop loss
- POT: Probability of reaching target
- Expected Value: POP * avg_win - (1-POP) * avg_loss

Usage:
    from probability_analysis import calculate_pop, expected_value, analyze_trade_probability

    pop = calculate_pop(spot, strike, iv, dte, side="LONG_CALL")
    ev = expected_value(pop, max_profit, max_loss)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from enum import Enum

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TradeType(Enum):
    LONG_CALL = "LONG_CALL"
    LONG_PUT = "LONG_PUT"
    SHORT_CALL = "SHORT_CALL"
    SHORT_PUT = "SHORT_PUT"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    BULL_PUT_SPREAD = "BULL_PUT_SPREAD"  # Credit spread
    BEAR_CALL_SPREAD = "BEAR_CALL_SPREAD"  # Credit spread
    IRON_CONDOR = "IRON_CONDOR"
    LONG_STRADDLE = "LONG_STRADDLE"
    LONG_STRANGLE = "LONG_STRANGLE"


@dataclass
class ProbabilityResult:
    """Result of probability analysis."""
    trade_type: str
    pop: float                    # Probability of profit
    pop_method: str               # "delta", "lognormal", or "monte_carlo"
    pos: float                    # Probability of touching stop
    pot: float                    # Probability of reaching target
    expected_value: float         # E[V] in currency
    expected_return_pct: float    # E[V] as % of risk
    risk_reward_ratio: float      # max_profit / max_loss
    breakeven_spot: float         # Spot price at breakeven
    details: Dict


@dataclass
class TradeParameters:
    """Parameters for probability calculation."""
    spot: float
    strike: float                 # Primary strike (for single-leg)
    strike2: Optional[float]      # Second strike (for spreads)
    iv: float                     # Implied volatility (annual)
    dte: int                      # Days to expiry
    premium: float                # Premium paid (positive) or received (negative)
    trade_type: TradeType
    stop_loss: Optional[float] = None    # Stop loss price
    target: Optional[float] = None       # Target price
    risk_free_rate: float = 0.06


# Environment configuration
PROB_METHOD = os.getenv("TRABOT_PROB_METHOD", "lognormal")  # delta, lognormal, monte_carlo
MIN_POP_THRESHOLD = float(os.getenv("TRABOT_MIN_POP_THRESHOLD", "0.40"))
MIN_EXPECTED_VALUE = float(os.getenv("TRABOT_MIN_EXPECTED_VALUE", "0"))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (fallback if scipy not available)."""
    if HAS_SCIPY:
        return stats.norm.cdf(x)
    # Approximation using error function
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    if HAS_SCIPY:
        return stats.norm.pdf(x)
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def calculate_d1_d2(
    spot: float,
    strike: float,
    iv: float,
    dte: int,
    r: float = 0.06,
) -> Tuple[float, float]:
    """Calculate Black-Scholes d1 and d2."""
    if dte <= 0:
        dte = 1
    T = dte / 365.0
    if T <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0, 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T

    return d1, d2


def prob_below_strike(
    spot: float,
    strike: float,
    iv: float,
    dte: int,
    r: float = 0.06,
) -> float:
    """Probability that spot ends below strike at expiry (log-normal model)."""
    _, d2 = calculate_d1_d2(spot, strike, iv, dte, r)
    return _norm_cdf(-d2)


def prob_above_strike(
    spot: float,
    strike: float,
    iv: float,
    dte: int,
    r: float = 0.06,
) -> float:
    """Probability that spot ends above strike at expiry."""
    return 1.0 - prob_below_strike(spot, strike, iv, dte, r)


def delta_to_prob(delta: float, trade_type: TradeType) -> float:
    """Quick POP approximation from delta.

    For calls: POP ≈ delta (for long), 1-delta (for short)
    For puts: POP ≈ |delta| (for long), 1-|delta| (for short)
    """
    abs_delta = abs(delta)

    if trade_type in (TradeType.LONG_CALL, TradeType.SHORT_PUT):
        # Need spot above strike
        return abs_delta
    elif trade_type in (TradeType.LONG_PUT, TradeType.SHORT_CALL):
        # Need spot below strike
        return abs_delta

    return 0.5  # Default


def calculate_pop_single_leg(
    params: TradeParameters,
    method: str = "lognormal",
) -> float:
    """Calculate POP for single-leg options.

    For long options: Need to recoup premium
    For short options: Need spot to stay away from strike
    """
    spot = params.spot
    strike = params.strike
    iv = params.iv
    dte = params.dte
    premium = params.premium
    r = params.risk_free_rate

    if params.trade_type == TradeType.LONG_CALL:
        # Breakeven = strike + premium paid
        breakeven = strike + abs(premium)
        return prob_above_strike(spot, breakeven, iv, dte, r)

    elif params.trade_type == TradeType.LONG_PUT:
        # Breakeven = strike - premium paid
        breakeven = strike - abs(premium)
        return prob_below_strike(spot, breakeven, iv, dte, r)

    elif params.trade_type == TradeType.SHORT_CALL:
        # Breakeven = strike + premium received
        breakeven = strike + abs(premium)
        return prob_below_strike(spot, breakeven, iv, dte, r)

    elif params.trade_type == TradeType.SHORT_PUT:
        # Breakeven = strike - premium received
        breakeven = strike - abs(premium)
        return prob_above_strike(spot, breakeven, iv, dte, r)

    return 0.5


def calculate_pop_spread(
    params: TradeParameters,
    method: str = "lognormal",
) -> float:
    """Calculate POP for vertical spreads."""
    spot = params.spot
    iv = params.iv
    dte = params.dte
    r = params.risk_free_rate
    strike1 = params.strike
    strike2 = params.strike2 or params.strike

    if params.trade_type == TradeType.BULL_CALL_SPREAD:
        # Debit spread: Buy lower call, sell higher call
        # Max profit when spot >= higher strike
        # Breakeven = lower strike + net debit
        lower_strike = min(strike1, strike2)
        breakeven = lower_strike + abs(params.premium)
        return prob_above_strike(spot, breakeven, iv, dte, r)

    elif params.trade_type == TradeType.BEAR_PUT_SPREAD:
        # Debit spread: Buy higher put, sell lower put
        # Max profit when spot <= lower strike
        # Breakeven = higher strike - net debit
        higher_strike = max(strike1, strike2)
        breakeven = higher_strike - abs(params.premium)
        return prob_below_strike(spot, breakeven, iv, dte, r)

    elif params.trade_type == TradeType.BULL_PUT_SPREAD:
        # Credit spread: Sell higher put, buy lower put
        # Max profit when spot >= higher strike (both expire worthless)
        # Breakeven = higher strike - net credit
        higher_strike = max(strike1, strike2)
        breakeven = higher_strike - abs(params.premium)
        return prob_above_strike(spot, breakeven, iv, dte, r)

    elif params.trade_type == TradeType.BEAR_CALL_SPREAD:
        # Credit spread: Sell lower call, buy higher call
        # Max profit when spot <= lower strike
        # Breakeven = lower strike + net credit
        lower_strike = min(strike1, strike2)
        breakeven = lower_strike + abs(params.premium)
        return prob_below_strike(spot, breakeven, iv, dte, r)

    return 0.5


def calculate_pop_iron_condor(
    params: TradeParameters,
    put_strikes: Tuple[float, float],  # (short_put, long_put)
    call_strikes: Tuple[float, float],  # (short_call, long_call)
) -> float:
    """Calculate POP for iron condor.

    Iron condor profits if spot stays between short strikes.
    """
    spot = params.spot
    iv = params.iv
    dte = params.dte
    r = params.risk_free_rate
    premium = abs(params.premium)  # Net credit received

    short_put, _ = put_strikes
    short_call, _ = call_strikes

    # Breakevens
    lower_breakeven = short_put - premium / 2  # Approximate split
    upper_breakeven = short_call + premium / 2

    # POP = P(lower_be < spot < upper_be)
    prob_above_lower = prob_above_strike(spot, lower_breakeven, iv, dte, r)
    prob_below_upper = prob_below_strike(spot, upper_breakeven, iv, dte, r)

    return max(0, prob_above_lower + prob_below_upper - 1.0)


def calculate_pop_straddle(
    params: TradeParameters,
) -> float:
    """Calculate POP for long straddle.

    Straddle profits if spot moves beyond breakevens in either direction.
    """
    spot = params.spot
    strike = params.strike
    iv = params.iv
    dte = params.dte
    r = params.risk_free_rate
    premium = abs(params.premium)

    # Breakevens on both sides
    upper_breakeven = strike + premium
    lower_breakeven = strike - premium

    # POP = P(spot > upper_be) + P(spot < lower_be)
    prob_above_upper = prob_above_strike(spot, upper_breakeven, iv, dte, r)
    prob_below_lower = prob_below_strike(spot, lower_breakeven, iv, dte, r)

    return prob_above_upper + prob_below_lower


def calculate_pop_strangle(
    params: TradeParameters,
) -> float:
    """Calculate POP for long strangle.

    Similar to straddle but with OTM strikes.
    """
    spot = params.spot
    iv = params.iv
    dte = params.dte
    r = params.risk_free_rate
    premium = abs(params.premium)
    strike1 = params.strike
    strike2 = params.strike2 or params.strike

    call_strike = max(strike1, strike2)
    put_strike = min(strike1, strike2)

    # Breakevens
    upper_breakeven = call_strike + premium
    lower_breakeven = put_strike - premium

    prob_above_upper = prob_above_strike(spot, upper_breakeven, iv, dte, r)
    prob_below_lower = prob_below_strike(spot, lower_breakeven, iv, dte, r)

    return prob_above_upper + prob_below_lower


def calculate_pop(params: TradeParameters, method: str = None) -> float:
    """Main POP calculation function.

    Routes to appropriate calculator based on trade type.
    """
    if method is None:
        method = PROB_METHOD

    trade_type = params.trade_type

    # Single-leg options
    if trade_type in (TradeType.LONG_CALL, TradeType.LONG_PUT,
                      TradeType.SHORT_CALL, TradeType.SHORT_PUT):
        return calculate_pop_single_leg(params, method)

    # Vertical spreads
    elif trade_type in (TradeType.BULL_CALL_SPREAD, TradeType.BEAR_PUT_SPREAD,
                        TradeType.BULL_PUT_SPREAD, TradeType.BEAR_CALL_SPREAD):
        return calculate_pop_spread(params, method)

    # Straddles/Strangles
    elif trade_type == TradeType.LONG_STRADDLE:
        return calculate_pop_straddle(params)

    elif trade_type == TradeType.LONG_STRANGLE:
        return calculate_pop_strangle(params)

    return 0.5


def prob_of_touching(
    spot: float,
    target: float,
    iv: float,
    dte: int,
) -> float:
    """Probability of spot touching a given level during the trade.

    Uses reflection principle for barrier crossing probability.
    POT ≈ 2 * P(S_T > target) for target > spot
    """
    if dte <= 0:
        return 0.0

    T = dte / 365.0
    if iv <= 0 or spot <= 0:
        return 0.0

    # Standard approach: POT ≈ 2 * terminal probability
    # (reflection principle approximation)
    term_prob = prob_above_strike(spot, target, iv, dte) if target > spot else \
                prob_below_strike(spot, target, iv, dte)

    # Probability of touching is approximately 2x terminal probability
    # Capped at 1.0
    return min(1.0, 2.0 * term_prob)


def expected_value(
    pop: float,
    max_profit: float,
    max_loss: float,
) -> float:
    """Calculate expected value of a trade.

    E[V] = POP * max_profit - (1 - POP) * max_loss
    """
    return pop * max_profit - (1 - pop) * abs(max_loss)


def expected_return_pct(
    pop: float,
    max_profit: float,
    max_loss: float,
) -> float:
    """Calculate expected return as percentage of risk."""
    if max_loss == 0:
        return 0.0
    ev = expected_value(pop, max_profit, max_loss)
    return ev / abs(max_loss) * 100


def analyze_trade_probability(
    params: TradeParameters,
    max_profit: float,
    max_loss: float,
) -> ProbabilityResult:
    """Comprehensive probability analysis for a trade.

    Args:
        params: Trade parameters
        max_profit: Maximum possible profit
        max_loss: Maximum possible loss (positive number)

    Returns:
        ProbabilityResult with all metrics
    """
    # Calculate POP
    pop = calculate_pop(params)

    # Probability of touching stop/target
    pos = 0.0
    pot = 0.0

    if params.stop_loss and params.stop_loss > 0:
        pos = prob_of_touching(params.spot, params.stop_loss, params.iv, params.dte)

    if params.target and params.target > 0:
        pot = prob_of_touching(params.spot, params.target, params.iv, params.dte)

    # Expected value
    ev = expected_value(pop, max_profit, max_loss)
    ev_pct = expected_return_pct(pop, max_profit, max_loss)

    # Risk-reward ratio
    rr = max_profit / abs(max_loss) if max_loss != 0 else 0.0

    # Calculate breakeven
    breakeven = calculate_breakeven(params)

    details = {
        "spot": params.spot,
        "strike": params.strike,
        "iv": params.iv,
        "dte": params.dte,
        "premium": params.premium,
    }

    return ProbabilityResult(
        trade_type=params.trade_type.value,
        pop=pop,
        pop_method=PROB_METHOD,
        pos=pos,
        pot=pot,
        expected_value=ev,
        expected_return_pct=ev_pct,
        risk_reward_ratio=rr,
        breakeven_spot=breakeven,
        details=details,
    )


def calculate_breakeven(params: TradeParameters) -> float:
    """Calculate breakeven spot price for the trade."""
    if params.trade_type == TradeType.LONG_CALL:
        return params.strike + abs(params.premium)
    elif params.trade_type == TradeType.LONG_PUT:
        return params.strike - abs(params.premium)
    elif params.trade_type == TradeType.SHORT_CALL:
        return params.strike + abs(params.premium)
    elif params.trade_type == TradeType.SHORT_PUT:
        return params.strike - abs(params.premium)
    elif params.trade_type == TradeType.BULL_CALL_SPREAD:
        return min(params.strike, params.strike2 or params.strike) + abs(params.premium)
    elif params.trade_type == TradeType.BEAR_PUT_SPREAD:
        return max(params.strike, params.strike2 or params.strike) - abs(params.premium)
    elif params.trade_type == TradeType.BULL_PUT_SPREAD:
        return max(params.strike, params.strike2 or params.strike) - abs(params.premium)
    elif params.trade_type == TradeType.BEAR_CALL_SPREAD:
        return min(params.strike, params.strike2 or params.strike) + abs(params.premium)

    return params.spot


def meets_probability_filters(
    result: ProbabilityResult,
    min_pop: float = None,
    min_ev: float = None,
    min_rr: float = 0.5,
) -> Tuple[bool, str]:
    """Check if trade meets probability filters.

    Returns (passes, reason)
    """
    if min_pop is None:
        min_pop = MIN_POP_THRESHOLD
    if min_ev is None:
        min_ev = MIN_EXPECTED_VALUE

    if result.pop < min_pop:
        return False, f"POP {result.pop:.1%} < min {min_pop:.1%}"

    if result.expected_value < min_ev:
        return False, f"Expected value {result.expected_value:.0f} < min {min_ev:.0f}"

    if result.risk_reward_ratio < min_rr:
        return False, f"Risk-reward {result.risk_reward_ratio:.2f} < min {min_rr:.2f}"

    return True, ""


def get_probability_summary(result: ProbabilityResult) -> Dict:
    """Get summary dict for logging/display."""
    return {
        "trade_type": result.trade_type,
        "pop": f"{result.pop:.1%}",
        "expected_value": f"Rs {result.expected_value:,.0f}",
        "expected_return": f"{result.expected_return_pct:.1f}%",
        "risk_reward": f"{result.risk_reward_ratio:.2f}",
        "breakeven": f"Rs {result.breakeven_spot:,.0f}",
        "pos": f"{result.pos:.1%}" if result.pos else "N/A",
        "pot": f"{result.pot:.1%}" if result.pot else "N/A",
    }


if __name__ == "__main__":
    # Demo
    print("Probability Analysis Demo")
    print("=" * 50)

    # Example: Long Call
    params = TradeParameters(
        spot=20000,
        strike=20100,
        strike2=None,
        iv=0.15,
        dte=7,
        premium=150,
        trade_type=TradeType.LONG_CALL,
        stop_loss=19800,
        target=20500,
    )

    result = analyze_trade_probability(
        params,
        max_profit=800,  # Theoretical
        max_loss=150,    # Premium paid
    )

    print(f"\nTrade: {result.trade_type}")
    print(f"Spot: Rs {params.spot:,}")
    print(f"Strike: Rs {params.strike:,}")
    print(f"IV: {params.iv:.1%}")
    print(f"DTE: {params.dte}")
    print(f"Premium: Rs {params.premium}")

    print(f"\n{'='*50}")
    print(f"Probability of Profit: {result.pop:.1%}")
    print(f"Prob of Touching Stop: {result.pos:.1%}")
    print(f"Prob of Reaching Target: {result.pot:.1%}")
    print(f"Expected Value: Rs {result.expected_value:,.0f}")
    print(f"Expected Return: {result.expected_return_pct:.1f}%")
    print(f"Risk-Reward Ratio: {result.risk_reward_ratio:.2f}")
    print(f"Breakeven: Rs {result.breakeven_spot:,}")

    # Check filters
    passes, reason = meets_probability_filters(result)
    print(f"\nPasses Filters: {passes}")
    if reason:
        print(f"Reason: {reason}")
