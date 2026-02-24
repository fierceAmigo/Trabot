"""greeks_monitor.py

P2: Greeks-based position monitoring and stop loss management.

This module provides Greeks-driven exit conditions beyond simple premium stops.
For options positions, Greek thresholds can be more meaningful than price stops.

Key Concepts:
- Delta Stop: Exit when delta exceeds threshold (position getting too directional)
- Theta Decay: Exit when theta decay exceeds max daily loss tolerance
- Vega Exposure: Exit if IV moves against position beyond tolerance
- Gamma Risk: Exit near expiry if gamma creates excessive overnight risk

Usage:
    from greeks_monitor import check_greeks_stops, GreeksThresholds

    thresholds = GreeksThresholds(max_delta=0.70, max_theta_daily_loss=500)
    should_exit, reason = check_greeks_stops(position_greeks, thresholds)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ExitReason(Enum):
    DELTA_BREACH = "DELTA_BREACH"
    THETA_BREACH = "THETA_BREACH"
    VEGA_BREACH = "VEGA_BREACH"
    GAMMA_BREACH = "GAMMA_BREACH"
    DTE_BREACH = "DTE_BREACH"
    COMPOSITE = "COMPOSITE"
    NONE = "NONE"


@dataclass
class GreeksThresholds:
    """Thresholds for Greeks-based stops."""
    # Delta thresholds (absolute value)
    max_delta_long: float = 0.75      # Exit long option if delta > this
    min_delta_long: float = 0.15      # Exit long option if delta < this (worthless)
    max_delta_short: float = 0.70     # Exit short option if |delta| > this

    # Theta thresholds
    max_theta_daily_pct: float = 0.02  # Max daily theta as % of position value
    min_theta_remaining: float = 0.05  # Exit if theta < this (no decay left)

    # Vega thresholds
    max_vega_loss_pct: float = 0.30   # Max loss from IV move as % of position

    # Gamma thresholds (near expiry)
    max_gamma_dollar: float = 500     # Max $ gamma exposure per point move
    gamma_warning_dte: int = 2        # Warn when DTE <= this

    # DTE thresholds
    min_dte_long: int = 1             # Exit long options at this DTE
    min_dte_short: int = 0            # Let short options expire


@dataclass
class PositionGreeks:
    """Greeks for a position (may be multi-leg)."""
    underlying: str
    net_delta: float              # Sum of deltas * contracts
    net_gamma: float              # Sum of gammas * contracts
    net_vega: float               # Sum of vegas * contracts (per 1% IV)
    net_theta: float              # Sum of thetas * contracts (per day)
    position_value: float         # Current mark-to-market value
    contracts: int                # Total contracts
    dte: int                      # Days to expiry (nearest leg)
    is_long_premium: bool         # True if net long premium


@dataclass
class GreeksStopResult:
    """Result of Greeks stop check."""
    should_exit: bool
    exit_reason: ExitReason
    details: str
    urgency: str                  # "IMMEDIATE", "SOON", "MONITOR"
    greeks_snapshot: Dict


# Environment configuration
GREEKS_STOPS_ENABLE = os.getenv("TRABOT_GREEKS_STOPS_ENABLE", "1").strip() == "1"
MAX_DELTA_LONG = float(os.getenv("TRABOT_MAX_DELTA_LONG", "0.75"))
MIN_DELTA_LONG = float(os.getenv("TRABOT_MIN_DELTA_LONG", "0.15"))
MAX_DELTA_SHORT = float(os.getenv("TRABOT_MAX_DELTA_SHORT", "0.70"))
GAMMA_WARNING_DTE = int(os.getenv("TRABOT_GAMMA_WARNING_DTE", "2"))


def _default_thresholds() -> GreeksThresholds:
    """Get thresholds from environment or defaults."""
    return GreeksThresholds(
        max_delta_long=MAX_DELTA_LONG,
        min_delta_long=MIN_DELTA_LONG,
        max_delta_short=MAX_DELTA_SHORT,
        gamma_warning_dte=GAMMA_WARNING_DTE,
    )


def check_delta_stop(
    position: PositionGreeks,
    thresholds: GreeksThresholds,
) -> Tuple[bool, str]:
    """Check if delta exceeds thresholds.

    Returns (should_exit, reason)
    """
    abs_delta = abs(position.net_delta)

    if position.is_long_premium:
        # Long premium positions
        if abs_delta > thresholds.max_delta_long:
            return True, f"Delta {abs_delta:.2f} > max {thresholds.max_delta_long:.2f} (deep ITM, consider taking profits)"

        if abs_delta < thresholds.min_delta_long and position.dte > 2:
            return True, f"Delta {abs_delta:.2f} < min {thresholds.min_delta_long:.2f} (near worthless)"
    else:
        # Short premium positions
        if abs_delta > thresholds.max_delta_short:
            return True, f"Delta {abs_delta:.2f} > max {thresholds.max_delta_short:.2f} (position getting risky)"

    return False, ""


def check_theta_stop(
    position: PositionGreeks,
    thresholds: GreeksThresholds,
) -> Tuple[bool, str]:
    """Check if theta decay is problematic.

    For long premium: High theta = rapid decay, may want to exit
    For short premium: Low theta = little decay benefit remaining
    """
    if position.position_value <= 0:
        return False, ""

    theta_daily_pct = abs(position.net_theta) / position.position_value

    if position.is_long_premium:
        # Long premium bleeds theta
        if theta_daily_pct > thresholds.max_theta_daily_pct:
            return True, f"Daily theta decay {theta_daily_pct:.1%} > max {thresholds.max_theta_daily_pct:.1%}"
    else:
        # Short premium collects theta
        # But if theta is very low, not much left to collect
        if abs(position.net_theta) < thresholds.min_theta_remaining and position.dte > 3:
            return True, f"Theta {position.net_theta:.2f} too low - decay benefit exhausted"

    return False, ""


def check_gamma_stop(
    position: PositionGreeks,
    thresholds: GreeksThresholds,
    spot: float,
) -> Tuple[bool, str]:
    """Check if gamma risk is excessive.

    Gamma increases near expiry and ATM. High gamma = P&L swings rapidly.
    """
    if position.dte > thresholds.gamma_warning_dte:
        return False, ""

    # Dollar gamma = change in delta-dollars per point move
    dollar_gamma = abs(position.net_gamma) * spot * position.contracts

    if dollar_gamma > thresholds.max_gamma_dollar:
        return True, f"Gamma risk ${dollar_gamma:.0f}/point at {position.dte} DTE (near-expiry risk)"

    return False, ""


def check_vega_stop(
    position: PositionGreeks,
    thresholds: GreeksThresholds,
    iv_change: float = 0.0,
) -> Tuple[bool, str]:
    """Check if vega exposure caused excessive loss.

    Args:
        position: Current position Greeks
        thresholds: Stop thresholds
        iv_change: Change in IV since entry (e.g., 0.05 for 5% IV increase)
    """
    if position.position_value <= 0 or iv_change == 0:
        return False, ""

    # Vega P&L = vega * IV change (vega is per 1% IV)
    vega_pnl = position.net_vega * (iv_change * 100)  # Convert to percentage points

    vega_pnl_pct = vega_pnl / position.position_value

    if position.is_long_premium:
        # Long vega - IV drop hurts
        if iv_change < 0 and abs(vega_pnl_pct) > thresholds.max_vega_loss_pct:
            return True, f"IV crush loss {vega_pnl_pct:.1%} > max {thresholds.max_vega_loss_pct:.1%}"
    else:
        # Short vega - IV spike hurts
        if iv_change > 0 and abs(vega_pnl_pct) > thresholds.max_vega_loss_pct:
            return True, f"IV spike loss {vega_pnl_pct:.1%} > max {thresholds.max_vega_loss_pct:.1%}"

    return False, ""


def check_dte_stop(
    position: PositionGreeks,
    thresholds: GreeksThresholds,
) -> Tuple[bool, str]:
    """Check if position should exit based on DTE."""
    if position.is_long_premium:
        if position.dte <= thresholds.min_dte_long:
            return True, f"DTE {position.dte} <= min {thresholds.min_dte_long} for long premium"
    else:
        if position.dte <= thresholds.min_dte_short:
            return True, f"DTE {position.dte} <= min {thresholds.min_dte_short} for short premium"

    return False, ""


def check_greeks_stops(
    position: PositionGreeks,
    spot: float,
    thresholds: Optional[GreeksThresholds] = None,
    iv_change: float = 0.0,
) -> GreeksStopResult:
    """Main function: Check all Greeks-based stops.

    Args:
        position: Current position Greeks
        spot: Current spot price
        thresholds: Stop thresholds (uses defaults if None)
        iv_change: Change in IV since entry

    Returns:
        GreeksStopResult with exit decision and details
    """
    if not GREEKS_STOPS_ENABLE:
        return GreeksStopResult(
            should_exit=False,
            exit_reason=ExitReason.NONE,
            details="Greeks stops disabled",
            urgency="MONITOR",
            greeks_snapshot={},
        )

    if thresholds is None:
        thresholds = _default_thresholds()

    reasons = []
    urgencies = []

    # Check each Greek stop
    delta_exit, delta_reason = check_delta_stop(position, thresholds)
    if delta_exit:
        reasons.append(("DELTA", delta_reason))
        urgencies.append("SOON" if abs(position.net_delta) < 0.85 else "IMMEDIATE")

    theta_exit, theta_reason = check_theta_stop(position, thresholds)
    if theta_exit:
        reasons.append(("THETA", theta_reason))
        urgencies.append("SOON")

    gamma_exit, gamma_reason = check_gamma_stop(position, thresholds, spot)
    if gamma_exit:
        reasons.append(("GAMMA", gamma_reason))
        urgencies.append("IMMEDIATE")

    vega_exit, vega_reason = check_vega_stop(position, thresholds, iv_change)
    if vega_exit:
        reasons.append(("VEGA", vega_reason))
        urgencies.append("SOON")

    dte_exit, dte_reason = check_dte_stop(position, thresholds)
    if dte_exit:
        reasons.append(("DTE", dte_reason))
        urgencies.append("IMMEDIATE")

    # Determine overall result
    should_exit = len(reasons) > 0

    if len(reasons) == 0:
        exit_reason = ExitReason.NONE
        details = "All Greeks within thresholds"
        urgency = "MONITOR"
    elif len(reasons) == 1:
        exit_reason = ExitReason[reasons[0][0] + "_BREACH"]
        details = reasons[0][1]
        urgency = urgencies[0]
    else:
        exit_reason = ExitReason.COMPOSITE
        details = "; ".join(r[1] for r in reasons)
        urgency = "IMMEDIATE" if "IMMEDIATE" in urgencies else "SOON"

    greeks_snapshot = {
        "delta": position.net_delta,
        "gamma": position.net_gamma,
        "vega": position.net_vega,
        "theta": position.net_theta,
        "dte": position.dte,
        "position_value": position.position_value,
        "is_long_premium": position.is_long_premium,
    }

    return GreeksStopResult(
        should_exit=should_exit,
        exit_reason=exit_reason,
        details=details,
        urgency=urgency,
        greeks_snapshot=greeks_snapshot,
    )


def get_greeks_health_score(
    position: PositionGreeks,
    thresholds: Optional[GreeksThresholds] = None,
) -> float:
    """Calculate a health score for the position based on Greeks.

    Returns 0-1 where:
    - 1.0 = All Greeks well within thresholds
    - 0.5 = Some Greeks approaching thresholds
    - 0.0 = Multiple Greeks breaching thresholds
    """
    if thresholds is None:
        thresholds = _default_thresholds()

    score = 1.0

    # Delta score
    abs_delta = abs(position.net_delta)
    if position.is_long_premium:
        if abs_delta > thresholds.max_delta_long:
            score -= 0.3
        elif abs_delta > thresholds.max_delta_long * 0.8:
            score -= 0.1
        if abs_delta < thresholds.min_delta_long:
            score -= 0.3
        elif abs_delta < thresholds.min_delta_long * 1.5:
            score -= 0.1

    # DTE score
    if position.dte <= 1:
        score -= 0.3
    elif position.dte <= 3:
        score -= 0.1

    # Theta score (for long premium)
    if position.is_long_premium and position.position_value > 0:
        theta_pct = abs(position.net_theta) / position.position_value
        if theta_pct > thresholds.max_theta_daily_pct:
            score -= 0.2
        elif theta_pct > thresholds.max_theta_daily_pct * 0.7:
            score -= 0.1

    return max(0.0, min(1.0, score))


def suggest_adjustment(
    position: PositionGreeks,
    stop_result: GreeksStopResult,
) -> List[str]:
    """Suggest adjustments based on Greeks stop triggers.

    Returns list of possible adjustments.
    """
    suggestions = []

    if not stop_result.should_exit:
        return suggestions

    if stop_result.exit_reason == ExitReason.DELTA_BREACH:
        if abs(position.net_delta) > 0.7:
            suggestions.append("Roll to further OTM strike to reduce delta")
            suggestions.append("Add hedge leg (opposite delta)")
            suggestions.append("Close position and re-enter at better level")

    elif stop_result.exit_reason == ExitReason.THETA_BREACH:
        if position.is_long_premium:
            suggestions.append("Roll out to further expiry to reduce theta decay")
            suggestions.append("Convert to spread to offset theta")
        else:
            suggestions.append("Close position - theta collection complete")

    elif stop_result.exit_reason == ExitReason.GAMMA_BREACH:
        suggestions.append("Close position - gamma risk too high near expiry")
        suggestions.append("Roll to next expiry to reduce gamma")

    elif stop_result.exit_reason == ExitReason.VEGA_BREACH:
        if position.is_long_premium:
            suggestions.append("IV crushed - close or convert to spread")
        else:
            suggestions.append("IV spiked - consider rolling or closing")

    elif stop_result.exit_reason == ExitReason.DTE_BREACH:
        suggestions.append("Close position - approaching expiry")
        suggestions.append("Roll to next expiry if still bullish/bearish")

    return suggestions


def compute_break_even_greeks(
    entry_price: float,
    current_greeks: PositionGreeks,
    target_pnl: float = 0.0,
) -> Dict:
    """Compute spot/IV levels needed to break even.

    Returns break-even levels for each Greek.
    """
    current_pnl = current_greeks.position_value - entry_price

    result = {}

    # Delta break-even (spot move needed)
    if abs(current_greeks.net_delta) > 0.01:
        spot_move_needed = (target_pnl - current_pnl) / current_greeks.net_delta
        result["spot_move_needed"] = spot_move_needed

    # Theta break-even (days needed)
    if abs(current_greeks.net_theta) > 0.01:
        if not current_greeks.is_long_premium:  # Short premium collects theta
            days_to_be = (target_pnl - current_pnl) / abs(current_greeks.net_theta)
            result["days_to_breakeven"] = max(0, days_to_be)

    # Vega break-even (IV change needed)
    if abs(current_greeks.net_vega) > 0.01:
        iv_change_needed = (target_pnl - current_pnl) / current_greeks.net_vega / 100
        result["iv_change_needed"] = iv_change_needed

    return result


if __name__ == "__main__":
    # Demo
    print("Greeks Monitor Demo")
    print("=" * 50)

    # Sample position
    position = PositionGreeks(
        underlying="NIFTY",
        net_delta=0.65,
        net_gamma=0.02,
        net_vega=15.0,
        net_theta=-8.5,
        position_value=5000,
        contracts=1,
        dte=5,
        is_long_premium=True,
    )

    print(f"\nPosition: {position.underlying}")
    print(f"  Delta: {position.net_delta:.2f}")
    print(f"  Gamma: {position.net_gamma:.3f}")
    print(f"  Vega: {position.net_vega:.2f}")
    print(f"  Theta: {position.net_theta:.2f}")
    print(f"  DTE: {position.dte}")
    print(f"  Value: Rs {position.position_value:,.0f}")

    # Check stops
    result = check_greeks_stops(position, spot=20000, iv_change=-0.02)

    print(f"\n{'='*50}")
    print(f"Should Exit: {result.should_exit}")
    print(f"Reason: {result.exit_reason.value}")
    print(f"Details: {result.details}")
    print(f"Urgency: {result.urgency}")

    # Health score
    health = get_greeks_health_score(position)
    print(f"\nHealth Score: {health:.0%}")

    # Suggestions
    if result.should_exit:
        print("\nSuggested Adjustments:")
        for s in suggest_adjustment(position, result):
            print(f"  - {s}")
