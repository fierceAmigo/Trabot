"""adjustment_engine.py

P2: Simplified adjustment framework for options positions.

When a position moves against you, adjustments can:
- Reduce risk
- Lower breakeven
- Convert losing trades to winning ones
- Lock in partial profits

This module provides adjustment suggestions and mechanics for:
- Rolling (same structure, different strike/expiry)
- Converting (change structure type)
- Adding hedges
- Closing partial positions

Usage:
    from adjustment_engine import suggest_adjustments, AdjustmentType

    adjustments = suggest_adjustments(position, current_pnl, spot)
    for adj in adjustments:
        print(f"{adj.type}: {adj.description}")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class AdjustmentType(Enum):
    ROLL_OUT = "ROLL_OUT"           # Same strike, later expiry
    ROLL_UP = "ROLL_UP"             # Higher strike (calls)
    ROLL_DOWN = "ROLL_DOWN"         # Lower strike (puts)
    ROLL_UP_OUT = "ROLL_UP_OUT"     # Higher strike + later expiry
    ROLL_DOWN_OUT = "ROLL_DOWN_OUT" # Lower strike + later expiry
    CONVERT_TO_SPREAD = "CONVERT_TO_SPREAD"
    ADD_HEDGE = "ADD_HEDGE"
    CLOSE_PARTIAL = "CLOSE_PARTIAL"
    CLOSE_FULL = "CLOSE_FULL"
    DOUBLE_DOWN = "DOUBLE_DOWN"     # Add to winning position
    NO_ACTION = "NO_ACTION"


class PositionStatus(Enum):
    WINNING = "WINNING"             # Current profit
    SCRATCH = "SCRATCH"             # Near breakeven
    LOSING_SMALL = "LOSING_SMALL"   # <25% of max loss
    LOSING_MEDIUM = "LOSING_MEDIUM" # 25-50% of max loss
    LOSING_LARGE = "LOSING_LARGE"   # >50% of max loss
    NEAR_STOP = "NEAR_STOP"         # Approaching stop loss
    STOPPED_OUT = "STOPPED_OUT"     # At or past stop


@dataclass
class Position:
    """Simplified position representation."""
    underlying: str
    structure: str              # BUY_CE, BULL_CALL_SPREAD, etc.
    strike: float               # Primary strike
    strike2: Optional[float]    # Secondary strike for spreads
    expiry: str                 # YYYY-MM-DD
    dte: int
    entry_price: float          # Premium paid/received per unit
    current_price: float        # Current premium
    contracts: int
    is_long: bool               # True if net long premium
    entry_spot: float           # Spot at entry
    current_spot: float         # Current spot


@dataclass
class Adjustment:
    """Recommended adjustment."""
    type: AdjustmentType
    description: str
    mechanics: str              # Step-by-step instructions
    expected_outcome: str
    cost_estimate: float        # Net debit/credit
    risk_impact: str            # How it changes risk profile
    urgency: str                # "IMMEDIATE", "SOON", "OPTIONAL"
    confidence: float           # 0-1 confidence in recommendation


# Environment configuration
ADJUSTMENT_ENABLE = os.getenv("TRABOT_ADJUSTMENT_ENABLE", "1").strip() == "1"
LOSING_SMALL_THRESHOLD = float(os.getenv("TRABOT_LOSING_SMALL_PCT", "0.25"))
LOSING_MEDIUM_THRESHOLD = float(os.getenv("TRABOT_LOSING_MEDIUM_PCT", "0.50"))
NEAR_STOP_THRESHOLD = float(os.getenv("TRABOT_NEAR_STOP_PCT", "0.80"))


def classify_position_status(
    position: Position,
    max_loss: float,
    stop_loss: Optional[float] = None,
) -> PositionStatus:
    """Classify position status based on current P&L.

    Args:
        position: Current position
        max_loss: Maximum possible loss
        stop_loss: Stop loss price (premium level)

    Returns:
        PositionStatus classification
    """
    current_pnl = (position.current_price - position.entry_price) * position.contracts
    if not position.is_long:
        current_pnl = -current_pnl  # Reverse for short positions

    pnl_pct = abs(current_pnl / max_loss) if max_loss != 0 else 0

    # Check stop loss first
    if stop_loss:
        if position.is_long:
            if position.current_price <= stop_loss:
                return PositionStatus.STOPPED_OUT
            elif position.current_price <= stop_loss * 1.1:
                return PositionStatus.NEAR_STOP
        else:
            if position.current_price >= stop_loss:
                return PositionStatus.STOPPED_OUT
            elif position.current_price >= stop_loss * 0.9:
                return PositionStatus.NEAR_STOP

    # Classify by P&L
    if current_pnl > 0:
        return PositionStatus.WINNING
    elif abs(current_pnl) < max_loss * 0.05:
        return PositionStatus.SCRATCH
    elif pnl_pct < LOSING_SMALL_THRESHOLD:
        return PositionStatus.LOSING_SMALL
    elif pnl_pct < LOSING_MEDIUM_THRESHOLD:
        return PositionStatus.LOSING_MEDIUM
    else:
        return PositionStatus.LOSING_LARGE


def _suggest_roll_out(position: Position) -> Adjustment:
    """Suggest rolling to later expiry."""
    return Adjustment(
        type=AdjustmentType.ROLL_OUT,
        description=f"Roll {position.structure} to next weekly/monthly expiry",
        mechanics=(
            f"1. Close current position at {position.current_price:.2f}\n"
            f"2. Open same strike ({position.strike}) at later expiry\n"
            f"3. Net debit/credit depends on IV term structure"
        ),
        expected_outcome="More time for trade thesis to play out",
        cost_estimate=position.current_price * 0.1,  # Rough estimate
        risk_impact="Extends time risk, may increase theta decay",
        urgency="SOON" if position.dte <= 3 else "OPTIONAL",
        confidence=0.7,
    )


def _suggest_roll_strike(position: Position, direction: str) -> Adjustment:
    """Suggest rolling to different strike."""
    adj_type = AdjustmentType.ROLL_UP if direction == "UP" else AdjustmentType.ROLL_DOWN

    if direction == "UP":
        new_strike = position.strike * 1.02  # ~2% higher
        mechanics = (
            f"1. Close current {position.strike} position\n"
            f"2. Open new position at {new_strike:.0f}\n"
            f"3. May receive credit (rolling up calls) or pay debit (rolling up puts)"
        )
        outcome = "Higher strike reduces delta, lowers breakeven for shorts"
    else:
        new_strike = position.strike * 0.98  # ~2% lower
        mechanics = (
            f"1. Close current {position.strike} position\n"
            f"2. Open new position at {new_strike:.0f}\n"
            f"3. May receive credit (rolling down puts) or pay debit (rolling down calls)"
        )
        outcome = "Lower strike reduces delta, lowers breakeven for longs"

    return Adjustment(
        type=adj_type,
        description=f"Roll {position.structure} {direction.lower()} to {new_strike:.0f}",
        mechanics=mechanics,
        expected_outcome=outcome,
        cost_estimate=position.current_price * 0.15,
        risk_impact="Changes delta exposure, may affect theta",
        urgency="SOON",
        confidence=0.6,
    )


def _suggest_convert_to_spread(position: Position) -> Adjustment:
    """Suggest converting single-leg to spread."""
    if "CE" in position.structure:
        spread_type = "Bull Call Spread" if position.is_long else "Bear Call Spread"
        action = "Sell OTM call" if position.is_long else "Buy OTM call"
        otm_strike = position.strike * 1.02
    else:
        spread_type = "Bear Put Spread" if position.is_long else "Bull Put Spread"
        action = "Sell OTM put" if position.is_long else "Buy OTM put"
        otm_strike = position.strike * 0.98

    return Adjustment(
        type=AdjustmentType.CONVERT_TO_SPREAD,
        description=f"Convert to {spread_type} by adding leg at {otm_strike:.0f}",
        mechanics=(
            f"1. Keep current {position.structure} at {position.strike}\n"
            f"2. {action} at {otm_strike:.0f}\n"
            f"3. Reduces cost basis, caps profit potential"
        ),
        expected_outcome="Lower breakeven, defined risk, collected premium offsets loss",
        cost_estimate=-position.entry_price * 0.3,  # Credit from selling
        risk_impact="Caps upside, reduces theta decay, defines max loss",
        urgency="SOON",
        confidence=0.75,
    )


def _suggest_add_hedge(position: Position) -> Adjustment:
    """Suggest adding a hedge leg."""
    if position.is_long:
        if "CE" in position.structure:
            hedge = "Buy protective put below current spot"
            hedge_strike = position.current_spot * 0.98
        else:
            hedge = "Buy protective call above current spot"
            hedge_strike = position.current_spot * 1.02
    else:
        if "CE" in position.structure:
            hedge = "Buy call at higher strike to define risk"
            hedge_strike = position.strike * 1.03
        else:
            hedge = "Buy put at lower strike to define risk"
            hedge_strike = position.strike * 0.97

    return Adjustment(
        type=AdjustmentType.ADD_HEDGE,
        description=hedge,
        mechanics=(
            f"1. Keep existing position\n"
            f"2. Add hedge at {hedge_strike:.0f}\n"
            f"3. Creates defined-risk structure"
        ),
        expected_outcome="Limits max loss, provides downside protection",
        cost_estimate=position.entry_price * 0.4,  # Cost of hedge
        risk_impact="Reduces potential loss, may reduce profit potential",
        urgency="OPTIONAL",
        confidence=0.6,
    )


def _suggest_close_partial(position: Position) -> Adjustment:
    """Suggest closing partial position."""
    close_pct = 50 if position.contracts > 1 else 100

    return Adjustment(
        type=AdjustmentType.CLOSE_PARTIAL,
        description=f"Close {close_pct}% of position to lock in P&L",
        mechanics=(
            f"1. Close {close_pct}% ({position.contracts // 2 or 1} contracts)\n"
            f"2. Let remaining position run with tighter stop\n"
            f"3. Reduces exposure while maintaining upside"
        ),
        expected_outcome="Locks in partial gain/limits loss, reduces position size",
        cost_estimate=0,
        risk_impact="Halves exposure, reduces both risk and reward",
        urgency="OPTIONAL",
        confidence=0.65,
    )


def _suggest_close_full(position: Position, reason: str) -> Adjustment:
    """Suggest closing entire position."""
    return Adjustment(
        type=AdjustmentType.CLOSE_FULL,
        description=f"Close entire position: {reason}",
        mechanics=(
            f"1. Market order to close all {position.contracts} contracts\n"
            f"2. Lock in current P&L\n"
            f"3. Wait for better setup"
        ),
        expected_outcome="Preserve capital, avoid further loss",
        cost_estimate=0,
        risk_impact="Eliminates all position risk",
        urgency="IMMEDIATE" if "stop" in reason.lower() else "SOON",
        confidence=0.8,
    )


def suggest_adjustments(
    position: Position,
    max_loss: float,
    stop_loss: Optional[float] = None,
    target: Optional[float] = None,
) -> List[Adjustment]:
    """Main function: Suggest adjustments for a position.

    Args:
        position: Current position details
        max_loss: Maximum loss for the position
        stop_loss: Stop loss premium level
        target: Target premium level

    Returns:
        List of suggested adjustments, ordered by recommendation
    """
    if not ADJUSTMENT_ENABLE:
        return []

    adjustments = []
    status = classify_position_status(position, max_loss, stop_loss)

    # Handle stopped out
    if status == PositionStatus.STOPPED_OUT:
        adjustments.append(_suggest_close_full(position, "Stop loss triggered"))
        return adjustments

    # Handle near stop
    if status == PositionStatus.NEAR_STOP:
        adjustments.append(_suggest_close_full(position, "Approaching stop loss"))
        if position.dte > 7:
            adjustments.append(_suggest_roll_out(position))
        return adjustments

    # Handle winning positions
    if status == PositionStatus.WINNING:
        current_pnl = (position.current_price - position.entry_price) * position.contracts
        if not position.is_long:
            current_pnl = -current_pnl

        # Check if near target
        if target and position.is_long and position.current_price >= target * 0.9:
            adjustments.append(_suggest_close_partial(position))
            adjustments.append(Adjustment(
                type=AdjustmentType.DOUBLE_DOWN,
                description="Add to winning position on pullback",
                mechanics="Wait for 2-3% pullback, add 50% more contracts",
                expected_outcome="Increase profit potential",
                cost_estimate=position.current_price,
                risk_impact="Increases exposure and risk",
                urgency="OPTIONAL",
                confidence=0.5,
            ))
        else:
            adjustments.append(_suggest_close_partial(position))
            adjustments.append(Adjustment(
                type=AdjustmentType.NO_ACTION,
                description="Hold position - trade working",
                mechanics="Monitor position, adjust stop to breakeven",
                expected_outcome="Let winner run",
                cost_estimate=0,
                risk_impact="No change",
                urgency="OPTIONAL",
                confidence=0.7,
            ))
        return adjustments

    # Handle scratch
    if status == PositionStatus.SCRATCH:
        if position.dte <= 3:
            adjustments.append(_suggest_roll_out(position))
        adjustments.append(Adjustment(
            type=AdjustmentType.NO_ACTION,
            description="Hold - position at breakeven",
            mechanics="Monitor for directional move",
            expected_outcome="Wait for thesis to play out",
            cost_estimate=0,
            risk_impact="No change",
            urgency="OPTIONAL",
            confidence=0.6,
        ))
        return adjustments

    # Handle losing positions
    if status == PositionStatus.LOSING_SMALL:
        # Minor loss - consider rolling or converting
        if position.dte <= 5:
            adjustments.append(_suggest_roll_out(position))

        if "SPREAD" not in position.structure:
            adjustments.append(_suggest_convert_to_spread(position))

        adjustments.append(Adjustment(
            type=AdjustmentType.NO_ACTION,
            description="Hold - loss is small and manageable",
            mechanics="Monitor position, maintain original stop",
            expected_outcome="Give trade time to work",
            cost_estimate=0,
            risk_impact="No change",
            urgency="OPTIONAL",
            confidence=0.55,
        ))

    elif status == PositionStatus.LOSING_MEDIUM:
        # Medium loss - more aggressive adjustments needed
        if position.dte > 5:
            if "CE" in position.structure:
                adjustments.append(_suggest_roll_strike(position, "DOWN"))
            else:
                adjustments.append(_suggest_roll_strike(position, "UP"))

        if "SPREAD" not in position.structure:
            adjustments.append(_suggest_convert_to_spread(position))

        adjustments.append(_suggest_close_partial(position))
        adjustments.append(_suggest_add_hedge(position))

    elif status == PositionStatus.LOSING_LARGE:
        # Large loss - prioritize capital preservation
        adjustments.append(_suggest_close_full(position, "Large loss - preserve capital"))

        # Only suggest roll if significant time left
        if position.dte > 14:
            adj_roll = _suggest_roll_out(position)
            adj_roll.urgency = "SOON"
            adjustments.append(adj_roll)

    return adjustments


def get_adjustment_summary(adjustments: List[Adjustment]) -> Dict:
    """Get summary of adjustments for logging."""
    if not adjustments:
        return {"count": 0, "recommendations": []}

    return {
        "count": len(adjustments),
        "recommendations": [
            {
                "type": adj.type.value,
                "description": adj.description,
                "urgency": adj.urgency,
                "confidence": f"{adj.confidence:.0%}",
            }
            for adj in adjustments
        ],
        "top_recommendation": adjustments[0].type.value if adjustments else None,
    }


def should_auto_adjust(
    position: Position,
    max_loss: float,
    stop_loss: Optional[float] = None,
) -> Tuple[bool, Optional[AdjustmentType]]:
    """Check if position should be auto-adjusted.

    Returns (should_adjust, recommended_type)
    """
    status = classify_position_status(position, max_loss, stop_loss)

    # Auto-close on stop
    if status == PositionStatus.STOPPED_OUT:
        return True, AdjustmentType.CLOSE_FULL

    # Auto-roll on very low DTE if still has value
    if position.dte <= 1 and position.current_price > position.entry_price * 0.2:
        return True, AdjustmentType.ROLL_OUT

    # Suggest close on large loss
    if status == PositionStatus.LOSING_LARGE:
        return True, AdjustmentType.CLOSE_FULL

    return False, None


if __name__ == "__main__":
    # Demo
    print("Adjustment Engine Demo")
    print("=" * 50)

    # Sample losing position
    position = Position(
        underlying="NIFTY",
        structure="BUY_CE",
        strike=20100,
        strike2=None,
        expiry="2024-01-18",
        dte=5,
        entry_price=200,
        current_price=120,  # 40% loss
        contracts=2,
        is_long=True,
        entry_spot=20000,
        current_spot=19900,
    )

    print(f"\nPosition: {position.structure} {position.strike}")
    print(f"Entry: Rs {position.entry_price}, Current: Rs {position.current_price}")
    print(f"P&L: Rs {(position.current_price - position.entry_price) * position.contracts}")
    print(f"DTE: {position.dte}")

    # Get adjustments
    adjustments = suggest_adjustments(
        position,
        max_loss=position.entry_price * position.contracts,
        stop_loss=100,  # Stop at Rs 100
        target=350,     # Target at Rs 350
    )

    print(f"\n{'='*50}")
    print(f"Position Status: {classify_position_status(position, 400, 100).value}")
    print(f"\nRecommended Adjustments ({len(adjustments)}):")

    for i, adj in enumerate(adjustments, 1):
        print(f"\n{i}. {adj.type.value} [{adj.urgency}] ({adj.confidence:.0%} confidence)")
        print(f"   {adj.description}")
        print(f"   Expected: {adj.expected_outcome}")
        if adj.cost_estimate != 0:
            print(f"   Est. Cost: Rs {adj.cost_estimate:.0f}")
