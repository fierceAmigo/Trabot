"""circuit_breaker.py

P0: Daily loss limit and circuit breaker for capital protection.

This module tracks daily P&L and trips a circuit breaker when losses
exceed configured thresholds, preventing further trading.

States:
- ACTIVE: Normal trading allowed
- WARNING: Approaching limit (50%, 70%, 90% thresholds)
- TRIPPED: Limit exceeded, no new entries allowed

Features:
- Tracks realized + unrealized P&L
- Auto-resets at market open each day (9:15 IST)
- Persists state to JSON for crash recovery
- Consecutive loss tracking

Environment Variables:
- TRABOT_CIRCUIT_BREAKER_ENABLE: 0/1 (default 1)
- TRABOT_MAX_DAILY_LOSS_PCT: Max daily loss % (default 2.0)
- TRABOT_MAX_CONSECUTIVE_LOSSES: Max consecutive losses (default 5)
- TRABOT_CIRCUIT_BREAKER_STATE_PATH: State file path

Usage:
    from circuit_breaker import check_circuit_breaker, record_trade_result

    ok, status, reason = check_circuit_breaker(capital=100000)
    if not ok:
        print(f"Circuit breaker tripped: {reason}")
        return None
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, date, time as dtime, timedelta
from typing import Optional, Tuple, Dict, List
from enum import Enum

try:
    from dateutil import tz
    IST = tz.gettz("Asia/Kolkata")
except ImportError:
    IST = None


class BreakerStatus(Enum):
    ACTIVE = "ACTIVE"
    WARNING_50 = "WARNING_50"
    WARNING_70 = "WARNING_70"
    WARNING_90 = "WARNING_90"
    TRIPPED = "TRIPPED"


@dataclass
class DailyState:
    date: str  # YYYY-MM-DD
    realized_pnl: float  # Closed trade P&L
    unrealized_pnl: float  # Open position P&L (mark-to-market)
    trade_count: int
    win_count: int
    loss_count: int
    consecutive_losses: int
    status: str  # BreakerStatus value
    last_updated: str  # ISO timestamp
    trades: List[Dict]  # List of trade records


# Environment configuration
CIRCUIT_BREAKER_ENABLE = os.getenv("TRABOT_CIRCUIT_BREAKER_ENABLE", "1").strip() == "1"
MAX_DAILY_LOSS_PCT = float(os.getenv("TRABOT_MAX_DAILY_LOSS_PCT", "2.0"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("TRABOT_MAX_CONSECUTIVE_LOSSES", "5"))
STATE_FILE = os.getenv("TRABOT_CIRCUIT_BREAKER_STATE_PATH", "data/circuit_breaker_state.json")

# Market timing (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15


def _now_ist() -> datetime:
    """Get current time in IST."""
    if IST:
        return datetime.now(IST)
    return datetime.now()


def _today_ist() -> date:
    """Get today's date in IST."""
    return _now_ist().date()


def _is_market_open() -> bool:
    """Check if market is currently open."""
    now = _now_ist()
    # Market hours: 9:15 AM to 3:30 PM IST, Monday-Friday
    if now.weekday() >= 5:  # Weekend
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def _should_reset(state: DailyState) -> bool:
    """Check if state should be reset for new trading day."""
    if not state.date:
        return True

    state_date = date.fromisoformat(state.date)
    today = _today_ist()

    # Reset if it's a new day
    if state_date < today:
        return True

    return False


def _empty_state() -> DailyState:
    """Create empty state for new day."""
    return DailyState(
        date=_today_ist().isoformat(),
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        trade_count=0,
        win_count=0,
        loss_count=0,
        consecutive_losses=0,
        status=BreakerStatus.ACTIVE.value,
        last_updated=_now_ist().isoformat(),
        trades=[],
    )


def load_state() -> DailyState:
    """Load circuit breaker state from file."""
    if not os.path.exists(STATE_FILE):
        return _empty_state()

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        state = DailyState(
            date=data.get("date", ""),
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
            trade_count=int(data.get("trade_count", 0)),
            win_count=int(data.get("win_count", 0)),
            loss_count=int(data.get("loss_count", 0)),
            consecutive_losses=int(data.get("consecutive_losses", 0)),
            status=data.get("status", BreakerStatus.ACTIVE.value),
            last_updated=data.get("last_updated", ""),
            trades=data.get("trades", []),
        )

        # Reset if new day
        if _should_reset(state):
            return _empty_state()

        return state

    except (json.JSONDecodeError, IOError, KeyError):
        return _empty_state()


def save_state(state: DailyState) -> None:
    """Save circuit breaker state to file."""
    os.makedirs(os.path.dirname(STATE_FILE) or "data", exist_ok=True)

    state.last_updated = _now_ist().isoformat()

    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2, ensure_ascii=False)
    except IOError:
        pass  # Best effort


def _calculate_status(
    total_pnl: float,
    capital: float,
    consecutive_losses: int,
) -> BreakerStatus:
    """Determine circuit breaker status based on P&L."""
    if capital <= 0:
        return BreakerStatus.ACTIVE

    # Check consecutive losses
    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        return BreakerStatus.TRIPPED

    # Calculate loss percentage
    loss_pct = abs(min(0, total_pnl)) / capital * 100

    if loss_pct >= MAX_DAILY_LOSS_PCT:
        return BreakerStatus.TRIPPED
    elif loss_pct >= MAX_DAILY_LOSS_PCT * 0.90:
        return BreakerStatus.WARNING_90
    elif loss_pct >= MAX_DAILY_LOSS_PCT * 0.70:
        return BreakerStatus.WARNING_70
    elif loss_pct >= MAX_DAILY_LOSS_PCT * 0.50:
        return BreakerStatus.WARNING_50

    return BreakerStatus.ACTIVE


def check_circuit_breaker(
    capital: float,
    unrealized_pnl: Optional[float] = None,
) -> Tuple[bool, BreakerStatus, str]:
    """Check if circuit breaker allows new entries.

    Args:
        capital: Total trading capital
        unrealized_pnl: Current unrealized P&L (optional, uses stored value if None)

    Returns:
        (ok, status, reason) where ok=True means trading allowed
    """
    if not CIRCUIT_BREAKER_ENABLE:
        return True, BreakerStatus.ACTIVE, ""

    state = load_state()

    # Update unrealized P&L if provided
    if unrealized_pnl is not None:
        state.unrealized_pnl = float(unrealized_pnl)
        save_state(state)

    # Calculate total P&L
    total_pnl = state.realized_pnl + state.unrealized_pnl

    # Determine status
    status = _calculate_status(total_pnl, capital, state.consecutive_losses)
    state.status = status.value
    save_state(state)

    # Generate reason message
    if status == BreakerStatus.TRIPPED:
        if state.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            reason = f"Consecutive losses ({state.consecutive_losses}) exceeded limit ({MAX_CONSECUTIVE_LOSSES})"
        else:
            loss_pct = abs(min(0, total_pnl)) / capital * 100
            reason = f"Daily loss ({loss_pct:.2f}%) exceeded limit ({MAX_DAILY_LOSS_PCT}%)"
        return False, status, reason

    if status in (BreakerStatus.WARNING_50, BreakerStatus.WARNING_70, BreakerStatus.WARNING_90):
        loss_pct = abs(min(0, total_pnl)) / capital * 100
        reason = f"Daily loss at {loss_pct:.2f}% of {MAX_DAILY_LOSS_PCT}% limit"
        return True, status, reason

    return True, status, ""


def record_trade_result(
    trade_id: str,
    pnl: float,
    is_closed: bool = True,
) -> DailyState:
    """Record a trade result and update state.

    Args:
        trade_id: Unique identifier for the trade
        pnl: Profit/loss amount (positive = profit, negative = loss)
        is_closed: True if trade is closed (realized P&L)

    Returns:
        Updated state
    """
    state = load_state()

    # Record trade
    trade_record = {
        "trade_id": trade_id,
        "pnl": float(pnl),
        "is_closed": is_closed,
        "timestamp": _now_ist().isoformat(),
    }
    state.trades.append(trade_record)
    state.trade_count += 1

    if is_closed:
        state.realized_pnl += float(pnl)

        if pnl >= 0:
            state.win_count += 1
            state.consecutive_losses = 0
        else:
            state.loss_count += 1
            state.consecutive_losses += 1

    save_state(state)
    return state


def update_unrealized_pnl(unrealized_pnl: float) -> DailyState:
    """Update unrealized P&L from portfolio mark-to-market.

    Args:
        unrealized_pnl: Current unrealized P&L across all open positions

    Returns:
        Updated state
    """
    state = load_state()
    state.unrealized_pnl = float(unrealized_pnl)
    save_state(state)
    return state


def get_daily_summary(capital: float) -> Dict:
    """Get summary of daily trading activity."""
    state = load_state()
    total_pnl = state.realized_pnl + state.unrealized_pnl

    return {
        "date": state.date,
        "realized_pnl": state.realized_pnl,
        "unrealized_pnl": state.unrealized_pnl,
        "total_pnl": total_pnl,
        "total_pnl_pct": (total_pnl / capital * 100) if capital > 0 else 0,
        "trade_count": state.trade_count,
        "win_count": state.win_count,
        "loss_count": state.loss_count,
        "win_rate": (state.win_count / state.trade_count * 100) if state.trade_count > 0 else 0,
        "consecutive_losses": state.consecutive_losses,
        "status": state.status,
        "limit_pct": MAX_DAILY_LOSS_PCT,
        "limit_used_pct": (abs(min(0, total_pnl)) / capital / MAX_DAILY_LOSS_PCT * 100) if capital > 0 else 0,
    }


def reset_circuit_breaker(force: bool = False) -> bool:
    """Reset circuit breaker state.

    Args:
        force: If True, reset even if market is open

    Returns:
        True if reset was performed
    """
    if not force and _is_market_open():
        return False

    state = _empty_state()
    save_state(state)
    return True


def get_remaining_risk_budget(capital: float) -> float:
    """Get remaining risk budget for today.

    Returns the amount that can still be lost before circuit breaker trips.
    """
    state = load_state()
    total_pnl = state.realized_pnl + state.unrealized_pnl

    max_loss = capital * MAX_DAILY_LOSS_PCT / 100
    current_loss = abs(min(0, total_pnl))
    remaining = max_loss - current_loss

    return max(0, remaining)


def should_reduce_size(capital: float) -> Tuple[bool, float]:
    """Check if position size should be reduced due to approaching limits.

    Returns (should_reduce, multiplier) where multiplier is 0-1
    """
    state = load_state()
    total_pnl = state.realized_pnl + state.unrealized_pnl

    if capital <= 0:
        return False, 1.0

    loss_pct = abs(min(0, total_pnl)) / capital * 100
    limit_used = loss_pct / MAX_DAILY_LOSS_PCT

    if limit_used >= 0.90:
        return True, 0.25  # 25% of normal size
    elif limit_used >= 0.70:
        return True, 0.50  # 50% of normal size
    elif limit_used >= 0.50:
        return True, 0.75  # 75% of normal size

    # Also reduce if consecutive losses are high
    if state.consecutive_losses >= 3:
        mult = max(0.25, 1.0 - (state.consecutive_losses - 2) * 0.25)
        return True, mult

    return False, 1.0


if __name__ == "__main__":
    # Demo
    capital = 100000

    print("Circuit Breaker Status Check")
    print("=" * 50)

    ok, status, reason = check_circuit_breaker(capital)
    print(f"Trading allowed: {ok}")
    print(f"Status: {status.value}")
    if reason:
        print(f"Reason: {reason}")

    print("\nDaily Summary:")
    summary = get_daily_summary(capital)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nRemaining risk budget: ₹{get_remaining_risk_budget(capital):,.2f}")

    should_reduce, mult = should_reduce_size(capital)
    if should_reduce:
        print(f"Size reduction recommended: {mult:.0%} of normal")
