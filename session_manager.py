"""session_manager.py

P1: Session-aware trading rules for Indian markets.

Different sessions have different volatility profiles and optimal strategies.
This module classifies time into sessions and provides session-specific rules.

Sessions (IST):
- OPENING (9:15-10:00): High volatility, wide spreads, scalps only
- MORNING (10:00-12:00): Good for trend entries
- LUNCH (12:00-13:30): Low volume, avoid new entries
- AFTERNOON (13:30-14:30): Moderate, selective entries
- CLOSING (14:30-15:30): Strong momentum only, avoid last 15 min

Environment Variables:
- TRABOT_SESSION_RULES_ENABLE: 0/1 (default 1)
- TRABOT_BLOCK_LUNCH_ENTRIES: 0/1 (default 1)
- TRABOT_BLOCK_CLOSING_NEW: 0/1 (default 1)
- TRABOT_OPENING_SCALP_ONLY: 0/1 (default 0)

Usage:
    from session_manager import get_session_config, should_allow_entry

    config = get_session_config()
    if not should_allow_entry(signal_strength=1.2):
        return None  # Skip entry
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Optional, Dict, Tuple
from enum import Enum

try:
    from dateutil import tz
    IST = tz.gettz("Asia/Kolkata")
except ImportError:
    IST = None


class Session(Enum):
    PRE_MARKET = "PRE_MARKET"
    OPENING = "OPENING"
    MORNING = "MORNING"
    LUNCH = "LUNCH"
    AFTERNOON = "AFTERNOON"
    CLOSING = "CLOSING"
    LAST_15_MIN = "LAST_15_MIN"
    POST_MARKET = "POST_MARKET"


@dataclass
class SessionConfig:
    """Configuration for a trading session."""
    session: Session
    allow_new_entries: bool
    min_signal_strength: float  # Minimum signal strength required
    stop_multiplier: float  # Multiply stop distance by this
    target_multiplier: float  # Multiply target distance by this
    size_multiplier: float  # Multiply position size by this
    prefer_scalps: bool  # Prefer quick in-and-out trades
    avoid_multi_leg: bool  # Avoid complex structures
    notes: str


# Environment configuration
SESSION_RULES_ENABLE = os.getenv("TRABOT_SESSION_RULES_ENABLE", "1").strip() == "1"
BLOCK_LUNCH_ENTRIES = os.getenv("TRABOT_BLOCK_LUNCH_ENTRIES", "1").strip() == "1"
BLOCK_CLOSING_NEW = os.getenv("TRABOT_BLOCK_CLOSING_NEW", "1").strip() == "1"
OPENING_SCALP_ONLY = os.getenv("TRABOT_OPENING_SCALP_ONLY", "0").strip() == "1"


# Session boundaries (IST)
SESSION_TIMES = {
    Session.PRE_MARKET: (dtime(0, 0), dtime(9, 14)),
    Session.OPENING: (dtime(9, 15), dtime(9, 59)),
    Session.MORNING: (dtime(10, 0), dtime(11, 59)),
    Session.LUNCH: (dtime(12, 0), dtime(13, 29)),
    Session.AFTERNOON: (dtime(13, 30), dtime(14, 29)),
    Session.CLOSING: (dtime(14, 30), dtime(15, 14)),
    Session.LAST_15_MIN: (dtime(15, 15), dtime(15, 29)),
    Session.POST_MARKET: (dtime(15, 30), dtime(23, 59)),
}


# Default session configurations
DEFAULT_CONFIGS: Dict[Session, SessionConfig] = {
    Session.PRE_MARKET: SessionConfig(
        session=Session.PRE_MARKET,
        allow_new_entries=False,
        min_signal_strength=0,
        stop_multiplier=1.0,
        target_multiplier=1.0,
        size_multiplier=0,
        prefer_scalps=False,
        avoid_multi_leg=True,
        notes="Market closed",
    ),
    Session.OPENING: SessionConfig(
        session=Session.OPENING,
        allow_new_entries=True,
        min_signal_strength=1.3,  # Require stronger signals
        stop_multiplier=0.8,  # Tighter stops (volatile)
        target_multiplier=0.7,  # Lower targets (quick profits)
        size_multiplier=0.6,  # Smaller size (higher risk)
        prefer_scalps=True,
        avoid_multi_leg=True,  # Spreads hard to fill
        notes="High volatility, wide spreads, quick scalps",
    ),
    Session.MORNING: SessionConfig(
        session=Session.MORNING,
        allow_new_entries=True,
        min_signal_strength=1.0,  # Normal threshold
        stop_multiplier=1.0,
        target_multiplier=1.0,
        size_multiplier=1.0,
        prefer_scalps=False,
        avoid_multi_leg=False,
        notes="Best session for trend entries",
    ),
    Session.LUNCH: SessionConfig(
        session=Session.LUNCH,
        allow_new_entries=not BLOCK_LUNCH_ENTRIES,
        min_signal_strength=1.4,  # Require very strong signals
        stop_multiplier=1.2,  # Wider stops (low liquidity)
        target_multiplier=0.8,  # Lower targets
        size_multiplier=0.5,  # Half size
        prefer_scalps=False,
        avoid_multi_leg=True,  # Poor fills
        notes="Low volume, avoid unless very strong signal",
    ),
    Session.AFTERNOON: SessionConfig(
        session=Session.AFTERNOON,
        allow_new_entries=True,
        min_signal_strength=1.1,
        stop_multiplier=1.0,
        target_multiplier=1.0,
        size_multiplier=0.9,
        prefer_scalps=False,
        avoid_multi_leg=False,
        notes="Moderate session, selective entries",
    ),
    Session.CLOSING: SessionConfig(
        session=Session.CLOSING,
        allow_new_entries=not BLOCK_CLOSING_NEW,
        min_signal_strength=1.4,  # Strong momentum only
        stop_multiplier=0.7,  # Tight stops
        target_multiplier=0.6,  # Quick targets
        size_multiplier=0.5,
        prefer_scalps=True,
        avoid_multi_leg=True,
        notes="Strong momentum only, avoid complex structures",
    ),
    Session.LAST_15_MIN: SessionConfig(
        session=Session.LAST_15_MIN,
        allow_new_entries=False,
        min_signal_strength=0,
        stop_multiplier=1.0,
        target_multiplier=1.0,
        size_multiplier=0,
        prefer_scalps=False,
        avoid_multi_leg=True,
        notes="Do not enter - closing auction, unpredictable",
    ),
    Session.POST_MARKET: SessionConfig(
        session=Session.POST_MARKET,
        allow_new_entries=False,
        min_signal_strength=0,
        stop_multiplier=1.0,
        target_multiplier=1.0,
        size_multiplier=0,
        prefer_scalps=False,
        avoid_multi_leg=True,
        notes="Market closed",
    ),
}


def _now_ist() -> datetime:
    """Get current time in IST."""
    if IST:
        return datetime.now(IST)
    return datetime.now()


def get_current_session(timestamp: Optional[datetime] = None) -> Session:
    """Determine current trading session.

    Args:
        timestamp: Time to check (defaults to now)

    Returns:
        Current Session enum
    """
    if timestamp is None:
        timestamp = _now_ist()

    # Extract time component
    if hasattr(timestamp, 'time'):
        t = timestamp.time()
    else:
        t = timestamp

    # Check each session boundary
    for session, (start, end) in SESSION_TIMES.items():
        if start <= t <= end:
            return session

    return Session.POST_MARKET


def get_session_config(
    session: Optional[Session] = None,
    timestamp: Optional[datetime] = None,
) -> SessionConfig:
    """Get configuration for a session.

    Args:
        session: Session to get config for (defaults to current)
        timestamp: Time to determine session (if session not provided)

    Returns:
        SessionConfig for the session
    """
    if session is None:
        session = get_current_session(timestamp)

    return DEFAULT_CONFIGS.get(session, DEFAULT_CONFIGS[Session.POST_MARKET])


def should_allow_entry(
    signal_strength: float = 1.0,
    strategy_type: str = "SINGLE_LEG",
    timestamp: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """Check if entry is allowed in current session.

    Args:
        signal_strength: Strength of the signal (abs value of score)
        strategy_type: Type of strategy (SINGLE_LEG, SPREAD, IRON_CONDOR, etc.)
        timestamp: Time to check

    Returns:
        (allowed, reason)
    """
    if not SESSION_RULES_ENABLE:
        return True, ""

    config = get_session_config(timestamp=timestamp)

    # Check if entries allowed
    if not config.allow_new_entries:
        return False, f"No new entries in {config.session.value}: {config.notes}"

    # Check signal strength
    if signal_strength < config.min_signal_strength:
        return False, f"Signal strength {signal_strength:.2f} < {config.min_signal_strength:.2f} required for {config.session.value}"

    # Check multi-leg restriction
    if config.avoid_multi_leg and strategy_type not in ("SINGLE_LEG", "BUY_CE", "BUY_PE"):
        return False, f"Multi-leg strategies not recommended in {config.session.value}"

    return True, ""


def get_adjusted_levels(
    entry: float,
    stop: float,
    target: float,
    timestamp: Optional[datetime] = None,
) -> Tuple[float, float, float]:
    """Adjust entry/stop/target based on session.

    Args:
        entry: Original entry price
        stop: Original stop price
        target: Original target price
        timestamp: Time to determine session

    Returns:
        (adjusted_entry, adjusted_stop, adjusted_target)
    """
    if not SESSION_RULES_ENABLE:
        return entry, stop, target

    config = get_session_config(timestamp=timestamp)

    # Calculate distances
    stop_distance = abs(entry - stop)
    target_distance = abs(target - entry)

    # Adjust distances
    adj_stop_distance = stop_distance * config.stop_multiplier
    adj_target_distance = target_distance * config.target_multiplier

    # Apply adjustments
    if stop < entry:  # Long position
        adj_stop = entry - adj_stop_distance
        adj_target = entry + adj_target_distance
    else:  # Short position
        adj_stop = entry + adj_stop_distance
        adj_target = entry - adj_target_distance

    return entry, adj_stop, adj_target


def get_size_multiplier(timestamp: Optional[datetime] = None) -> float:
    """Get position size multiplier for current session.

    Returns:
        Multiplier (0-1) to apply to calculated position size
    """
    if not SESSION_RULES_ENABLE:
        return 1.0

    config = get_session_config(timestamp=timestamp)
    return config.size_multiplier


def get_session_factor(timestamp: Optional[datetime] = None) -> float:
    """Get scoring factor for current session.

    This is used to adjust signal scores based on session quality.
    Higher factor = better session for trading.

    Returns:
        Factor (0-1.2) to multiply signal score
    """
    config = get_session_config(timestamp=timestamp)

    factors = {
        Session.PRE_MARKET: 0.0,
        Session.OPENING: 0.8,
        Session.MORNING: 1.2,  # Best session
        Session.LUNCH: 0.6,
        Session.AFTERNOON: 1.0,
        Session.CLOSING: 0.7,
        Session.LAST_15_MIN: 0.0,
        Session.POST_MARKET: 0.0,
    }

    return factors.get(config.session, 0.5)


def get_recommended_time_stop(
    base_time_stop_min: int = 90,
    timestamp: Optional[datetime] = None,
) -> int:
    """Get recommended time-stop duration based on session.

    Args:
        base_time_stop_min: Base time stop in minutes
        timestamp: Current time

    Returns:
        Adjusted time stop in minutes
    """
    if not SESSION_RULES_ENABLE:
        return base_time_stop_min

    config = get_session_config(timestamp=timestamp)

    # Shorter time stops in volatile/closing sessions
    if config.prefer_scalps:
        return int(base_time_stop_min * 0.5)

    # Longer time stops in morning
    if config.session == Session.MORNING:
        return int(base_time_stop_min * 1.2)

    return base_time_stop_min


def get_session_summary(timestamp: Optional[datetime] = None) -> Dict:
    """Get summary of current session for logging."""
    config = get_session_config(timestamp=timestamp)

    return {
        "session": config.session.value,
        "allow_entries": config.allow_new_entries,
        "min_signal_strength": config.min_signal_strength,
        "size_multiplier": config.size_multiplier,
        "stop_multiplier": config.stop_multiplier,
        "target_multiplier": config.target_multiplier,
        "prefer_scalps": config.prefer_scalps,
        "avoid_multi_leg": config.avoid_multi_leg,
        "notes": config.notes,
        "session_factor": get_session_factor(timestamp),
    }


def is_good_entry_window(timestamp: Optional[datetime] = None) -> bool:
    """Quick check if current time is a good entry window.

    Returns True for MORNING and AFTERNOON sessions.
    """
    session = get_current_session(timestamp)
    return session in (Session.MORNING, Session.AFTERNOON)


def minutes_until_session_change(timestamp: Optional[datetime] = None) -> int:
    """Get minutes until next session change."""
    if timestamp is None:
        timestamp = _now_ist()

    current = get_current_session(timestamp)
    current_end = SESSION_TIMES[current][1]

    current_time = timestamp.time() if hasattr(timestamp, 'time') else timestamp

    # Calculate minutes until session end
    current_mins = current_time.hour * 60 + current_time.minute
    end_mins = current_end.hour * 60 + current_end.minute

    return max(0, end_mins - current_mins)


if __name__ == "__main__":
    # Demo
    print("Session Manager Demo")
    print("=" * 50)

    # Current session
    session = get_current_session()
    config = get_session_config()

    print(f"\nCurrent Session: {session.value}")
    print(f"Time: {_now_ist().strftime('%H:%M')}")

    print("\nSession Config:")
    for k, v in get_session_summary().items():
        print(f"  {k}: {v}")

    print(f"\nMinutes until session change: {minutes_until_session_change()}")

    # Test entry check
    print("\nEntry Check (signal_strength=1.2, strategy=SPREAD):")
    allowed, reason = should_allow_entry(1.2, "SPREAD")
    print(f"  Allowed: {allowed}")
    if reason:
        print(f"  Reason: {reason}")

    # All sessions
    print("\nAll Sessions:")
    for s in Session:
        cfg = DEFAULT_CONFIGS.get(s)
        if cfg:
            status = "✓" if cfg.allow_new_entries else "✗"
            print(f"  {status} {s.value:15} - {cfg.notes[:40]}")
