"""event_calendar.py

P0: Event calendar for avoiding earnings, RBI policy, and expiry events.

This module helps avoid entering positions that would span high-impact events
where IV crush or gap risk could cause significant losses.

Event Types:
- EARNINGS: Company quarterly results
- RBI_POLICY: RBI monetary policy announcements
- MONTHLY_EXPIRY: Last Thursday of month (F&O expiry)
- WEEKLY_EXPIRY: Thursday weekly expiry (for indices)
- BUDGET: Union budget day
- CUSTOM: User-defined events

Data Sources:
- data/events.json: Manual/fetched event data
- Computed: Expiry dates (algorithmic)

Usage:
    from event_calendar import should_avoid_entry, get_upcoming_events

    if should_avoid_entry(underlying, date, dte):
        return None  # Skip this recommendation

Environment Variables:
- TRABOT_EVENT_CHECK_ENABLE: 0/1 (default 1)
- TRABOT_EVENT_EARNINGS_BUFFER_HRS: Hours before/after earnings (default 24)
- TRABOT_EVENT_RBI_BUFFER_HRS: Hours before/after RBI (default 12)
- TRABOT_EVENT_EXPIRY_BUFFER_HRS: Hours before expiry (default 4)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum

try:
    from dateutil import tz
    IST = tz.gettz("Asia/Kolkata")
except ImportError:
    IST = None


class EventType(Enum):
    EARNINGS = "EARNINGS"
    RBI_POLICY = "RBI_POLICY"
    MONTHLY_EXPIRY = "MONTHLY_EXPIRY"
    WEEKLY_EXPIRY = "WEEKLY_EXPIRY"
    BUDGET = "BUDGET"
    FED_FOMC = "FED_FOMC"
    CUSTOM = "CUSTOM"


@dataclass
class Event:
    underlying: str  # "NIFTY", "RELIANCE", "ALL" for market-wide
    event_type: EventType
    event_date: date
    event_time: Optional[str]  # "14:00" or None for all-day
    description: str
    buffer_hours_before: int
    buffer_hours_after: int


# Environment configuration
EVENT_CHECK_ENABLE = os.getenv("TRABOT_EVENT_CHECK_ENABLE", "1").strip() == "1"
EARNINGS_BUFFER_HRS = int(os.getenv("TRABOT_EVENT_EARNINGS_BUFFER_HRS", "24"))
RBI_BUFFER_HRS = int(os.getenv("TRABOT_EVENT_RBI_BUFFER_HRS", "12"))
EXPIRY_BUFFER_HRS = int(os.getenv("TRABOT_EVENT_EXPIRY_BUFFER_HRS", "4"))
BUDGET_BUFFER_HRS = int(os.getenv("TRABOT_EVENT_BUDGET_BUFFER_HRS", "24"))

# Default events file path
EVENTS_FILE = os.getenv("TRABOT_EVENTS_FILE", "data/events.json")


def _now_ist() -> datetime:
    """Get current time in IST."""
    if IST:
        return datetime.now(IST)
    return datetime.now()


def _date_to_datetime(d: date, time_str: Optional[str] = None) -> datetime:
    """Convert date to datetime, optionally with time."""
    if time_str:
        try:
            h, m = map(int, time_str.split(":"))
            return datetime(d.year, d.month, d.day, h, m)
        except (ValueError, AttributeError):
            pass
    return datetime(d.year, d.month, d.day, 9, 15)  # Default market open


def _last_thursday(year: int, month: int) -> date:
    """Get last Thursday of a month (monthly expiry)."""
    # Start from last day of month
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)

    # Find last Thursday (weekday 3)
    days_since_thursday = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=days_since_thursday)


def _thursdays_in_month(year: int, month: int) -> List[date]:
    """Get all Thursdays in a month (weekly expiries)."""
    thursdays = []
    # Start from first day
    d = date(year, month, 1)
    # Find first Thursday
    days_until_thursday = (3 - d.weekday()) % 7
    d = d + timedelta(days=days_until_thursday)

    while d.month == month:
        thursdays.append(d)
        d = d + timedelta(days=7)

    return thursdays


def _generate_expiry_events(start_date: date, months_ahead: int = 3) -> List[Event]:
    """Generate expiry events for upcoming months."""
    events = []
    current = start_date

    for _ in range(months_ahead):
        year, month = current.year, current.month

        # Monthly expiry
        monthly = _last_thursday(year, month)
        if monthly >= start_date:
            events.append(Event(
                underlying="ALL",
                event_type=EventType.MONTHLY_EXPIRY,
                event_date=monthly,
                event_time="15:30",
                description=f"Monthly F&O Expiry {monthly.strftime('%b %Y')}",
                buffer_hours_before=EXPIRY_BUFFER_HRS,
                buffer_hours_after=0,
            ))

        # Weekly expiries (for indices like NIFTY, BANKNIFTY)
        for thursday in _thursdays_in_month(year, month):
            if thursday >= start_date and thursday != monthly:
                events.append(Event(
                    underlying="NIFTY",  # Weekly expiry primarily for indices
                    event_type=EventType.WEEKLY_EXPIRY,
                    event_date=thursday,
                    event_time="15:30",
                    description=f"Weekly Expiry {thursday.strftime('%d %b')}",
                    buffer_hours_before=EXPIRY_BUFFER_HRS,
                    buffer_hours_after=0,
                ))

        # Move to next month
        if month == 12:
            current = date(year + 1, 1, 1)
        else:
            current = date(year, month + 1, 1)

    return events


def _load_events_from_file(filepath: str) -> List[Event]:
    """Load events from JSON file."""
    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        events = []
        for item in data.get("events", []):
            try:
                event_type = EventType(item.get("type", "CUSTOM"))
                event_date = date.fromisoformat(item["date"])

                # Get buffer hours based on event type
                if event_type == EventType.EARNINGS:
                    buffer_before = item.get("buffer_hours_before", EARNINGS_BUFFER_HRS)
                    buffer_after = item.get("buffer_hours_after", EARNINGS_BUFFER_HRS)
                elif event_type == EventType.RBI_POLICY:
                    buffer_before = item.get("buffer_hours_before", RBI_BUFFER_HRS)
                    buffer_after = item.get("buffer_hours_after", RBI_BUFFER_HRS)
                elif event_type == EventType.BUDGET:
                    buffer_before = item.get("buffer_hours_before", BUDGET_BUFFER_HRS)
                    buffer_after = item.get("buffer_hours_after", BUDGET_BUFFER_HRS)
                else:
                    buffer_before = item.get("buffer_hours_before", 12)
                    buffer_after = item.get("buffer_hours_after", 12)

                events.append(Event(
                    underlying=item.get("underlying", "ALL").upper(),
                    event_type=event_type,
                    event_date=event_date,
                    event_time=item.get("time"),
                    description=item.get("description", ""),
                    buffer_hours_before=buffer_before,
                    buffer_hours_after=buffer_after,
                ))
            except (KeyError, ValueError) as e:
                continue  # Skip malformed entries

        return events
    except (json.JSONDecodeError, IOError):
        return []


def load_all_events(include_computed: bool = True) -> List[Event]:
    """Load all events from file and compute expiry dates."""
    events = _load_events_from_file(EVENTS_FILE)

    if include_computed:
        today = _now_ist().date()
        expiry_events = _generate_expiry_events(today, months_ahead=3)
        events.extend(expiry_events)

    return events


def get_events_for_underlying(
    underlying: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Event]:
    """Get events affecting a specific underlying."""
    underlying = underlying.upper()
    events = load_all_events()

    if start_date is None:
        start_date = _now_ist().date()
    if end_date is None:
        end_date = start_date + timedelta(days=30)

    filtered = []
    for e in events:
        # Check if event applies to this underlying
        if e.underlying not in (underlying, "ALL"):
            # Special case: Index events affect index-linked underlyings
            if e.underlying == "NIFTY" and underlying in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
                pass
            else:
                continue

        # Check date range
        if e.event_date < start_date or e.event_date > end_date:
            continue

        filtered.append(e)

    return sorted(filtered, key=lambda x: x.event_date)


def is_within_event_window(
    event: Event,
    check_time: datetime,
) -> Tuple[bool, str]:
    """Check if a time is within an event's buffer window."""
    event_dt = _date_to_datetime(event.event_date, event.event_time)

    window_start = event_dt - timedelta(hours=event.buffer_hours_before)
    window_end = event_dt + timedelta(hours=event.buffer_hours_after)

    # Make check_time timezone-naive for comparison
    if hasattr(check_time, 'replace') and check_time.tzinfo:
        check_time = check_time.replace(tzinfo=None)

    if window_start <= check_time <= window_end:
        return True, f"{event.event_type.value}: {event.description}"

    return False, ""


def position_spans_event(
    underlying: str,
    entry_date: date,
    dte: int,
) -> Tuple[bool, Optional[Event]]:
    """Check if a position would span any high-impact event.

    Returns (spans_event, event) where event is the first conflicting event.
    """
    underlying = underlying.upper()
    expiry_date = entry_date + timedelta(days=dte)

    events = get_events_for_underlying(
        underlying,
        start_date=entry_date,
        end_date=expiry_date,
    )

    # Only block for high-impact events
    blocking_types = {
        EventType.EARNINGS,
        EventType.RBI_POLICY,
        EventType.BUDGET,
        EventType.FED_FOMC,
    }

    for event in events:
        if event.event_type in blocking_types:
            return True, event

    return False, None


def should_avoid_entry(
    underlying: str,
    entry_time: Optional[datetime] = None,
    dte: int = 7,
) -> Tuple[bool, str]:
    """Main function: Should we avoid entering this position?

    Checks:
    1. Is current time within any event's buffer window?
    2. Would the position span any high-impact event?

    Returns (should_avoid, reason)
    """
    if not EVENT_CHECK_ENABLE:
        return False, ""

    underlying = underlying.upper()

    if entry_time is None:
        entry_time = _now_ist()

    entry_date = entry_time.date() if hasattr(entry_time, 'date') else entry_time

    # Check 1: Current time within event window
    events = load_all_events()
    for event in events:
        if event.underlying not in (underlying, "ALL", "NIFTY"):
            continue

        in_window, reason = is_within_event_window(event, entry_time)
        if in_window:
            return True, f"Within event window: {reason}"

    # Check 2: Position would span high-impact event
    spans, event = position_spans_event(underlying, entry_date, dte)
    if spans and event:
        return True, f"Position spans {event.event_type.value}: {event.description} on {event.event_date}"

    return False, ""


def get_upcoming_events(
    underlying: Optional[str] = None,
    days_ahead: int = 14,
) -> List[Dict]:
    """Get upcoming events for display/logging."""
    start = _now_ist().date()
    end = start + timedelta(days=days_ahead)

    if underlying:
        events = get_events_for_underlying(underlying, start, end)
    else:
        events = [e for e in load_all_events()
                  if start <= e.event_date <= end]

    return [
        {
            "underlying": e.underlying,
            "type": e.event_type.value,
            "date": e.event_date.isoformat(),
            "time": e.event_time or "All Day",
            "description": e.description,
            "days_away": (e.event_date - start).days,
        }
        for e in sorted(events, key=lambda x: x.event_date)
    ]


def save_event(
    underlying: str,
    event_type: str,
    event_date: str,
    description: str,
    event_time: Optional[str] = None,
) -> bool:
    """Add a new event to the events file."""
    os.makedirs(os.path.dirname(EVENTS_FILE) or "data", exist_ok=True)

    # Load existing
    try:
        if os.path.exists(EVENTS_FILE):
            with open(EVENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"events": []}
    except (json.JSONDecodeError, IOError):
        data = {"events": []}

    # Add new event
    data["events"].append({
        "underlying": underlying.upper(),
        "type": event_type,
        "date": event_date,
        "time": event_time,
        "description": description,
    })

    # Save
    try:
        with open(EVENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except IOError:
        return False


# Pre-populated RBI policy dates for 2024-2025 (update periodically)
RBI_POLICY_DATES_2024_2025 = [
    # 2024
    "2024-02-08", "2024-04-05", "2024-06-07",
    "2024-08-08", "2024-10-09", "2024-12-06",
    # 2025
    "2025-02-07", "2025-04-09", "2025-06-06",
    "2025-08-06", "2025-10-08", "2025-12-05",
    # 2026
    "2026-02-06", "2026-04-08", "2026-06-05",
]


def initialize_default_events():
    """Initialize events file with known market events."""
    if os.path.exists(EVENTS_FILE):
        return  # Don't overwrite existing

    events = []

    # Add RBI policy dates
    for date_str in RBI_POLICY_DATES_2024_2025:
        events.append({
            "underlying": "ALL",
            "type": "RBI_POLICY",
            "date": date_str,
            "time": "10:00",
            "description": f"RBI Monetary Policy {date_str}",
        })

    # Budget days (approximate - update annually)
    events.append({
        "underlying": "ALL",
        "type": "BUDGET",
        "date": "2025-02-01",
        "time": "11:00",
        "description": "Union Budget 2025-26",
    })
    events.append({
        "underlying": "ALL",
        "type": "BUDGET",
        "date": "2026-02-01",
        "time": "11:00",
        "description": "Union Budget 2026-27",
    })

    os.makedirs(os.path.dirname(EVENTS_FILE) or "data", exist_ok=True)
    with open(EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"events": events}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Initialize default events if not present
    initialize_default_events()

    # Demo: Show upcoming events
    print("Upcoming Events (next 14 days):")
    print("-" * 60)
    for e in get_upcoming_events(days_ahead=14):
        print(f"  {e['date']} | {e['type']:15} | {e['underlying']:10} | {e['description']}")

    # Demo: Check if should avoid entry
    print("\nEntry Check for RELIANCE (7 DTE):")
    avoid, reason = should_avoid_entry("RELIANCE", dte=7)
    print(f"  Avoid: {avoid}, Reason: {reason or 'None'}")
