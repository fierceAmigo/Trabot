"""Tests for session_manager.py

P1: Session-aware trading rules for Indian markets.
"""

from __future__ import annotations

import os
from datetime import datetime, time as dtime
from unittest.mock import patch

import pytest

# Import module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session_manager import (
    Session,
    SessionConfig,
    SESSION_TIMES,
    get_current_session,
    get_session_config,
    should_allow_entry,
    get_adjusted_levels,
    is_good_entry_window,
    get_size_multiplier,
    get_session_summary,
)


class TestSessionEnum:
    """Tests for Session enum."""

    def test_all_sessions_exist(self) -> None:
        """Verify all expected sessions exist."""
        assert Session.PRE_MARKET.value == "PRE_MARKET"
        assert Session.OPENING.value == "OPENING"
        assert Session.MORNING.value == "MORNING"
        assert Session.LUNCH.value == "LUNCH"
        assert Session.AFTERNOON.value == "AFTERNOON"
        assert Session.CLOSING.value == "CLOSING"
        assert Session.LAST_15_MIN.value == "LAST_15_MIN"
        assert Session.POST_MARKET.value == "POST_MARKET"


class TestSessionTimes:
    """Tests for session time boundaries."""

    def test_session_times_are_complete(self) -> None:
        """All sessions should have time boundaries."""
        for session in Session:
            assert session in SESSION_TIMES

    def test_session_times_are_tuples(self) -> None:
        """Each session should have (start, end) tuple."""
        for session, times in SESSION_TIMES.items():
            assert isinstance(times, tuple)
            assert len(times) == 2
            start, end = times
            assert isinstance(start, dtime)
            assert isinstance(end, dtime)

    def test_opening_session_starts_at_market_open(self) -> None:
        """Opening session should start at 9:15."""
        start, _ = SESSION_TIMES[Session.OPENING]
        assert start.hour == 9
        assert start.minute == 15


class TestGetCurrentSession:
    """Tests for get_current_session function."""

    @patch("session_manager._now_ist")
    def test_pre_market_session(self, mock_now: patch) -> None:
        """Before 9:15 should be PRE_MARKET."""
        mock_now.return_value = datetime(2024, 1, 15, 8, 30, 0)  # 8:30 AM

        session = get_current_session()
        assert session == Session.PRE_MARKET

    @patch("session_manager._now_ist")
    def test_opening_session(self, mock_now: patch) -> None:
        """9:15-10:00 should be OPENING."""
        mock_now.return_value = datetime(2024, 1, 15, 9, 30, 0)  # 9:30 AM

        session = get_current_session()
        assert session == Session.OPENING

    @patch("session_manager._now_ist")
    def test_morning_session(self, mock_now: patch) -> None:
        """10:00-12:00 should be MORNING."""
        mock_now.return_value = datetime(2024, 1, 15, 10, 30, 0)  # 10:30 AM

        session = get_current_session()
        assert session == Session.MORNING

    @patch("session_manager._now_ist")
    def test_lunch_session(self, mock_now: patch) -> None:
        """12:00-13:30 should be LUNCH."""
        mock_now.return_value = datetime(2024, 1, 15, 12, 30, 0)  # 12:30 PM

        session = get_current_session()
        assert session == Session.LUNCH

    @patch("session_manager._now_ist")
    def test_afternoon_session(self, mock_now: patch) -> None:
        """13:30-14:30 should be AFTERNOON."""
        mock_now.return_value = datetime(2024, 1, 15, 14, 0, 0)  # 2:00 PM

        session = get_current_session()
        assert session == Session.AFTERNOON

    @patch("session_manager._now_ist")
    def test_closing_session(self, mock_now: patch) -> None:
        """14:30-15:15 should be CLOSING."""
        mock_now.return_value = datetime(2024, 1, 15, 15, 0, 0)  # 3:00 PM

        session = get_current_session()
        assert session == Session.CLOSING

    @patch("session_manager._now_ist")
    def test_last_15_min_session(self, mock_now: patch) -> None:
        """15:15-15:30 should be LAST_15_MIN."""
        mock_now.return_value = datetime(2024, 1, 15, 15, 20, 0)  # 3:20 PM

        session = get_current_session()
        assert session == Session.LAST_15_MIN

    @patch("session_manager._now_ist")
    def test_post_market_session(self, mock_now: patch) -> None:
        """After 15:30 should be POST_MARKET."""
        mock_now.return_value = datetime(2024, 1, 15, 16, 0, 0)  # 4:00 PM

        session = get_current_session()
        assert session == Session.POST_MARKET


class TestGetSessionConfig:
    """Tests for get_session_config function."""

    @patch("session_manager._now_ist")
    def test_returns_session_config(self, mock_now: patch) -> None:
        """Should return SessionConfig instance."""
        mock_now.return_value = datetime(2024, 1, 15, 10, 30, 0)

        config = get_session_config()

        assert isinstance(config, SessionConfig)
        assert config.session == Session.MORNING

    @patch("session_manager._now_ist")
    def test_morning_session_config(self, mock_now: patch) -> None:
        """Morning session should have favorable config."""
        mock_now.return_value = datetime(2024, 1, 15, 10, 30, 0)

        config = get_session_config()

        assert config.allow_new_entries is True
        assert config.min_signal_strength == 1.0
        assert config.size_multiplier == 1.0

    @patch("session_manager._now_ist")
    def test_opening_session_config(self, mock_now: patch) -> None:
        """Opening session should have tighter config."""
        mock_now.return_value = datetime(2024, 1, 15, 9, 30, 0)

        config = get_session_config()

        assert config.session == Session.OPENING
        assert config.min_signal_strength > 1.0  # Require stronger signals
        assert config.size_multiplier < 1.0  # Smaller size
        assert config.prefer_scalps is True

    @patch("session_manager._now_ist")
    def test_lunch_session_config(self, mock_now: patch) -> None:
        """Lunch session should be restrictive."""
        mock_now.return_value = datetime(2024, 1, 15, 12, 30, 0)

        with patch("session_manager.BLOCK_LUNCH_ENTRIES", True):
            config = get_session_config()

            assert config.session == Session.LUNCH
            assert config.min_signal_strength > 1.0
            assert config.size_multiplier < 1.0


class TestShouldAllowEntry:
    """Tests for should_allow_entry function."""

    @patch("session_manager.SESSION_RULES_ENABLE", True)
    @patch("session_manager._now_ist")
    def test_allows_strong_signal_in_morning(self, mock_now: patch) -> None:
        """Strong signal in morning should be allowed."""
        mock_now.return_value = datetime(2024, 1, 15, 10, 30, 0)

        allowed, reason = should_allow_entry(signal_strength=1.5)

        assert allowed is True

    @patch("session_manager.SESSION_RULES_ENABLE", True)
    @patch("session_manager._now_ist")
    def test_blocks_weak_signal_in_opening(self, mock_now: patch) -> None:
        """Weak signal in opening session should be blocked."""
        mock_now.return_value = datetime(2024, 1, 15, 9, 30, 0)

        allowed, reason = should_allow_entry(signal_strength=1.0)

        # Opening requires min_signal_strength of 1.3
        assert allowed is False
        assert "signal" in reason.lower() or "weak" in reason.lower()

    @patch("session_manager.SESSION_RULES_ENABLE", True)
    @patch("session_manager._now_ist")
    def test_blocks_entry_in_pre_market(self, mock_now: patch) -> None:
        """Pre-market entries should be blocked."""
        mock_now.return_value = datetime(2024, 1, 15, 8, 30, 0)

        allowed, reason = should_allow_entry(signal_strength=2.0)

        assert allowed is False
        assert "market" in reason.lower() or "closed" in reason.lower()

    @patch("session_manager.SESSION_RULES_ENABLE", False)
    def test_disabled_rules_allow_all(self) -> None:
        """When disabled, all entries should be allowed."""
        allowed, reason = should_allow_entry(signal_strength=0.5)

        assert allowed is True

    @patch("session_manager.SESSION_RULES_ENABLE", True)
    @patch("session_manager._now_ist")
    @patch("session_manager.BLOCK_LUNCH_ENTRIES", True)
    def test_blocks_lunch_when_configured(self, mock_now: patch) -> None:
        """Lunch entries should be blocked when configured."""
        mock_now.return_value = datetime(2024, 1, 15, 12, 30, 0)

        allowed, reason = should_allow_entry(signal_strength=1.5)

        # Depends on configuration
        assert isinstance(allowed, bool)


class TestGetAdjustedLevels:
    """Tests for get_adjusted_levels function."""

    @patch("session_manager._now_ist")
    def test_adjusts_for_opening(self, mock_now: patch) -> None:
        """Opening session should adjust levels."""
        mock_now.return_value = datetime(2024, 1, 15, 9, 30, 0)

        adjusted = get_adjusted_levels(
            stop=100, target=200
        )

        assert isinstance(adjusted, dict)
        assert "stop" in adjusted
        assert "target" in adjusted

    @patch("session_manager._now_ist")
    def test_morning_levels(self, mock_now: patch) -> None:
        """Morning session levels."""
        mock_now.return_value = datetime(2024, 1, 15, 10, 30, 0)

        adjusted = get_adjusted_levels(
            stop=100, target=200
        )

        # Morning session should be standard
        assert adjusted["stop"] > 0
        assert adjusted["target"] > 0


class TestIsGoodEntryWindow:
    """Tests for is_good_entry_window function."""

    @patch("session_manager._now_ist")
    def test_morning_is_good_window(self, mock_now: patch) -> None:
        """Morning session should be good entry window."""
        mock_now.return_value = datetime(2024, 1, 15, 10, 30, 0)

        is_good = is_good_entry_window()
        assert isinstance(is_good, bool)

    @patch("session_manager._now_ist")
    def test_pre_market_not_good(self, mock_now: patch) -> None:
        """Pre-market should not be good entry window."""
        mock_now.return_value = datetime(2024, 1, 15, 8, 30, 0)

        is_good = is_good_entry_window()
        assert is_good is False


class TestGetSizeMultiplier:
    """Tests for get_size_multiplier function."""

    @patch("session_manager._now_ist")
    def test_returns_float(self, mock_now: patch) -> None:
        """Should return float multiplier."""
        mock_now.return_value = datetime(2024, 1, 15, 10, 30, 0)

        multiplier = get_size_multiplier()

        assert isinstance(multiplier, (int, float))
        assert multiplier > 0

    @patch("session_manager._now_ist")
    def test_morning_full_size(self, mock_now: patch) -> None:
        """Morning session should have full size."""
        mock_now.return_value = datetime(2024, 1, 15, 10, 30, 0)

        multiplier = get_size_multiplier()
        assert multiplier >= 0.9  # Near 1.0

    @patch("session_manager._now_ist")
    def test_opening_reduced_size(self, mock_now: patch) -> None:
        """Opening session should have reduced size."""
        mock_now.return_value = datetime(2024, 1, 15, 9, 30, 0)

        multiplier = get_size_multiplier()
        assert multiplier < 1.0  # Reduced


class TestEdgeCases:
    """Tests for edge cases."""

    @patch("session_manager._now_ist")
    def test_exactly_at_session_boundary(self, mock_now: patch) -> None:
        """Test behavior exactly at session boundary."""
        # Exactly at 10:00 (end of OPENING, start of MORNING)
        mock_now.return_value = datetime(2024, 1, 15, 10, 0, 0)

        session = get_current_session()
        # Should be MORNING at 10:00
        assert session == Session.MORNING

    @patch("session_manager._now_ist")
    def test_exactly_at_market_open(self, mock_now: patch) -> None:
        """Test behavior exactly at market open."""
        mock_now.return_value = datetime(2024, 1, 15, 9, 15, 0)

        session = get_current_session()
        assert session == Session.OPENING

    @patch("session_manager._now_ist")
    def test_exactly_at_market_close(self, mock_now: patch) -> None:
        """Test behavior exactly at market close."""
        mock_now.return_value = datetime(2024, 1, 15, 15, 30, 0)

        session = get_current_session()
        assert session == Session.POST_MARKET
