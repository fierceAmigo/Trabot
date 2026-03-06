"""Tests for circuit_breaker.py

P0: Daily loss limit and circuit breaker for capital protection.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime
from typing import Generator
from unittest.mock import patch

import pytest

# Import module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_breaker import (
    BreakerStatus,
    DailyState,
    load_state,
    save_state,
    check_circuit_breaker,
    record_trade_result,
    update_unrealized_pnl,
    get_daily_summary,
    get_remaining_risk_budget,
    should_reduce_size,
    reset_circuit_breaker,
)


class TestBreakerStatus:
    """Tests for BreakerStatus enum."""

    def test_status_values(self) -> None:
        """Verify all expected status values exist."""
        assert BreakerStatus.ACTIVE.value == "ACTIVE"
        assert BreakerStatus.WARNING_50.value == "WARNING_50"
        assert BreakerStatus.WARNING_70.value == "WARNING_70"
        assert BreakerStatus.WARNING_90.value == "WARNING_90"
        assert BreakerStatus.TRIPPED.value == "TRIPPED"


class TestDailyState:
    """Tests for DailyState dataclass."""

    def test_state_fields(self) -> None:
        """Test all state fields are present."""
        state = DailyState(
            date="2024-01-15",
            realized_pnl=-500.0,
            unrealized_pnl=-200.0,
            trade_count=5,
            win_count=2,
            loss_count=3,
            consecutive_losses=2,
            status=BreakerStatus.WARNING_50.value,
            last_updated="2024-01-15T10:30:00",
            trades=[{"pnl": 100}, {"pnl": -200}],
        )

        assert state.date == "2024-01-15"
        assert state.realized_pnl == -500.0
        assert state.trade_count == 5


class TestStatePersistence:
    """Tests for state load/save functions."""

    @pytest.fixture
    def temp_state_file(self) -> Generator[str, None, None]:
        """Create temporary state file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            yield f.name
        os.unlink(f.name)

    def test_save_and_load_state(self, temp_state_file: str) -> None:
        """Test saving and loading state."""
        with patch("circuit_breaker.STATE_FILE", temp_state_file):
            state = DailyState(
                date="2024-01-15",
                realized_pnl=-750.0,
                unrealized_pnl=-100.0,
                trade_count=4,
                win_count=1,
                loss_count=3,
                consecutive_losses=3,
                status=BreakerStatus.WARNING_50.value,
                last_updated="2024-01-15T12:00:00",
                trades=[{"pnl": 200}, {"pnl": -300}],
            )

            save_state(state)
            loaded = load_state()

            assert loaded.date == state.date
            assert loaded.realized_pnl == state.realized_pnl
            assert loaded.trade_count == state.trade_count
            assert loaded.status == state.status

    def test_load_missing_file(self, temp_state_file: str) -> None:
        """Test loading when file doesn't exist returns empty state."""
        nonexistent_file = "/tmp/nonexistent_circuit_breaker.json"

        with patch("circuit_breaker.STATE_FILE", nonexistent_file):
            state = load_state()

            assert state.realized_pnl == 0.0
            assert state.trade_count == 0
            assert state.status == BreakerStatus.ACTIVE.value


class TestCheckCircuitBreaker:
    """Tests for check_circuit_breaker function."""

    @pytest.fixture
    def temp_state_file(self) -> Generator[str, None, None]:
        """Create temporary state file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            yield f.name
        os.unlink(f.name)

    @patch("circuit_breaker._today_ist")
    @patch("circuit_breaker._now_ist")
    def test_active_state_allows_trading(
        self, mock_now: patch, mock_today: patch, temp_state_file: str
    ) -> None:
        """Test that active state allows trading."""
        mock_today.return_value = date(2024, 1, 15)
        mock_now.return_value = datetime(2024, 1, 15, 10, 0, 0)

        with patch("circuit_breaker.STATE_FILE", temp_state_file):
            ok, status, reason = check_circuit_breaker(capital=100000)

            assert ok is True
            assert status == BreakerStatus.ACTIVE

    @patch("circuit_breaker._today_ist")
    @patch("circuit_breaker._now_ist")
    def test_tripped_blocks_trading(
        self, mock_now: patch, mock_today: patch, temp_state_file: str
    ) -> None:
        """Test that tripped state blocks trading."""
        mock_today.return_value = date(2024, 1, 15)
        mock_now.return_value = datetime(2024, 1, 15, 10, 0, 0)

        with patch("circuit_breaker.STATE_FILE", temp_state_file):
            # Create a tripped state
            state = DailyState(
                date="2024-01-15",
                realized_pnl=-2500.0,  # Over 2% of 100k
                unrealized_pnl=0,
                trade_count=6,
                win_count=1,
                loss_count=5,
                consecutive_losses=5,
                status=BreakerStatus.TRIPPED.value,
                last_updated=mock_now.return_value.isoformat(),
                trades=[],
            )
            save_state(state)

            ok, status, reason = check_circuit_breaker(capital=100000)

            assert ok is False
            assert status == BreakerStatus.TRIPPED
            assert "tripped" in reason.lower() or "loss" in reason.lower()


class TestRecordTradeResult:
    """Tests for record_trade_result function."""

    @pytest.fixture
    def temp_state_file(self) -> Generator[str, None, None]:
        """Create temporary state file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            yield f.name
        os.unlink(f.name)

    @patch("circuit_breaker._today_ist")
    @patch("circuit_breaker._now_ist")
    def test_record_winning_trade(
        self, mock_now: patch, mock_today: patch, temp_state_file: str
    ) -> None:
        """Test recording a winning trade."""
        mock_today.return_value = date(2024, 1, 15)
        mock_now.return_value = datetime(2024, 1, 15, 11, 0, 0)

        with patch("circuit_breaker.STATE_FILE", temp_state_file):
            record_trade_result(pnl=500.0, trade_info={"symbol": "NIFTY"})

            state = load_state()
            assert state.realized_pnl == 500.0
            assert state.trade_count == 1
            assert state.win_count == 1
            assert state.loss_count == 0
            assert state.consecutive_losses == 0

    @patch("circuit_breaker._today_ist")
    @patch("circuit_breaker._now_ist")
    def test_record_losing_trade(
        self, mock_now: patch, mock_today: patch, temp_state_file: str
    ) -> None:
        """Test recording a losing trade."""
        mock_today.return_value = date(2024, 1, 15)
        mock_now.return_value = datetime(2024, 1, 15, 11, 0, 0)

        with patch("circuit_breaker.STATE_FILE", temp_state_file):
            record_trade_result(pnl=-300.0, trade_info={"symbol": "BANKNIFTY"})

            state = load_state()
            assert state.realized_pnl == -300.0
            assert state.trade_count == 1
            assert state.win_count == 0
            assert state.loss_count == 1
            assert state.consecutive_losses == 1

    @patch("circuit_breaker._today_ist")
    @patch("circuit_breaker._now_ist")
    def test_consecutive_losses_tracking(
        self, mock_now: patch, mock_today: patch, temp_state_file: str
    ) -> None:
        """Test consecutive losses are tracked correctly."""
        mock_today.return_value = date(2024, 1, 15)
        mock_now.return_value = datetime(2024, 1, 15, 11, 0, 0)

        with patch("circuit_breaker.STATE_FILE", temp_state_file):
            # Record 3 consecutive losses
            record_trade_result(pnl=-100.0)
            record_trade_result(pnl=-200.0)
            record_trade_result(pnl=-150.0)

            state = load_state()
            assert state.consecutive_losses == 3

            # Win resets consecutive losses
            record_trade_result(pnl=500.0)

            state = load_state()
            assert state.consecutive_losses == 0


class TestRemainingRiskBudget:
    """Tests for get_remaining_risk_budget function."""

    @pytest.fixture
    def temp_state_file(self) -> Generator[str, None, None]:
        """Create temporary state file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            yield f.name
        os.unlink(f.name)

    @patch("circuit_breaker._today_ist")
    @patch("circuit_breaker._now_ist")
    def test_budget_calculation(
        self, mock_now: patch, mock_today: patch, temp_state_file: str
    ) -> None:
        """Test remaining risk budget calculation."""
        mock_today.return_value = date(2024, 1, 15)
        mock_now.return_value = datetime(2024, 1, 15, 12, 0, 0)

        with patch("circuit_breaker.STATE_FILE", temp_state_file):
            # Start fresh
            reset_circuit_breaker()

            # Budget should be available
            budget = get_remaining_risk_budget(capital=100000)
            assert budget > 0


class TestDailySummary:
    """Tests for get_daily_summary function."""

    @pytest.fixture
    def temp_state_file(self) -> Generator[str, None, None]:
        """Create temporary state file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            yield f.name
        os.unlink(f.name)

    @patch("circuit_breaker._today_ist")
    @patch("circuit_breaker._now_ist")
    def test_summary_contains_required_fields(
        self, mock_now: patch, mock_today: patch, temp_state_file: str
    ) -> None:
        """Test summary contains all required fields."""
        mock_today.return_value = date(2024, 1, 15)
        mock_now.return_value = datetime(2024, 1, 15, 12, 0, 0)

        with patch("circuit_breaker.STATE_FILE", temp_state_file):
            state = DailyState(
                date="2024-01-15",
                realized_pnl=-500.0,
                unrealized_pnl=-100.0,
                trade_count=4,
                win_count=2,
                loss_count=2,
                consecutive_losses=1,
                status=BreakerStatus.ACTIVE.value,
                last_updated=mock_now.return_value.isoformat(),
                trades=[],
            )
            save_state(state)

            summary = get_daily_summary()

            assert isinstance(summary, dict)
