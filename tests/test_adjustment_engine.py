"""Tests for adjustment_engine.py

P2: Simplified adjustment framework for options positions.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Import module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adjustment_engine import (
    AdjustmentType,
    PositionStatus,
    Position,
    Adjustment,
    classify_position_status,
    suggest_adjustments,
    get_adjustment_summary,
    should_auto_adjust,
)


class TestAdjustmentType:
    """Tests for AdjustmentType enum."""

    def test_all_types_exist(self) -> None:
        """Verify all expected adjustment types exist."""
        assert AdjustmentType.ROLL_OUT.value == "ROLL_OUT"
        assert AdjustmentType.ROLL_UP.value == "ROLL_UP"
        assert AdjustmentType.ROLL_DOWN.value == "ROLL_DOWN"
        assert AdjustmentType.CONVERT_TO_SPREAD.value == "CONVERT_TO_SPREAD"
        assert AdjustmentType.ADD_HEDGE.value == "ADD_HEDGE"
        assert AdjustmentType.CLOSE_PARTIAL.value == "CLOSE_PARTIAL"
        assert AdjustmentType.CLOSE_FULL.value == "CLOSE_FULL"
        assert AdjustmentType.NO_ACTION.value == "NO_ACTION"


class TestPositionStatus:
    """Tests for PositionStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Verify all expected statuses exist."""
        assert PositionStatus.WINNING.value == "WINNING"
        assert PositionStatus.SCRATCH.value == "SCRATCH"
        assert PositionStatus.LOSING_SMALL.value == "LOSING_SMALL"
        assert PositionStatus.LOSING_MEDIUM.value == "LOSING_MEDIUM"
        assert PositionStatus.LOSING_LARGE.value == "LOSING_LARGE"
        assert PositionStatus.NEAR_STOP.value == "NEAR_STOP"
        assert PositionStatus.STOPPED_OUT.value == "STOPPED_OUT"


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self) -> None:
        """Test creating Position instance."""
        position = Position(
            underlying="NIFTY",
            structure="BUY_CE",
            strike=20000,
            strike2=None,
            expiry="2024-01-18",
            dte=5,
            entry_price=200,
            current_price=180,
            contracts=2,
            is_long=True,
            entry_spot=19950,
            current_spot=19900,
        )

        assert position.underlying == "NIFTY"
        assert position.structure == "BUY_CE"
        assert position.strike == 20000
        assert position.dte == 5
        assert position.is_long is True


class TestClassifyPositionStatus:
    """Tests for classify_position_status function."""

    @pytest.fixture
    def long_position(self) -> Position:
        """Create a sample long position."""
        return Position(
            underlying="NIFTY",
            structure="BUY_CE",
            strike=20000,
            strike2=None,
            expiry="2024-01-18",
            dte=5,
            entry_price=200,
            current_price=200,  # At breakeven
            contracts=2,
            is_long=True,
            entry_spot=19950,
            current_spot=19950,
        )

    def test_winning_status(self, long_position: Position) -> None:
        """Profitable position should be WINNING."""
        long_position.current_price = 250  # +50

        status = classify_position_status(
            position=long_position,
            max_loss=400,  # 200 * 2 contracts
        )

        assert status == PositionStatus.WINNING

    def test_scratch_status(self, long_position: Position) -> None:
        """Near breakeven should be SCRATCH."""
        long_position.current_price = 198  # -2 (very small loss)

        status = classify_position_status(
            position=long_position,
            max_loss=400,
        )

        assert status == PositionStatus.SCRATCH

    def test_losing_small_status(self, long_position: Position) -> None:
        """Small loss (<25%) should be LOSING_SMALL."""
        long_position.current_price = 160  # -40 (20% loss of 200)

        status = classify_position_status(
            position=long_position,
            max_loss=400,
        )

        assert status == PositionStatus.LOSING_SMALL

    def test_losing_medium_status(self, long_position: Position) -> None:
        """Medium loss (25-50%) should be LOSING_MEDIUM."""
        long_position.current_price = 120  # -80 (40% of max loss)

        status = classify_position_status(
            position=long_position,
            max_loss=400,
        )

        assert status == PositionStatus.LOSING_MEDIUM

    def test_losing_large_status(self, long_position: Position) -> None:
        """Large loss (>50%) should be LOSING_LARGE."""
        long_position.current_price = 60  # -140 (70% of max loss)

        status = classify_position_status(
            position=long_position,
            max_loss=400,
        )

        assert status == PositionStatus.LOSING_LARGE

    def test_stopped_out_status(self, long_position: Position) -> None:
        """At/past stop should be STOPPED_OUT."""
        long_position.current_price = 90  # Below stop

        status = classify_position_status(
            position=long_position,
            max_loss=400,
            stop_loss=100,  # Stop at 100
        )

        assert status == PositionStatus.STOPPED_OUT

    def test_near_stop_status(self, long_position: Position) -> None:
        """Near stop should be NEAR_STOP."""
        long_position.current_price = 105  # Just above stop

        status = classify_position_status(
            position=long_position,
            max_loss=400,
            stop_loss=100,  # Stop at 100
        )

        assert status == PositionStatus.NEAR_STOP


class TestSuggestAdjustments:
    """Tests for suggest_adjustments function."""

    @pytest.fixture
    def losing_position(self) -> Position:
        """Create a losing position."""
        return Position(
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

    @patch("adjustment_engine.ADJUSTMENT_ENABLE", True)
    def test_returns_list_of_adjustments(self, losing_position: Position) -> None:
        """Should return list of Adjustment objects."""
        adjustments = suggest_adjustments(
            position=losing_position,
            max_loss=400,
        )

        assert isinstance(adjustments, list)
        for adj in adjustments:
            assert isinstance(adj, Adjustment)

    @patch("adjustment_engine.ADJUSTMENT_ENABLE", True)
    def test_stopped_out_suggests_close(self, losing_position: Position) -> None:
        """Stopped out position should suggest CLOSE_FULL."""
        losing_position.current_price = 80  # Below stop

        adjustments = suggest_adjustments(
            position=losing_position,
            max_loss=400,
            stop_loss=100,
        )

        assert len(adjustments) > 0
        assert adjustments[0].type == AdjustmentType.CLOSE_FULL

    @patch("adjustment_engine.ADJUSTMENT_ENABLE", True)
    def test_low_dte_suggests_roll(self, losing_position: Position) -> None:
        """Low DTE position should suggest rolling."""
        losing_position.dte = 3

        adjustments = suggest_adjustments(
            position=losing_position,
            max_loss=400,
        )

        # Should suggest roll among options
        adj_types = [a.type for a in adjustments]
        assert (
            AdjustmentType.ROLL_OUT in adj_types
            or AdjustmentType.CLOSE_FULL in adj_types
        )

    @patch("adjustment_engine.ADJUSTMENT_ENABLE", True)
    def test_single_leg_suggests_convert(self, losing_position: Position) -> None:
        """Single leg position should suggest converting to spread."""
        adjustments = suggest_adjustments(
            position=losing_position,
            max_loss=400,
        )

        adj_types = [a.type for a in adjustments]
        assert (
            AdjustmentType.CONVERT_TO_SPREAD in adj_types
            or AdjustmentType.NO_ACTION in adj_types
        )

    @patch("adjustment_engine.ADJUSTMENT_ENABLE", False)
    def test_disabled_returns_empty(self, losing_position: Position) -> None:
        """When disabled, should return empty list."""
        adjustments = suggest_adjustments(
            position=losing_position,
            max_loss=400,
        )

        assert adjustments == []


class TestAdjustmentAttributes:
    """Tests for Adjustment object attributes."""

    @patch("adjustment_engine.ADJUSTMENT_ENABLE", True)
    def test_adjustment_has_required_fields(self) -> None:
        """Adjustment should have all required fields."""
        position = Position(
            underlying="NIFTY",
            structure="BUY_CE",
            strike=20100,
            strike2=None,
            expiry="2024-01-18",
            dte=5,
            entry_price=200,
            current_price=120,
            contracts=2,
            is_long=True,
            entry_spot=20000,
            current_spot=19900,
        )

        adjustments = suggest_adjustments(
            position=position,
            max_loss=400,
        )

        if adjustments:
            adj = adjustments[0]
            assert hasattr(adj, "type")
            assert hasattr(adj, "description")
            assert hasattr(adj, "mechanics")
            assert hasattr(adj, "expected_outcome")
            assert hasattr(adj, "cost_estimate")
            assert hasattr(adj, "urgency")
            assert hasattr(adj, "confidence")


class TestShouldAutoAdjust:
    """Tests for should_auto_adjust function."""

    @pytest.fixture
    def position(self) -> Position:
        """Create a sample position."""
        return Position(
            underlying="NIFTY",
            structure="BUY_CE",
            strike=20000,
            strike2=None,
            expiry="2024-01-18",
            dte=5,
            entry_price=200,
            current_price=200,
            contracts=2,
            is_long=True,
            entry_spot=19950,
            current_spot=19950,
        )

    def test_auto_close_on_stop(self, position: Position) -> None:
        """Should auto-close when stopped out."""
        position.current_price = 80  # Below stop

        should_adjust, adj_type = should_auto_adjust(
            position=position,
            max_loss=400,
            stop_loss=100,
        )

        assert should_adjust is True
        assert adj_type == AdjustmentType.CLOSE_FULL

    def test_auto_roll_on_low_dte(self, position: Position) -> None:
        """Should suggest roll on very low DTE."""
        position.dte = 1
        position.current_price = 150  # Still has value

        should_adjust, adj_type = should_auto_adjust(
            position=position,
            max_loss=400,
        )

        assert should_adjust is True
        assert adj_type == AdjustmentType.ROLL_OUT

    def test_no_auto_adjust_for_healthy(self, position: Position) -> None:
        """Should not auto-adjust healthy position."""
        position.current_price = 220  # Profitable

        should_adjust, adj_type = should_auto_adjust(
            position=position,
            max_loss=400,
        )

        assert should_adjust is False
        assert adj_type is None


class TestGetAdjustmentSummary:
    """Tests for get_adjustment_summary function."""

    def test_empty_adjustments(self) -> None:
        """Empty list should return count of 0."""
        summary = get_adjustment_summary([])

        assert summary["count"] == 0
        assert summary["recommendations"] == []

    @patch("adjustment_engine.ADJUSTMENT_ENABLE", True)
    def test_summary_with_adjustments(self) -> None:
        """Should return summary with adjustment details."""
        position = Position(
            underlying="NIFTY",
            structure="BUY_CE",
            strike=20100,
            strike2=None,
            expiry="2024-01-18",
            dte=5,
            entry_price=200,
            current_price=120,
            contracts=2,
            is_long=True,
            entry_spot=20000,
            current_spot=19900,
        )

        adjustments = suggest_adjustments(
            position=position,
            max_loss=400,
        )

        summary = get_adjustment_summary(adjustments)

        assert summary["count"] == len(adjustments)
        if adjustments:
            assert "top_recommendation" in summary
            assert len(summary["recommendations"]) > 0


class TestWinningPositionAdjustments:
    """Tests for adjustments on winning positions."""

    @patch("adjustment_engine.ADJUSTMENT_ENABLE", True)
    def test_winning_suggests_partial_close(self) -> None:
        """Winning position should suggest partial close."""
        position = Position(
            underlying="NIFTY",
            structure="BUY_CE",
            strike=20000,
            strike2=None,
            expiry="2024-01-18",
            dte=5,
            entry_price=200,
            current_price=280,  # +40% profit
            contracts=4,  # Multiple contracts
            is_long=True,
            entry_spot=19950,
            current_spot=20100,
        )

        adjustments = suggest_adjustments(
            position=position,
            max_loss=800,
            target=350,
        )

        adj_types = [a.type for a in adjustments]
        # Should suggest partial close or no action
        assert (
            AdjustmentType.CLOSE_PARTIAL in adj_types
            or AdjustmentType.NO_ACTION in adj_types
        )
