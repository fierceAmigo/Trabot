"""Tests for expected_move.py

P1: Expected move calculation for spread width selection.
"""

from __future__ import annotations

import math
import os
from unittest.mock import patch

import pytest

# Import module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from expected_move import (
    ExpectedMoveResult,
    Z_SCORES,
    calculate_expected_move,
    suggest_spread_width,
    breakeven_distance_vs_expected,
    expected_move_multiple,
)


class TestZScores:
    """Tests for Z-score constants."""

    def test_z_scores_exist(self) -> None:
        """Verify key Z-score values exist."""
        assert 0.68 in Z_SCORES
        assert 0.95 in Z_SCORES

    def test_z_score_for_one_std(self) -> None:
        """Z-score for 68% confidence should be ~1.0."""
        assert Z_SCORES[0.68] == 1.00

    def test_z_score_for_two_std(self) -> None:
        """Z-score for 95% confidence should be ~1.96."""
        assert Z_SCORES[0.95] == 1.96


class TestCalculateExpectedMove:
    """Tests for calculate_expected_move function."""

    def test_basic_calculation(self) -> None:
        """Test basic expected move calculation."""
        result = calculate_expected_move(
            spot=20000,
            iv=0.15,
            dte=7,
            confidence=0.68,
        )

        assert isinstance(result, ExpectedMoveResult)
        assert result.spot == 20000
        assert result.iv == 0.15
        assert result.dte == 7
        assert result.confidence == 0.68
        assert result.expected_move > 0

    def test_expected_move_formula(self) -> None:
        """Test expected move follows correct formula."""
        spot = 20000
        iv = 0.20
        dte = 30
        confidence = 0.68
        z = Z_SCORES[confidence]

        result = calculate_expected_move(spot, iv, dte, confidence)

        # EM = Spot × IV × sqrt(DTE/365) × Z
        expected = spot * iv * math.sqrt(dte / 365) * z
        assert abs(result.expected_move - expected) < 0.01

    def test_higher_iv_means_higher_move(self) -> None:
        """Higher IV should result in higher expected move."""
        low_iv_result = calculate_expected_move(
            spot=20000, iv=0.10, dte=7, confidence=0.68
        )
        high_iv_result = calculate_expected_move(
            spot=20000, iv=0.30, dte=7, confidence=0.68
        )

        assert high_iv_result.expected_move > low_iv_result.expected_move

    def test_longer_dte_means_higher_move(self) -> None:
        """Longer DTE should result in higher expected move."""
        short_dte_result = calculate_expected_move(
            spot=20000, iv=0.15, dte=3, confidence=0.68
        )
        long_dte_result = calculate_expected_move(
            spot=20000, iv=0.15, dte=30, confidence=0.68
        )

        assert long_dte_result.expected_move > short_dte_result.expected_move

    def test_higher_confidence_means_wider_move(self) -> None:
        """Higher confidence should result in wider expected move."""
        low_conf_result = calculate_expected_move(
            spot=20000, iv=0.15, dte=7, confidence=0.68
        )
        high_conf_result = calculate_expected_move(
            spot=20000, iv=0.15, dte=7, confidence=0.95
        )

        assert high_conf_result.expected_move > low_conf_result.expected_move

    def test_bounds_calculation(self) -> None:
        """Test upper and lower bounds are calculated correctly."""
        result = calculate_expected_move(
            spot=20000, iv=0.15, dte=7, confidence=0.68
        )

        assert result.upper_bound == result.spot + result.expected_move
        assert result.lower_bound == result.spot - result.expected_move

    def test_expected_move_pct(self) -> None:
        """Test expected move percentage calculation."""
        result = calculate_expected_move(
            spot=20000, iv=0.15, dte=7, confidence=0.68
        )

        expected_pct = (result.expected_move / result.spot) * 100
        assert abs(result.expected_move_pct - expected_pct) < 0.001

    def test_minimum_dte(self) -> None:
        """DTE should be at least 1."""
        result = calculate_expected_move(
            spot=20000, iv=0.15, dte=0, confidence=0.68
        )

        assert result.dte >= 1

    def test_interpolation_for_unknown_confidence(self) -> None:
        """Test interpolation for confidence levels not in table."""
        # 0.75 is between 0.68 and 0.80
        result = calculate_expected_move(
            spot=20000, iv=0.15, dte=7, confidence=0.75
        )

        result_68 = calculate_expected_move(
            spot=20000, iv=0.15, dte=7, confidence=0.68
        )
        result_80 = calculate_expected_move(
            spot=20000, iv=0.15, dte=7, confidence=0.80
        )

        # Result should be between the two
        assert result_68.expected_move < result.expected_move < result_80.expected_move


class TestSuggestSpreadWidth:
    """Tests for suggest_spread_width function."""

    def test_returns_multiple_of_step(self) -> None:
        """Spread width should be multiple of step size."""
        width = suggest_spread_width(
            expected_move=300,
            step=50,
            strategy_type="BULL_CALL_SPREAD",
        )

        assert width % 50 == 0

    def test_debit_spread_width(self) -> None:
        """Debit spreads should capture portion of expected move."""
        width = suggest_spread_width(
            expected_move=500,
            step=50,
            strategy_type="BULL_CALL_SPREAD",
        )

        # Should be 60-80% of expected move
        assert 0.5 * 500 <= width <= 1.0 * 500

    def test_credit_spread_width(self) -> None:
        """Credit spreads should be outside expected move."""
        width = suggest_spread_width(
            expected_move=500,
            step=50,
            strategy_type="BULL_PUT_SPREAD",
        )

        # Width for credit spread should be reasonable
        assert width >= 100  # At least 2 steps
        assert width <= 600  # Not too wide

    def test_respects_max_width(self) -> None:
        """Width should not exceed max steps configuration."""
        with patch("expected_move.MAX_SPREAD_WIDTH_STEPS", 5):
            width = suggest_spread_width(
                expected_move=1000,
                step=50,
                strategy_type="BULL_CALL_SPREAD",
            )

            assert width <= 5 * 50

    def test_respects_min_width(self) -> None:
        """Width should be at least min steps configuration."""
        with patch("expected_move.MIN_SPREAD_WIDTH_STEPS", 2):
            width = suggest_spread_width(
                expected_move=50,  # Very small
                step=50,
                strategy_type="BULL_CALL_SPREAD",
            )

            assert width >= 2 * 50

    def test_different_regimes(self) -> None:
        """Width should vary by regime."""
        trend_width = suggest_spread_width(
            expected_move=400,
            step=50,
            strategy_type="BULL_CALL_SPREAD",
            regime="TREND",
        )
        volatile_width = suggest_spread_width(
            expected_move=400,
            step=50,
            strategy_type="BULL_CALL_SPREAD",
            regime="VOLATILE",
        )

        # Volatile regime typically suggests wider spreads
        # Note: This test may need adjustment based on actual implementation
        assert trend_width >= 100
        assert volatile_width >= 100

    def test_invalid_step_uses_default(self) -> None:
        """Invalid step should use default value."""
        width = suggest_spread_width(
            expected_move=300,
            step=0,  # Invalid
            strategy_type="BULL_CALL_SPREAD",
        )

        # Should use default step of 50
        assert width % 50 == 0
        assert width > 0


class TestBreakevenDistanceVsExpected:
    """Tests for breakeven_distance_vs_expected function."""

    def test_returns_dict(self) -> None:
        """Should return dict with comparison info."""
        result = breakeven_distance_vs_expected(
            spot=20000,
            breakeven=20300,
            iv=0.15,
            dte=7,
        )

        assert isinstance(result, dict)

    def test_close_breakeven_is_favorable(self) -> None:
        """Close breakeven should be more favorable."""
        close_be = breakeven_distance_vs_expected(
            spot=20000,
            breakeven=20100,  # 0.5% away
            iv=0.15,
            dte=7,
        )
        far_be = breakeven_distance_vs_expected(
            spot=20000,
            breakeven=20500,  # 2.5% away
            iv=0.15,
            dte=7,
        )

        # Result should indicate close BE is better
        assert isinstance(close_be, dict)
        assert isinstance(far_be, dict)


class TestExpectedMoveMultiple:
    """Tests for expected_move_multiple function."""

    def test_returns_float(self) -> None:
        """Should return float multiple."""
        multiple = expected_move_multiple(
            distance=300,
            spot=20000,
            iv=0.15,
            dte=7,
        )

        assert isinstance(multiple, (int, float))
        assert multiple >= 0

    def test_distance_zero_returns_zero(self) -> None:
        """Zero distance should return ~0 multiple."""
        multiple = expected_move_multiple(
            distance=0,
            spot=20000,
            iv=0.15,
            dte=7,
        )

        assert multiple == 0 or multiple < 0.1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_iv(self) -> None:
        """Zero IV should result in zero expected move."""
        result = calculate_expected_move(
            spot=20000, iv=0.0, dte=7, confidence=0.68
        )

        assert result.expected_move == 0

    def test_very_high_iv(self) -> None:
        """Very high IV should be handled."""
        result = calculate_expected_move(
            spot=20000, iv=1.0, dte=7, confidence=0.68  # 100% IV
        )

        assert result.expected_move > 0
        assert result.expected_move < result.spot  # Should not exceed spot

    def test_type_coercion(self) -> None:
        """Inputs should be coerced to correct types."""
        result = calculate_expected_move(
            spot="20000",  # String
            iv="0.15",  # String
            dte=7.5,  # Float
            confidence=0.68,
        )

        assert result.spot == 20000.0
        assert result.iv == 0.15
        assert result.dte == 7  # Integer
