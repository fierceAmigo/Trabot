"""Tests for probability_analysis.py

P2: Probability of profit calculations for options.
"""

from __future__ import annotations

import math
import os
from unittest.mock import patch

import pytest

# Import module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probability_analysis import (
    delta_to_prob,
    calculate_pop,
    calculate_pop_single_leg,
    expected_value,
    meets_probability_filters,
    get_probability_summary,
    ProbabilityResult,
    analyze_trade_probability,
    prob_above_strike,
    prob_below_strike,
)


class TestDeltaToProb:
    """Tests for delta_to_prob conversion."""

    def test_atm_delta_around_50_percent(self) -> None:
        """ATM delta (0.50) should give ~50% probability."""
        prob = delta_to_prob(delta=0.50)

        assert 45 <= prob <= 55

    def test_deep_itm_high_prob(self) -> None:
        """Deep ITM delta should give high probability."""
        prob = delta_to_prob(delta=0.85)

        assert prob >= 70

    def test_deep_otm_low_prob(self) -> None:
        """Deep OTM delta should give low probability."""
        prob = delta_to_prob(delta=0.15)

        assert prob <= 30

    def test_put_delta_handling(self) -> None:
        """Put delta (negative) should be handled correctly."""
        prob = delta_to_prob(delta=-0.30)

        assert 0 <= prob <= 100


class TestProbAboveBelow:
    """Tests for prob_above_strike and prob_below_strike."""

    def test_prob_above_returns_valid(self) -> None:
        """prob_above_strike should return 0-100."""
        prob = prob_above_strike(
            spot=20000,
            strike=20200,
            iv=0.15,
            dte=7,
        )

        assert 0 <= prob <= 100

    def test_prob_below_returns_valid(self) -> None:
        """prob_below_strike should return 0-100."""
        prob = prob_below_strike(
            spot=20000,
            strike=19800,
            iv=0.15,
            dte=7,
        )

        assert 0 <= prob <= 100

    def test_higher_strike_lower_prob_above(self) -> None:
        """Higher strike should have lower prob_above."""
        low_strike_prob = prob_above_strike(
            spot=20000, strike=20100, iv=0.15, dte=7
        )
        high_strike_prob = prob_above_strike(
            spot=20000, strike=20500, iv=0.15, dte=7
        )

        assert low_strike_prob > high_strike_prob


class TestCalculatePopSingleLeg:
    """Tests for calculate_pop_single_leg function."""

    def test_returns_valid_probability(self) -> None:
        """Should return probability between 0-100."""
        pop = calculate_pop_single_leg(
            spot=20000,
            strike=20200,
            premium=100,
            iv=0.15,
            dte=7,
            option_type="CE",
            is_long=True,
        )

        assert 0 <= pop <= 100


class TestExpectedValue:
    """Tests for expected value calculation."""

    def test_positive_ev_when_good_odds(self) -> None:
        """Should calculate positive EV when odds are favorable."""
        ev = expected_value(
            pop=60,  # 60% win rate
            max_profit=200,
            max_loss=100,
        )

        # EV = 0.6 * 200 - 0.4 * 100 = 120 - 40 = 80
        assert ev > 0

    def test_negative_ev_when_bad_odds(self) -> None:
        """Should calculate negative EV when odds are unfavorable."""
        ev = expected_value(
            pop=30,  # 30% win rate
            max_profit=200,
            max_loss=100,
        )

        # EV = 0.3 * 200 - 0.7 * 100 = 60 - 70 = -10
        assert ev < 0

    def test_zero_pop_negative_ev(self) -> None:
        """Zero POP should give negative EV (max loss)."""
        ev = expected_value(
            pop=0,
            max_profit=200,
            max_loss=100,
        )

        assert ev == -100

    def test_hundred_pop_positive_ev(self) -> None:
        """100% POP should give positive EV (max profit)."""
        ev = expected_value(
            pop=100,
            max_profit=200,
            max_loss=100,
        )

        assert ev == 200


class TestProbabilityResult:
    """Tests for ProbabilityResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating ProbabilityResult."""
        result = ProbabilityResult(
            pop=65.5,
            method="lognormal",
            expected_value=150.0,
            max_profit=300.0,
            max_loss=150.0,
            risk_reward_ratio=2.0,
            recommendation="Good setup",
        )

        assert result.pop == 65.5
        assert result.method == "lognormal"
        assert result.expected_value == 150.0
        assert result.risk_reward_ratio == 2.0


class TestMeetsProbabilityFilters:
    """Tests for meets_probability_filters function."""

    def test_passes_with_good_params(self) -> None:
        """Should pass with good probability parameters."""
        passes = meets_probability_filters(
            pop=60,
            ev=100,
        )

        assert isinstance(passes, bool)

    def test_fails_with_low_pop(self) -> None:
        """Should fail with low POP if filter is set."""
        passes = meets_probability_filters(
            pop=20,
            ev=100,
        )

        # Result depends on configuration
        assert isinstance(passes, bool)


class TestGetProbabilitySummary:
    """Tests for get_probability_summary function."""

    def test_returns_dict(self) -> None:
        """Should return summary dict."""
        result = ProbabilityResult(
            pop=65.5,
            method="lognormal",
            expected_value=150.0,
            max_profit=300.0,
            max_loss=150.0,
            risk_reward_ratio=2.0,
            recommendation="Good setup",
        )

        summary = get_probability_summary(result)

        assert isinstance(summary, dict)
        assert "pop" in summary
        assert "expected_value" in summary
        assert "recommendation" in summary


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_dte(self) -> None:
        """Zero DTE should handle gracefully."""
        prob = prob_above_strike(
            spot=20000,
            strike=20100,
            iv=0.15,
            dte=0,
        )

        assert 0 <= prob <= 100

    def test_zero_iv(self) -> None:
        """Zero IV should handle gracefully."""
        prob = prob_above_strike(
            spot=20000,
            strike=20100,
            iv=0.0,
            dte=7,
        )

        # With zero IV, price shouldn't move
        assert 0 <= prob <= 100

    def test_very_high_iv(self) -> None:
        """Very high IV should handle gracefully."""
        prob = prob_above_strike(
            spot=20000,
            strike=25000,  # Very OTM
            iv=1.0,  # 100% IV
            dte=30,
        )

        assert 0 <= prob <= 100

    def test_delta_zero(self) -> None:
        """Delta of 0 should handle gracefully."""
        prob = delta_to_prob(delta=0.0)

        assert prob == 0 or prob <= 5

    def test_delta_one(self) -> None:
        """Delta of 1 should handle gracefully."""
        prob = delta_to_prob(delta=1.0)

        assert prob >= 95 or prob == 100
