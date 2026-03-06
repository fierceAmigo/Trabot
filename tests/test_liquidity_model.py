"""Tests for liquidity_model.py

P3: Advanced liquidity scoring for options selection.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Import module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liquidity_model import (
    LiquidityGrade,
    LiquidityMetrics,
    LiquidityScore,
    score_spread,
    score_depth,
    score_volume,
    score_oi,
    score_oi_trend,
    calculate_liquidity_score,
    estimate_slippage_pct,
    estimate_slippage,
    rank_strikes_by_liquidity,
    filter_by_liquidity,
    get_best_liquid_strike,
    liquidity_adjusted_price,
    get_liquidity_summary,
)


class TestLiquidityGrade:
    """Tests for LiquidityGrade enum."""

    def test_all_grades_exist(self) -> None:
        """Verify all expected grades exist."""
        assert LiquidityGrade.EXCELLENT.value == "A"
        assert LiquidityGrade.GOOD.value == "B"
        assert LiquidityGrade.FAIR.value == "C"
        assert LiquidityGrade.POOR.value == "D"
        assert LiquidityGrade.AVOID.value == "F"


class TestLiquidityMetrics:
    """Tests for LiquidityMetrics dataclass."""

    def test_metrics_creation(self) -> None:
        """Test creating metrics instance."""
        metrics = LiquidityMetrics(
            bid_ask_spread=2.0,
            spread_pct=0.01,
            bid_size=500,
            ask_size=600,
            volume=5000,
            oi=15000,
            oi_change=1200,
            last_trade_time="14:30:00",
        )

        assert metrics.bid_ask_spread == 2.0
        assert metrics.spread_pct == 0.01
        assert metrics.bid_size == 500
        assert metrics.ask_size == 600
        assert metrics.volume == 5000
        assert metrics.oi == 15000


class TestScoreSpread:
    """Tests for score_spread function."""

    def test_zero_spread_perfect_score(self) -> None:
        """Zero spread should get perfect score."""
        score = score_spread(0.0)
        assert score == 100.0

    def test_high_spread_low_score(self) -> None:
        """High spread (10%+) should get zero score."""
        score = score_spread(0.10)
        assert score == 0.0

    def test_moderate_spread_moderate_score(self) -> None:
        """Moderate spread should get moderate score."""
        score = score_spread(0.05)  # 5%
        assert 40 <= score <= 60

    def test_negative_spread_treated_as_zero(self) -> None:
        """Negative spread should be treated as zero."""
        score = score_spread(-0.01)
        assert score == 100.0

    def test_spread_score_is_inverse(self) -> None:
        """Lower spread should get higher score."""
        low_spread_score = score_spread(0.01)
        high_spread_score = score_spread(0.05)

        assert low_spread_score > high_spread_score


class TestScoreDepth:
    """Tests for score_depth function."""

    def test_high_depth_high_score(self) -> None:
        """High depth should get high score."""
        score = score_depth(5000, 5000)
        assert score >= 80

    def test_low_depth_low_score(self) -> None:
        """Low depth should get low score."""
        score = score_depth(10, 10)
        assert score <= 30

    def test_moderate_depth_moderate_score(self) -> None:
        """Moderate depth should get moderate score."""
        score = score_depth(300, 300)
        assert 40 <= score <= 70


class TestScoreVolume:
    """Tests for score_volume function."""

    def test_high_volume_high_score(self) -> None:
        """High volume should get high score."""
        score = score_volume(50000)
        assert score >= 90

    def test_low_volume_low_score(self) -> None:
        """Low volume should get low score."""
        score = score_volume(50)
        assert score <= 25

    def test_moderate_volume_moderate_score(self) -> None:
        """Moderate volume should get moderate score."""
        score = score_volume(3000)
        assert 50 <= score <= 75


class TestScoreOI:
    """Tests for score_oi function."""

    def test_high_oi_high_score(self) -> None:
        """High OI should get high score."""
        score = score_oi(100000)
        assert score >= 90

    def test_low_oi_low_score(self) -> None:
        """Low OI should get low score."""
        score = score_oi(100)
        assert score <= 20

    def test_moderate_oi_moderate_score(self) -> None:
        """Moderate OI should get moderate score."""
        score = score_oi(8000)
        assert 55 <= score <= 75


class TestScoreOITrend:
    """Tests for score_oi_trend function."""

    def test_rising_oi_high_score(self) -> None:
        """Rising OI should get high score."""
        score = score_oi_trend(10000, 2500)  # +25%
        assert score >= 85

    def test_falling_oi_low_score(self) -> None:
        """Falling OI should get low score."""
        score = score_oi_trend(10000, -1500)  # -15%
        assert score <= 40

    def test_flat_oi_moderate_score(self) -> None:
        """Flat OI should get moderate score."""
        score = score_oi_trend(10000, 0)
        assert 50 <= score <= 60

    def test_zero_oi_neutral_score(self) -> None:
        """Zero OI should get neutral score."""
        score = score_oi_trend(0, 0)
        assert score == 50.0


class TestCalculateLiquidityScore:
    """Tests for calculate_liquidity_score function."""

    def test_returns_liquidity_score(self) -> None:
        """Should return LiquidityScore instance."""
        metrics = LiquidityMetrics(
            bid_ask_spread=2.0,
            spread_pct=0.01,
            bid_size=500,
            ask_size=600,
            volume=5000,
            oi=15000,
            oi_change=1200,
            last_trade_time="14:30:00",
        )

        result = calculate_liquidity_score(metrics)

        assert isinstance(result, LiquidityScore)
        assert 0 <= result.score <= 100
        assert isinstance(result.grade, LiquidityGrade)
        assert isinstance(result.factors, dict)

    def test_excellent_liquidity(self) -> None:
        """Excellent liquidity should get A grade."""
        metrics = LiquidityMetrics(
            bid_ask_spread=1.0,
            spread_pct=0.005,  # 0.5%
            bid_size=5000,
            ask_size=5000,
            volume=50000,
            oi=100000,
            oi_change=5000,
            last_trade_time="14:30:00",
        )

        result = calculate_liquidity_score(metrics)

        assert result.grade == LiquidityGrade.EXCELLENT
        assert result.score >= 80

    def test_poor_liquidity(self) -> None:
        """Poor liquidity should get D or F grade."""
        metrics = LiquidityMetrics(
            bid_ask_spread=10.0,
            spread_pct=0.08,  # 8%
            bid_size=20,
            ask_size=30,
            volume=50,
            oi=200,
            oi_change=-50,
            last_trade_time="10:00:00",
        )

        result = calculate_liquidity_score(metrics)

        assert result.grade in (LiquidityGrade.POOR, LiquidityGrade.AVOID)
        assert result.score <= 40

    def test_all_factors_included(self) -> None:
        """All scoring factors should be in result."""
        metrics = LiquidityMetrics(
            bid_ask_spread=2.0,
            spread_pct=0.01,
            bid_size=500,
            ask_size=600,
            volume=5000,
            oi=15000,
            oi_change=1200,
            last_trade_time="14:30:00",
        )

        result = calculate_liquidity_score(metrics)

        assert "spread" in result.factors
        assert "depth" in result.factors
        assert "volume" in result.factors
        assert "oi" in result.factors
        assert "oi_trend" in result.factors


class TestEstimateSlippage:
    """Tests for slippage estimation functions."""

    def test_estimate_slippage_pct(self) -> None:
        """Test slippage percentage estimation."""
        metrics = LiquidityMetrics(
            bid_ask_spread=2.0,
            spread_pct=0.02,  # 2%
            bid_size=500,
            ask_size=600,
            volume=5000,
            oi=15000,
            oi_change=0,
            last_trade_time="14:30:00",
        )

        slippage_pct = estimate_slippage_pct(metrics)

        assert slippage_pct > 0
        assert slippage_pct <= 0.10  # Max 10%

    def test_high_liquidity_low_slippage(self) -> None:
        """High liquidity should have low slippage."""
        high_liq = LiquidityMetrics(
            bid_ask_spread=1.0,
            spread_pct=0.005,
            bid_size=5000,
            ask_size=5000,
            volume=50000,
            oi=100000,
            oi_change=0,
            last_trade_time="14:30:00",
        )
        low_liq = LiquidityMetrics(
            bid_ask_spread=5.0,
            spread_pct=0.05,
            bid_size=50,
            ask_size=50,
            volume=500,
            oi=1000,
            oi_change=0,
            last_trade_time="14:30:00",
        )

        high_liq_slippage = estimate_slippage_pct(high_liq)
        low_liq_slippage = estimate_slippage_pct(low_liq)

        assert high_liq_slippage < low_liq_slippage

    def test_estimate_slippage_for_contracts(self) -> None:
        """Test slippage for specific contract count."""
        metrics = LiquidityMetrics(
            bid_ask_spread=2.0,
            spread_pct=0.02,
            bid_size=500,
            ask_size=600,
            volume=5000,
            oi=15000,
            oi_change=0,
            last_trade_time="14:30:00",
        )

        slippage = estimate_slippage(
            contracts=5,
            metrics=metrics,
            is_entry=True,
        )

        assert slippage > 0


class TestRankStrikesByLiquidity:
    """Tests for rank_strikes_by_liquidity function."""

    def test_ranks_by_score_descending(self) -> None:
        """Strikes should be ranked by score descending."""
        options = [
            {
                "strike": 20000,
                "bid": 100,
                "ask": 102,
                "spread_pct": 0.02,
                "bid_qty": 100,
                "ask_qty": 100,
                "volume": 500,
                "oi": 1000,
                "oi_change": 0,
            },
            {
                "strike": 20100,
                "bid": 50,
                "ask": 51,
                "spread_pct": 0.02,
                "bid_qty": 1000,
                "ask_qty": 1000,
                "volume": 10000,
                "oi": 50000,
                "oi_change": 5000,
            },
        ]

        ranked = rank_strikes_by_liquidity(options)

        assert len(ranked) == 2
        # Higher liquidity strike should be first
        assert ranked[0][0] == 20100
        assert ranked[0][1].score > ranked[1][1].score


class TestFilterByLiquidity:
    """Tests for filter_by_liquidity function."""

    def test_filters_low_oi(self) -> None:
        """Should filter out options with low OI."""
        options = [
            {"strike": 20000, "oi": 100, "spread_pct": 0.01, "volume": 1000,
             "bid": 100, "ask": 101, "bid_qty": 500, "ask_qty": 500},
            {"strike": 20100, "oi": 5000, "spread_pct": 0.01, "volume": 5000,
             "bid": 50, "ask": 51, "bid_qty": 500, "ask_qty": 500},
        ]

        filtered = filter_by_liquidity(options, min_oi=1000)

        assert len(filtered) == 1
        assert filtered[0]["strike"] == 20100

    def test_filters_high_spread(self) -> None:
        """Should filter out options with high spread."""
        options = [
            {"strike": 20000, "oi": 5000, "spread_pct": 0.10, "volume": 1000,
             "bid": 100, "ask": 110, "bid_qty": 500, "ask_qty": 500},
            {"strike": 20100, "oi": 5000, "spread_pct": 0.02, "volume": 5000,
             "bid": 50, "ask": 51, "bid_qty": 500, "ask_qty": 500},
        ]

        filtered = filter_by_liquidity(options, max_spread_pct=0.05)

        assert len(filtered) == 1
        assert filtered[0]["strike"] == 20100


class TestGetBestLiquidStrike:
    """Tests for get_best_liquid_strike function."""

    def test_finds_best_strike_near_target(self) -> None:
        """Should find most liquid strike near target."""
        options = [
            {"strike": 19950, "oi": 1000, "spread_pct": 0.02, "volume": 500,
             "bid": 150, "ask": 153, "bid_qty": 100, "ask_qty": 100},
            {"strike": 20000, "oi": 50000, "spread_pct": 0.01, "volume": 20000,
             "bid": 100, "ask": 101, "bid_qty": 2000, "ask_qty": 2000},
            {"strike": 20050, "oi": 2000, "spread_pct": 0.02, "volume": 1000,
             "bid": 50, "ask": 51, "bid_qty": 200, "ask_qty": 200},
        ]

        best = get_best_liquid_strike(
            options=options,
            near_strike=20000,
            max_distance=2,
        )

        assert best is not None
        assert best["strike"] == 20000  # Most liquid

    def test_returns_none_when_no_nearby(self) -> None:
        """Should return None when no options nearby."""
        options = [
            {"strike": 19000, "oi": 5000, "spread_pct": 0.01, "volume": 5000,
             "bid": 500, "ask": 505, "bid_qty": 500, "ask_qty": 500},
        ]

        best = get_best_liquid_strike(
            options=options,
            near_strike=20000,
            max_distance=2,
        )

        assert best is None


class TestLiquidityAdjustedPrice:
    """Tests for liquidity_adjusted_price function."""

    def test_buy_price_higher_than_ask(self) -> None:
        """Buy price should be >= ask (includes slippage)."""
        option = {
            "bid": 100,
            "ask": 102,
            "ltp": 101,
            "bid_qty": 500,
            "ask_qty": 500,
            "volume": 5000,
            "oi": 10000,
        }

        price = liquidity_adjusted_price(option, side="BUY", contracts=1)

        assert price >= 102

    def test_sell_price_lower_than_bid(self) -> None:
        """Sell price should be <= bid (includes slippage)."""
        option = {
            "bid": 100,
            "ask": 102,
            "ltp": 101,
            "bid_qty": 500,
            "ask_qty": 500,
            "volume": 5000,
            "oi": 10000,
        }

        price = liquidity_adjusted_price(option, side="SELL", contracts=1)

        assert price <= 100


class TestGetLiquiditySummary:
    """Tests for get_liquidity_summary function."""

    def test_returns_dict(self) -> None:
        """Should return dict with summary info."""
        score = LiquidityScore(
            score=75.0,
            grade=LiquidityGrade.GOOD,
            factors={"spread": 80, "depth": 70, "volume": 75, "oi": 80, "oi_trend": 60},
            estimated_slippage_pct=0.015,
            recommendation="Good liquidity - use limit orders",
        )

        summary = get_liquidity_summary(score)

        assert isinstance(summary, dict)
        assert "score" in summary
        assert "grade" in summary
        assert "recommendation" in summary
