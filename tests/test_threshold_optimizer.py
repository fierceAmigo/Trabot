"""Tests for threshold_optimizer.py

P3: Walk-forward threshold calibration for adaptive signal tuning.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import patch

import pytest

# Import module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from threshold_optimizer import (
    OptimizationResult,
    PerformanceMetrics,
    load_best_params,
    save_best_params,
    calculate_metrics,
    filter_trades_by_params,
    simulate_with_params,
    grid_search,
    walk_forward_optimize,
    optimize_thresholds,
    get_regime_specific_params,
    compare_param_sets,
)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_metrics_creation(self) -> None:
        """Test creating PerformanceMetrics instance."""
        metrics = PerformanceMetrics(
            total_pnl=5000,
            trade_count=20,
            win_count=12,
            loss_count=8,
            avg_win=500,
            avg_loss=200,
            sharpe=1.5,
            sortino=2.0,
            profit_factor=1.8,
            max_drawdown=0.10,
            win_rate=0.60,
        )

        assert metrics.total_pnl == 5000
        assert metrics.trade_count == 20
        assert metrics.win_rate == 0.60


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_empty_trades(self) -> None:
        """Empty trades should return zero metrics."""
        metrics = calculate_metrics([])

        assert metrics.total_pnl == 0
        assert metrics.trade_count == 0
        assert metrics.win_rate == 0

    def test_all_winning_trades(self) -> None:
        """All winning trades should have 100% win rate."""
        trades = [
            {"pnl": 100},
            {"pnl": 200},
            {"pnl": 150},
        ]

        metrics = calculate_metrics(trades)

        assert metrics.win_count == 3
        assert metrics.loss_count == 0
        assert metrics.win_rate == 1.0

    def test_all_losing_trades(self) -> None:
        """All losing trades should have 0% win rate."""
        trades = [
            {"pnl": -100},
            {"pnl": -200},
            {"pnl": -150},
        ]

        metrics = calculate_metrics(trades)

        assert metrics.win_count == 0
        assert metrics.loss_count == 3
        assert metrics.win_rate == 0.0

    def test_mixed_trades(self) -> None:
        """Mixed trades should calculate correct metrics."""
        trades = [
            {"pnl": 500},
            {"pnl": -200},
            {"pnl": 300},
            {"pnl": -100},
            {"pnl": 400},
        ]

        metrics = calculate_metrics(trades)

        assert metrics.total_pnl == 900  # 500 - 200 + 300 - 100 + 400
        assert metrics.trade_count == 5
        assert metrics.win_count == 3
        assert metrics.loss_count == 2
        assert metrics.win_rate == 0.6

    def test_profit_factor_calculation(self) -> None:
        """Test profit factor is calculated correctly."""
        trades = [
            {"pnl": 1000},  # Gross profit = 1000
            {"pnl": -500},  # Gross loss = 500
        ]

        metrics = calculate_metrics(trades)

        assert metrics.profit_factor == 2.0  # 1000 / 500

    def test_avg_win_loss_calculation(self) -> None:
        """Test average win/loss calculation."""
        trades = [
            {"pnl": 200},
            {"pnl": 400},
            {"pnl": -100},
            {"pnl": -200},
        ]

        metrics = calculate_metrics(trades)

        assert metrics.avg_win == 300  # (200 + 400) / 2
        assert metrics.avg_loss == 150  # (100 + 200) / 2


class TestSharpeCalculation:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_positive_for_winning(self) -> None:
        """Sharpe should be positive for profitable trades."""
        trades = [{"pnl": 100 + i * 10} for i in range(20)]

        metrics = calculate_metrics(trades)

        assert metrics.sharpe > 0

    def test_sharpe_negative_for_losing(self) -> None:
        """Sharpe should be negative for losing trades."""
        trades = [{"pnl": -100 - i * 10} for i in range(20)]

        metrics = calculate_metrics(trades)

        assert metrics.sharpe < 0


class TestFilterTradesByParams:
    """Tests for filter_trades_by_params function."""

    def test_filters_by_adx(self) -> None:
        """Should filter trades by ADX threshold."""
        trades = [
            {"side": "LONG", "signal_metrics": {"adx": 15, "rsi": 55}},
            {"side": "LONG", "signal_metrics": {"adx": 25, "rsi": 55}},
            {"side": "LONG", "signal_metrics": {"adx": 30, "rsi": 55}},
        ]
        params = {"ADX_MIN": 20}

        filtered = filter_trades_by_params(trades, params)

        assert len(filtered) == 2
        for trade in filtered:
            assert trade["signal_metrics"]["adx"] >= 20

    def test_filters_by_rsi_long(self) -> None:
        """Should filter LONG trades by RSI threshold."""
        trades = [
            {"side": "LONG", "signal_metrics": {"adx": 25, "rsi": 48}},
            {"side": "LONG", "signal_metrics": {"adx": 25, "rsi": 55}},
        ]
        params = {"ADX_MIN": 18, "RSI_LONG_MIN": 52}

        filtered = filter_trades_by_params(trades, params)

        assert len(filtered) == 1
        assert filtered[0]["signal_metrics"]["rsi"] == 55

    def test_filters_by_rsi_short(self) -> None:
        """Should filter SHORT trades by RSI threshold."""
        trades = [
            {"side": "SHORT", "signal_metrics": {"adx": 25, "rsi": 52}},
            {"side": "SHORT", "signal_metrics": {"adx": 25, "rsi": 45}},
        ]
        params = {"ADX_MIN": 18, "RSI_SHORT_MAX": 48}

        filtered = filter_trades_by_params(trades, params)

        assert len(filtered) == 1
        assert filtered[0]["signal_metrics"]["rsi"] == 45

    def test_filters_by_iv_percentile(self) -> None:
        """Should filter by IV percentile range."""
        trades = [
            {"side": "LONG", "signal_metrics": {"adx": 25, "rsi": 55, "iv_percentile": 0.15}},
            {"side": "LONG", "signal_metrics": {"adx": 25, "rsi": 55, "iv_percentile": 0.45}},
            {"side": "LONG", "signal_metrics": {"adx": 25, "rsi": 55, "iv_percentile": 0.85}},
        ]
        params = {
            "ADX_MIN": 18,
            "RSI_LONG_MIN": 52,
            "MIN_IV_PERCENTILE": 0.20,
            "MAX_IV_PERCENTILE": 0.70,
        }

        filtered = filter_trades_by_params(trades, params)

        assert len(filtered) == 1
        assert filtered[0]["signal_metrics"]["iv_percentile"] == 0.45


class TestSimulateWithParams:
    """Tests for simulate_with_params function."""

    def test_simulates_stop_hit(self) -> None:
        """Should simulate trade hitting stop."""
        trades = [
            {
                "pnl": 100,
                "atr": 100,
                "entry_price": 20000,
                "contracts": 1,
                "mfe": 50,  # Didn't reach target
                "mae": 200,  # Hit stop
            }
        ]
        params = {"STOP_ATR_MULT": 1.5, "TARGET_ATR_MULT": 2.0}

        simulated = simulate_with_params(trades, params)

        # Should have stopped out at 150 (1.5 * 100 ATR)
        assert simulated[0]["pnl"] == -150

    def test_simulates_target_hit(self) -> None:
        """Should simulate trade hitting target."""
        trades = [
            {
                "pnl": 100,
                "atr": 100,
                "entry_price": 20000,
                "contracts": 1,
                "mfe": 250,  # Hit target
                "mae": 50,  # Didn't hit stop
            }
        ]
        params = {"STOP_ATR_MULT": 1.5, "TARGET_ATR_MULT": 2.0}

        simulated = simulate_with_params(trades, params)

        # Should have hit target at 200 (2.0 * 100 ATR)
        assert simulated[0]["pnl"] == 200


class TestGridSearch:
    """Tests for grid_search function."""

    def test_returns_best_params(self) -> None:
        """Should return best parameters."""
        # Create sample trades that would pass filters
        trades = [
            {
                "entry_time": f"2024-01-{i+1:02d}T10:00:00",
                "side": "LONG",
                "pnl": 200 if i % 3 != 0 else -100,
                "atr": 100,
                "entry_price": 20000,
                "contracts": 1,
                "mfe": 250,
                "mae": 80,
                "signal_metrics": {"adx": 22, "rsi": 55, "iv_percentile": 0.45},
            }
            for i in range(30)
        ]

        # Simple param grid
        param_grid = {
            "ADX_MIN": [18, 22],
            "RSI_LONG_MIN": [52, 55],
        }

        best_params, metrics = grid_search(trades, param_grid)

        assert isinstance(best_params, dict)
        assert isinstance(metrics, PerformanceMetrics)

    def test_returns_empty_for_no_valid_trades(self) -> None:
        """Should handle case with no valid trades."""
        trades = [
            {"side": "LONG", "pnl": 100, "signal_metrics": {"adx": 10, "rsi": 45}},
        ]

        param_grid = {"ADX_MIN": [25, 30]}

        best_params, metrics = grid_search(trades, param_grid)

        # Should return empty or default
        assert isinstance(metrics, PerformanceMetrics)


class TestBestParamsPersistence:
    """Tests for load/save best params functions."""

    @pytest.fixture
    def temp_params_file(self) -> Generator[str, None, None]:
        """Create temporary params file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            yield f.name
        os.unlink(f.name)

    def test_save_and_load_params(self, temp_params_file: str) -> None:
        """Test saving and loading params."""
        with patch("threshold_optimizer.BEST_PARAMS_PATH", temp_params_file):
            params = {"ADX_MIN": 22, "RSI_LONG_MIN": 55}
            metrics = PerformanceMetrics(
                total_pnl=5000,
                trade_count=20,
                win_count=12,
                loss_count=8,
                avg_win=500,
                avg_loss=200,
                sharpe=1.5,
                sortino=2.0,
                profit_factor=1.8,
                max_drawdown=0.10,
                win_rate=0.60,
            )

            save_best_params(params, metrics)
            loaded = load_best_params()

            assert loaded["ADX_MIN"] == 22
            assert loaded["RSI_LONG_MIN"] == 55

    def test_load_missing_file(self) -> None:
        """Loading missing file should return empty dict."""
        with patch("threshold_optimizer.BEST_PARAMS_PATH", "/nonexistent/path.json"):
            loaded = load_best_params()
            assert loaded == {}


class TestOptimizeThresholds:
    """Tests for optimize_thresholds main function."""

    @patch("threshold_optimizer.OPTIMIZER_ENABLE", True)
    def test_returns_params_or_none(self) -> None:
        """Should return params dict or None."""
        trades = [
            {
                "entry_time": f"2024-01-{i+1:02d}T10:00:00",
                "side": "LONG",
                "pnl": 200,
                "signal_metrics": {"adx": 25, "rsi": 55},
            }
            for i in range(30)
        ]

        result = optimize_thresholds(trades, lookback_days=60)

        assert result is None or isinstance(result, dict)

    @patch("threshold_optimizer.OPTIMIZER_ENABLE", False)
    def test_disabled_returns_none(self) -> None:
        """When disabled, should return None."""
        trades = [{"entry_time": "2024-01-01T10:00:00", "pnl": 100}]

        result = optimize_thresholds(trades)

        assert result is None


class TestGetRegimeSpecificParams:
    """Tests for get_regime_specific_params function."""

    @pytest.fixture
    def temp_params_file(self) -> Generator[str, None, None]:
        """Create temporary params file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            yield f.name
        os.unlink(f.name)

    def test_returns_dict(self, temp_params_file: str) -> None:
        """Should always return a dict."""
        with patch("threshold_optimizer.BEST_PARAMS_PATH", temp_params_file):
            params = get_regime_specific_params("TREND")
            assert isinstance(params, dict)


class TestCompareParamSets:
    """Tests for compare_param_sets function."""

    def test_sorts_by_sharpe(self) -> None:
        """Results should be sorted by Sharpe descending."""
        trades = [
            {
                "side": "LONG",
                "pnl": 200,
                "atr": 100,
                "entry_price": 20000,
                "contracts": 1,
                "mfe": 250,
                "mae": 50,
                "signal_metrics": {"adx": 25, "rsi": 55},
            }
        ] * 25

        param_sets = [
            {"ADX_MIN": 18, "RSI_LONG_MIN": 50},
            {"ADX_MIN": 22, "RSI_LONG_MIN": 55},
        ]

        results = compare_param_sets(trades, param_sets)

        # Should return sorted list
        assert len(results) == 2
        if results[0][1].sharpe != 0 and results[1][1].sharpe != 0:
            assert results[0][1].sharpe >= results[1][1].sharpe


class TestWalkForwardOptimize:
    """Tests for walk_forward_optimize function."""

    def test_returns_list_of_results(self) -> None:
        """Should return list of OptimizationResult."""
        # Create trades spanning multiple months
        trades = []
        base_date = datetime(2024, 1, 1)
        for i in range(100):
            trade_date = base_date + timedelta(days=i)
            trades.append({
                "entry_time": trade_date.isoformat(),
                "side": "LONG" if i % 2 == 0 else "SHORT",
                "pnl": 200 if i % 3 != 0 else -100,
                "atr": 100,
                "entry_price": 20000,
                "contracts": 1,
                "mfe": 250,
                "mae": 80,
                "signal_metrics": {"adx": 22, "rsi": 55 if i % 2 == 0 else 45},
            })

        results = walk_forward_optimize(
            trades,
            train_days=30,
            test_days=7,
        )

        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, OptimizationResult)

    def test_empty_trades_returns_empty(self) -> None:
        """Empty trades should return empty list."""
        results = walk_forward_optimize([])
        assert results == []
