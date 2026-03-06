"""Pytest configuration and shared fixtures for Trabot tests."""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, date
from typing import Dict, Generator, List
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_ist_time():
    """Fixture to mock IST time functions."""
    def _mock_time(year: int, month: int, day: int, hour: int, minute: int = 0):
        with patch("circuit_breaker._now_ist") as mock_now, \
             patch("circuit_breaker._today_ist") as mock_today, \
             patch("session_manager._now_ist") as mock_session_now:
            dt = datetime(year, month, day, hour, minute, 0)
            mock_now.return_value = dt
            mock_today.return_value = date(year, month, day)
            mock_session_now.return_value = dt
            yield dt
    return _mock_time


@pytest.fixture
def temp_json_file() -> Generator[str, None, None]:
    """Create a temporary JSON file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def sample_option_data() -> Dict:
    """Sample option data for testing."""
    return {
        "strike": 20000,
        "expiry": "2024-01-18",
        "option_type": "CE",
        "bid": 100,
        "ask": 102,
        "ltp": 101,
        "bid_qty": 500,
        "ask_qty": 600,
        "volume": 5000,
        "oi": 15000,
        "oi_change": 1200,
        "iv": 0.15,
        "delta": 0.50,
        "gamma": 0.001,
        "theta": -5.0,
        "vega": 10.0,
    }


@pytest.fixture
def sample_trade_data() -> Dict:
    """Sample trade data for testing."""
    return {
        "entry_time": "2024-01-15T10:00:00",
        "exit_time": "2024-01-15T14:00:00",
        "underlying": "NIFTY",
        "symbol": "NIFTY24JAN20000CE",
        "side": "LONG",
        "entry_price": 200,
        "exit_price": 250,
        "contracts": 2,
        "pnl": 100,
        "atr": 100,
        "mfe": 80,
        "mae": 30,
        "signal_metrics": {
            "adx": 22,
            "rsi": 55,
            "iv_percentile": 0.45,
        },
    }


@pytest.fixture
def sample_trades(sample_trade_data: Dict) -> List[Dict]:
    """Generate list of sample trades."""
    trades = []
    base_pnl = [200, -100, 150, -80, 300, -120, 180, -90, 250, -150]

    for i, pnl in enumerate(base_pnl):
        trade = sample_trade_data.copy()
        trade["entry_time"] = f"2024-01-{i+1:02d}T10:00:00"
        trade["pnl"] = pnl
        trade["signal_metrics"] = {
            "adx": 20 + (i % 5),
            "rsi": 50 + (i % 10),
            "iv_percentile": 0.30 + (i * 0.05),
        }
        trades.append(trade)

    return trades


@pytest.fixture
def high_liquidity_metrics():
    """High liquidity option metrics."""
    from liquidity_model import LiquidityMetrics
    return LiquidityMetrics(
        bid_ask_spread=1.0,
        spread_pct=0.005,
        bid_size=5000,
        ask_size=5000,
        volume=50000,
        oi=100000,
        oi_change=5000,
        last_trade_time="14:30:00",
    )


@pytest.fixture
def low_liquidity_metrics():
    """Low liquidity option metrics."""
    from liquidity_model import LiquidityMetrics
    return LiquidityMetrics(
        bid_ask_spread=10.0,
        spread_pct=0.08,
        bid_size=20,
        ask_size=30,
        volume=50,
        oi=200,
        oi_change=-50,
        last_trade_time="10:00:00",
    )


@pytest.fixture
def sample_position():
    """Sample position for adjustment testing."""
    from adjustment_engine import Position
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


# Markers for test categorization
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
