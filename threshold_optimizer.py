"""threshold_optimizer.py

P3: Walk-forward threshold calibration for adaptive signal tuning.

This module implements rolling optimization of signal thresholds based on
historical performance, adapting to changing market conditions.

Key Features:
- Rolling window optimization (train on N days, test on M days)
- Parameter grid search across key thresholds
- Performance metrics: Sharpe, win rate, profit factor
- Regime-specific threshold adjustments
- Exports best params to JSON for scan-time use

Usage:
    from threshold_optimizer import optimize_thresholds, load_best_params

    # Run optimization
    best_params = optimize_thresholds(history_df, lookback_days=60)

    # Load for use in scanner
    params = load_best_params()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import math


@dataclass
class OptimizationResult:
    """Result of threshold optimization."""
    best_params: Dict[str, Any]
    sharpe: float
    win_rate: float
    profit_factor: float
    total_trades: int
    train_period: str
    test_period: str
    regime: str


@dataclass
class PerformanceMetrics:
    """Performance metrics for a parameter set."""
    total_pnl: float
    trade_count: int
    win_count: int
    loss_count: int
    avg_win: float
    avg_loss: float
    sharpe: float
    sortino: float
    profit_factor: float
    max_drawdown: float
    win_rate: float


# Environment configuration
OPTIMIZER_ENABLE = os.getenv("TRABOT_OPTIMIZER_ENABLE", "1").strip() == "1"
BEST_PARAMS_PATH = os.getenv("TRABOT_BEST_PARAMS_PATH", "data/best_params.json")
TRAIN_WINDOW_DAYS = int(os.getenv("TRABOT_TRAIN_WINDOW_DAYS", "60"))
TEST_WINDOW_DAYS = int(os.getenv("TRABOT_TEST_WINDOW_DAYS", "14"))
MIN_TRADES_FOR_VALID = int(os.getenv("TRABOT_MIN_TRADES_FOR_VALID", "20"))

# Default parameter grid
DEFAULT_PARAM_GRID = {
    "ADX_MIN": [15, 18, 20, 22, 25],
    "RSI_LONG_MIN": [50, 52, 55],
    "RSI_SHORT_MAX": [45, 48, 50],
    "STOP_ATR_MULT": [1.2, 1.5, 1.8, 2.0],
    "TARGET_ATR_MULT": [1.8, 2.0, 2.2, 2.5, 3.0],
    "MIN_IV_PERCENTILE": [0.20, 0.30, 0.40],
    "MAX_IV_PERCENTILE": [0.70, 0.80, 0.90],
}


def load_best_params() -> Dict[str, Any]:
    """Load best parameters from JSON file.

    Returns empty dict if file doesn't exist.
    """
    if not os.path.exists(BEST_PARAMS_PATH):
        return {}

    try:
        with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("params", {})
    except (json.JSONDecodeError, IOError):
        return {}


def save_best_params(
    params: Dict[str, Any],
    metrics: PerformanceMetrics,
    regime: str = "ALL",
) -> bool:
    """Save best parameters to JSON file."""
    os.makedirs(os.path.dirname(BEST_PARAMS_PATH) or "data", exist_ok=True)

    data = {
        "params": params,
        "metrics": {
            "sharpe": metrics.sharpe,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "trade_count": metrics.trade_count,
        },
        "regime": regime,
        "updated_at": datetime.now().isoformat(),
    }

    try:
        with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except IOError:
        return False


def calculate_metrics(trades: List[Dict]) -> PerformanceMetrics:
    """Calculate performance metrics from trade list.

    Each trade dict should have: pnl, entry_time, exit_time
    """
    if not trades:
        return PerformanceMetrics(
            total_pnl=0, trade_count=0, win_count=0, loss_count=0,
            avg_win=0, avg_loss=0, sharpe=0, sortino=0, profit_factor=0,
            max_drawdown=0, win_rate=0
        )

    pnls = [t.get("pnl", 0) for t in trades]
    total_pnl = sum(pnls)
    trade_count = len(pnls)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_count = len(wins)
    loss_count = len(losses)

    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0

    win_rate = win_count / trade_count if trade_count > 0 else 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1  # Avoid div by zero
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Sharpe ratio (assuming daily returns)
    if len(pnls) > 1:
        mean_pnl = total_pnl / trade_count
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_pnl = math.sqrt(variance) if variance > 0 else 1
        sharpe = (mean_pnl / std_pnl) * math.sqrt(252) if std_pnl > 0 else 0
    else:
        sharpe = 0

    # Sortino ratio (downside deviation only)
    negative_pnls = [p for p in pnls if p < 0]
    if negative_pnls and len(negative_pnls) > 1:
        mean_pnl = total_pnl / trade_count
        downside_variance = sum(p ** 2 for p in negative_pnls) / len(negative_pnls)
        downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 1
        sortino = (mean_pnl / downside_std) * math.sqrt(252) if downside_std > 0 else 0
    else:
        sortino = sharpe  # Fallback to Sharpe if no losses

    # Max drawdown
    equity = 0
    peak = 0
    max_dd = 0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return PerformanceMetrics(
        total_pnl=total_pnl,
        trade_count=trade_count,
        win_count=win_count,
        loss_count=loss_count,
        avg_win=avg_win,
        avg_loss=avg_loss,
        sharpe=sharpe,
        sortino=sortino,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        win_rate=win_rate,
    )


def filter_trades_by_params(
    trades: List[Dict],
    params: Dict[str, Any],
) -> List[Dict]:
    """Filter trades that would have passed given parameters.

    Requires trades to have signal metrics stored.
    """
    filtered = []

    for trade in trades:
        metrics = trade.get("signal_metrics", {})

        # ADX filter
        adx = metrics.get("adx", 0)
        if adx < params.get("ADX_MIN", 18):
            continue

        # RSI filter
        rsi = metrics.get("rsi", 50)
        side = trade.get("side", "")
        if side == "LONG" and rsi < params.get("RSI_LONG_MIN", 52):
            continue
        if side == "SHORT" and rsi > params.get("RSI_SHORT_MAX", 48):
            continue

        # IV percentile filter (if available)
        ivp = metrics.get("iv_percentile", 0.5)
        min_ivp = params.get("MIN_IV_PERCENTILE")
        max_ivp = params.get("MAX_IV_PERCENTILE")
        if min_ivp and ivp < min_ivp:
            continue
        if max_ivp and ivp > max_ivp:
            continue

        filtered.append(trade)

    return filtered


def simulate_with_params(
    trades: List[Dict],
    params: Dict[str, Any],
) -> List[Dict]:
    """Simulate trades with adjusted stop/target.

    Recalculates P&L based on stop/target multipliers.
    """
    stop_mult = params.get("STOP_ATR_MULT", 1.5)
    target_mult = params.get("TARGET_ATR_MULT", 2.2)

    simulated = []
    for trade in trades:
        sim_trade = trade.copy()

        # Get ATR and entry
        atr = trade.get("atr", 0)
        entry = trade.get("entry_price", 0)

        if atr > 0 and entry > 0:
            # Recalculate stop/target
            new_stop_dist = atr * stop_mult
            new_target_dist = atr * target_mult

            # Check if original trade hit stop or target
            mfe = trade.get("mfe", 0)  # Max favorable excursion
            mae = trade.get("mae", 0)  # Max adverse excursion

            # Determine outcome with new parameters
            if mae >= new_stop_dist:
                # Would have stopped out
                sim_trade["pnl"] = -new_stop_dist * trade.get("contracts", 1)
            elif mfe >= new_target_dist:
                # Would have hit target
                sim_trade["pnl"] = new_target_dist * trade.get("contracts", 1)
            else:
                # Use actual P&L (time stop or other exit)
                pass

        simulated.append(sim_trade)

    return simulated


def grid_search(
    trades: List[Dict],
    param_grid: Dict[str, List] = None,
    metric: str = "sharpe",
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """Grid search over parameter combinations.

    Args:
        trades: Historical trades with signal metrics
        param_grid: Parameter grid to search
        metric: Optimization metric ("sharpe", "profit_factor", "win_rate")

    Returns:
        (best_params, best_metrics)
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    best_params = {}
    best_score = float("-inf")
    best_metrics = None

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    def generate_combinations(index: int, current: Dict) -> List[Dict]:
        if index == len(param_names):
            return [current.copy()]

        results = []
        for value in param_values[index]:
            current[param_names[index]] = value
            results.extend(generate_combinations(index + 1, current))
        return results

    combinations = generate_combinations(0, {})

    for params in combinations:
        # Filter trades by signal parameters
        filtered = filter_trades_by_params(trades, params)

        if len(filtered) < MIN_TRADES_FOR_VALID:
            continue

        # Simulate with stop/target parameters
        simulated = simulate_with_params(filtered, params)

        # Calculate metrics
        metrics = calculate_metrics(simulated)

        # Get score based on chosen metric
        if metric == "sharpe":
            score = metrics.sharpe
        elif metric == "profit_factor":
            score = metrics.profit_factor
        elif metric == "win_rate":
            score = metrics.win_rate
        else:
            score = metrics.sharpe

        # Apply penalty for low trade count
        if metrics.trade_count < MIN_TRADES_FOR_VALID * 1.5:
            score *= 0.8

        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_metrics = metrics

    return best_params, best_metrics or PerformanceMetrics(
        total_pnl=0, trade_count=0, win_count=0, loss_count=0,
        avg_win=0, avg_loss=0, sharpe=0, sortino=0, profit_factor=0,
        max_drawdown=0, win_rate=0
    )


def walk_forward_optimize(
    trades: List[Dict],
    train_days: int = None,
    test_days: int = None,
    param_grid: Dict[str, List] = None,
) -> List[OptimizationResult]:
    """Walk-forward optimization with rolling windows.

    Args:
        trades: All historical trades (must have 'entry_time' field)
        train_days: Training window size
        test_days: Testing window size
        param_grid: Parameter grid

    Returns:
        List of optimization results for each window
    """
    if train_days is None:
        train_days = TRAIN_WINDOW_DAYS
    if test_days is None:
        test_days = TEST_WINDOW_DAYS

    # Sort trades by entry time
    sorted_trades = sorted(
        trades,
        key=lambda t: t.get("entry_time", "")
    )

    if not sorted_trades:
        return []

    # Get date range
    try:
        first_date = datetime.fromisoformat(sorted_trades[0]["entry_time"].split("T")[0])
        last_date = datetime.fromisoformat(sorted_trades[-1]["entry_time"].split("T")[0])
    except (KeyError, ValueError):
        return []

    results = []
    current_start = first_date

    while current_start + timedelta(days=train_days + test_days) <= last_date:
        train_end = current_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)

        # Filter trades for train and test periods
        train_trades = [
            t for t in sorted_trades
            if current_start.isoformat() <= t.get("entry_time", "")[:10] < train_end.isoformat()
        ]

        test_trades = [
            t for t in sorted_trades
            if train_end.isoformat() <= t.get("entry_time", "")[:10] < test_end.isoformat()
        ]

        if len(train_trades) >= MIN_TRADES_FOR_VALID:
            # Optimize on training data
            best_params, train_metrics = grid_search(train_trades, param_grid)

            if best_params:
                # Validate on test data
                filtered_test = filter_trades_by_params(test_trades, best_params)
                simulated_test = simulate_with_params(filtered_test, best_params)
                test_metrics = calculate_metrics(simulated_test)

                results.append(OptimizationResult(
                    best_params=best_params,
                    sharpe=test_metrics.sharpe,
                    win_rate=test_metrics.win_rate,
                    profit_factor=test_metrics.profit_factor,
                    total_trades=test_metrics.trade_count,
                    train_period=f"{current_start.date()} to {train_end.date()}",
                    test_period=f"{train_end.date()} to {test_end.date()}",
                    regime="ALL",
                ))

        # Slide window
        current_start += timedelta(days=test_days)

    return results


def optimize_thresholds(
    trades: List[Dict],
    param_grid: Dict[str, List] = None,
    lookback_days: int = None,
    regime: str = "ALL",
) -> Optional[Dict[str, Any]]:
    """Main optimization function.

    Args:
        trades: Historical trades
        param_grid: Parameter grid to search
        lookback_days: Only use trades from last N days
        regime: Filter by regime (TREND, CHOP, VOLATILE, ALL)

    Returns:
        Best parameters or None if optimization failed
    """
    if not OPTIMIZER_ENABLE:
        return None

    if lookback_days is None:
        lookback_days = TRAIN_WINDOW_DAYS

    # Filter by date
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    recent_trades = [
        t for t in trades
        if t.get("entry_time", "") >= cutoff
    ]

    # Filter by regime
    if regime != "ALL":
        recent_trades = [
            t for t in recent_trades
            if t.get("regime", "ALL") == regime
        ]

    if len(recent_trades) < MIN_TRADES_FOR_VALID:
        return None

    # Run grid search
    best_params, metrics = grid_search(recent_trades, param_grid)

    if best_params and metrics.sharpe > 0:
        save_best_params(best_params, metrics, regime)
        return best_params

    return None


def get_regime_specific_params(regime: str) -> Dict[str, Any]:
    """Get optimized parameters for a specific regime.

    Falls back to global params if regime-specific not available.
    """
    # Try regime-specific file first
    regime_path = BEST_PARAMS_PATH.replace(".json", f"_{regime.lower()}.json")

    if os.path.exists(regime_path):
        try:
            with open(regime_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("params", {})
        except (json.JSONDecodeError, IOError):
            pass

    # Fall back to global params
    return load_best_params()


def compare_param_sets(
    trades: List[Dict],
    param_sets: List[Dict[str, Any]],
) -> List[Tuple[Dict, PerformanceMetrics]]:
    """Compare multiple parameter sets on same trade data.

    Returns list of (params, metrics) sorted by Sharpe.
    """
    results = []

    for params in param_sets:
        filtered = filter_trades_by_params(trades, params)
        simulated = simulate_with_params(filtered, params)
        metrics = calculate_metrics(simulated)
        results.append((params, metrics))

    # Sort by Sharpe descending
    results.sort(key=lambda x: x[1].sharpe, reverse=True)
    return results


if __name__ == "__main__":
    # Demo
    print("Threshold Optimizer Demo")
    print("=" * 50)

    # Sample trade data
    sample_trades = [
        {
            "entry_time": "2024-01-01T10:00:00",
            "side": "LONG",
            "pnl": 500,
            "atr": 100,
            "entry_price": 20000,
            "contracts": 1,
            "mfe": 250,
            "mae": 80,
            "signal_metrics": {"adx": 22, "rsi": 55, "iv_percentile": 0.45},
        },
        {
            "entry_time": "2024-01-03T11:00:00",
            "side": "LONG",
            "pnl": -200,
            "atr": 90,
            "entry_price": 20100,
            "contracts": 1,
            "mfe": 100,
            "mae": 200,
            "signal_metrics": {"adx": 19, "rsi": 52, "iv_percentile": 0.50},
        },
        {
            "entry_time": "2024-01-05T09:30:00",
            "side": "SHORT",
            "pnl": 300,
            "atr": 110,
            "entry_price": 20200,
            "contracts": 1,
            "mfe": 350,
            "mae": 50,
            "signal_metrics": {"adx": 25, "rsi": 45, "iv_percentile": 0.35},
        },
    ] * 10  # Multiply for minimum trade count

    print(f"\nSample trades: {len(sample_trades)}")

    # Run optimization
    best_params, metrics = grid_search(sample_trades)

    print(f"\nBest Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print(f"\nMetrics:")
    print(f"  Sharpe: {metrics.sharpe:.2f}")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Trade Count: {metrics.trade_count}")

    # Load current best params
    current_params = load_best_params()
    if current_params:
        print(f"\nLoaded saved params: {current_params}")
