# Trabot

**Options Research Bot for Indian Markets**

An educational options scanner built on Zerodha Kite Connect. Scans NSE/NFO underlyings, generates directional signals, selects option structures, applies risk caps, and journals recommendations for later analysis.

> **Disclaimer:** Educational tool only. Not financial advice. Use at your own risk.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Module Reference](#module-reference)
- [Environment Variables](#environment-variables)
- [Data Files](#data-files)
- [Usage Examples](#usage-examples)

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Kite API

Create a `.env` file:

```bash
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
```

Generate access token:

```bash
python kite_login.py
```

### 3. Run Health Check

```bash
python doctor.py
```

### 4. Run Scanner

```bash
# Intraday mode (default)
python scan_options_v22.py --mode intraday

# Swing mode
python scan_options_v22.py --mode swing
```

---

## Architecture

Trabot is organized into phases, each adding capabilities:

```
Phase 1: Data Integrity          Phase 2: Regime Detection
+-----------------------+        +-----------------------+
| trabot_schema.py      |        | regime.py             |
| run_manifest.py       |        | - TREND/CHOP/VOLATILE |
| io_utils.py           |        | - Multi-TF alignment  |
| journal.py            |        +-----------------------+
+-----------------------+
                                 Phase 3: Strategy Engine
Phase 4: Portfolio Risk          +-----------------------+
+-----------------------+        | strategy_engine.py    |
| portfolio.py          |        | - Debit spreads       |
| correlation.py        |        | - Credit spreads      |
| portfolio_kite.py     |        | - Iron condors        |
+-----------------------+        | - Straddles/strangles |
                                 +-----------------------+
Phase 5: Execution Realism
+-----------------------+        Phase 6: Analysis
| execution.py          |        +-----------------------+
| quotes.py             |        | reco_analyzer_v22.py  |
| throttle.py           |        | walkforward_tuner.py  |
| kite_client.py        |        | tuning.py             |
+-----------------------+        +-----------------------+

P0-P3: Advanced Risk & Trading Intelligence
+-----------------------+        +-----------------------+
| circuit_breaker.py    |        | expected_move.py      |
| event_calendar.py     |        | iv_term_structure.py  |
| session_manager.py    |        | probability_analysis.py|
+-----------------------+        +-----------------------+
| greeks_monitor.py     |        | threshold_optimizer.py|
| adjustment_engine.py  |        | liquidity_model.py    |
+-----------------------+        +-----------------------+
```

### Signal Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ market_data │───>│  strategy   │───>│   regime    │───>│  strategy   │
│  (candles)  │    │  (signal)   │    │ (detection) │    │   engine    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│  portfolio  │<───│  risk_caps  │<───│  iv_greeks  │<──────────┘
│   (caps)    │    │  (sizing)   │    │  (Greeks)   │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                          v
                   ┌─────────────┐
                   │   journal   │
                   │  (output)   │
                   └─────────────┘
```

---

## Module Reference

### Core Scanner

#### `scan_options_v22.py`
**Main universe scanner with all phases integrated.**

- Builds universe from NFO instruments
- Fetches candles via `market_data.py`
- Computes signal via `strategy.py`
- Detects regime via `regime.py` (TREND/CHOP/VOLATILE)
- Selects structure via `strategy_engine.py` (spreads, condors, single-leg)
- Computes IV + Greeks via `iv_greeks.py`
- Applies position sizing via `risk_caps.py`
- Checks portfolio caps via `portfolio.py` (optional)
- Deduplicates with cooldown window
- Outputs: CSV files + appends to `reco_history.csv`

Modes: `intraday` (15m candles, 0-7 DTE) or `swing` (60m candles, 4-14 DTE)

---

### Signal Generation

#### `strategy.py`
**Trend-follow signal generator.**

Signal logic:
- **LONG**: EMA_fast > EMA_slow AND ADX >= 18 AND RSI >= 52
- **SHORT**: EMA_fast < EMA_slow AND ADX >= 18 AND RSI <= 48
- **NO_TRADE**: Filters not met (but watch levels provided)

Returns `Signal` dataclass with:
- `side`: LONG / SHORT / NO_TRADE
- `entry`, `stop`, `target`: Price levels (ATR-based)
- `metrics`: Dict with EMA, RSI, ADX, ATR values + watch info

#### `regime.py`
**Candle-driven regime classifier.**

Labels:
- **TREND**: High ADX + directional alignment across timeframes
- **CHOP**: Low ADX + poor alignment
- **VOLATILE**: High ATR% or high IV percentile

Functions:
- `timeframe_signature(df)`: Computes direction, ADX, ATR% for one timeframe
- `alignment_gate(ltf, htf, dtf)`: Returns True if all timeframes agree on direction
- `detect_regime(...)`: Returns `RegimeResult` with label + confidence (0-1)

#### `strategy_engine.py`
**Multi-leg strategy selector (Phase 3).**

Selects structure based on:
- Signal side (LONG/SHORT)
- Regime (TREND/CHOP/VOLATILE)
- IV percentile (high/low)
- Signal strength

Structures supported:
| Action | Description |
|--------|-------------|
| `BUY_CE` / `BUY_PE` | Single-leg directional |
| `BULL_CALL_SPREAD` | Buy ATM call, sell OTM call |
| `BEAR_PUT_SPREAD` | Buy ATM put, sell OTM put |
| `BULL_PUT_CREDIT` | Sell OTM put, buy further OTM put |
| `BEAR_CALL_CREDIT` | Sell OTM call, buy further OTM call |
| `IRON_CONDOR` | Bull put + bear call credit spreads |
| `LONG_STRADDLE` | Buy ATM call + ATM put |
| `LONG_STRANGLE` | Buy OTM call + OTM put |

---

### Greeks & IV

#### `iv_greeks.py`
**Black-Scholes IV solver and Greeks calculator.**

Functions:
- `implied_volatility(price, S, K, T, r, right)`: Bisection solver, returns `(iv, ok_flag)`
- `greeks(S, K, T, r, iv, right)`: Returns dict with `delta`, `gamma`, `vega_1pct`, `theta_day`
- `time_to_expiry_years(expiry)`: Converts expiry string to T in years

Notes:
- Theta is per calendar day (for risk caps)
- Vega is per 1% IV move
- Risk-free rate hardcoded at 6%

#### `iv_store.py`
**IV history storage and percentile calculation.**

- Appends IV snapshots to `data/iv_history.csv`
- Computes rolling IV percentile with EWMA smoothing
- Returns percentile in [0, 1] range

---

### Risk Management

#### `risk_caps.py`
**Greeks-based position sizing.**

Caps (per trade):
| Cap | Default % of Capital |
|-----|---------------------|
| Premium | 8% (high) / 5% (moderate) |
| Delta-notional | 100% |
| Vega (1% IV move) | 2% |
| Theta (daily) | 0.6% |

Multipliers applied:
- **Regime**: TREND (1.0x), VOLATILE (0.65-0.85x), CHOP (0.80-1.05x)
- **Confidence**: High (1.0x), Low (0.70x)
- **DTE**: <=1 (0.65x), 2 (0.75x), 3 (0.85x), 4+ (1.0x)

Final lots = `min(by_premium, by_delta, by_vega, by_theta, max_lots_hard)`

#### `portfolio.py`
**Portfolio-level risk management (Phase 4).**

Aggregate caps:
| Cap | Default | Formula |
|-----|---------|---------|
| Premium at risk | 35% | sum(premium_at_risk) <= frac * capital |
| Delta-notional | 60% | sum(\|net_delta\| * spot * contracts) <= frac * capital |
| Vega | 60% | sum(\|vega_1pct\| * contracts) <= frac * capital / 1000 |
| Gamma | 50% | sum(\|gamma\| * contracts) <= frac * capital / 1000 |
| Theta | 80% | sum(\|theta_day\| * contracts) <= frac * capital / 1000 |
| Positions per underlying | 2 | Hard count limit |
| Positions per cluster | 4 | Hard count limit |

Correlation cap (optional):
- Computes correlation of daily returns between new position and existing portfolio
- Rejects if max correlation > threshold (default 0.85)

State file: `data/portfolio_state.json`
Cluster mapping: `data/clusters.json`

#### `correlation.py`
**Correlation-based portfolio diversification.**

- Fetches daily candles for underlyings
- Computes pairwise correlation of close-to-close returns
- Returns max correlation with existing portfolio positions
- LRU cached for performance

---

### Data & Kite Integration

#### `kite_client.py`
**Authenticated Kite client with rate limiting.**

Features:
- Token bucket rate limiter (default: 3 RPS, burst 6)
- Exponential backoff on 429 errors
- Wrapper functions: `kite_quote_safe()`, `kite_ltp_safe()`, `kite_historical_safe()`, `kite_positions_safe()`

#### `kite_chain.py`
**Option chain slice builder.**

- Loads NFO instruments (cached to CSV)
- Picks expiry within DTE band
- Fetches quotes for strikes around ATM
- Returns `ChainSlice` with calls/puts DataFrames + spot/ATM

#### `kite_login.py`
**One-time token generator.**

- Prints Kite login URL
- Exchanges request_token for access_token
- Writes to `.env` file

#### `market_data.py`
**Historical candle fetcher with caching.**

- Maps interval strings to Kite format (e.g., `15m` -> `15minute`)
- Fetches in safe chunks (respects Kite limits)
- Caches to `data/candle_cache/` with TTL
- Tracks cache hit/miss stats

#### `quotes.py`
**Quote caching layer (Phase 5).**

- TTL-based cache (default 3 seconds)
- Batch fetch with chunking
- Normalizes bid/ask/mid/ltp/spread_pct

---

### Market Context

#### `market_sentiment.py`
**Market context snapshot builder.**

Computes:
- **VIX**: Level + 30-day percentile
- **PCR**: Put-Call ratio from OI and volume
- **Skew**: OTM put IV - OTM call IV (proxy)
- **OI Walls**: Strikes with highest OI (support/resistance)
- **Index Trend**: Direction using same signal logic

Returns `MarketContext` with bias (BULLISH/BEARISH/NEUTRAL) and strength score.

---

### Output & Journaling

#### `journal.py`
**Append-only recommendation journal.**

- Appends to `data/reco_history.csv` (never overwrites)
- Writes per-run snapshots
- Schema-stable output

#### `trabot_schema.py`
**Schema versioning for CSV stability (Phase 1).**

- Defines 71 columns for reco rows
- `normalize_row()` ensures all rows match schema
- Unknown keys captured in `extra_json` column
- Current schema version: 3

#### `run_manifest.py`
**Per-run reproducibility manifest (Phase 1).**

Writes `data/runs/<run_id>/manifest.json` with:
- All `TRABOT_*` environment variables
- Config snapshot
- Universe info
- Runtime stats
- Output file paths

---

### Analysis & Tuning

#### `reco_analyzer_v22.py`
**Recommendation evaluator (Phase 6).**

- Fetches option candles for each reco
- Simulates entry at first candle open after `ts_reco`
- Tracks SL/target/time-stop hits
- Computes MFE/MAE (max favorable/adverse excursion)
- Reports: win rate, profit factor, Sharpe, Sortino, max drawdown
- Loss attribution tags

#### `walkforward_tuner.py`
**Walk-forward parameter tuner (Phase 6).**

- Reads evaluated CSV
- Tests parameter combinations on rolling windows
- Outputs `data/best_params.json` for scan-time overrides

#### `execution.py`
**Execution realism primitives (Phase 5).**

Fill models:
| Model | Penalty |
|-------|---------|
| `mid` / `optimistic` | 0% of spread |
| `bid` / `ask` | 50% of spread |
| `realistic` / `mid_k` | k% of spread (default k=0.25) |
| `pessimistic` | 100% of spread |

---

### Utilities

#### `io_utils.py`
**Atomic file operations.**

- `atomic_write_text()`: Write via temp file + rename
- `atomic_write_json()`: JSON with atomic write
- `append_csv_row()`: Schema-stable CSV append

#### `logging_utils.py`
**Structured JSON logging.**

- JSON Lines format for production
- Falls back to plain text if JSON fails
- Configurable via `TRABOT_LOG_LEVEL` and `TRABOT_LOG_JSON`

#### `stats.py`
**Lightweight stats collector.**

Tracks:
- API calls, retries, 429 errors
- API latency (sum, count, avg)
- Cache hits/misses (candle, quote)

#### `throttle.py`
**Token bucket rate limiter.**

- Used by `kite_client.py` for API rate limiting
- Configurable rate and burst size

#### `doctor.py`
**Environment health check.**

Checks:
- Python version
- Schema version
- Required env vars (KITE_API_KEY, KITE_ACCESS_TOKEN)
- Module imports

---

### P0-P3: Advanced Risk & Trading Intelligence

#### `circuit_breaker.py` (P0)
**Daily loss limit and circuit breaker for capital protection.**

States:
- `ACTIVE`: Normal trading allowed
- `WARNING_50/70/90`: Approaching limit
- `TRIPPED`: Limit exceeded, no new entries

Features:
- Tracks realized + unrealized P&L
- Auto-resets at market open (9:15 IST)
- Consecutive loss tracking
- Persists state to JSON

Functions:
- `check_circuit_breaker(capital)`: Returns `(ok, status, reason)`
- `record_trade_result(trade_id, pnl)`: Record closed trade
- `get_remaining_risk_budget(capital)`: Remaining risk budget

#### `event_calendar.py` (P0)
**Event calendar for avoiding earnings, RBI policy, and expiry events.**

Event Types:
- `EARNINGS`: Company quarterly results
- `RBI_POLICY`: RBI monetary policy announcements
- `MONTHLY_EXPIRY`: Last Thursday of month
- `WEEKLY_EXPIRY`: Thursday weekly expiry
- `BUDGET`: Union budget day

Functions:
- `should_avoid_entry(underlying, dte)`: Returns `(avoid, reason)`
- `position_spans_event(underlying, entry_date, dte)`: Check for conflicts
- `get_upcoming_events(days_ahead)`: List upcoming events

#### `session_manager.py` (P1)
**Session-aware trading rules for Indian markets.**

Sessions (IST):
- `OPENING` (9:15-10:00): High volatility, scalps only
- `MORNING` (10:00-12:00): Best for trend entries
- `LUNCH` (12:00-13:30): Low volume, avoid entries
- `AFTERNOON` (13:30-14:30): Moderate, selective entries
- `CLOSING` (14:30-15:30): Strong momentum only

Functions:
- `should_allow_entry(signal_strength, strategy_type)`: Returns `(allowed, reason)`
- `get_adjusted_levels(entry, stop, target)`: Session-adjusted levels
- `get_size_multiplier()`: Position size multiplier

#### `expected_move.py` (P1)
**Expected move calculation for spread width selection.**

Formula: `EM = Spot × IV × sqrt(DTE/365) × Z-score`

Z-scores:
- 68% (1 std): 1.00
- 80%: 1.28
- 90%: 1.64
- 95% (2 std): 1.96

Functions:
- `calculate_expected_move(spot, iv, dte, confidence)`: Returns `ExpectedMoveResult`
- `suggest_spread_width(expected_move, step, strategy_type)`: Optimal width
- `get_credit_spread_strikes(spot, iv, dte, side, step)`: Strike selection

#### `iv_term_structure.py` (P1)
**IV term structure analysis for strategy selection.**

Structure Types:
- `CONTANGO`: Near IV < Far IV (normal, favor front-month selling)
- `BACKWARDATION`: Near IV > Far IV (avoid calendars)
- `FLAT`: Neutral

Functions:
- `analyze_term_structure(underlying, options_data, spot)`: Returns `TermStructureResult`
- `get_term_structure_bias(ts_result)`: Returns `(bias, adjustment_factor)`
- `get_optimal_expiry(ts_result, side, default_dte)`: Optimal DTE

#### `greeks_monitor.py` (P2)
**Greeks-based position monitoring and stop loss management.**

Stop Types:
- `DELTA_BREACH`: Position too directional
- `THETA_BREACH`: Excessive daily decay
- `VEGA_BREACH`: IV move against position
- `GAMMA_BREACH`: Near-expiry risk

Functions:
- `check_greeks_stops(position, spot, thresholds)`: Returns `GreeksStopResult`
- `get_greeks_health_score(position)`: 0-1 health score
- `suggest_adjustment(position, stop_result)`: Adjustment suggestions

#### `probability_analysis.py` (P2)
**Probability of Profit (POP) and Expected Value calculations.**

Methods:
- Delta-approximated (quick)
- Log-normal distribution (accurate)

Functions:
- `calculate_pop(params)`: Probability of profit
- `expected_value(pop, max_profit, max_loss)`: E[V] calculation
- `analyze_trade_probability(params, max_profit, max_loss)`: Full analysis
- `meets_probability_filters(result, min_pop, min_ev)`: Filter check

#### `adjustment_engine.py` (P2)
**Simplified adjustment framework for options positions.**

Adjustment Types:
- `ROLL_OUT`: Same strike, later expiry
- `ROLL_UP/DOWN`: Different strike
- `CONVERT_TO_SPREAD`: Add leg to define risk
- `ADD_HEDGE`: Protective leg
- `CLOSE_PARTIAL/FULL`: Exit position

Functions:
- `suggest_adjustments(position, max_loss, stop_loss)`: List of suggestions
- `classify_position_status(position, max_loss)`: Status classification
- `should_auto_adjust(position, max_loss)`: Auto-adjustment trigger

#### `threshold_optimizer.py` (P3)
**Walk-forward threshold calibration for adaptive tuning.**

Features:
- Rolling window optimization
- Parameter grid search
- Performance metrics: Sharpe, win rate, profit factor

Functions:
- `optimize_thresholds(trades, param_grid, lookback_days)`: Find best params
- `load_best_params()`: Load from JSON
- `walk_forward_optimize(trades, train_days, test_days)`: Full walk-forward

#### `liquidity_model.py` (P3)
**Advanced liquidity scoring for options selection.**

Factors (weighted):
- Bid-ask spread (30%)
- Order book depth (20%)
- Volume (20%)
- Open interest (20%)
- OI trend (10%)

Grades: A (Excellent) → F (Avoid)

Functions:
- `calculate_liquidity_score(metrics)`: Returns `LiquidityScore`
- `estimate_slippage(contracts, metrics)`: Slippage estimate
- `filter_by_liquidity(options, min_score)`: Filter options

---

### Legacy/Support Modules

| Module | Description |
|--------|-------------|
| `backtest.py` | Candle-based forward simulator |
| `config.py` | Default configuration values |
| `indicators.py` | EMA, RSI, ATR, ADX helpers |
| `main.py` | Single-underlying demo runner |
| `option_chain.py` | Backwards-compatible wrapper |
| `patterns.py` | Pattern/setup idea generator |
| `recommender.py` | Signal -> recommendation converter |
| `reco_analyzer.py` | Legacy analyzer (use v22) |
| `scan_options.py` | Legacy scanner (use v22) |
| `scan_options_global.py` | Legacy global scanner |
| `scan_options_global_v22.py` | Global scanner v22 |
| `tuning.py` | Parameter tuning helpers |
| `portfolio_kite.py` | Hydrate portfolio from Kite positions |

---

## Environment Variables

### Kite API

| Variable | Default | Description |
|----------|---------|-------------|
| `KITE_API_KEY` | (required) | Kite Connect API key |
| `KITE_ACCESS_TOKEN` | (required) | Daily access token |
| `KITE_API_SECRET` | (for login) | API secret for token generation |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_KITE_RPS` | 3.0 | Requests per second limit |
| `TRABOT_KITE_BURST` | 6 | Burst allowance |
| `TRABOT_KITE_MAX_TRIES` | 6 | Max retry attempts |
| `TRABOT_KITE_BACKOFF_BASE` | 0.8 | Backoff base (seconds) |
| `TRABOT_QUOTE_TTL_SEC` | 3 | Quote cache TTL |
| `TRABOT_CACHE_TTL_MIN` | 5 | Candle cache TTL (minutes) |

### Scanner Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_MODE` | intraday | Scanner mode (intraday/swing) |
| `TRABOT_CAPITAL` | 100000 | Capital for sizing |
| `TRABOT_RISK_PROFILE` | high | Risk profile (high/moderate) |
| `LOOKBACK_DAYS` | 180 | Candle history lookback |
| `INTERVAL` | day | Candle interval |
| `UNIVERSE_START` | 0 | Universe slice start |
| `UNIVERSE_COUNT` | (all) | Universe slice count |
| `STRIKES_AROUND_ATM` | 12 | Strikes to fetch per side |

### Signal Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `EMA_FAST` | 20 | Fast EMA period |
| `EMA_SLOW` | 50 | Slow EMA period |
| `RSI_PERIOD` | 14 | RSI period |
| `ADX_PERIOD` | 14 | ADX period |
| `ADX_MIN` | 18 | Minimum ADX for signal |
| `ATR_PERIOD` | 14 | ATR period |
| `STOP_ATR_MULT` | 1.5 | Stop distance (ATR multiple) |
| `TARGET_ATR_MULT` | 2.2 | Target distance (ATR multiple) |

### Regime & Alignment

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_REQUIRE_HTF_ALIGN` | 0 | Require HTF alignment (0/1) |
| `TRABOT_SKIP_CHOP` | 0 | Skip CHOP regime (0/1) |
| `TRABOT_BLOCK_LONG_PREMIUM_IVP` | 0.0 | Block long premium above IVP |
| `ALIGN_MODE` | (auto) | Alignment mode (off/soft/hard) |

### Portfolio Caps

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_PORTFOLIO_ENABLE` | 0 | Enable portfolio caps (0/1) |
| `TRABOT_PORTFOLIO_RESERVE` | 0 | Reserve positions during scan (0/1) |
| `TRABOT_PORTFOLIO_STATE_PATH` | data/portfolio_state.json | State file path |
| `TRABOT_PORTFOLIO_MAX_PREMIUM_FRAC` | 0.35 | Max premium at risk (fraction) |
| `TRABOT_PORTFOLIO_MAX_DELTA_NOTIONAL_FRAC` | 0.60 | Max delta-notional (fraction) |
| `TRABOT_PORTFOLIO_MAX_VEGA_FRAC` | 0.60 | Max vega exposure (fraction) |
| `TRABOT_PORTFOLIO_MAX_GAMMA_FRAC` | 0.50 | Max gamma exposure (fraction) |
| `TRABOT_PORTFOLIO_MAX_THETA_FRAC` | 0.80 | Max theta exposure (fraction) |
| `TRABOT_PORTFOLIO_MAX_POS_PER_UNDERLYING` | 2 | Max positions per underlying |
| `TRABOT_PORTFOLIO_MAX_POS_PER_CLUSTER` | 4 | Max positions per cluster |
| `TRABOT_PORTFOLIO_CORR_ENABLE` | 0 | Enable correlation cap (0/1) |
| `TRABOT_PORTFOLIO_MAX_CORR` | 0.85 | Max correlation threshold |
| `TRABOT_PORTFOLIO_CORR_LOOKBACK_DAYS` | 120 | Correlation lookback |

### Stop/Target Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_SINGLE_LEG_SL_TGT_MODE` | premium | SL/target mode (premium/delta) |
| `TRABOT_MAX_TARGET_MOVE_PCT` | 0.08 | Max target move (% of spot) |
| `TRABOT_MAX_STOP_MOVE_PCT` | 0.04 | Max stop move (% of spot) |
| `TRABOT_TGT_CAP_MULT` | 3.0 | Target cap (multiple of entry) |
| `TRABOT_WIDTH_MAX_MOVE_PCT` | 0.03 | Max spread width (% of spot) |
| `TIME_STOP_MIN` | 90 | Time stop (minutes, intraday) |

### Deduplication

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_DEDUP_COOLDOWN_MIN` | 30 | Cooldown between same reco (minutes) |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_LOG_LEVEL` | INFO | Log level |
| `TRABOT_LOG_JSON` | 1 | JSON log format (0/1) |

### Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_APPLY_BEST_PARAMS` | 0 | Apply best_params.json (0/1) |
| `TRABOT_BEST_PARAMS_PATH` | data/best_params.json | Best params file |

### P0: Circuit Breaker

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_CIRCUIT_BREAKER_ENABLE` | 1 | Enable circuit breaker (0/1) |
| `TRABOT_MAX_DAILY_LOSS_PCT` | 2.0 | Max daily loss as % of capital |
| `TRABOT_MAX_CONSECUTIVE_LOSSES` | 5 | Max consecutive losses before trip |
| `TRABOT_CIRCUIT_BREAKER_STATE_PATH` | data/circuit_breaker_state.json | State file |

### P0: Event Calendar

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_EVENT_CHECK_ENABLE` | 1 | Enable event calendar check (0/1) |
| `TRABOT_EVENT_EARNINGS_BUFFER_HRS` | 24 | Buffer hours for earnings |
| `TRABOT_EVENT_RBI_BUFFER_HRS` | 12 | Buffer hours for RBI policy |
| `TRABOT_EVENT_EXPIRY_BUFFER_HRS` | 4 | Buffer hours before expiry |
| `TRABOT_EVENTS_FILE` | data/events.json | Events data file |

### P1: Session Manager

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_SESSION_RULES_ENABLE` | 1 | Enable session rules (0/1) |
| `TRABOT_BLOCK_LUNCH_ENTRIES` | 1 | Block entries during lunch (0/1) |
| `TRABOT_BLOCK_CLOSING_NEW` | 1 | Block new entries during closing (0/1) |
| `TRABOT_OPENING_SCALP_ONLY` | 0 | Opening session scalps only (0/1) |

### P1: Expected Move

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_EXPECTED_MOVE_ENABLE` | 1 | Enable expected move calc (0/1) |
| `TRABOT_MAX_SPREAD_WIDTH_STEPS` | 10 | Max spread width in steps |
| `TRABOT_MIN_SPREAD_WIDTH_STEPS` | 2 | Min spread width in steps |

### P1: IV Term Structure

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_TERM_STRUCTURE_ENABLE` | 1 | Enable term structure analysis (0/1) |
| `TRABOT_CONTANGO_THRESHOLD` | 0.02 | Contango threshold (2%) |
| `TRABOT_BACKWARDATION_THRESHOLD` | -0.02 | Backwardation threshold |

### P2: Greeks Monitor

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_GREEKS_STOPS_ENABLE` | 1 | Enable Greeks-based stops (0/1) |
| `TRABOT_MAX_DELTA_LONG` | 0.75 | Max delta for long options |
| `TRABOT_MIN_DELTA_LONG` | 0.15 | Min delta for long options |
| `TRABOT_MAX_DELTA_SHORT` | 0.70 | Max delta for short options |
| `TRABOT_GAMMA_WARNING_DTE` | 2 | DTE for gamma warnings |

### P2: Probability Analysis

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_PROBABILITY_FILTER_ENABLE` | 0 | Enable POP filter (0/1) |
| `TRABOT_MIN_POP_THRESHOLD` | 0.40 | Minimum probability of profit |
| `TRABOT_MIN_EXPECTED_VALUE` | 0 | Minimum expected value |
| `TRABOT_PROB_METHOD` | lognormal | POP method (delta/lognormal) |

### P3: Liquidity Model

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_LIQUIDITY_MODEL_ENABLE` | 1 | Enable liquidity scoring (0/1) |
| `TRABOT_MIN_LIQUIDITY_SCORE` | 40 | Minimum liquidity score (0-100) |

### P3: Threshold Optimizer

| Variable | Default | Description |
|----------|---------|-------------|
| `TRABOT_OPTIMIZER_ENABLE` | 1 | Enable optimizer (0/1) |
| `TRABOT_TRAIN_WINDOW_DAYS` | 60 | Training window days |
| `TRABOT_TEST_WINDOW_DAYS` | 14 | Testing window days |
| `TRABOT_MIN_TRADES_FOR_VALID` | 20 | Min trades for valid optimization |

---

## Data Files

### Outputs

| Path | Description |
|------|-------------|
| `data/reco_history.csv` | Append-only recommendation journal |
| `data/reco_latest_v22.csv` | Latest run recommendations |
| `data/options_scan_results_v22.csv` | Full scan results |
| `data/options_top10_v22.csv` | Top 10 recommendations |
| `data/runs/<run_id>/manifest.json` | Run manifest with config snapshot |
| `data/reco_evaluated_v22_latest.csv` | Evaluated recommendations |

### State

| Path | Description |
|------|-------------|
| `data/portfolio_state.json` | Portfolio position state |
| `data/clusters.json` | Underlying -> cluster mapping |
| `data/best_params.json` | Walk-forward tuned parameters |
| `data/circuit_breaker_state.json` | Circuit breaker daily state |
| `data/events.json` | Event calendar data |

### Cache

| Path | Description |
|------|-------------|
| `data/candle_cache/` | Cached historical candles |
| `data/kite_instruments_NFO.csv` | Cached NFO instruments |
| `data/kite_instruments_NSE.csv` | Cached NSE instruments |
| `data/iv_history.csv` | IV snapshot history |
| `data/market_sentiment_history.csv` | Market context history |

---

## Usage Examples

### Basic Scan

```bash
# Intraday scan
python scan_options_v22.py --mode intraday

# Swing scan
python scan_options_v22.py --mode swing
```

### With Portfolio Caps

```bash
export TRABOT_PORTFOLIO_ENABLE=1
export TRABOT_PORTFOLIO_RESERVE=1
export TRABOT_CAPITAL=500000
python scan_options_v22.py
```

### With Correlation Check

```bash
export TRABOT_PORTFOLIO_ENABLE=1
export TRABOT_PORTFOLIO_CORR_ENABLE=1
export TRABOT_PORTFOLIO_MAX_CORR=0.80
python scan_options_v22.py
```

### Skip CHOP Regime

```bash
export TRABOT_SKIP_CHOP=1
python scan_options_v22.py
```

### Require HTF Alignment

```bash
export TRABOT_REQUIRE_HTF_ALIGN=1
python scan_options_v22.py
```

### Analyze Recommendations

```bash
python reco_analyzer_v22.py

# With realistic fills
python reco_analyzer_v22.py --fill_model realistic --fill_k 0.25
```

### Walk-Forward Tuning

```bash
python walkforward_tuner.py --csv data/reco_evaluated_v22_latest.csv
```

---

## License

Educational use only. No warranty. Not financial advice.
