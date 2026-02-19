
# Trabot

## Project desc

Trabot is an educational options “research bot” for Indian markets built around Zerodha Kite Connect.

At a high level it:
- Pulls **underlying price candles** (via Kite historical data), computes a **directional signal** using indicators (EMA/RSI/ADX/ATR).
- Pulls a **small option-chain slice around ATM** (calls + puts) from Kite (NFO instruments + live quotes).
- Converts the signal into a **human-readable trade plan** (watch trigger, entry/SL/target on underlying levels) and suggests **which option contract(s)** to consider (CE/PE or spread in some cases).
- Provides **universe scanners** that iterate through many underlyings, apply liquidity/quality gates, compute IV + Greeks, apply risk/position sizing caps, and write results into CSVs.
- Keeps an **append-only journal** of “recommendations” so you can later evaluate how they performed.

> ⚠️ Educational tool only. Not financial advice.

---

## backtest.py

**What it does:** A tiny candle-based forward simulator used for quick research/backtests.

**Logic (plain English):**
- For each “trade entry” candle index, it assumes entry at the **next candle open**.
- It then scans forward candle-by-candle until it hits **stop**, **target**, or a **time-based exit**.
- Produces an “R multiple” outcome (risk-normalized return) and summary stats.

---

## config.py

**What it does:** Central place to configure the default underlying, timeframe, indicator periods, strike step, and backtest knobs.

**Logic (plain English):**
- You set things like `UNDERLYING`, `KITE_SYMBOL`, `INTERVAL`, EMA/RSI/ADX/ATR periods, stop/target ATR multipliers, strike step, and chain window width.
- Other scripts import these defaults so you don’t hardcode values everywhere.

---

## indicators.py

**What it does:** Lightweight indicator helpers (EMA, RSI, ATR, ADX, z-score) with flexible column handling.

**Logic (plain English):**
- Accepts dataframes with either `open/high/low/close` or `Open/High/Low/Close`.
- Computes indicators used by pattern detection and helper logic.

---

## iv_greeks.py

**What it does:** Black–Scholes implied volatility and Greeks approximation.

**Logic (plain English):**
- Uses a robust **bisection solver** to estimate IV from the option price.
- Computes **delta/gamma/vega/theta**; theta is returned per day (handy for risk caps).
- Also includes helpers for time-to-expiry in years.

---

## iv_store.py

**What it does:** Stores IV snapshots and computes rolling IV percentile.

**Logic (plain English):**
- Appends IV snapshots to a CSV history (e.g. `data/iv_history.csv`).
- Cleans data (skips invalid IV values) and keeps a “daily-ish” history.
- Computes **IV percentile** over a rolling window with EWMA smoothing, returning a value in `[0, 1]`.

---

## journal.py

**What it does:** Append-only journal of recommendations.

**Logic (plain English):**
- Appends each run’s recommendations into a single `data/reco_history.csv` (never overwritten).
- Also writes per-run “snapshot” CSVs so you can diff runs over time.

---

## kite_client.py

**What it does:** Creates an authenticated KiteConnect client.

**Logic (plain English):**
- Reads `KITE_API_KEY` and `KITE_ACCESS_TOKEN` from `.env`.
- Returns a ready-to-use `KiteConnect` client instance for the rest of the modules.

---

## kite_login.py

**What it does:** One-time helper to generate and store a Kite access token.

**Logic (plain English):**
- Prints the Kite login URL, asks you to paste back the `request_token`.
- Exchanges it for `access_token` using `KITE_API_SECRET`.
- Writes/updates `KITE_ACCESS_TOKEN=...` into `.env`.

---

## kite_chain.py

**What it does:** Builds a tight option-chain slice around ATM using Kite.

**Logic (plain English):**
- Loads NFO instruments (cached to `data/kite_instruments_NFO.csv` for speed).
- Picks an expiry (supports DTE band: `min_dte_days` / `max_dte_days`).
- Finds spot price + ATM strike and fetches quotes for strikes around ATM.
- Returns a `ChainSlice` containing calls/puts dataframes + spot/ATM.

---

## main.py

**What it does:** Single-underlying “demo runner” that prints market state + signal + suggested option legs.

**Logic (plain English):**
1. Fetch recent candles for the configured symbol and interval.
2. Compute signal (trend/strength/levels) from indicators.
3. Fetch option chain slice and “align” entry/SL/target to live spot (because candle close may lag).
4. Build a recommendation (CE/PE/spread/watch) and print it.
5. Optionally runs a walk-forward backtest over the fetched candle window.

---

## market_data.py

**What it does:** Fetches and caches historical candles using Kite.

**Logic (plain English):**
- Converts friendly interval strings (`15m`, `day`, etc.) into Kite’s interval names.
- Resolves instrument tokens (prefers `ltp()` token, falls back to instruments dump).
- Fetches historical data in safe chunks (Kite has limits).
- Caches candles in `data/candle_cache/` so repeated scans don’t hammer the API.

---

## market_sentiment.py

**What it does:** Builds a lightweight “market context” snapshot using only Kite-available signals.

**Logic (plain English):**
- Uses **India VIX** level + percentile as a regime proxy.
- Computes option-chain aggregates like **PCR** (from OI/volume) and a simple **skew proxy**.
- Finds “OI walls” (strikes with largest OI) as rough support/resistance markers.
- Adds index trend direction using the same `strategy.compute_signal()` logic.
- Appends snapshots to `data/market_sentiment_history.csv`.

---

## option_chain.py

**What it does:** Backwards-compatible wrapper around `kite_chain.get_kite_chain_slice()`.

**Logic (plain English):**
- Keeps the old `get_chain_slice()` API so the rest of the code can stay stable.
- Pulls defaults from `config.py`.

---

## patterns.py

**What it does:** Pattern / setup idea generator (trade ideas, not orders).

**Logic (plain English):**
- Normalizes candle columns and datetime handling.
- Runs a few setup detectors:
  - **Donchian breakout**: looks for range breakouts.
  - **Pullback trend**: trend + retracement style setup.
  - **Bollinger mean reversion**: “stretch then snap back” logic.
  - **Volatility squeeze**: compression then expansion potential.
- Outputs a list of `TradeIdea` objects with entry/stop/target and confidence.

---

## reco_analyzer.py

**What it does:** Evaluates historical recommendations from `data/reco_history.csv` using option candles.

**Logic (plain English):**
- Treats each journal row as “entered immediately after the recommendation time”.
- Uses the first candle OPEN after `ts_reco` as entry.
- Checks each candle for SL/target hits; flags ambiguous “both hit same bar”.
- Produces evaluated CSV + a text summary (win-rate, common loss tags, etc.).

---

## reco_analyzer_v22.py

**What it does:** Updated evaluator with more detailed metrics.

**Logic (plain English):**
- Same concept as `reco_analyzer.py`, but adds:
  - Better IST safety,
  - MFE/MAE tracking,
  - Target reach fraction,
  - Uses per-row `time_stop_min` if available.
- Writes “latest” plus timestamped evaluated outputs.

---

## recommender.py

**What it does:** Converts a `Signal` + `ChainSlice` into a “what to buy/watch” recommendation.

**Logic (plain English):**
- Builds a merged “preview table” (CE + PE side-by-side by strike).
- If the signal is `NO_TRADE`, outputs a **WATCH** recommendation with a trigger condition.
- If signal is LONG/SHORT, picks a contract near ATM (and may pick spread width based on expected move).
- Returns a structured `Recommendation` with legs, levels, reason, and a chain preview.

---

## risk_caps.py

**What it does:** Greeks-based position sizing and safety caps.

**Logic (plain English):**
- Calculates max lots based on:
  - Premium exposure,
  - Delta-notional exposure,
  - Vega (1% IV move sensitivity),
  - Theta (daily decay).
- Scales caps by capital + regime multipliers + confidence multipliers.
- Optionally tightens size for very near expiry (DTE penalty).

---

## scan_options.py

**What it does:** Full-universe scanner (older V2.1 style).

**Logic (plain English):**
- Builds a universe of optionable underlyings from the NFO instruments list.
- For each underlying:
  - fetches history (cached),
  - computes a directional signal,
  - grabs an ATM-centered chain slice,
  - scores candidate strikes with liquidity filters,
  - estimates IV + Greeks,
  - writes ranked results + top picks.
- Appends Top picks into `data/reco_history.csv` and writes run snapshots.

---

## scan_options_global.py

**What it does:** Global scan variant that can optionally incorporate “global context”.

**Logic (plain English):**
- Same scanning core, but may adjust scoring with a “risk-on/risk-off” bias if `yfinance` is available.
- Adds additional heuristics (liquidity thresholds by index vs stock, slippage model, “learned edge” style helper).
- Produces candidate recommendations across the universe.

---

## scan_options_global_v22.py

**What it does:** Lean V2.2 global scan.

**Logic (plain English):**
- Uses a directional confidence gate + high IV gate.
- Picks strikes using delta targeting and liquidity scoring.
- Builds option-based SL/target and time stop.
- Writes global scan results + reco latest + appends to reco_history.

---

## scan_options_v22.py

**What it does:** Full-universe scanner with V2.2 upgrades (IV percentile + regime + sentiment + risk caps).

**Logic (plain English):**
- Builds a universe from NFO instruments and iterates through underlyings.
- For each underlying:
  1. Fetch candles → compute signal (trend, levels, confidence proxies).
  2. Fetch chain slice → score strikes using liquidity gates (spread/OI/volume/min premium).
  3. Compute IV + Greeks for the chosen contract.
  4. Compute IV percentile and label regime (TREND/CHOP/VOLATILE).
  5. Pull market context (VIX/PCR/skew/OI-walls) and adjust risk/scoring.
  6. Compute position sizing via `risk_caps`.
- Writes scan result CSVs, “top 10” outputs, `reco_latest`, and appends to `reco_history.csv`.

---

## requirements.txt

**What it does:** Minimal Python dependencies list.

**Logic (plain English):**
- Lists the core libs needed for dataframes, math, env loading, and Kite API.

---

## Phase 4–6 (production upgrades)

### Phase 4: Portfolio-level risk caps (optional)
Trabot can (optionally) block recommendations that would violate portfolio caps.

Enable:
```bash
export TRABOT_PORTFOLIO_ENABLE=1
# if you want the scanner to reserve accepted recos into portfolio_state.json during the run:
export TRABOT_PORTFOLIO_RESERVE=1
```

State file (default): `data/portfolio_state.json`  
Cluster mapping (optional): `data/clusters.json` (e.g. `{"SBIN":"BANKS","AUROPHARMA":"PHARMA"}`)

Main caps:
- `TRABOT_PORTFOLIO_MAX_PREMIUM_FRAC` (default 0.35)
- `TRABOT_PORTFOLIO_MAX_DELTA_NOTIONAL_FRAC` (default 0.60)
- `TRABOT_PORTFOLIO_MAX_POS_PER_UNDERLYING` (default 2)
- `TRABOT_PORTFOLIO_MAX_POS_PER_CLUSTER` (default 4)

### Phase 5: Execution realism (fills + spread-aware triggers)
Analyzer uses `execution.py` for:
- entry worse by `k * spread_pct`
- exit worse by `k * spread_pct`
- SL/Target detection on *approx executable* high/low (spread-adjusted)

You can still override analyzer fill model:
```bash
python reco_analyzer_v22.py --fill_model realistic --fill_k 0.25
```

### Phase 6: Analyzer + tuning
Analyzer summary includes:
- Profit factor, Sharpe, Sortino, max drawdown
- loss attribution tags (SPREAD_WIDE, MULTI_LEG, CREDIT_STRUCTURE, STALE_QUOTES, etc.)

Walk-forward tuning on the analyzer output (no candle refetch):
```bash
python walkforward_tuner.py --csv data/reco_evaluated_v22_latest.csv
```
Outputs: `data/walkforward_report_latest.txt`
