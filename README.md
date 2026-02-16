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

## Quickstart (local)

### Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p data
.env (DO NOT COMMIT)
Required keys:

KITE_API_KEY=...

KITE_ACCESS_TOKEN=...

Only needed for login helper:

KITE_API_SECRET=...

Generate / refresh token:

python kite_login.py
How to run scanners (intraday vs swing)
V2.2 scanners (recommended)
Intraday

python scan_options_v22.py --mode intraday
python scan_options_global_v22.py --mode intraday
Swing

python scan_options_v22.py --mode swing
python scan_options_global_v22.py --mode swing
You can also set mode via env:

TRABOT_MODE=intraday python scan_options_v22.py
TRABOT_MODE=swing    python scan_options_v22.py
Set risk profile + capital
TRABOT_RISK_PROFILE=high     TRABOT_CAPITAL=100000 python scan_options_v22.py --mode intraday
TRABOT_RISK_PROFILE=moderate TRABOT_CAPITAL=400000 python scan_options_v22.py --mode intraday
Updated default capital buckets (if TRABOT_CAPITAL is not set)
If you do NOT set TRABOT_CAPITAL, defaults are:

high → ₹100,000

moderate → ₹400,000

If you set TRABOT_CAPITAL, that value always overrides defaults.

Rate limits: recommended knobs
Kite rate limits can trigger “Too many requests” when scanning too many symbols/strikes too quickly.
Use these knobs:

UNIVERSE_START, UNIVERSE_COUNT — scan a slice instead of full universe

STRIKES_AROUND_ATM — reduce strikes fetched per underlying

SLEEP_BETWEEN_SYMBOLS — slow down the scan

Example (faster + safer):

TRABOT_SENTIMENT_ENABLED=0 \
UNIVERSE_COUNT=100 \
STRIKES_AROUND_ATM=6 \
SLEEP_BETWEEN_SYMBOLS=0.35 \
python scan_options_v22.py --mode intraday
Liquidity gates (why “No candidates found” happens)
These filters can eliminate all candidates (especially off-hours/weekends):

TRABOT_MIN_MID_PRICE

TRABOT_MAX_SPREAD_PCT

TRABOT_MIN_OI

TRABOT_MIN_VOL

Example (looser gates for research):

TRABOT_MIN_MID_PRICE=1 \
TRABOT_MAX_SPREAD_PCT=0.50 \
TRABOT_MIN_OI=0 TRABOT_MIN_VOL=0 \
python scan_options_v22.py --mode intraday
“pass_caps”, “max_lots”, and unsized picks
max_lots = maximum lots allowed by the risk model (premium, delta-notional, vega/theta, etc.)

pass_caps = True if you can take at least 1 lot within caps

If a row has max_lots=0 / pass_caps=False, it is an unsized pick:
the system is saying “don’t trade this with this capital/risk settings.”

Outputs (data folder)
Typical outputs under data/:

options_scan_results_v22_<mode>.csv (latest)

options_scan_results_v22_<timestamp>.csv (snapshot)

options_top10_v22_<mode>.csv (latest)

reco_latest_v22_<mode>.csv (latest recommendations)

reco_v22_<timestamp>.csv (snapshot recommendations)

reco_history.csv (append-only)

⚠️ Note: if you have run multiple versions over time, reco_history.csv may contain mixed schemas.

How to evaluate results (SL hits / target hits / time exits)
Evaluate one run file (cleanest)
python reco_analyzer_v22.py --history data/reco_v22_<RUN_ID>.csv
It writes:

data/reco_evaluated_v22_latest.csv

data/reco_eval_v22_summary_latest.txt
(and timestamped copies)

Past 1 week (merge v22 files)
If reco_history.csv is mixed-schema, do NOT feed it to pandas/analyzer directly.
Merge reco_v22_*.csv:

python - <<'PY'
import glob, pandas as pd
files = sorted(glob.glob("data/reco_v22_*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
df.to_csv("data/reco_v22_merged.csv", index=False)
print("merged rows:", len(df), "files:", len(files))
PY

python reco_analyzer_v22.py --history data/reco_v22_merged.csv --from_date 2026-02-09
backtest.py
What it does: A tiny candle-based forward simulator used for quick research/backtests.

Logic (plain English):

For each “trade entry” candle index, it assumes entry at the next candle open.

It then scans forward candle-by-candle until it hits stop, target, or a time-based exit.

Produces an “R multiple” outcome (risk-normalized return) and summary stats.

config.py
What it does: Central place to configure the default underlying, timeframe, indicator periods, strike step, and backtest knobs.

Logic (plain English):

You set things like UNDERLYING, KITE_SYMBOL, INTERVAL, EMA/RSI/ADX/ATR periods, stop/target ATR multipliers, strike step, and chain window width.

Other scripts import these defaults so you don’t hardcode values everywhere.

indicators.py
What it does: Lightweight indicator helpers (EMA, RSI, ATR, ADX, z-score) with flexible column handling.

Logic (plain English):

Accepts dataframes with either open/high/low/close or Open/High/Low/Close.

Computes indicators used by pattern detection and helper logic.

iv_greeks.py
What it does: Black–Scholes implied volatility and Greeks approximation.

Logic (plain English):

Uses a robust bisection solver to estimate IV from the option price.

Computes delta/gamma/vega/theta; theta is returned per day (handy for risk caps).

Also includes helpers for time-to-expiry in years.

iv_store.py
What it does: Stores IV snapshots and computes rolling IV percentile.

Logic (plain English):

Appends IV snapshots to a CSV history (e.g. data/iv_history.csv).

Cleans data (skips invalid IV values) and keeps a “daily-ish” history.

Computes IV percentile over a rolling window with EWMA smoothing, returning a value in [0, 1].

journal.py
What it does: Append-only journal of recommendations.

Logic (plain English):

Appends each run’s recommendations into a single data/reco_history.csv (never overwritten).

Also writes per-run “snapshot” CSVs so you can diff runs over time.

kite_client.py
What it does: Creates an authenticated KiteConnect client.

Logic (plain English):

Reads KITE_API_KEY and KITE_ACCESS_TOKEN from .env.

Returns a ready-to-use KiteConnect client instance for the rest of the modules.

kite_login.py
What it does: One-time helper to generate and store a Kite access token.

Logic (plain English):

Prints the Kite login URL, asks you to paste back the request_token.

Exchanges it for access_token using KITE_API_SECRET.

Writes/updates KITE_ACCESS_TOKEN=... into .env.

kite_chain.py
What it does: Builds a tight option-chain slice around ATM using Kite.

Logic (plain English):

Loads NFO instruments from data/kite_instruments_NFO.csv when available (reduces API calls / rate limits).

Picks an expiry (supports DTE band: min_dte_days / max_dte_days).

Finds spot price + ATM strike and fetches quotes for strikes around ATM.

Returns a ChainSlice containing calls/puts dataframes + spot/ATM.

Includes spot mapping for index symbols where spot tradingsymbol differs from F&O symbol.

main.py
What it does: Single-underlying “demo runner” that prints market state + signal + suggested option legs.

Logic (plain English):

Fetch recent candles for the configured symbol and interval.

Compute signal (trend/strength/levels) from indicators.

Fetch option chain slice and “align” entry/SL/target to live spot (because candle close may lag).

Build a recommendation (CE/PE/spread/watch) and print it.

Optionally runs a walk-forward backtest over the fetched candle window.

market_data.py
What it does: Fetches and caches historical candles using Kite.

Logic (plain English):

Converts friendly interval strings (15m, day, etc.) into Kite’s interval names.

Resolves instrument tokens (prefers ltp() token, falls back to instruments dump).

Fetches historical data in safe chunks (Kite has limits).

Caches candles briefly (TTL) to reduce rate limits while staying “fresh enough” for scans.

market_sentiment.py
What it does: Builds a lightweight “market context” snapshot using only Kite-available signals.

Logic (plain English):

Uses India VIX level + percentile as a regime proxy.

Computes option-chain aggregates like PCR (from OI/volume) and a simple skew proxy.

Finds “OI walls” (strikes with largest OI) as rough support/resistance markers.

Adds index trend direction using the same strategy.compute_signal() logic.

Appends snapshots to data/market_sentiment_history.csv.

option_chain.py
What it does: Backwards-compatible wrapper around kite_chain.get_kite_chain_slice().

Logic (plain English):

Keeps the old get_chain_slice() API so the rest of the code can stay stable.

Pulls defaults from config.py.

patterns.py
What it does: Pattern / setup idea generator (trade ideas, not orders).

Logic (plain English):

Normalizes candle columns and datetime handling.

Runs a few setup detectors:

Donchian breakout: looks for range breakouts.

Pullback trend: trend + retracement style setup.

Bollinger mean reversion: “stretch then snap back” logic.

Volatility squeeze: compression then expansion potential.

Outputs a list of TradeIdea objects with entry/stop/target and confidence.

reco_analyzer.py
What it does: Evaluates historical recommendations from data/reco_history.csv using option candles.

Logic (plain English):

Treats each journal row as “entered immediately after the recommendation time”.

Uses the first candle OPEN after ts_reco as entry.

Checks each candle for SL/target hits; flags ambiguous “both hit same bar”.

Produces evaluated CSV + a text summary (win-rate, common loss tags, etc.).

reco_analyzer_v22.py
What it does: Updated evaluator with more detailed metrics.

Logic (plain English):

Same concept as reco_analyzer.py, but adds:

Better IST safety,

MFE/MAE tracking,

Target reach fraction,

Uses per-row time_stop_min if available.

Writes “latest” plus timestamped evaluated outputs.

Recommended usage is to evaluate reco_v22_*.csv files (consistent schema) rather than mixed reco_history.csv.

recommender.py
What it does: Converts a Signal + ChainSlice into a “what to buy/watch” recommendation.

Logic (plain English):

Builds a merged “preview table” (CE + PE side-by-side by strike).

If the signal is NO_TRADE, outputs a WATCH recommendation with a trigger condition.

If signal is LONG/SHORT, picks a contract near ATM (and may pick spread width based on expected move).

Returns a structured Recommendation with legs, levels, reason, and a chain preview.

risk_caps.py
What it does: Greeks-based position sizing and safety caps.

Logic (plain English):

Calculates max lots based on:

Premium exposure,

Delta-notional exposure,

Vega (1% IV move sensitivity),

Theta (daily decay).

Scales caps by capital + regime multipliers + confidence multipliers.

Optionally tightens size for very near expiry (DTE penalty).

scan_options.py
What it does: Full-universe scanner (older V2.1 style).

Logic (plain English):

Builds a universe of optionable underlyings from the NFO instruments list.

For each underlying:

fetches history (cached),

computes a directional signal,

grabs an ATM-centered chain slice,

scores candidate strikes with liquidity filters,

estimates IV + Greeks,

writes ranked results + top picks.

Appends Top picks into data/reco_history.csv and writes run snapshots.

scan_options_global.py
What it does: Global scan variant that can optionally incorporate “global context”.

Logic (plain English):

Same scanning core, but may adjust scoring with a “risk-on/risk-off” bias if yfinance is available.

Adds additional heuristics (liquidity thresholds by index vs stock, slippage model, “learned edge” style helper).

Produces candidate recommendations across the universe.

scan_options_global_v22.py
What it does: Lean V2.2 global scan.

Logic (plain English):

Uses a directional confidence gate + high IV gate.

Picks strikes using delta targeting and liquidity scoring.

Builds option-based SL/target and time stop.

Writes global scan results + reco latest + appends to reco_history.

scan_options_v22.py
What it does: Full-universe scanner with V2.2 upgrades (IV percentile + regime + sentiment + risk caps).

Logic (plain English):

Builds a universe from NFO instruments and iterates through underlyings.

For each underlying:

Fetch candles → compute signal (trend, levels, confidence proxies).

Fetch chain slice → score strikes using liquidity gates (spread/OI/volume/min premium).

Compute IV + Greeks for the chosen contract.

Compute IV percentile and label regime (TREND/CHOP/VOLATILE).

(Optional) Pull market context and adjust risk/scoring.

Compute position sizing via risk_caps and prefer trades that PASS caps (pass_caps=True).

Writes scan result CSVs, “top 10” outputs, reco_latest, and appends to reco_history.csv.

requirements.txt
What it does: Minimal Python dependencies list.

Logic (plain English):

Lists the core libs needed for dataframes, math, env loading, and Kite API.
