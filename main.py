import math
import pandas as pd

from config import (
    UNDERLYING, YF_TICKER, LOOKBACK_DAYS, INTERVAL,
    EMA_FAST, EMA_SLOW, RSI_PERIOD, ADX_PERIOD, ADX_MIN,
    ATR_PERIOD, STOP_ATR_MULT, TARGET_ATR_MULT,
    STRIKE_STEP, STRIKES_AROUND_ATM, HIGH_VOL_ATR_PCT,
    BACKTEST_ENABLED, BACKTEST_MAX_HOLD_BARS_INTRADAY, BACKTEST_MAX_HOLD_BARS_DAILY
)

from market_data import fetch_history
from option_chain import get_chain_slice
from strategy import compute_signal
from recommender import build_recommendation
from backtest import _simulate_forward, summarize


def _choose_hold_bars(interval: str) -> int:
    return BACKTEST_MAX_HOLD_BARS_DAILY if interval == "1d" else BACKTEST_MAX_HOLD_BARS_INTRADAY


def run_backtest_walkforward(df: pd.DataFrame, used_interval: str):
    hold_bars = _choose_hold_bars(used_interval)
    min_bars = max(EMA_SLOW, RSI_PERIOD, ADX_PERIOD, ATR_PERIOD) + 5
    trades = []

    for i in range(min_bars, len(df) - 2):
        hist = df.iloc[: i + 1].copy()

        sig = compute_signal(
            df=hist,
            ema_fast=EMA_FAST,
            ema_slow=EMA_SLOW,
            rsi_period=RSI_PERIOD,
            adx_period=ADX_PERIOD,
            adx_min=ADX_MIN,
            atr_period=ATR_PERIOD,
            stop_atr_mult=STOP_ATR_MULT,
            target_atr_mult=TARGET_ATR_MULT,
        )

        if sig.side == "NO_TRADE":
            continue

        entry_i = i + 1
        t = _simulate_forward(df, entry_i, sig.side, sig.stop, sig.target, hold_bars)
        if t:
            trades.append(t)

    return summarize(trades), trades


def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _shift_level(x, delta: float):
    if x is None:
        return None
    try:
        return float(x) + float(delta)
    except Exception:
        return x


def _shift_metric(metrics: dict, key: str, delta: float):
    if key not in metrics:
        return
    v = metrics[key]
    if v is None or _is_nan(v):
        return
    try:
        metrics[key] = float(v) + float(delta)
    except Exception:
        pass


def main():
    print("Fetching last ~1 month market data...")
    df, used_interval = fetch_history(YF_TICKER, lookback_days=LOOKBACK_DAYS, interval=INTERVAL)
    print(f"Fetched {len(df)} candles for {YF_TICKER} (requested={INTERVAL}, used={used_interval})")

    # 1) Compute signal from candles (indicators)
    signal = compute_signal(
        df=df,
        ema_fast=EMA_FAST,
        ema_slow=EMA_SLOW,
        rsi_period=RSI_PERIOD,
        adx_period=ADX_PERIOD,
        adx_min=ADX_MIN,
        atr_period=ATR_PERIOD,
        stop_atr_mult=STOP_ATR_MULT,
        target_atr_mult=TARGET_ATR_MULT,
    )

    # 2) Fetch chain (gives live spot + ATM) so we can align levels to live market
    print("\nFetching option chain from Zerodha (Kite)...")
    chain = get_chain_slice(UNDERLYING, strike_step=STRIKE_STEP, strikes_around_atm=STRIKES_AROUND_ATM)

    # 3) Rebase levels to Kite spot (because candle close can be delayed)
    candle_close = float(signal.metrics.get("close", 0.0))
    kite_spot = float(chain.spot)
    delta = kite_spot - candle_close

    # Shift entry/stop/target (if present)
    signal.entry = _shift_level(signal.entry, delta)
    signal.stop = _shift_level(signal.stop, delta)
    signal.target = _shift_level(signal.target, delta)

    # Shift watch levels in metrics (if present)
    _shift_metric(signal.metrics, "watch_entry", delta)
    _shift_metric(signal.metrics, "watch_stop", delta)
    _shift_metric(signal.metrics, "watch_target", delta)

    print("\n=== MARKET STATE (latest) ===")
    print(f"{'candle_close':>12}: {candle_close:.4f}")
    print(f"{'kite_spot':>12}: {kite_spot:.4f}")
    print(f"{'spot_delta':>12}: {delta:.4f}  (kite_spot - candle_close)")
    for k in ["ema_fast", "ema_slow", "rsi", "adx", "atr", "atr_pct"]:
        v = signal.metrics.get(k)
        if isinstance(v, float):
            print(f"{k:>12}: {v:.4f}")
        else:
            print(f"{k:>12}: {v}")

    # Watch block (only if present)
    ws = signal.metrics.get("watch_side")
    wt = signal.metrics.get("watch_trigger")
    we = signal.metrics.get("watch_entry")
    wsl = signal.metrics.get("watch_stop")
    wtg = signal.metrics.get("watch_target")

    if ws and ws != "NONE" and wt:
        print("\n=== WATCH MODE (pre-signal) ===")
        print(f"{'watch_side':>12}: {ws}")
        print(f"{'trigger':>12}: {wt}")
        if isinstance(we, float) and not _is_nan(we):
            print(f"{'entry':>12}: {we:.2f}")
        if isinstance(wsl, float) and not _is_nan(wsl):
            print(f"{'stop':>12}: {wsl:.2f}")
        if isinstance(wtg, float) and not _is_nan(wtg):
            print(f"{'target':>12}: {wtg:.2f}")

    print("\n=== SIGNAL (latest) ===")
    print(f"side   : {signal.side}")
    print(f"reason : {signal.reason}")
    if signal.side != "NO_TRADE":
        print(f"entry  : {signal.entry:.2f}  (aligned to kite_spot)")
        print(f"stop   : {signal.stop:.2f}   (aligned to kite_spot)")
        print(f"target : {signal.target:.2f} (aligned to kite_spot)")
        print("\nEntry rule: Enter when price breaks the entry level on this timeframe.")
        print("Exit rule : Exit at stop or target (underlying levels).")

    # 4) Build recommendation (will output WATCH_CE / WATCH_PE if NO_TRADE)
    rec = build_recommendation(UNDERLYING, signal, chain, high_vol_atr_pct=HIGH_VOL_ATR_PCT)

    print("\n=== WHAT TO BUY (educational output) ===")
    print(f"Underlying : {rec.underlying}")
    print(f"Expiry     : {rec.expiry}")
    print(f"Spot/ATM   : spot≈{chain.spot:.2f} | ATM={chain.atm}")
    print(f"Action     : {rec.action}")
    print(f"Reason     : {rec.reason}")

    if rec.trigger:
        print("\n✅ WATCH TRIGGER:")
        print(f"  {rec.trigger}")

    if rec.legs:
        print("\nSuggested option legs (tradingsymbols for later order placement):")
        for leg in rec.legs:
            ltp = f" | LTP≈{leg.ltp:.2f}" if leg.ltp is not None else ""
            oi = f" | OI≈{leg.oi:.0f}" if leg.oi is not None else ""
            print(f"  {leg.side:>4} {leg.tradingsymbol or '???'}  ({leg.strike} {leg.right}){ltp}{oi}")

    if rec.entry_underlying is not None:
        print("\nPlan (underlying levels, aligned to kite_spot):")
        print(f"  Enter near/through: {rec.entry_underlying:.2f}")
        print(f"  Stop at:          {rec.stop_underlying:.2f}")
        print(f"  Target at:        {rec.target_underlying:.2f}")

    print("\nOption chain slice around ATM (Kite quotes):")
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 80)
    print(rec.chain_preview.to_string(index=False))

    if BACKTEST_ENABLED:
        print("\n=== BACKTEST (walk-forward on last ~month; candle-based) ===")
        stats, _ = run_backtest_walkforward(df, used_interval)
        for k, v in stats.items():
            print(f"{k}: {v}")

    print("\nNOTE: This is for research/education. Not financial advice.")


if __name__ == "__main__":
    main()
