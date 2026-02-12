# ========= DEFAULT INSTRUMENT (Kite-only) =========

# Underlying name as used in NFO instruments ("NIFTY" / "BANKNIFTY")
UNDERLYING = "NIFTY"

# Single source of truth for spot + historical candles (Kite symbol)
KITE_SYMBOL = "NSE:NIFTY 50"

# Keep backward-compat variable name used by main.py (so you don’t edit other files)
YF_TICKER = KITE_SYMBOL

# Used for live spot/ATM
KITE_SPOT_SYMBOL = KITE_SYMBOL

LOOKBACK_DAYS = 35
INTERVAL = "15m"

# ========= SIGNAL =========
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ADX_PERIOD = 14
ADX_MIN = 18

ATR_PERIOD = 14
STOP_ATR_MULT = 1.5
TARGET_ATR_MULT = 2.2

# ========= OPTION CHAIN (from Kite) =========
STRIKE_STEP = 50              # NIFTY: 50
STRIKES_AROUND_ATM = 8

HIGH_VOL_ATR_PCT = 1.5

# Cache Kite instruments CSV locally (faster)
INSTRUMENTS_CACHE_PATH = "data/kite_instruments_NFO.csv"

# ========= BACKTEST =========
BACKTEST_ENABLED = True
BACKTEST_MAX_HOLD_BARS_INTRADAY = 24
BACKTEST_MAX_HOLD_BARS_DAILY = 5
