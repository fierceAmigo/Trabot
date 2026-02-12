from kite_chain import get_kite_chain_slice, ChainSlice
from config import (
    UNDERLYING, KITE_SPOT_SYMBOL, STRIKE_STEP, STRIKES_AROUND_ATM, INSTRUMENTS_CACHE_PATH
)


def get_chain_slice(symbol: str = UNDERLYING, strike_step: int = STRIKE_STEP, strikes_around_atm: int = STRIKES_AROUND_ATM) -> ChainSlice:
    """
    Backwards-compatible wrapper so main.py can keep calling get_chain_slice().
    Uses Zerodha Kite Connect instruments + quotes (no NSE scraping).
    """
    return get_kite_chain_slice(
        underlying=symbol,
        kite_spot_symbol=KITE_SPOT_SYMBOL,
        strike_step=strike_step,
        strikes_around_atm=strikes_around_atm,
        cache_path=INSTRUMENTS_CACHE_PATH
    )
