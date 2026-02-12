"""
iv_greeks.py

Black–Scholes implied volatility + Greeks (approximation).

- Uses a robust bisection solver for IV (stable even with noisy prices)
- Theta is returned per DAY (calendar day) for easier risk caps.
- This is an approximation; stock options can deviate due to dividends/skew.
"""

from __future__ import annotations
import math
from datetime import datetime, date, time as dtime

SQRT_2PI = math.sqrt(2.0 * math.pi)

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    vsqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2

def bs_price(S: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    right = right.upper()
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if right == "CE":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def implied_volatility(price: float, S: float, K: float, T: float, r: float, right: str,
                       lo: float = 1e-6, hi: float = 5.0, tol: float = 1e-4, max_iter: int = 80):
    """
    Returns (iv, ok_flag).
    """
    if price is None or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None, False

    right = right.upper()
    intrinsic = max(0.0, S - K) if right == "CE" else max(0.0, K - S)
    if price < intrinsic * 0.98:
        return None, False

    f_lo = bs_price(S, K, T, r, lo, right) - price
    f_hi = bs_price(S, K, T, r, hi, right) - price

    if f_lo * f_hi > 0:
        return None, False

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(S, K, T, r, mid, right) - price
        if abs(f_mid) < tol:
            return mid, True
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return mid, True

def greeks(S: float, K: float, T: float, r: float, iv: float, right: str):
    """
    Returns dict: delta, gamma, vega_1pct, theta_day.
    - vega_1pct = price change for 1% IV move
    - theta_day = price decay per day (typically negative for long options)
    """
    right = right.upper()
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega_1pct": 0.0, "theta_day": 0.0}

    d1, d2 = _d1_d2(S, K, T, r, iv)
    pdf = norm_pdf(d1)

    gamma = pdf / (S * iv * math.sqrt(T))
    vega = S * pdf * math.sqrt(T)      # per 1.0 vol (100%)
    vega_1pct = vega / 100.0

    if right == "CE":
        delta = norm_cdf(d1)
        theta = -(S * pdf * iv) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm_cdf(d2)
    else:
        delta = norm_cdf(d1) - 1.0
        theta = -(S * pdf * iv) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm_cdf(-d2)

    theta_day = theta / 365.0
    return {"delta": delta, "gamma": gamma, "vega_1pct": vega_1pct, "theta_day": theta_day}

def time_to_expiry_years(expiry: str, now: datetime | None = None, close_hour: int = 15, close_min: int = 30):
    """
    expiry: YYYY-MM-DD
    Returns T in years.
    """
    if now is None:
        now = datetime.now()
    try:
        y, m, d = [int(x) for x in expiry.split("-")]
        exp_dt = datetime(y, m, d, close_hour, close_min)
        secs = (exp_dt - now).total_seconds()
        if secs <= 0:
            return 1e-8
        return secs / (365.0 * 24 * 3600)
    except Exception:
        return 1e-8
