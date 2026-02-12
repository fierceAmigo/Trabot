"""risk_caps.py

Greeks-based *position sizing* + safety caps.

This module is deliberately simple and deterministic (data-centric):
- It sizes BUY option positions in *lots* (lot_size comes from instruments dump).
- It enforces multiple caps (premium, delta-notional, vega, theta).
- Caps scale by: capital, market regime, and Greeks confidence.
- Optional DTE penalty tightens size in expiry week.

Exposure model (for N lots):
  contracts = lot_size * lots
  delta_notional  ~= |delta| * spot * contracts
  vega_1pct_total ~= |vega_1pct| * contracts        (PnL change for 1% IV move)
  theta_day_total ~= |theta_day| * contracts        (daily decay magnitude)

NOTE: This is not financial advice. Educational tool only.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, List, Optional


# -------------------------
# Multipliers / profiles
# -------------------------

def _regime_multipliers(regime: str) -> Dict[str, float]:
    """Multipliers applied to base caps."""
    r = (regime or "").upper()
    if r == "TREND":
        return {"premium": 1.00, "delta": 1.25, "vega": 1.00, "theta": 1.00}
    if r == "VOLATILE":
        return {"premium": 0.65, "delta": 0.85, "vega": 0.75, "theta": 0.85}
    # CHOP/default
    return {"premium": 0.80, "delta": 0.95, "vega": 1.05, "theta": 0.95}


def _confidence_multiplier(confidence: str) -> float:
    c = (confidence or "high").strip().lower()
    return 1.0 if c == "high" else 0.70


def _dte_multiplier(dte: Optional[int]) -> float:
    """Tighten size near expiry."""
    if dte is None:
        return 1.0
    if dte <= 1:
        return 0.65
    if dte == 2:
        return 0.75
    if dte == 3:
        return 0.85
    return 1.0


def _premium_risk_pct(risk_profile: str) -> float:
    """Max premium at risk as % of capital for BUY options."""
    rp = (risk_profile or "high").strip().lower()
    if rp.startswith("mod"):
        return 0.05
    return 0.08  # high


# -------------------------
# Caps computation
# -------------------------

def compute_caps(
    *,
    capital: float,
    regime: str,
    confidence: str = "high",
    risk_profile: str = "high",
    dte: Optional[int] = None,
) -> Dict[str, float]:
    """Return absolute caps in INR units."""
    capital = float(capital or 0.0)
    m_reg = _regime_multipliers(regime)
    m_conf = _confidence_multiplier(confidence)
    m_dte = _dte_multiplier(dte)

    # Premium risk cap
    max_premium = capital * _premium_risk_pct(risk_profile) * m_reg["premium"] * m_conf * m_dte

    # Base greek caps as fraction of capital
    max_delta = capital * 1.00 * m_reg["delta"] * m_conf * m_dte
    max_vega = capital * 0.020 * m_reg["vega"] * m_conf * m_dte
    max_theta = capital * 0.006 * m_reg["theta"] * m_conf * m_dte

    return {
        "max_premium": float(max_premium),
        "max_delta_notional": float(max_delta),
        "max_vega_1pct": float(max_vega),
        "max_theta_day": float(max_theta),
    }


def compute_max_lots(
    *,
    capital: float,
    regime: str,
    confidence: str,
    risk_profile: str,
    dte: Optional[int],
    spot: float,
    option_price: float,
    lot_size: int,
    delta: float,
    vega_1pct: float,
    theta_day: float,
    max_lots_hard: int = 6,
) -> Tuple[int, Dict[str, float], List[str]]:
    """Compute max lots allowed by caps.

    Returns:
      (max_lots, exposures_per_lot, messages)
    """
    msgs: List[str] = []
    lot_size = int(lot_size or 1)
    if lot_size <= 0:
        lot_size = 1

    if capital <= 0:
        return 0, {}, ["Capital <= 0 -> cannot size position."]
    if option_price <= 0 or spot <= 0:
        return 0, {}, ["Invalid prices -> cannot size position."]

    caps = compute_caps(
        capital=capital,
        regime=regime,
        confidence=confidence,
        risk_profile=risk_profile,
        dte=dte,
    )

    # Exposures for 1 lot
    contracts_1lot = float(lot_size)
    delta_notional_1lot = abs(float(delta)) * float(spot) * contracts_1lot
    vega_1pct_1lot = abs(float(vega_1pct)) * contracts_1lot
    theta_day_1lot = abs(float(theta_day)) * contracts_1lot
    premium_1lot = float(option_price) * contracts_1lot

    exposures = {
        "premium_1lot": premium_1lot,
        "delta_notional_1lot": delta_notional_1lot,
        "vega_1pct_1lot": vega_1pct_1lot,
        "theta_day_1lot": theta_day_1lot,
        **caps,
    }

    def lots_by(cap: float, per_lot: float) -> int:
        if per_lot <= 0:
            return max_lots_hard
        return int(math.floor(cap / per_lot))

    by_premium = lots_by(caps["max_premium"], premium_1lot)
    by_delta = lots_by(caps["max_delta_notional"], delta_notional_1lot)
    by_vega = lots_by(caps["max_vega_1pct"], vega_1pct_1lot)
    by_theta = lots_by(caps["max_theta_day"], theta_day_1lot)

    max_lots = max(0, min(by_premium, by_delta, by_vega, by_theta, int(max_lots_hard)))

    if confidence.strip().lower() == "low":
        msgs.append("Low Greeks confidence -> tighter size.")

    msgs.append(
        f"Lots caps: premium={by_premium}, delta={by_delta}, vega={by_vega}, theta={by_theta} -> max_lots={max_lots}"
    )

    # Helpful warnings
    if premium_1lot > caps["max_premium"]:
        msgs.append("1 lot premium exceeds premium cap.")
    if delta_notional_1lot > caps["max_delta_notional"]:
        msgs.append("1 lot delta-notional exceeds cap.")
    if vega_1pct_1lot > caps["max_vega_1pct"]:
        msgs.append("1 lot vega exceeds cap.")
    if theta_day_1lot > caps["max_theta_day"]:
        msgs.append("1 lot theta/day exceeds cap.")

    return max_lots, exposures, msgs


def check_greeks_caps(exposure: dict, capital: float, regime: str, confidence: str = "high"):
    """Backward-compatible: pass/fail check using the old exposure dict.

    exposure keys:
      delta_notional, vega_1pct_total, theta_day_total

    This retains the original interface used by older scripts.
    """
    msgs: List[str] = []
    caps = compute_caps(capital=capital, regime=regime, confidence=confidence, risk_profile="high", dte=None)

    d = abs(float(exposure.get("delta_notional", 0.0)))
    v = abs(float(exposure.get("vega_1pct_total", 0.0)))
    t = abs(float(exposure.get("theta_day_total", 0.0)))

    ok = True
    if d > caps["max_delta_notional"]:
        ok = False
        msgs.append(f"Delta notional too high: {d:,.0f} > cap {caps['max_delta_notional']:,.0f}")
    if v > caps["max_vega_1pct"]:
        ok = False
        msgs.append(f"Vega too high (PnL per 1% IV): {v:,.0f} > cap {caps['max_vega_1pct']:,.0f}")
    if t > caps["max_theta_day"]:
        ok = False
        msgs.append(f"Theta decay too high (per day): {t:,.0f} > cap {caps['max_theta_day']:,.0f}")

    if confidence.strip().lower() == "low":
        msgs.append("Low confidence Greeks -> caps were tightened.")

    if ok:
        msgs.append("✅ Greeks exposure within caps.")
    return ok, msgs
