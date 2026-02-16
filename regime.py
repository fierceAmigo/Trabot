"""regime.py

Phase-2 foundation: regime detection + multi-timeframe alignment gate.

Goals (Phase 2)
- Replace IVP-only regime label with a candle-driven classifier:
  TREND / CHOP / VOLATILE (+ confidence 0..1)
- Provide a *hard* multi-timeframe alignment gate.

Design principles
- Must be deterministic and cheap.
- Must not require extra third-party deps.
- Must degrade gracefully when candles are missing (return UNKNOWN, low confidence).

This classifier is intentionally simple; it will be iterated later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from indicators import ema, adx, atr


@dataclass
class RegimeResult:
    label: str  # TREND / CHOP / VOLATILE / UNKNOWN
    confidence: float  # 0..1
    details: Dict[str, Any]


def _safe_last(series: pd.Series) -> Optional[float]:
    try:
        if series is None or series.empty:
            return None
        v = series.iloc[-1]
        if v is None:
            return None
        v = float(v)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def timeframe_signature(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute a minimal directional/strength/volatility signature for a timeframe."""
    if df is None or df.empty or len(df) < 30:
        return {
            "dir": "FLAT",
            "dir_score": 0.0,
            "adx": None,
            "atr_pct": None,
            "bars": 0,
        }

    close = df["close"].astype(float)

    e_fast = ema(close, 20)
    e_slow = ema(close, 50)

    ef = _safe_last(e_fast)
    es = _safe_last(e_slow)
    last_close = _safe_last(close)

    if ef is None or es is None or last_close is None or last_close == 0:
        direction = "FLAT"
        dir_score = 0.0
    else:
        diff = ef - es
        # Normalize by price (roughly scale invariant)
        dir_score = float(diff / last_close)
        if dir_score > 0.0005:
            direction = "UP"
        elif dir_score < -0.0005:
            direction = "DOWN"
        else:
            direction = "FLAT"

    a = adx(df, period=14)
    a_last = _safe_last(a)

    at = atr(df, period=14)
    at_last = _safe_last(at)
    atr_pct = None
    if at_last is not None and last_close:
        atr_pct = float(at_last / last_close * 100.0)

    return {
        "dir": direction,
        "dir_score": float(dir_score),
        "adx": a_last,
        "atr_pct": atr_pct,
        "bars": int(len(df)),
    }


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def alignment_gate(
    *,
    ltf: Dict[str, Any],
    htf: Optional[Dict[str, Any]] = None,
    dtf: Optional[Dict[str, Any]] = None,
    allow_flat: bool = False,
) -> bool:
    """Hard alignment gate.

    - Requires non-FLAT directional alignment between LTF and HTF (and DTF if provided).
    - If allow_flat=True, a FLAT higher timeframe does not fail the gate.
    """

    if not ltf or ltf.get("dir") in (None, "FLAT"):
        return False

    ldir = ltf.get("dir")

    for tf in (htf, dtf):
        if not tf:
            continue
        tdir = tf.get("dir")
        if tdir in (None, "FLAT"):
            if allow_flat:
                continue
            return False
        if tdir != ldir:
            return False

    return True


def detect_regime(
    *,
    df_ltf: pd.DataFrame,
    df_htf: Optional[pd.DataFrame] = None,
    df_dtf: Optional[pd.DataFrame] = None,
    ivp: Optional[float] = None,
    high_vol_atr_pct: float = 1.2,
) -> RegimeResult:
    """Detect regime from candles, using IVP as a *secondary* input.

    Parameters
    - high_vol_atr_pct: approximate threshold for labeling VOLATILE using ATR%%.
    """

    sig_l = timeframe_signature(df_ltf)
    sig_h = timeframe_signature(df_htf) if df_htf is not None else None
    sig_d = timeframe_signature(df_dtf) if df_dtf is not None else None

    # Strength proxy from ADX.
    adx_vals = [v for v in [sig_l.get("adx"), (sig_h or {}).get("adx"), (sig_d or {}).get("adx")] if isinstance(v, (int, float))]
    adx_avg = float(sum(adx_vals) / len(adx_vals)) if adx_vals else 0.0

    # Vol proxy from ATR%%.
    atr_vals = [v for v in [sig_l.get("atr_pct"), (sig_h or {}).get("atr_pct"), (sig_d or {}).get("atr_pct")] if isinstance(v, (int, float))]
    atr_max = float(max(atr_vals)) if atr_vals else 0.0

    # Alignment.
    align_strict = alignment_gate(ltf=sig_l, htf=sig_h, dtf=sig_d, allow_flat=False)
    align_allow_flat = alignment_gate(ltf=sig_l, htf=sig_h, dtf=sig_d, allow_flat=True)

    # Normalize metrics -> 0..1.
    trend_strength = _clamp01(_sigmoid((adx_avg - 18.0) / 6.0))  # ADX ~18 baseline

    # Volatility score: ATR%% relative to threshold.
    vol_score = 0.0
    if atr_max and high_vol_atr_pct:
        vol_score = _clamp01(atr_max / float(high_vol_atr_pct) - 0.8)  # start rising slightly below threshold

    # IVP influences volatility (soft).
    ivp_score = 0.0
    if ivp is not None and not (isinstance(ivp, float) and math.isnan(ivp)):
        ivp_score = _clamp01((float(ivp) - 0.55) / 0.25)  # 0 at 0.55, 1 at 0.80

    volatile_conf = _clamp01(max(vol_score, ivp_score))

    # Trend confidence requires direction + alignment.
    align_bonus = 1.0 if align_strict else (0.6 if align_allow_flat else 0.0)
    dir_ok = 1.0 if sig_l.get("dir") in ("UP", "DOWN") else 0.0
    trend_conf = _clamp01(0.55 * trend_strength + 0.25 * align_bonus + 0.20 * dir_ok)

    # Chop confidence: low trend_strength, low vol, and poor alignment.
    chop_conf = _clamp01(
        (1.0 - trend_strength) * 0.55
        + (1.0 - volatile_conf) * 0.25
        + (0.8 if not align_allow_flat else 0.2) * 0.20
    )

    # Decide label.
    label = "UNKNOWN"
    conf = 0.0
    scores = {
        "VOLATILE": volatile_conf,
        "TREND": trend_conf,
        "CHOP": chop_conf,
    }
    label = max(scores, key=lambda k: scores[k])
    conf = float(scores[label])

    # If insufficient bars, mark UNKNOWN.
    if sig_l.get("bars", 0) < 30:
        label = "UNKNOWN"
        conf = 0.0

    details: Dict[str, Any] = {
        "ltf": sig_l,
        "htf": sig_h,
        "dtf": sig_d,
        "adx_avg": adx_avg,
        "atr_max_pct": atr_max,
        "ivp": ivp,
        "trend_strength": trend_strength,
        "volatile_score": volatile_conf,
        "align_strict": align_strict,
        "align_allow_flat": align_allow_flat,
    }

    return RegimeResult(label=label, confidence=conf, details=details)
