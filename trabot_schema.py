"""trabot_schema.py

Phase-1 foundation: stable, versioned recommendation (reco) schema.

Why this exists
- Mixed CSV schemas across runs can corrupt reco_history.csv (pandas ParserError).
- We fix that by:
  1) pinning a schema_version for reco rows,
  2) writing to a versioned history file (reco_history_v<schema>.csv),
  3) normalizing every row to a stable, ordered column set,
  4) capturing any unexpected extra keys into a single JSON column.

This module is intentionally dependency-light.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional


# Bump MAJOR when column semantics change.
RECO_SCHEMA_VERSION: str = os.getenv("TRABOT_RECO_SCHEMA_VERSION", "3").strip() or "1"

DATA_DIR = os.getenv("TRABOT_DATA_DIR", "data")
DEFAULT_HISTORY_PATH = os.path.join(DATA_DIR, f"reco_history_v{RECO_SCHEMA_VERSION}.csv")

# Stable, ordered columns (append-only history expects the same header forever).
# If you add a new column, append it to the end to preserve backward compatibility.
RECO_COLUMNS: List[str] = [
    # identity
    "ts_reco",
    "schema_version",
    "run_id",
    "source",
    "bucket",
    "mode",

    # timeframe context
    "interval",
    "htf_interval",
    "dtf_interval",
    "ts_signal",

    # quotes / execution model
    "quote_ts",
    "entry_model",
    "bid",
    "ask",
    "mid",
    "ltp",
    "spread_pct",

    # instrument
    "underlying",
    "spot_symbol",
    "expiry",
    "dte",
    "action",
    "side",
    "tradingsymbol",
    "kite_symbol",
    "strike",
    "right",

    # plan
    "entry",
    "sl",
    "target",
    "time_stop_min",

    # scoring + regime
    "score",
    "regime",
    "regime_conf",
    "align_mode",
    "ltf_dir",
    "htf_dir",
    "dtf_dir",
    "htf_align",

    # IV / Greeks
    "iv",
    "iv_pct",
    "iv_samples",
    "greeks_conf",
    "delta",
    "vega_1pct",
    "theta_day",

    # sizing
    "lot_size",
    "max_lots",
    "pass_caps",

    # market context overlay
    "sent_mult",
    "mkt_bias",
    "mkt_strength",
    "mkt_risk_off",

    # explainability
    "reason",
    "notes",

    # Phase-3: multi-leg strategy fields (populated when applicable)
    "strategy_type",
    "legs_json",
    "net_premium",
    "max_loss",
    "max_profit",
    "breakevens",
    "margin_est",
    "signal_strength",
    "gamma",
    "legs_count",
    "quote_age_s_max",
    "quote_age_s_mean",
    "cluster",
    "fill_model",
    "fill_k",

    # overflow for forward-compat
    "extra_json",
]


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _normalize_scalar(v: Any) -> Any:
    """Convert values into CSV-safe scalar cells."""
    if v is None or _is_nan(v):
        return ""

    # Preserve booleans/ints/floats/strings.
    if isinstance(v, (bool, int, float, str)):
        return v

    # Common containers: stringify.
    if isinstance(v, (list, tuple, set)):
        return " | ".join(str(x) for x in v)

    # Dicts: deterministic JSON.
    if isinstance(v, dict):
        try:
            return json.dumps(v, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(v)

    return str(v)


def normalize_row(row: Dict[str, Any], *, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Normalize a reco row to the stable schema.

    - Ensures all columns exist.
    - Captures unknown keys in extra_json.
    - Ensures schema_version is set.
    """
    cols = columns or RECO_COLUMNS

    out: Dict[str, Any] = {}

    # Capture unknown keys.
    extra = {k: v for k, v in row.items() if k not in cols and k != "extra_json"}

    for c in cols:
        if c == "schema_version":
            out[c] = RECO_SCHEMA_VERSION
            continue

        if c == "extra_json":
            if "extra_json" in row and row["extra_json"] not in (None, ""):
                out[c] = _normalize_scalar(row.get("extra_json"))
            elif extra:
                try:
                    out[c] = json.dumps(extra, ensure_ascii=False, sort_keys=True)
                except Exception:
                    out[c] = str(extra)
            else:
                out[c] = ""
            continue

        out[c] = _normalize_scalar(row.get(c, ""))

    return out


def normalize_rows(rows: Iterable[Dict[str, Any]], *, columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    cols = columns or RECO_COLUMNS
    return [normalize_row(r, columns=cols) for r in rows]
