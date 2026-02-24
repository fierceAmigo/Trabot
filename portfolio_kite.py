"""portfolio_kite.py

Phase-4: hydrate portfolio state from Kite positions() when available.

This is optional and best-effort. If Kite is not available or fields differ,
we simply return the existing state unchanged.

Enable:
  TRABOT_PORTFOLIO_SYNC_KITE=1
"""

from __future__ import annotations

import os
from typing import Dict, Any

from kite_client import kite_positions_safe


def hydrate_from_kite(state: Dict[str, Any]) -> Dict[str, Any]:
    if os.getenv("TRABOT_PORTFOLIO_SYNC_KITE", "0").strip() != "1":
        return state
    try:
        pos = kite_positions_safe()
        # Kite returns dict: {"net": [...], "day": [...]}
        net = pos.get("net") or []
        # Keep only open positions
        open_pos = []
        for p in net:
            try:
                qty = float(p.get("quantity", 0))
                if abs(qty) < 1e-9:
                    continue
                ts = str(p.get("tradingsymbol") or "")
                if not ts:
                    continue
                open_pos.append({
                    "tradingsymbol": ts,
                    "exchange": p.get("exchange"),
                    "quantity": qty,
                    "product": p.get("product"),
                    "average_price": p.get("average_price"),
                    "last_price": p.get("last_price"),
                })
            except Exception:
                continue
        state = dict(state or {})
        state["kite_positions_raw"] = open_pos
        return state
    except Exception:
        return state
