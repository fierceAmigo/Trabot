"""portfolio.py

Phase-4: portfolio-level risk management.

This module maintains a lightweight portfolio state and validates whether a new
recommendation would violate aggregate exposure caps.

Design goals:
- deterministic + file-backed (JSON)
- works even without live broker positions (you can hydrate state manually)
- integrates with Trabot scanner via feature flags

State file format (data/portfolio_state.json by default):
{
  "asof": "2026-02-18T10:30:00+05:30",
  "positions": [
    {
      "id": "AUROPHARMA_20260226_BULL_CALL_SPREAD",
      "underlying": "AUROPHARMA",
      "cluster": "PHARMA",
      "opened_ts": "...",
      "status": "OPEN",
      "legs": [
        {"side": "BUY", "right": "CE", "strike": 1200, "tradingsymbol": "...", "delta": 0.52, "gamma": 0.001, "vega_1pct": 0.12, "theta_day": -0.08, "price": 20.0},
        {"side": "SELL", ...}
      ],
      "lots": 1,
      "lot_size": 550,
      "premium_at_risk": 11000.0
    }
  ]
}

Caps (env):
- TRABOT_PORTFOLIO_ENABLE=1
- TRABOT_PORTFOLIO_STATE_PATH=data/portfolio_state.json
- TRABOT_PORTFOLIO_MAX_PREMIUM_FRAC=0.35   (sum premium_at_risk <= frac * capital)
- TRABOT_PORTFOLIO_MAX_DELTA_NOTIONAL_FRAC=0.60  (|net_delta|*spot*contracts <= frac*capital)
- TRABOT_PORTFOLIO_MAX_VEGA_FRAC=0.60      (sum |vega_1pct|*contracts <= frac*capital/1000)
- TRABOT_PORTFOLIO_MAX_GAMMA_FRAC=0.50     (sum |gamma|*contracts <= frac*capital/10000)
- TRABOT_PORTFOLIO_MAX_POS_PER_UNDERLYING=2
- TRABOT_PORTFOLIO_MAX_POS_PER_CLUSTER=4

Cluster mapping:
- Provide data/clusters.json, e.g. {"AUROPHARMA":"PHARMA", "SBIN":"BANKS"}
- Fallback: "DEFAULT"

NOTE: Educational tool only – not financial advice.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone

try:
    from dateutil import tz
except Exception:  # pragma: no cover
    tz = None


@dataclass
class Leg:
    side: str                 # BUY/SELL
    right: str                # CE/PE
    strike: int
    tradingsymbol: str
    delta: float = 0.0
    gamma: float = 0.0
    vega_1pct: float = 0.0
    theta_day: float = 0.0
    price: float = 0.0        # per unit option price used for risk calc


@dataclass
class Position:
    id: str
    underlying: str
    cluster: str
    opened_ts: str
    status: str               # OPEN/CLOSED
    legs: List[Leg]
    lots: int
    lot_size: int
    premium_at_risk: float = 0.0


@dataclass
class PortfolioState:
    asof: str
    positions: List[Position]


def _now_ist_iso() -> str:
    if tz:
        return datetime.now(tz.gettz("Asia/Kolkata")).isoformat()
    return datetime.now(timezone.utc).isoformat()


def load_clusters(path: str = "data/clusters.json") -> Dict[str, str]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                m = json.load(f)
            if isinstance(m, dict):
                return {str(k).upper(): str(v).upper() for k, v in m.items()}
    except Exception:
        pass
    return {}


def load_portfolio(path: str) -> PortfolioState:
    if not os.path.exists(path):
        return PortfolioState(asof=_now_ist_iso(), positions=[])
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f) or {}
    positions = []
    for p in raw.get("positions", []) or []:
        legs = []
        for lg in p.get("legs", []) or []:
            try:
                legs.append(
                    Leg(
                        side=str(lg.get("side", "BUY")).upper(),
                        right=str(lg.get("right", "CE")).upper(),
                        strike=int(lg.get("strike", 0)),
                        tradingsymbol=str(lg.get("tradingsymbol", "")),
                        delta=float(lg.get("delta", 0.0) or 0.0),
                        gamma=float(lg.get("gamma", 0.0) or 0.0),
                        vega_1pct=float(lg.get("vega_1pct", 0.0) or 0.0),
                        theta_day=float(lg.get("theta_day", 0.0) or 0.0),
                        price=float(lg.get("price", 0.0) or 0.0),
                    )
                )
            except Exception:
                continue
        try:
            positions.append(
                Position(
                    id=str(p.get("id") or ""),
                    underlying=str(p.get("underlying") or "").upper(),
                    cluster=str(p.get("cluster") or "DEFAULT").upper(),
                    opened_ts=str(p.get("opened_ts") or ""),
                    status=str(p.get("status") or "OPEN").upper(),
                    legs=legs,
                    lots=int(p.get("lots") or 0),
                    lot_size=int(p.get("lot_size") or 1),
                    premium_at_risk=float(p.get("premium_at_risk") or 0.0),
                )
            )
        except Exception:
            continue
    return PortfolioState(asof=str(raw.get("asof") or _now_ist_iso()), positions=positions)


def save_portfolio(state: PortfolioState, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "asof": state.asof,
        "positions": [
            {
                "id": p.id,
                "underlying": p.underlying,
                "cluster": p.cluster,
                "opened_ts": p.opened_ts,
                "status": p.status,
                "legs": [asdict(lg) for lg in p.legs],
                "lots": p.lots,
                "lot_size": p.lot_size,
                "premium_at_risk": p.premium_at_risk,
            }
            for p in state.positions
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _pos_contracts(p: Position) -> int:
    return int(max(0, p.lots) * max(1, p.lot_size))


def aggregate_exposures(state: PortfolioState, spot_by_underlying: Dict[str, float] | None = None) -> Dict[str, float]:
    """Return aggregated exposures across OPEN positions."""
    spot_by_underlying = spot_by_underlying or {}
    prem = 0.0
    delta_notional = 0.0
    vega = 0.0
    gamma = 0.0
    theta = 0.0

    for p in state.positions:
        if p.status != "OPEN":
            continue
        contracts = _pos_contracts(p)
        prem += float(p.premium_at_risk or 0.0)

        # Aggregate net greeks from legs
        net_delta = 0.0
        net_gamma = 0.0
        net_vega = 0.0
        net_theta = 0.0
        for lg in p.legs:
            sgn = 1.0 if lg.side == "BUY" else -1.0
            net_delta += sgn * float(lg.delta or 0.0)
            net_gamma += sgn * float(lg.gamma or 0.0)
            net_vega += sgn * float(lg.vega_1pct or 0.0)
            net_theta += sgn * float(lg.theta_day or 0.0)

        spot = float(spot_by_underlying.get(p.underlying, 0.0) or 0.0)
        if spot > 0:
            delta_notional += abs(net_delta) * spot * contracts
        vega += abs(net_vega) * contracts
        gamma += abs(net_gamma) * contracts
        theta += abs(net_theta) * contracts

    return {
        "premium_at_risk": prem,
        "delta_notional": delta_notional,
        "vega_1pct_total": vega,
        "gamma_total": gamma,
        "theta_day_total": theta,
    }


def count_open_positions(state: PortfolioState) -> Tuple[Dict[str, int], Dict[str, int]]:
    by_under = {}
    by_cluster = {}
    for p in state.positions:
        if p.status != "OPEN":
            continue
        by_under[p.underlying] = by_under.get(p.underlying, 0) + 1
        by_cluster[p.cluster] = by_cluster.get(p.cluster, 0) + 1
    return by_under, by_cluster


def make_position_from_reco(
    reco_row: dict,
    *,
    capital: float,
    clusters: Dict[str, str],
) -> Position:
    underlying = str(reco_row.get("underlying") or "").upper()
    cluster = clusters.get(underlying, "DEFAULT")
    legs = []
    try:
        raw = reco_row.get("legs_json", "")
        lg_list = json.loads(raw) if raw else []
    except Exception:
        lg_list = []
    for lg in lg_list:
        try:
            legs.append(
                Leg(
                    side=str(lg.get("side", "BUY")).upper(),
                    right=str(lg.get("right", "CE")).upper(),
                    strike=int(lg.get("strike", 0)),
                    tradingsymbol=str(lg.get("tradingsymbol", "")),
                    delta=float(lg.get("delta", 0.0) or 0.0),
                    gamma=float(lg.get("gamma", 0.0) or 0.0),
                    vega_1pct=float(lg.get("vega_1pct", 0.0) or 0.0),
                    theta_day=float(lg.get("theta_day", 0.0) or 0.0),
                    price=float(lg.get("price_used", 0.0) or 0.0),
                )
            )
        except Exception:
            continue

    lot_size = int(reco_row.get("lot_size") or 1)
    lots = int(reco_row.get("lots") or 0)

    # premium_at_risk: prefer max_loss * contracts if available; fallback to abs(net_premium)*contracts
    prem_risk = 0.0
    try:
        ml = reco_row.get("max_loss", None)
        if ml not in ("", None):
            prem_risk = abs(float(ml)) * lot_size * lots
    except Exception:
        prem_risk = 0.0
    if prem_risk <= 0:
        try:
            nprem = reco_row.get("net_premium", None)
            if nprem not in ("", None):
                prem_risk = abs(float(nprem)) * lot_size * lots
        except Exception:
            prem_risk = 0.0
    if prem_risk <= 0:
        try:
            prem_risk = abs(float(reco_row.get("entry", 0.0))) * lot_size * lots
        except Exception:
            prem_risk = 0.0

    pos_id = f"{underlying}_{reco_row.get('expiry','')}_{reco_row.get('strategy_type','')}"
    return Position(
        id=pos_id,
        underlying=underlying,
        cluster=cluster,
        opened_ts=str(reco_row.get("ts_reco") or _now_ist_iso()),
        status="OPEN",
        legs=legs,
        lots=lots,
        lot_size=lot_size,
        premium_at_risk=float(prem_risk),
    )


def check_portfolio_caps(
    state: PortfolioState,
    new_pos: Position,
    *,
    capital: float,
    spot: float,
    max_premium_frac: float,
    max_delta_notional_frac: float,
    max_vega_frac: float,
    max_gamma_frac: float,
    max_theta_frac: float,
    max_pos_per_underlying: int,
    max_pos_per_cluster: int,
) -> Tuple[bool, str]:
    """Return (ok, reason)."""
    if capital <= 0:
        return True, "CAPITAL_UNKNOWN"

    by_under, by_cluster = count_open_positions(state)
    if by_under.get(new_pos.underlying, 0) >= int(max_pos_per_underlying):
        return False, "PORTFOLIO_CAP_UNDERLYING_COUNT"
    if by_cluster.get(new_pos.cluster, 0) >= int(max_pos_per_cluster):
        return False, "PORTFOLIO_CAP_CLUSTER_COUNT"

    # aggregate including new
    temp = PortfolioState(asof=state.asof, positions=state.positions + [new_pos])
    exp = aggregate_exposures(temp, spot_by_underlying={new_pos.underlying: float(spot)})

    if exp["premium_at_risk"] > float(max_premium_frac) * capital:
        return False, "PORTFOLIO_CAP_PREMIUM"
    if exp["delta_notional"] > float(max_delta_notional_frac) * capital:
        return False, "PORTFOLIO_CAP_DELTA"
        # vega/gamma/theta are scaled; these are rough heuristics
    if exp["vega_1pct_total"] > float(max_vega_frac) * (capital / 1000.0):
        return False, "PORTFOLIO_CAP_VEGA"
    if exp["gamma_total"] > float(max_gamma_frac) * (capital / 1000.0):
        return False, "PORTFOLIO_CAP_GAMMA"
    if exp["theta_day_total"] > float(max_theta_frac) * (capital / 1000.0):
        return False, "PORTFOLIO_CAP_THETA"

    return True, "OK"



def maybe_reserve_position(state: PortfolioState, pos: Position, enabled: bool) -> PortfolioState:
    if not enabled:
        return state
    state.positions.append(pos)
    state.asof = _now_ist_iso()
    return state
