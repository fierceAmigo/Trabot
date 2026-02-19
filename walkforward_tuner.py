"""walkforward_tuner.py

Phase-6: walk-forward style tuning on evaluated reco CSV.

Usage:
  python walkforward_tuner.py --csv data/reco_evaluated_v22_latest.csv

It runs rolling windows:
- train: N days
- test: M days
Selects best parameter set on train (by score = profit_factor penalized by drawdown)
Evaluates it on the next test window.

Outputs:
- data/walkforward_report_latest.txt
"""

from __future__ import annotations

import argparse
import itertools
import os
import pandas as pd

from tuning import apply_filters, compute_metrics


def score_fn(m):
    # simple conservative objective
    pf = m.profit_factor if m.profit_factor == m.profit_factor else 0.0
    dd = abs(m.max_drawdown)
    n = m.n
    if n < 20:
        return -1e9
    return pf - 2.0 * dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/reco_evaluated_v22_latest.csv")
    ap.add_argument("--train_days", type=int, default=60)
    ap.add_argument("--test_days", type=int, default=20)
    ap.add_argument("--step_days", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "ts_reco" not in df.columns:
        raise SystemExit("Missing ts_reco in evaluated CSV")

    df["ts_reco"] = pd.to_datetime(df["ts_reco"], errors="coerce")
    df = df.dropna(subset=["ts_reco"]).sort_values("ts_reco")

    # parameter grid (keep small to avoid overfitting)
    grid = {
        "score": [0.8, 1.0, 1.2],
        "max_spread_pct": [0.05, 0.08, 0.12],
        "min_ivp": [None, 0.30],
        "max_quote_age_s": [None, 3.0, 5.0],
    }

    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]

    start = df["ts_reco"].min().normalize()
    end = df["ts_reco"].max().normalize()

    cur = start
    lines = []
    lines.append(f"Walk-forward tuning on: {args.csv}")
    lines.append(f"Data range: {start.date()} .. {end.date()}")
    lines.append(f"Train={args.train_days}d, Test={args.test_days}d, Step={args.step_days}d")
    lines.append("")

    overall = []

    while cur + pd.Timedelta(days=args.train_days + args.test_days) <= end + pd.Timedelta(days=1):
        train_start = cur
        train_end = cur + pd.Timedelta(days=args.train_days)
        test_end = train_end + pd.Timedelta(days=args.test_days)

        train = df[(df["ts_reco"] >= train_start) & (df["ts_reco"] < train_end)]
        test = df[(df["ts_reco"] >= train_end) & (df["ts_reco"] < test_end)]

        best = None
        best_score = -1e18
        best_m = None

        for p in combos:
            tr = apply_filters(train, p)
            m = compute_metrics(tr)
            sc = score_fn(m)
            if sc > best_score:
                best_score = sc
                best = p
                best_m = m

        te = apply_filters(test, best or {})
        m_te = compute_metrics(te)
        overall.append(m_te)

        lines.append(f"Window {train_start.date()}..{train_end.date()} train -> {train_end.date()}..{test_end.date()} test")
        lines.append(f"  best_params={best}")
        lines.append(f"  train: n={best_m.n} win={best_m.win_rate:.2f} pf={best_m.profit_factor:.2f} dd={best_m.max_drawdown:.2f}")
        lines.append(f"  test : n={m_te.n} win={m_te.win_rate:.2f} pf={m_te.profit_factor:.2f} dd={m_te.max_drawdown:.2f}")
        lines.append("")

        cur = cur + pd.Timedelta(days=args.step_days)

    # aggregate test windows (simple average)
    if overall:
        import numpy as np
        n = sum(m.n for m in overall)
        win = float(np.average([m.win_rate for m in overall], weights=[max(m.n,1) for m in overall]))
        pf = float(np.average([min(m.profit_factor, 20.0) for m in overall], weights=[max(m.n,1) for m in overall]))
        dd = float(min(m.max_drawdown for m in overall))
        lines.append("OVERALL TEST (avg across windows)")
        lines.append(f"  n={n} win={win:.2f} pf~={pf:.2f} worst_dd={dd:.2f}")

    out_path = "data/walkforward_report_latest.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
