#!/usr/bin/env python3
"""
Trade log analyzer

Given a backtest report directory (e.g., backtests/reports/fxify_phase1_15m/EURUSD.SIM),
compute:
- Top-N worst losers
- Core vs Pyramid vs Scalper breakdown (counts, PF, PnL)
- Drawdown segments: detect max drawdown window from equity_curve.csv and list trades in that window

Usage:
  python backtests/analyze_trade_log.py --report-dir backtests/reports/fxify_phase1_15m/EURUSD.SIM --top 10
"""
from __future__ import annotations
import os
import sys
import argparse
import math
import pandas as pd
from pathlib import Path

EPS = 1e-12

def load_report_dir(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades_p = report_dir / "trades.csv"
    equity_p = report_dir / "equity_curve.csv"
    if not trades_p.exists() or not equity_p.exists():
        raise FileNotFoundError(f"Expected trades.csv and equity_curve.csv in {report_dir}")
    tr = pd.read_csv(trades_p, parse_dates=["entry_time", "exit_time"]) if trades_p.exists() else pd.DataFrame()
    eq = pd.read_csv(equity_p, parse_dates=["time"]) if equity_p.exists() else pd.DataFrame()
    if not eq.empty:
        eq.set_index("time", inplace=True)
    return tr, eq


def worst_losers(tr: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    df = tr.copy()
    if df.empty:
        return df
    df = df.sort_values("pnl").head(top)
    return df


def pf(sum_wins: float, sum_losses: float) -> float:
    return (sum_wins / max(sum_losses, EPS)) if sum_losses > 0 else float("inf")


def core_vs_pyramid(tr: pd.DataFrame) -> pd.DataFrame:
    df = tr.copy()
    if df.empty:
        return pd.DataFrame()
    # Reason PyramidAdd denotes add rows (zero PnL bookkeeping for core layer); scalper has layer==scalper
    df["bucket"] = df.apply(lambda r: ("scalper" if str(r.get("layer", "")).lower()=="scalper" else ("pyramid" if str(r.get("reason", ""))=="PyramidAdd" else "core")), axis=1)
    rows = []
    for b in ["core", "pyramid", "scalper"]:
        sub = df[df["bucket"] == b]
        if sub.empty:
            rows.append({"bucket": b, "trades": 0, "wins": 0, "losses": 0, "win_rate_%": 0.0, "sum_wins": 0.0, "sum_losses": 0.0, "PF": 0.0, "PnL": 0.0})
            continue
        wins = sub[sub["pnl"] > 0]
        losses = sub[sub["pnl"] < 0]
        rows.append({
            "bucket": b,
            "trades": len(sub),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_%": 100.0 * len(wins) / max(len(sub), 1),
            "sum_wins": wins["pnl"].sum(),
            "sum_losses": -losses["pnl"].sum(),
            "PF": pf(wins["pnl"].sum(), -losses["pnl"].sum()),
            "PnL": sub["pnl"].sum(),
        })
    return pd.DataFrame(rows)


def max_drawdown_window(eq: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None, float]:
    if eq.empty:
        return None, None, 0.0
    s = eq["equity"].astype(float)
    roll_max = s.cummax()
    dd = s / roll_max - 1.0
    min_dd = dd.min()
    t2 = dd.idxmin()
    # Find preceding peak time
    peaks = roll_max[roll_max==roll_max.loc[:t2].max()]
    if peaks.empty:
        return None, None, float(min_dd)
    t1 = peaks.index[0]
    return t1, t2, float(min_dd)


def trades_in_window(tr: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if tr.empty or start is None or end is None:
        return pd.DataFrame()
    df = tr.copy()
    return df[(df["exit_time"]>=start) & (df["exit_time"]<=end)].sort_values("exit_time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", required=True, help="Path to symbol report dir containing trades.csv and equity_curve.csv")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    tr, eq = load_report_dir(report_dir)

    print(f"Report: {report_dir}")
    if tr.empty:
        print("No trades.csv found or it's empty.")
        sys.exit(0)

    # Worst losers
    worst = worst_losers(tr, top=args.top)
    print("\nWorst losers (top N):")
    print(worst.to_string(index=False))

    # Core vs Pyramid vs Scalper
    buckets = core_vs_pyramid(tr)
    print("\nCore vs Pyramid vs Scalper:")
    print(buckets.to_string(index=False))

    # Drawdown window
    if not eq.empty:
        t1, t2, mdd = max_drawdown_window(eq)
        print(f"\nMax Drawdown: {mdd*100:.2f}% from {t1} to {t2}")
        dd_trades = trades_in_window(tr, t1, t2)
        if not dd_trades.empty:
            print("Trades during max DD window:")
            print(dd_trades[["entry_time","exit_time","symbol","pnl","reason","layer","entry_adx","entry_atr_pct","entry_ma_bps","entry_edge_score","entry_spread_pips"]].to_string(index=False))
    else:
        print("\nEquity curve not found; skipping drawdown analysis.")

if __name__ == "__main__":
    main()
