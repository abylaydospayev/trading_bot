"""
Monte Carlo Resampling of Trade Outcomes

Runs a backtest to obtain a sequence of trades, then performs bootstrap
resampling of trade PnLs to estimate distribution of returns, max drawdown,
and profit factor. Saves CSV with iteration-level stats and a summary.
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd

# Project root
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtests.backtest_forex_fxify import download_forex, backtest_fxify


def profit_factor_from_pnls(pnls: List[float]) -> float:
    wins = sum([max(0.0, p) for p in pnls])
    losses = sum([max(0.0, -p) for p in pnls])
    if losses <= 1e-12:
        return float('inf') if wins > 0 else 1.0
    return wins / losses


def max_drawdown_pct_from_equity(eq: np.ndarray) -> float:
    run_max = np.maximum.accumulate(eq)
    dd = (eq - run_max) / (run_max + 1e-12)
    return float(np.min(dd) * 100.0)


def main():
    ap = argparse.ArgumentParser(description='Monte Carlo Resampling of Trades')
    ap.add_argument('--symbol', default='EURUSD')
    ap.add_argument('--start', default='2024-01-01')
    ap.add_argument('--interval', default='15m')
    ap.add_argument('--capital', type=float, default=10000.0)
    ap.add_argument('--mode', default='opportunistic', choices=['opportunistic', 'classic'])
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='backtests/reports/mc')
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    opportunistic = args.mode == 'opportunistic'

    # Obtain trades from a baseline backtest
    bars = download_forex(args.symbol, args.start, args.interval)
    equity, trades, _day = backtest_fxify(bars, args.symbol, args.capital, opportunistic)
    if len(trades) == 0:
        print('No trades; aborting MC.')
        return

    base_pnls = [float(getattr(t, 'pnl', 0.0) or 0.0) for t in trades]
    n = len(base_pnls)
    rng = np.random.default_rng(args.seed)

    rows = []
    for i in range(args.iters):
        # Sample with replacement same length as original
        idx = rng.integers(0, n, size=n)
        pnls = [base_pnls[j] for j in idx]
        eq = np.cumsum([args.capital] + pnls)  # prepend initial capital for curve
        eq = eq.astype(float)
        ret_pct = (eq[-1] / args.capital - 1.0) * 100.0
        maxdd = max_drawdown_pct_from_equity(eq)
        pf = profit_factor_from_pnls(pnls)
        rows.append({'iter': i+1, 'return_pct': ret_pct, 'maxdd_pct': maxdd, 'profit_factor': pf})

    df = pd.DataFrame(rows)
    path_all = Path(args.out) / f"mc_{args.symbol}_{args.interval}.csv"
    df.to_csv(path_all, index=False)
    print(f"Saved iterations: {path_all}")

    summary = pd.DataFrame({
        'symbol': [args.symbol],
        'interval': [args.interval],
        'iters': [args.iters],
        'ret_p50': [df['return_pct'].median()],
        'ret_p05': [df['return_pct'].quantile(0.05)],
        'ret_p95': [df['return_pct'].quantile(0.95)],
        'dd_p50': [df['maxdd_pct'].abs().median()],
        'dd_p95': [df['maxdd_pct'].abs().quantile(0.95)],
        'pf_p50': [df['profit_factor'].replace(np.inf, np.nan).median()],
        'pf_p05': [df['profit_factor'].replace(np.inf, np.nan).quantile(0.05)],
    })
    path_sum = Path(args.out) / f"mc_{args.symbol}_{args.interval}_summary.csv"
    summary.to_csv(path_sum, index=False)
    print(summary)


if __name__ == '__main__':
    main()
