"""
Walk-Forward Validation Harness for FX Strategy

Runs rolling walk-forward backtests across date segments and reports
per-segment performance metrics (Return, MaxDD, Profit Factor), along with
pass/fail against thresholds. Supports multiple symbols.

Enhancements:
- --engine {base,router} to choose the base backtester or the router-parity one
- Day-based segments when --align-months=false via --window-days/--step-days
    to avoid Yahoo 60-day 15m data limit issues
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Project root for imports
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtests.backtest_forex_fxify import download_forex as dl_base, backtest_fxify
from backtests.backtest_forex_fxify_router import (
    download_forex as dl_router,
    backtest_router,
)


def _month_floor(dt: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=dt.year, month=dt.month, day=1, tz='UTC')


def _add_months(dt: pd.Timestamp, months: int) -> pd.Timestamp:
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    day = 1
    return pd.Timestamp(year=year, month=month, day=day, tz='UTC')


def _segment_dates(start: str, end: str, window_months: int, step_months: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    s = pd.Timestamp(start, tz='UTC')
    e = pd.Timestamp(end, tz='UTC')
    s = _month_floor(s)
    e = _month_floor(e)
    out = []
    cur = s
    while True:
        seg_start = cur
        seg_end = _add_months(seg_start, window_months)
        if seg_end > e:
            break
        out.append((seg_start, seg_end))
        cur = _add_months(cur, step_months)
        if cur >= e:
            break
    return out


def _segment_dates_days(start: str, end: str, window_days: int, step_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    s = pd.Timestamp(start, tz='UTC')
    e = pd.Timestamp(end, tz='UTC')
    out = []
    cur = s
    while True:
        seg_start = cur
        seg_end = seg_start + pd.Timedelta(days=window_days)
        if seg_end > e:
            break
        out.append((seg_start, seg_end))
        cur = cur + pd.Timedelta(days=step_days)
        if cur >= e:
            break
    return out


def profit_factor_from_trades(trades) -> float:
    wins = sum([max(0.0, float(getattr(t, 'pnl', 0.0) or 0.0)) for t in trades])
    losses = sum([max(0.0, -float(getattr(t, 'pnl', 0.0) or 0.0)) for t in trades])
    if losses <= 1e-12:
        return float('inf') if wins > 0 else 1.0
    return wins / losses


def max_drawdown_pct(equity: pd.Series) -> float:
    run_max = equity.cummax()
    dd = (equity - run_max) / run_max
    return float(dd.min() * 100.0)


def ftmo_violations(equity: pd.Series, initial_capital: float,
                    daily_limit: float = 0.05, total_limit: float = 0.10) -> Tuple[bool, bool]:
    """Return (viol_daily, viol_total) flags based on equity series.
    - total_limit is evaluated vs initial capital.
    - daily_limit is evaluated per calendar day vs that day's start equity.
    """
    if len(equity) == 0 or initial_capital <= 0:
        return (False, False)
    # Total loss vs initial
    min_equity = float(equity.min())
    total_loss_pct = (initial_capital - min_equity) / initial_capital
    viol_total = bool(total_loss_pct > (total_limit + 1e-9))

    # Daily loss vs day start equity
    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()
    viol_daily = False
    for _, grp in eq.groupby(eq.index.date):
        day_start = float(grp.iloc[0])
        if day_start <= 0:
            continue
        min_day = float(grp.min())
        loss_pct = (day_start - min_day) / day_start
        if loss_pct > (daily_limit + 1e-9):
            viol_daily = True
            break
    return (viol_daily, viol_total)


def run_wf_for_symbol(symbol: str, start: str, end: str, interval: str, capital: float, opportunistic: bool,
                      window_months: int, step_months: int,
                      engine: str = 'base', align_months: bool = True,
                      window_days: int = 30, step_days: int = 30,
                      router_atr_min: float = 0.04, router_mr_z_entry: float = 1.2,
                      r1_adx_enter: float = 24.0, r1_mabps_enter: float = 2.0,
                      r1_adx_stay: float = 22.0, r1_mabps_stay: float = 1.8,
                      edge_buy_score: int = 60, edge_confirm_bars: int = 2,
                      edge_exit_score: int = 10,
                      mr_z_exit: float = 0.3, mr_time_stop_bars: int = 8,
                      r1_adx_gate: float = 27.0,
                      enforce_spread_cap: bool = False) -> pd.DataFrame:
    if align_months:
        segs = _segment_dates(start, end, window_months, step_months)
    else:
        segs = _segment_dates_days(start, end, window_days, step_days)
    rows = []
    for (seg_s, seg_e) in segs:
        try:
            # Use engine-specific downloader
            if engine == 'router':
                bars = dl_router(symbol, seg_s.strftime('%Y-%m-%d'), interval)
            else:
                bars = dl_base(symbol, seg_s.strftime('%Y-%m-%d'), interval)
            # Clip to seg end
            bars = bars[bars.index < seg_e.tz_convert(None)]
            if len(bars) < 200:
                continue
            if engine == 'router':
                equity, trades = backtest_router(
                    bars, symbol, interval, capital,
                    atr_min=router_atr_min,
                    mr_z_entry=router_mr_z_entry,
                    r1_adx_enter=r1_adx_enter,
                    r1_mabps_enter=r1_mabps_enter,
                    r1_adx_stay=r1_adx_stay,
                    r1_mabps_stay=r1_mabps_stay,
                    edge_buy_score=edge_buy_score,
                    edge_confirm_bars=edge_confirm_bars,
                    edge_exit_score=edge_exit_score,
                    mr_z_exit=mr_z_exit,
                    mr_time_stop_bars=mr_time_stop_bars,
                    r1_adx_gate=r1_adx_gate,
                    enforce_spread_cap=enforce_spread_cap,
                )
            else:
                equity, trades, _day = backtest_fxify(bars, symbol, capital, opportunistic)
            if len(equity) == 0:
                continue
            ret_pct = (float(equity.iloc[-1]) / capital - 1.0) * 100.0
            maxdd_pct = max_drawdown_pct(equity)
            pf = profit_factor_from_trades(trades)
            viol_daily, viol_total = ftmo_violations(equity, capital)
            rows.append({
                'symbol': symbol,
                'seg_start': seg_s.strftime('%Y-%m-%d'),
                'seg_end': seg_e.strftime('%Y-%m-%d'),
                'bars': len(bars),
                'return_pct': ret_pct,
                'maxdd_pct': maxdd_pct,
                'profit_factor': pf,
                'trades': len(trades),
                'engine': engine,
                'viol_daily': viol_daily,
                'viol_total': viol_total,
            })
        except Exception as e:
            rows.append({
                'symbol': symbol,
                'seg_start': seg_s.strftime('%Y-%m-%d'),
                'seg_end': seg_e.strftime('%Y-%m-%d'),
                'bars': 0,
                'return_pct': np.nan,
                'maxdd_pct': np.nan,
                'profit_factor': np.nan,
                'trades': 0,
                'error': str(e),
                'engine': engine,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Walk-Forward Validation')
    ap.add_argument('--symbols', default='EURUSD,XAUUSD')
    ap.add_argument('--start', default='2024-01-01')
    ap.add_argument('--end', default=datetime.now(timezone.utc).strftime('%Y-%m-%d'))
    ap.add_argument('--interval', default='15m')
    ap.add_argument('--capital', type=float, default=10000.0)
    ap.add_argument('--mode', default='opportunistic', choices=['opportunistic', 'classic'])
    ap.add_argument('--window-months', type=int, default=3)
    ap.add_argument('--step-months', type=int, default=1)
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--align-months', dest='align_months', action='store_true', help='Align segments to calendar months (default)')
    group.add_argument('--no-align-months', dest='align_months', action='store_false', help='Use day-based segments (--window-days/--step-days)')
    ap.set_defaults(align_months=True)
    ap.add_argument('--window-days', type=int, default=30)
    ap.add_argument('--step-days', type=int, default=30)
    ap.add_argument('--engine', default='base', choices=['base','router'])
    ap.add_argument('--router-atr-min', type=float, default=0.04, help='ATR%% lower band for router entries')
    ap.add_argument('--router-mr-z-entry', type=float, default=1.2, help='Router MR z-score entry threshold (abs)')
    ap.add_argument('--r1-adx-enter', type=float, default=24.0)
    ap.add_argument('--r1-mabps-enter', type=float, default=2.0)
    ap.add_argument('--r1-adx-stay', type=float, default=22.0)
    ap.add_argument('--r1-mabps-stay', type=float, default=1.8)
    ap.add_argument('--edge-buy-score', type=int, default=60)
    ap.add_argument('--edge-confirm-bars', type=int, default=2)
    ap.add_argument('--edge-exit-score', type=int, default=10)
    ap.add_argument('--mr-z-exit', type=float, default=0.3)
    ap.add_argument('--mr-time-stop-bars', type=int, default=8)
    ap.add_argument('--r1-adx-gate', type=float, default=27.0, help='R1 ADX quality gate threshold')
    ap.add_argument('--enforce-spread-cap', action='store_true', help='Enforce optional pair-specific spread hard cap gate in router backtests')
    ap.add_argument('--min-pf', type=float, default=1.2)
    ap.add_argument('--max-dd', type=float, default=10.0)
    ap.add_argument('--out', default='backtests/reports/wf')
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    opportunistic = args.mode == 'opportunistic'
    syms = [s.strip() for s in args.symbols.split(',') if s.strip()]

    all_rows = []
    for sym in syms:
        df = run_wf_for_symbol(sym, args.start, args.end, args.interval, args.capital, opportunistic,
                               args.window_months, args.step_months,
                               engine=args.engine, align_months=args.align_months,
                               window_days=args.window_days, step_days=args.step_days,
                               router_atr_min=args.router_atr_min,
                               router_mr_z_entry=args.router_mr_z_entry,
                               r1_adx_enter=args.r1_adx_enter,
                               r1_mabps_enter=args.r1_mabps_enter,
                               r1_adx_stay=args.r1_adx_stay,
                               r1_mabps_stay=args.r1_mabps_stay,
                               edge_buy_score=args.edge_buy_score,
                               edge_confirm_bars=args.edge_confirm_bars,
                               edge_exit_score=args.edge_exit_score,
                               mr_z_exit=args.mr_z_exit,
                               mr_time_stop_bars=args.mr_time_stop_bars,
                               r1_adx_gate=args.r1_adx_gate,
                               enforce_spread_cap=args.enforce_spread_cap)
        if len(df) == 0:
            continue
        df['pass_pf'] = df['profit_factor'] >= args.min_pf
        df['pass_dd'] = df['maxdd_pct'].abs() <= args.max_dd
        df['pass'] = df['pass_pf'] & df['pass_dd']
        suffix = f"{args.engine}_{'months' if args.align_months else 'days'}"
        out_path = Path(args.out) / f"wf_{sym}_{args.interval}_{suffix}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
        all_rows.append(df.assign(symbol=sym))

    if all_rows:
        cat = pd.concat(all_rows, ignore_index=True)
        agg = cat.groupby(['symbol','engine']).agg(
            segments=('symbol', 'count'),
            pass_rate=('pass', 'mean'),
            median_pf=('profit_factor', 'median'),
            median_dd=('maxdd_pct', lambda x: float(np.median(np.abs(x))))
        ).reset_index()
        agg.to_csv(Path(args.out) / f"wf_summary_{suffix}.csv", index=False)
        print(agg)


if __name__ == '__main__':
    main()
