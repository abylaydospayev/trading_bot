"""
Export 15m (or other timeframe) bars from MetaTrader 5 to CSV files
in the format expected by fxify_phase1_backtest.py.

Usage (PowerShell):
  python backtests\export_mt5_csv.py --symbols XAUUSD,USDJPY --timeframe 15m \
      --start 2024-01-01 --end 2025-10-29 --out backtests\data

If MT5 is already running and logged in, login details are optional.
Otherwise, set env vars or pass flags for MT5_LOGIN/MT5_PASSWORD/MT5_SERVER.
"""
from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception as e:
    print("MetaTrader5 module not installed. Install with: pip install MetaTrader5")
    sys.exit(1)

# Timeframe mapping
TF_MAP = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
}


def initialize(login: int|None, password: str|None, server: str|None, path: str|None) -> None:
    if path:
        ok = mt5.initialize(path)
    else:
        ok = mt5.initialize()
    if not ok:
        print(f"MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)
    if login and password and server:
        if not mt5.login(login, password, server):
            print(f"MT5 login failed: {mt5.last_error()}")
            sys.exit(1)


def export_symbol(symbol: str, tf_str: str, start: str, end: str, out_dir: Path) -> None:
    timeframe = TF_MAP.get(tf_str.lower())
    if timeframe is None:
        raise SystemExit(f"Unsupported timeframe: {tf_str}")

    # Ensure symbol visible
    info = mt5.symbol_info(symbol)
    if info is None or not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise SystemExit(f"Symbol not available: {symbol}")

    # Convert dates
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    # Pull bars
    rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        print(f"No data returned for {symbol} {tf_str}")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(None)
    df = df.rename(columns={
        'time': 'time',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
    })
    df = df[['time','open','high','low','close']].sort_values('time')

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"dataset_{symbol}_{start}_{end}_{tf_str.lower()}.csv"
    df.to_csv(fname, index=False)
    print(f"Saved: {fname}")


def main():
    ap = argparse.ArgumentParser(description="Export MT5 bars to CSV")
    ap.add_argument('--symbols', required=True, help='Comma separated symbols e.g. XAUUSD,USDJPY')
    ap.add_argument('--timeframe', default='15m')
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--out', default='backtests/data')
    ap.add_argument('--mt5-path', default=os.getenv('MT5_PATH', ''))
    ap.add_argument('--mt5-login', type=int, default=int(os.getenv('MT5_LOGIN', '0')))
    ap.add_argument('--mt5-password', default=os.getenv('MT5_PASSWORD', ''))
    ap.add_argument('--mt5-server', default=os.getenv('MT5_SERVER', ''))
    args = ap.parse_args()

    login = args.mt5_login if args.mt5_login != 0 else None
    password = args.mt5_password or None
    server = args.mt5_server or None
    path = args.mt5_path or None

    initialize(login, password, server, path)

    out_dir = Path(args.out)
    for s in [x.strip() for x in args.symbols.split(',') if x.strip()]:
        export_symbol(s.upper(), args.timeframe, args.start, args.end, out_dir)

    mt5.shutdown()


if __name__ == '__main__':
    main()
