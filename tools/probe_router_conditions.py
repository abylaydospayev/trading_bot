import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.backtest_forex_fxify_router import (
    download_forex,
    compute_indicators_router,
)


def main():
    ap = argparse.ArgumentParser(description='Probe router condition frequencies (R1/R2)')
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--interval', default='1h')
    ap.add_argument('--r1-adx-enter', type=float, default=24.0)
    ap.add_argument('--r1-mabps-enter', type=float, default=2.0)
    ap.add_argument('--r1-adx-stay', type=float, default=22.0)
    ap.add_argument('--r1-mabps-stay', type=float, default=1.8)
    ap.add_argument('--mr-z-entry', type=float, default=1.2)
    args = ap.parse_args()

    bars = download_forex(args.symbol, args.start, args.interval)
    df = compute_indicators_router(bars, args.interval)

    price = df['close'].astype(float)
    short_ma = df['short_ma'].astype(float)
    long_ma = df['long_ma'].astype(float)
    adx = df['adx'].astype(float)
    z = df['mr_z'].astype(float)
    atr_pct = (df['atr']/df['close']).replace([np.inf,-np.inf], np.nan).fillna(0.0) * 100.0

    ma_bps = (short_ma - long_ma).abs() / price.clip(lower=1e-12) * 10000.0

    r1_ok_enter = (adx >= args.r1_adx_enter) & (ma_bps >= args.r1_mabps_enter)
    r1_ok_stay = (adx >= args.r1_adx_stay) & (ma_bps >= args.r1_mabps_stay)
    r2_ok_enter = (adx <= args.r1_adx_stay) & (z.abs() >= abs(args.mr_z_entry))

    print(f"Symbol {args.symbol} {args.interval}")
    print(f"Bars: {len(df)}")
    print(f"ADX mean/med: {adx.mean():.2f}/{adx.median():.2f}")
    print(f"MA_bps mean/med: {ma_bps.mean():.3f}/{ma_bps.median():.3f}")
    print(f"Z mean/med: {z.mean():.3f}/{z.median():.3f}")
    print(f"ATR% mean/med: {atr_pct.mean():.3f}/{atr_pct.median():.3f}")
    print(f"r1_ok_enter rate: {r1_ok_enter.mean():.3f}")
    print(f"r1_ok_stay rate: {r1_ok_stay.mean():.3f}")
    print(f"r2_ok_enter rate: {r2_ok_enter.mean():.3f}")

if __name__ == '__main__':
    main()
