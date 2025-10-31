import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# project root
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.backtest_forex_fxify_router import (
    download_forex,
    compute_indicators_router,
    is_active_hour_utc,
    annualize_sigma_percent,
    pair_vol_floor_percent,
    ATR_PCT_MAX,
)


def main():
    ap = argparse.ArgumentParser(description='Probe router hard gates and ATR band stats')
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--interval', default='1h')
    args = ap.parse_args()

    bars = download_forex(args.symbol, args.start, args.interval)
    df = compute_indicators_router(bars, args.interval)

    # Compute hard gate components
    session_ok = pd.Series(df.index.tz_localize('UTC').map(lambda ts: is_active_hour_utc(ts, args.symbol)), index=df.index)
    sigma = df['mr_ewm_sigma'].astype(float).fillna(0.0)
    price = df['close'].astype(float).fillna(0.0)
    sigma_ann = (sigma / price).replace([np.inf,-np.inf], np.nan).fillna(0.0) * np.sqrt(252*24 if args.interval=='1h' else 252*24*4)
    vol_floor = pair_vol_floor_percent(args.symbol)
    vol_ok = sigma_ann >= (vol_floor/100.0)  # sigma_ann here expressed as fraction

    atr_pct = (df['atr'] / df['close']).replace([np.inf,-np.inf], np.nan).fillna(0.0) * 100.0

    print(f"Symbol {args.symbol} {args.interval}")
    print(f"Bars: {len(df)}")
    print(f"Session pass rate: {session_ok.mean():.3f}")
    print(f"Vol floor (pct): {vol_floor:.3f}")
    print(f"Sigma_ann pct mean/med: {sigma_ann.mean()*100:.3f}/{np.median(sigma_ann)*100:.3f}")
    print(f"Vol pass rate: {(vol_ok.mean()):.3f}")
    q = np.quantile(atr_pct, [0.1,0.25,0.5,0.75,0.9])
    print(f"ATR%% quantiles p10,p25,p50,p75,p90: {q}")
    print(f"ATR%% max cap: {ATR_PCT_MAX}")

if __name__ == '__main__':
    main()
