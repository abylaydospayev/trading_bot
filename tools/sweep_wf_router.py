import argparse
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import sys

from datetime import datetime

# Ensure imports work from project root
from pathlib import Path as _P
ROOT_DIR = _P(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtests.wf_validate import run_wf_for_symbol


def parse_float_list(s: str):
    vals = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


essential_cols = [
    'symbol','seg_start','seg_end','bars','return_pct','maxdd_pct','profit_factor','trades','engine'
]


def main():
    ap = argparse.ArgumentParser(description='Sweep router parameters via WF')
    ap.add_argument('--symbols', default='GBPUSD,USDJPY')
    ap.add_argument('--start', default='2025-09-20')
    ap.add_argument('--end', default=datetime.utcnow().strftime('%Y-%m-%d'))
    ap.add_argument('--interval', default='15m')
    ap.add_argument('--capital', type=float, default=10000.0)
    ap.add_argument('--window-days', type=int, default=30)
    ap.add_argument('--step-days', type=int, default=30)
    ap.add_argument('--atr-min-list', default='0.03,0.04')
    ap.add_argument('--mr-z-entry-list', default='0.8,1.0,1.2,1.4')
    ap.add_argument('--r1-adx-gate-list', default='24,27', help='List of R1 ADX gate thresholds to sweep')
    ap.add_argument('--edge-buy-score-list', default='50,60', help='List of edge buy score thresholds to sweep')
    ap.add_argument('--edge-confirm-bars-list', default='1,2', help='List of confirm bars to sweep')
    ap.add_argument('--enforce-spread-cap', action='store_true', help='Enable optional router spread cap gate during sweep')
    ap.add_argument('--out', default='backtests/reports/wf/wf_router_sweep.csv')
    args = ap.parse_args()

    syms = [s.strip() for s in args.symbols.split(',') if s.strip()]
    atr_list = parse_float_list(args.atr_min_list)
    z_list = parse_float_list(args.mr_z_entry_list)
    adx_gate_list = parse_float_list(args.r1_adx_gate_list)
    edge_score_list = parse_float_list(args.edge_buy_score_list)
    confirm_list = parse_float_list(args.edge_confirm_bars_list)

    rows = []
    for sym in syms:
        for atr_min, z_entry, adx_gate, edge_score, confirm_bars in itertools.product(atr_list, z_list, adx_gate_list, edge_score_list, confirm_list):
            try:
                df = run_wf_for_symbol(
                    symbol=sym,
                    start=args.start,
                    end=args.end,
                    interval=args.interval,
                    capital=args.capital,
                    opportunistic=True,
                    window_months=3,
                    step_months=1,
                    engine='router',
                    align_months=False,
                    window_days=args.window_days,
                    step_days=args.step_days,
                    router_atr_min=atr_min,
                    router_mr_z_entry=z_entry,
                    r1_adx_gate=adx_gate,
                    edge_buy_score=int(edge_score),
                    edge_confirm_bars=int(confirm_bars),
                    enforce_spread_cap=args.enforce_spread_cap,
                )
                if len(df) == 0:
                    rows.append({
                        'symbol': sym,
                        'interval': args.interval,
                        'atr_min': atr_min,
                        'mr_z_entry': z_entry,
                        'r1_adx_gate': adx_gate,
                        'edge_buy_score': int(edge_score),
                        'edge_confirm_bars': int(confirm_bars),
                        'segments': 0,
                        'pass_rate': np.nan,
                        'median_pf': np.nan,
                        'median_dd': np.nan,
                        'total_trades': 0,
                    })
                    continue
                df['pass_pf'] = df['profit_factor'] >= 1.2
                df['pass_dd'] = df['maxdd_pct'].abs() <= 10.0
                df['pass'] = df['pass_pf'] & df['pass_dd']
                rows.append({
                    'symbol': sym,
                    'interval': args.interval,
                    'atr_min': atr_min,
                    'mr_z_entry': z_entry,
                    'r1_adx_gate': adx_gate,
                    'edge_buy_score': int(edge_score),
                    'edge_confirm_bars': int(confirm_bars),
                    'segments': len(df),
                    'pass_rate': float(df['pass'].mean()),
                    'median_pf': float(df['profit_factor'].median()),
                    'median_dd': float(np.median(np.abs(df['maxdd_pct']))),
                    'total_trades': int(df['trades'].sum()),
                })
            except Exception as e:
                rows.append({
                    'symbol': sym,
                    'interval': args.interval,
                    'atr_min': atr_min,
                    'mr_z_entry': z_entry,
                    'r1_adx_gate': adx_gate,
                    'edge_buy_score': int(edge_score),
                    'edge_confirm_bars': int(confirm_bars),
                    'segments': 0,
                    'pass_rate': np.nan,
                    'median_pf': np.nan,
                    'median_dd': np.nan,
                    'total_trades': 0,
                    'error': str(e),
                })
    result = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    # Print quick pivot by symbol
    try:
        piv = result.pivot_table(index=['symbol','interval'], columns=['atr_min','mr_z_entry','r1_adx_gate','edge_buy_score','edge_confirm_bars'], values='median_pf')
        print('\nMedian PF by (atr_min, mr_z_entry, r1_adx_gate, edge_buy_score, edge_confirm_bars):')
        print(piv)
    except Exception:
        pass


if __name__ == '__main__':
    main()
