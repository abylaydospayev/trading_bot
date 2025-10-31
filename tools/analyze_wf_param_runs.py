import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np

def build_pattern(symbol: str):
    sym = symbol.lower()
    return re.compile(rf"{sym}_1h_r1adx(?P<adx>\d+)_edge(?P<edge>\d+)_conf(?P<conf>\d+)")


def main():
    ap = argparse.ArgumentParser(description='Summarize WF param runs for symbol 1h')
    ap.add_argument('--symbol', default='GBPUSD', help='Symbol used in WF CSV filenames (e.g., EURUSD)')
    ap.add_argument('--prefix', default=None, help='Folder prefix (e.g., eur, gbp, jpy, xau). If omitted, uses first three letters of symbol.')
    ap.add_argument('--base-dir', default='backtests/reports/wf/sweeps')
    ap.add_argument('--out', default='backtests/reports/wf/gbp_1h_param_sweep_summary.csv')
    args = ap.parse_args()

    base = Path(args.base_dir)
    rows = []
    prefix = (args.prefix or args.symbol[:3]).lower()
    pat = build_pattern(prefix)
    for sub in base.glob(f"{prefix}_1h_*"):
        m = pat.search(sub.name)
        if not m:
            continue
        adx = int(m.group('adx'))
        edge = int(m.group('edge'))
        conf = int(m.group('conf'))
        csv_path = sub / f'wf_{args.symbol}_1h_router_months.csv'
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        # Ensure columns
        if 'pass' not in df.columns:
            # reconstruct pass using defaults from wf_validate
            df['pass_pf'] = df['profit_factor'] >= 1.2
            df['pass_dd'] = df['maxdd_pct'].abs() <= 10.0
            df['pass'] = df['pass_pf'] & df['pass_dd']
        rows.append({
            'r1_adx_enter': adx,
            'edge_buy_score': edge,
            'edge_confirm_bars': conf,
            'segments': len(df),
            'pass_rate': float(df['pass'].mean()),
            'median_pf': float(df['profit_factor'].median()),
            'median_dd': float(np.median(np.abs(df['maxdd_pct']))),
            'total_trades': int(df['trades'].sum()) if 'trades' in df.columns else np.nan,
        })
    if not rows:
        print('No runs found.')
        return
    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    if out_path.name == 'gbp_1h_param_sweep_summary.csv':
        out_path = out_path.with_name(f"{args.symbol.lower()}_1h_param_sweep_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.sort_values(['pass_rate','median_pf'], ascending=[False, False], inplace=True)
    out_df.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')
    print(out_df.head(10))


if __name__ == '__main__':
    main()
