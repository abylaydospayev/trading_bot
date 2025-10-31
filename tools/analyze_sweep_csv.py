import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description='Analyze WF router sweep CSV and print top parameter combos')
    ap.add_argument('--path', required=True, help='Path to sweep CSV')
    ap.add_argument('--top', type=int, default=10, help='Number of top rows to show')
    ap.add_argument('--min-segments', type=int, default=1, help='Require at least this many segments')
    ap.add_argument('--min-pass-rate', type=float, default=None, help='Optional minimum pass rate filter (0..1)')
    ap.add_argument('--max-dd', type=float, default=None, help='Optional maximum abs median drawdown filter')
    args = ap.parse_args()

    p = Path(args.path)
    df = pd.read_csv(p)
    if 'pass_rate' not in df.columns:
        print('Sweep file missing aggregated columns; aborting')
        return
    df2 = df.copy()
    df2 = df2[df2['segments'] >= args.min_segments].copy()
    if args.min_pass_rate is not None:
        df2 = df2[df2['pass_rate'] >= args.min_pass_rate]
    if args.max_dd is not None:
        df2 = df2[df2['median_dd'].abs() <= args.max_dd]
    if len(df2) == 0:
        print('No rows after filtering')
        return
    df2['abs_median_dd'] = df2['median_dd'].abs()
    df2 = df2.sort_values(['pass_rate','median_pf','total_trades','abs_median_dd'], ascending=[False, False, False, True])
    cols = ['symbol','interval','atr_min','mr_z_entry','r1_adx_gate','edge_buy_score','edge_confirm_bars','segments','pass_rate','median_pf','median_dd','total_trades']
    cols = [c for c in cols if c in df2.columns]
    print(df2[cols].head(args.top).to_string(index=False))

if __name__ == '__main__':
    main()
