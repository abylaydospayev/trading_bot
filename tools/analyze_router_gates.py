import argparse
from pathlib import Path
import pandas as pd


def analyze_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    # derive hour
    ts_col = 'ts'
    try:
        dt = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    except Exception:
        return pd.DataFrame()
    df['hour'] = dt.dt.hour
    df['adx_ok'] = df['adx_ok'].astype(bool)
    df['ma_ok'] = df['ma_ok'].astype(bool)
    df['gate_ok'] = df['adx_ok'] & df['ma_ok']
    grp = df.groupby(['symbol','interval','hour']).agg(
        samples=('ts','count'),
        adx_pass_rate=('adx_ok','mean'),
        ma_pass_rate=('ma_ok','mean'),
        gate_pass_rate=('gate_ok','mean'),
        mean_adx=('adx','mean'),
        mean_mabps=('ma_bps','mean'),
        mean_mabps_thr=('mabps_thr','mean'),
        mean_ma_norm=('ma_norm','mean'),
        mean_atr_pct=('atr_pct','mean'),
        mean_edge_score=('edge_score','mean'),
    ).reset_index()
    grp['file'] = path.name
    return grp


def main():
    ap = argparse.ArgumentParser(description='Analyze router R1 gates CSVs')
    ap.add_argument('--reports-dir', default='backtests/reports')
    ap.add_argument('--out', default='backtests/reports/router_gates_summary.csv')
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    out_path = Path(args.out)
    frames = []
    for p in reports_dir.glob('router_gates_*_*.csv'):
        try:
            part = analyze_file(p)
            if not part.empty:
                frames.append(part)
        except Exception:
            continue
    if not frames:
        print('No router gates CSVs found or all empty.')
        return
    result = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')
    # Print quick rollup
    roll = result.groupby(['symbol','interval']).agg(
        hours=('hour','nunique'),
        samples=('samples','sum'),
        mean_gate_pass=('gate_pass_rate','mean'),
        median_gate_pass=('gate_pass_rate','median'),
    ).reset_index()
    print(roll)


if __name__ == '__main__':
    main()
