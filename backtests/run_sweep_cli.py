#!/usr/bin/env python3
"""
Parameter sweep runner for EURUSD.sim using the existing CLI backtester.
Runs a small grid over ADX, MIN_MA_DIST_BPS, and risk, and writes a CSV leaderboard.

Usage:
  python backtests/run_sweep_cli.py \
    --symbol EURUSD.sim \
    --start 2024-10-30 --end 2025-10-30 \
    --data-dir backtests/data \
    --out backtests/reports/sweep_eurusd_12m.csv
"""
import argparse
import csv
import re
import subprocess
from pathlib import Path

SUMMARY_RE = re.compile(
    r"Return:\s*([+-]?[0-9.]+)%\s*\|\s*MaxDD:\s*([+-]?[0-9.]+)%\s*\|\s*Sharpe:\s*([+-]?[0-9.]+)\s*\|\s*Trades:\s*(\d+)\s*\|\s*WinRate:\s*([0-9.]+)%\s*\|\s*PF:\s*([0-9.]+)"
)

DEFAULT_ADX = [18, 20, 22, 25]
DEFAULT_MABPS = [1.6, 1.8, 2.0, 2.2]
DEFAULT_RISK = [0.0025, 0.0030, 0.0035, 0.0040]

def run_case(symbol: str, start: str, end: str, data_dir: str, adx: float, mabps: float, risk: float) -> dict:
    args = [
        "python", "backtests/fxify_phase1_backtest.py",
        "--symbols", symbol,
        "--timeframe", "15m",
        "--data-dir", data_dir,
        "--capital", "100000",
        "--risk-pct", f"{risk}",
        "--atr-mult", "2.4",
        "--adx-threshold", f"{adx}",
        "--min-ma-dist-bps", f"{mabps}",
        "--sim-start", start,
        "--sim-end", end,
        "--ftmo",
    ]
    p = subprocess.run(args, capture_output=True, text=True)
    out = p.stdout + "\n" + p.stderr
    m = SUMMARY_RE.search(out)
    if not m:
        return {
            "return_pct": None, "maxdd_pct": None, "sharpe": None,
            "trades": None, "winrate_pct": None, "pf": None,
            "ok": False,
            "raw": out,
        }
    return {
        "return_pct": float(m.group(1)),
        "maxdd_pct": float(m.group(2)),
        "sharpe": float(m.group(3)),
        "trades": int(m.group(4)),
        "winrate_pct": float(m.group(5)),
        "pf": float(m.group(6)),
        "ok": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD.sim")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--data-dir", default="backtests/data")
    ap.add_argument("--out", default="backtests/reports/sweep_eurusd.csv")
    ap.add_argument("--adx", nargs="*", type=float, default=DEFAULT_ADX)
    ap.add_argument("--mabps", nargs="*", type=float, default=DEFAULT_MABPS)
    ap.add_argument("--risk", nargs="*", type=float, default=DEFAULT_RISK)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for adx in args.adx:
        for mabps in args.mabps:
            for risk in args.risk:
                res = run_case(args.symbol, args.start, args.end, args.data_dir, adx, mabps, risk)
                row = {
                    "symbol": args.symbol,
                    "start": args.start,
                    "end": args.end,
                    "adx": adx,
                    "ma_bps": mabps,
                    "risk": risk,
                    "return_pct": res["return_pct"],
                    "maxdd_pct": res["maxdd_pct"],
                    "sharpe": res["sharpe"],
                    "trades": res["trades"],
                    "winrate_pct": res["winrate_pct"],
                    "pf": res["pf"],
                    "ok": res["ok"],
                }
                rows.append(row)
                print(f"adx={adx} ma_bps={mabps} risk={risk} -> {row['return_pct']}% PF={row['pf']} ok={row['ok']}")

    # Sort by PF desc, then Return desc, then MaxDD asc
    rows_sorted = sorted(rows, key=lambda r: (
        float(r["pf"]) if r["pf"] is not None else -1.0,
        float(r["return_pct"]) if r["return_pct"] is not None else -1.0,
        -float(r["maxdd_pct"]) if r["maxdd_pct"] is not None else 1e9,
    ), reverse=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)
    print(f"Saved leaderboard to {out_path}")


if __name__ == "__main__":
    main()
