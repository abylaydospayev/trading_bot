#!/usr/bin/env python3
"""
Portfolio parameter sweep runner using the existing CLI backtester.
Maximizes profit under FTMO-style loss constraints by exploring a small grid
over core parameters while enforcing --ftmo (daily 5%, overall 10%).

Examples (PowerShell):
  python backtests/run_portfolio_sweep_cli.py \
    --symbols "EURUSD.sim,XAUZ25.sim" \
    --start 2024-10-30 --end 2025-10-30 \
    --data-dir backtests/data \
    --out backtests/reports/portfolio_sweep_12m.csv

Notes:
 - This runner parses the standard summary line printed by fxify_phase1_backtest.py.
 - For each grid point it calls the backtester with --ftmo to respect loss rules.
 - The grid is configurable via CLI; defaults are chosen to lift exposure prudently.
"""
import argparse
import csv
import itertools
import re
import subprocess
from pathlib import Path

SUMMARY_RE = re.compile(
    r"Return:\s*([+-]?[0-9.]+)%\s*\|\s*MaxDD:\s*([+-]?[0-9.]+)%\s*\|\s*Sharpe:\s*([+-]?[0-9.]+)\s*\|\s*Trades:\s*(\d+)\s*\|\s*WinRate:\s*([0-9.]+)%\s*\|\s*PF:\s*([0-9.]+)"
)

def run_case(symbols: str, start: str, end: str, data_dir: str,
             risk_pct: float, mcr: float, adx: float, mabps: float, atr_mult: float,
             extra_args: list[str] | None = None) -> dict:
    args = [
        "python", "backtests/fxify_phase1_backtest.py",
        "--symbols", symbols,
        "--timeframe", "15m",
        "--data-dir", data_dir,
        "--capital", "100000",
        "--risk-pct", f"{risk_pct}",
        "--atr-mult", f"{atr_mult}",
        "--adx-threshold", f"{adx}",
        "--min-ma-dist-bps", f"{mabps}",
        "--sim-start", start,
        "--sim-end", end,
        "--ftmo",
        "--max-total-risk", f"{mcr}",
    ]
    if extra_args:
        args.extend(extra_args)
    p = subprocess.run(args, capture_output=True, text=True)
    out = p.stdout + "\n" + p.stderr
    m = SUMMARY_RE.search(out)
    if not m:
        return {
            "return_pct": None, "maxdd_pct": None, "sharpe": None,
            "trades": None, "winrate_pct": None, "pf": None,
            "ok": False, "raw": out,
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
    ap.add_argument("--symbols", default="EURUSD.sim,XAUZ25.sim",
                    help="Comma-separated symbols for the portfolio sweep")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--data-dir", default="backtests/data")
    ap.add_argument("--out", default="backtests/reports/portfolio_sweep.csv")
    # Grids
    ap.add_argument("--risk", nargs="*", type=float, default=[0.0040, 0.0045])
    ap.add_argument("--mcr", nargs="*", type=float, default=[0.012, 0.015])
    ap.add_argument("--adx", nargs="*", type=float, default=[20, 22])
    ap.add_argument("--mabps", nargs="*", type=float, default=[1.8, 2.0])
    ap.add_argument("--atr", nargs="*", type=float, default=[2.2])
    # Extra passthrough (e.g., scalper enables), optional
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="Additional args to pass to backtester")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(args.risk, args.mcr, args.adx, args.mabps, args.atr))
    rows = []
    for (risk, mcr, adx, mabps, atr) in combos:
        res = run_case(args.symbols, args.start, args.end, args.data_dir,
                       risk_pct=risk, mcr=mcr, adx=adx, mabps=mabps, atr_mult=atr,
                       extra_args=(args.extra or []))
        row = {
            "symbols": args.symbols,
            "start": args.start,
            "end": args.end,
            "risk": risk,
            "mcr": mcr,
            "adx": adx,
            "ma_bps": mabps,
            "atr_mult": atr,
            "return_pct": res["return_pct"],
            "maxdd_pct": res["maxdd_pct"],
            "sharpe": res["sharpe"],
            "trades": res["trades"],
            "winrate_pct": res["winrate_pct"],
            "pf": res["pf"],
            "ok": res["ok"],
        }
        rows.append(row)
        print(f"risk={risk} mcr={mcr} adx={adx} ma_bps={mabps} atr={atr} -> {row['return_pct']}% PF={row['pf']} ok={row['ok']}")

    # Sort by PF desc, then Return desc, with lower MaxDD preferred
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
