import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import math


def summarize_trades_csv(path: Path, capital: float) -> Dict[str, float]:
    # trades.csv columns come from backtester Trade dataclass; minimally we need pnl and reason
    if not path.exists():
        return {}
    rows: List[Dict[str, str]] = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "pf": float("nan"),
            "trades_excl_adds": 0,
            "win_rate_excl_adds": 0.0,
            "pf_excl_adds": float("nan"),
            "pnl_sum": 0.0,
            "return_pct": 0.0,
        }

    def to_float(v: str, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    trades = len(rows)
    pnls = [to_float(r.get("pnl", "0"), 0.0) for r in rows]
    reasons = [r.get("reason", "") for r in rows]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = sum(-p for p in pnls if p < 0)
    pf = (gross_win / gross_loss) if gross_loss > 0 else math.inf

    # Exclude pyramid adds
    mask_excl = [reason != "PyramidAdd" for reason in reasons]
    pnls_excl = [p for p, m in zip(pnls, mask_excl) if m]
    trades_excl = len(pnls_excl)
    wins_excl = sum(1 for p in pnls_excl if p > 0)
    losses_excl = sum(1 for p in pnls_excl if p < 0)
    win_rate_excl = (wins_excl / trades_excl * 100.0) if trades_excl > 0 else 0.0
    gross_win_excl = sum(p for p in pnls_excl if p > 0)
    gross_loss_excl = sum(-p for p in pnls_excl if p < 0)
    pf_excl = (gross_win_excl / gross_loss_excl) if gross_loss_excl > 0 else math.inf

    pnl_sum = sum(pnls)
    return_pct = (pnl_sum / capital * 100.0) if capital > 0 else float("nan")

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "pf": pf,
        "trades_excl_adds": trades_excl,
        "win_rate_excl_adds": win_rate_excl,
        "pf_excl_adds": pf_excl,
        "pnl_sum": pnl_sum,
        "return_pct": return_pct,
    }


def find_symbols(output_dir: Path) -> List[str]:
    if not output_dir.exists():
        return []
    syms = []
    for child in output_dir.iterdir():
        if child.is_dir():
            trades_path = child / "trades.csv"
            if trades_path.exists():
                syms.append(child.name)
    return sorted(syms)


def main():
    ap = argparse.ArgumentParser(description="Summarize per-symbol trades.csv under an output directory")
    ap.add_argument("--dir", required=True, help="Output directory to scan (e.g., backtests/reports/fxify_ma1p0_recent_off45)")
    ap.add_argument("--capital", type=float, default=100000.0, help="Assumed capital used in per-symbol sims")
    args = ap.parse_args()

    base = Path(args.dir)
    syms = find_symbols(base)
    if not syms:
        print(f"No symbols with trades.csv found under {base}")
        return

    print("\nPer-symbol summary:")
    print("Symbol,Trades,WinRate(%),PF,TradesExclAdds,WinRateExclAdds(%),PFExclAdds,PNL($),Return(%)")
    for s in syms:
        stats = summarize_trades_csv(base / s / "trades.csv", args.capital)
        if not stats:
            continue
        print(
            f"{s},{stats['trades']},{stats['win_rate']:.1f},{stats['pf']:.2f},"
            f"{stats['trades_excl_adds']},{stats['win_rate_excl_adds']:.1f},{stats['pf_excl_adds']:.2f},"
            f"{stats['pnl_sum']:.2f},{stats['return_pct']:.2f}"
        )


if __name__ == "__main__":
    main()
