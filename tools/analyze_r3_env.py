"""
Analyze R3 scalper environment logs and produce concise summaries.

Input: logs/r3_env.csv (written by live/trading_bot_fxify.py)
Outputs:
- backtests/reports/r3_env_summary_by_hour.csv
- backtests/reports/r3_env_summary_overall.csv

Metrics:
- Counts and acceptance rate (env_ok) overall and per UTC hour
- Block reasons approximations: spread_fail (not sp_ok), sigma_fail (not sig_ok), session_fail (!in_london_ny)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import argparse
import pandas as pd


def load_logs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"No log file found at {path}")
    df = pd.read_csv(path)
    # Normalize booleans
    for col in ["in_london_ny", "sp_ok", "sig_ok", "env_ok"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    # Extract hour from ts
    if "ts" in df.columns:
        try:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df["hour"] = df["ts"].dt.hour
        except Exception:
            df["hour"] = -1
    else:
        df["hour"] = -1
    return df


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Approximate block reasons (non-exclusive)
    df["spread_fail"] = ~df.get("sp_ok", True)
    df["sigma_fail"] = ~df.get("sig_ok", True)
    df["session_fail"] = ~df.get("in_london_ny", True)

    def agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        ok = int(g.get("env_ok", pd.Series([False]*n)).sum())
        return pd.Series({
            "samples": n,
            "env_ok": ok,
            "accept_rate": ok / n if n > 0 else 0.0,
            "spread_fail": int(g["spread_fail"].sum()),
            "sigma_fail": int(g["sigma_fail"].sum()),
            "session_fail": int(g["session_fail"].sum()),
            "avg_spread": float(pd.to_numeric(g.get("spread"), errors="coerce").mean()),
            "avg_p20": float(pd.to_numeric(g.get("p20"), errors="coerce").mean()),
            "avg_sigma_ann": float(pd.to_numeric(g.get("sigma_ann"), errors="coerce").mean()),
            "avg_adx": float(pd.to_numeric(g.get("adx"), errors="coerce").mean()),
            "avg_atr_pct": float(pd.to_numeric(g.get("atr_pct"), errors="coerce").mean()),
        })

    overall = agg(df)
    overall_df = pd.DataFrame([overall])

    by_hour = df.groupby("hour").apply(agg).reset_index()
    by_hour = by_hour.sort_values("hour")

    return overall_df, by_hour


def main():
    ap = argparse.ArgumentParser(description="Analyze R3 env logs")
    ap.add_argument("--log", default="logs/r3_env.csv")
    ap.add_argument("--out", default="backtests/reports")
    args = ap.parse_args()

    path = Path(args.log)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_logs(path)
    overall, by_hour = summarize(df)

    overall.to_csv(out_dir / "r3_env_summary_overall.csv", index=False)
    by_hour.to_csv(out_dir / "r3_env_summary_by_hour.csv", index=False)
    print(f"Saved: {out_dir / 'r3_env_summary_overall.csv'}")
    print(f"Saved: {out_dir / 'r3_env_summary_by_hour.csv'}")


if __name__ == "__main__":
    main()
