#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.backtest_btc import prepare_bars, BASE_CONFIG  # reuse fetchers + indicators


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    # Derived helpers
    df["ma_dist_bps"] = ((df["short_ma"] - df["long_ma"]) / df["close"]).replace([np.inf, -np.inf], np.nan) * 10000.0
    df["above_trend"] = (df["close"] > df["trend_ma"]).astype(float)
    if "prev_high20" not in df.columns:
        lookback = 20
        df["highL"] = df["high"].rolling(lookback).max()
        df["prev_high20"] = df["highL"].shift(1)
    df["breakout_gap_bps"] = ((df["close"] - df["prev_high20"]) / df["close"]).replace([np.inf, -np.inf], np.nan) * 10000.0
    # Time-of-day cyclical features (UTC)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    hours = df.index.hour.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    # Active-session flag (aim to skip dead hours): 12:00–22:00 UTC
    df["is_active_hours"] = ((hours >= 12) & (hours <= 22)).astype(float)
    # ATR regime features: rolling percentile thresholds and flags
    df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan) * 100.0
    win = 200
    def _pctile(x: pd.Series, p: float) -> float:
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        return float(np.percentile(x, p))
    df["atr_pct_thr40"] = df["atr_pct"].rolling(win, min_periods=max(10, win//5)).apply(lambda x: _pctile(pd.Series(x), 40.0), raw=False).shift(1)
    df["atr_pct_thr60"] = df["atr_pct"].rolling(win, min_periods=max(10, win//5)).apply(lambda x: _pctile(pd.Series(x), 60.0), raw=False).shift(1)
    df["atr_above_40"] = (df["atr_pct"] >= df["atr_pct_thr40"]).astype(float)
    df["atr_above_60"] = (df["atr_pct"] >= df["atr_pct_thr60"]).astype(float)
    # Feature columns
    feat_cols = [
        "close","short_ma","long_ma","trend_ma","atr","adx","rsi",
        "adx_slope","atr_pct","atr_median20","prev_short_ma",
        "ma_dist_bps","above_trend","breakout_gap_bps",
        "hour_sin","hour_cos","is_active_hours",
        "atr_pct_thr40","atr_pct_thr60","atr_above_40","atr_above_60",
    ]
    df = df.dropna()
    return df[feat_cols]


def add_labels(bars: pd.DataFrame, horizon: int, thr_pct: float) -> pd.DataFrame:
    # Forward return label: 1 if close[t+h]/close[t]-1 >= thr_pct
    df = bars.copy()
    fwd = df["close"].shift(-horizon)
    ret = (fwd / df["close"] - 1.0)
    df["y"] = (ret >= thr_pct).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser(description="Prepare ML dataset from bars and indicators")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--data_source", default="alpaca", choices=["auto","yahoo","yfinance","alpaca"])
    ap.add_argument("--horizon", type=int, default=8, help="Label horizon in bars (e.g., 8=2 hours on 15m)")
    ap.add_argument("--thr_pct", type=float, default=0.003, help="Positive label if forward return >= this")
    ap.add_argument("--out", default=None, help="Output CSV path (defaults to ml/data/...)" )
    args = ap.parse_args()

    cfg = BASE_CONFIG.copy()
    cfg.update({
        "data_source": args.data_source,
        "short_ma": 10,
        "long_ma": 25,
        "trend_ma": 200,
    })

    bars = prepare_bars(args.symbol, args.start, args.end, args.interval, cfg)
    feat = build_features(bars)
    lab = add_labels(bars, args.horizon, args.thr_pct)[["y"]]
    data = feat.join(lab, how="inner").dropna()

    out_dir = ROOT / "ml" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (out_dir / f"dataset_{args.symbol.replace('-','_')}_{args.start}_{args.end}_{args.interval}.csv")
    data.to_csv(out_path, index=True)
    print(f"Saved dataset to {out_path} with shape {data.shape}")


if __name__ == "__main__":
    main()
