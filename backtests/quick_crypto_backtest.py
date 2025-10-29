"""
Minimal self-contained backtester for BTC/ETH/SOL using yfinance 1h bars.

- Strategy: trend-following with EMAs and ADX
  Entry: price > EMA(trend), EMA(short) > EMA(long), ADX > 25
  Exit: trailing stop (3x ATR) or EMA(short) < EMA(long)

- Position sizing: risk percent of equity per trade, sized by ATR stop distance
  Defaults: initial_capital=10000, risk_pct=0.5% per trade

Outputs per symbol:
  - Console summary (Return, MaxDD, Sharpe, Trades)
  - CSVs under backtests/reports/quick_crypto/<SYMBOL>_{interval}/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("Install yfinance: pip install yfinance")


EPS = 1e-12


def wilder_ewm(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1/period, adjust=False).mean()


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).fillna(0.0)
    return wilder_ewm(tr, period)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    atr = atr_wilder(df, period).replace(0, EPS)
    plus_di = 100 * (wilder_ewm(plus_dm, period) / atr)
    minus_di = 100 * (wilder_ewm(minus_dm, period) / atr)
    denom = (plus_di + minus_di).replace(0, EPS)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return wilder_ewm(dx, period)


def download(symbol: str, start: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, interval=interval, progress=False, auto_adjust=False, actions=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} {interval}")
    # Handle MultiIndex columns or index that sometimes appear in yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Try to pick the symbol level if present
        try:
            if symbol in df.columns.get_level_values(0):
                df = df.xs(symbol, axis=1, level=0)
        except Exception:
            # Fallback to first level
            try:
                df = df.droplevel(0, axis=1)
            except Exception:
                pass
    if isinstance(df.index, pd.MultiIndex):
        # Robustly collapse to the time level regardless of level names
        try:
            df.index = df.index.get_level_values(-1)
        except Exception:
            # Fallback: reset then set to Datetime if present
            try:
                tmp = df.reset_index()
                time_col = next((c for c in tmp.columns if str(c).lower() in ("datetime","date","time")), None)
                if time_col is not None:
                    tmp.set_index(time_col, inplace=True)
                    df = tmp
            except Exception:
                pass
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    cols = ["Open","High","Low","Close","Volume"]
    df = df[[c for c in cols if c in df.columns]].dropna()
    return df


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    reason: str = ""


def backtest(df: pd.DataFrame,
             short: int = 20,
             long: int = 30,
             trend: int = 200,
             adx_thr: float = 25,
             trail_mult: float = 3.0,
             initial_capital: float = 10000.0,
             risk_pct: float = 0.005) -> tuple[pd.Series, List[Trade]]:
    bars = df.copy()
    bars["short"] = bars["Close"].ewm(span=short, adjust=False).mean()
    bars["long"] = bars["Close"].ewm(span=long, adjust=False).mean()
    bars["trend"] = bars["Close"].ewm(span=trend, adjust=False).mean()
    bars["atr"] = atr_wilder(bars)
    bars["adx"] = adx(bars)
    bars = bars.dropna()

    cash = float(initial_capital)
    qty = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    trade_high = 0.0
    last_stop = 0.0
    trades: List[Trade] = []
    equity_curve: List[float] = []

    for i in range(1, len(bars)):
        row = bars.iloc[i]
        prev = bars.iloc[i-1]
        price = float(row["Close"])
        equity = cash + qty * price
        equity_curve.append(equity)

        # Position management
        if qty > 0:
            trade_high = max(trade_high, row["High"])  # type: ignore[index]
            stop = max(last_stop, trade_high - trail_mult * max(entry_atr, EPS))
            last_stop = stop
            stop_hit = price <= stop
            death_cross = row["short"] < row["long"]
            if stop_hit or death_cross:
                exit_price = price
                cash += qty * exit_price
                trades[-1].exit_time = row.name
                trades[-1].exit_price = float(exit_price)
                trades[-1].pnl = (exit_price - entry_price) * qty
                trades[-1].reason = "Trailing stop" if stop_hit else "Death cross"
                qty = 0.0
                last_stop = 0.0
                continue

        # Entry
        if qty == 0 and (price > row["trend"]) and (row["short"] > row["long"]) and (row["adx"] > adx_thr):
            entry_price = price
            entry_atr = float(row["atr"]) or 1e-6
            risk_dollars = equity * risk_pct
            risk_per_unit = trail_mult * entry_atr
            if risk_per_unit <= 0:
                continue
            raw_qty = risk_dollars / risk_per_unit
            qty = max(0.0, raw_qty)
            if qty <= 0:
                continue
            cash -= qty * entry_price
            trade_high = entry_price
            last_stop = 0.0
            trades.append(Trade(entry_time=row.name, entry_price=float(entry_price), qty=float(qty)))

    return pd.Series(equity_curve, index=bars.index[1:]), trades


def summarize(equity: pd.Series, trades: List[Trade]):
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0 if len(equity) > 1 else 0.0
    running_max = equity.cummax()
    max_dd = ((equity - running_max) / running_max).min() if len(equity) else 0.0
    rets = equity.pct_change().fillna(0.0)
    sharpe = (np.sqrt(252) * rets.mean() / (rets.std(ddof=0) + EPS)) if len(rets) else 0.0
    wins = sum(1 for t in trades if (t.pnl or 0) > 0)
    losses = sum(1 for t in trades if (t.pnl or 0) < 0)
    return total_return, max_dd, sharpe, wins, losses


def run_for(symbol: str, start: str, interval: str, out_root: Path):
    print(f"\n>>> {symbol} | {interval} | start={start}")
    df = download(symbol, start, interval)
    equity, trades = backtest(df)
    total_return, max_dd, sharpe, wins, losses = summarize(equity, trades)
    print(f"Return: {total_return*100:.2f}% | MaxDD: {max_dd*100:.2f}% | Sharpe: {sharpe:.2f} | Trades: {len(trades)} (W/L {wins}/{losses})")

    out_dir = out_root / f"{symbol.replace('/','-').replace(':','_')}_{interval}"
    out_dir.mkdir(parents=True, exist_ok=True)
    equity.to_csv(out_dir / "equity_curve.csv", header=["equity"])
    pd.DataFrame([{
        "entry_time": t.entry_time,
        "entry_price": t.entry_price,
        "qty": t.qty,
        "exit_time": t.exit_time,
        "exit_price": t.exit_price,
        "pnl": t.pnl,
        "reason": t.reason,
    } for t in trades]).to_csv(out_dir / "trades.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTC-USD,ETH-USD,SOL-USD")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--output", default="backtests/reports/quick_crypto")
    args = ap.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    for s in [x.strip() for x in args.symbols.split(',') if x.strip()]:
        run_for(s, args.start, args.interval, out_root)


if __name__ == "__main__":
    main()
