"""
FXIFY Phase 1 Backtester (15-minute focus)

Implements the user's 15m adaptation:
- Indicators: EMA(50/200), ADX~23, RSI (optional)
- Entries: Opportunistic edge score with confirmation, in active hours only
- Exits: Trailing stop 2.5x ATR, partial 50% at +2R, max duration 1 day
- Risk: 0.4% per trade default, max 6 trades/day per symbol, total concurrent risk cap 1.5%
- Assets: Works with XAUUSD, USDJPY, ETHUSD, EURUSD, BTCUSD etc.
- Data: Reads CSV datasets if present; otherwise falls back to yfinance (limited ~60 days for 15m)

Outputs per run:
- Portfolio summary and per-asset metrics
- CSVs under backtests/reports/fxify_phase1_15m/<SYMBOL>/

CSV schema expected (auto-detected):
- Required columns (case-insensitive): time|datetime,date, open, high, low, close[, volume]
- Timezone naïve or UTC; will be treated as UTC
"""
from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# Ensure project root on path for strategy imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Strategy utilities
from strategy.indicators import compute_atr_wilder, calculate_adx, calculate_rsi
from strategy.edge import compute_edge_features_and_score, EdgeResult
from ml.infer import predict_entry_prob

try:
    import yfinance as yf  # fallback only
except Exception:
    yf = None

EPS = 1e-12

# -------------------- Defaults per user spec --------------------
SHORT_EMA = 50
LONG_EMA = 200
ADX_THR = 23
ATR_MULT = 2.5
EDGE_BUY_SCORE = 60
EDGE_EXIT_SCORE = 10
EDGE_CONFIRM_BARS = 2
EDGE_EXIT_CONFIRM_BARS = 2

RISK_PCT_DEFAULT = 0.003   # 0.3%
MAX_TRADES_PER_DAY = 6
MAX_TOTAL_RISK = 0.010      # 1.0% concurrent
MAX_DURATION_MIN = 24 * 60  # 1 day
ACTIVE_HOUR_START = 6       # 06:00 UTC inclusive
ACTIVE_HOUR_END = 15        # 15:00 UTC inclusive
DAILY_STOP_PCT = 0.03       # Hard daily stop at -3.0%

# Quality filters (defaults within requested ranges)
ATR_PCT_THR = 0.40          # Default if no symbol-specific rule (% units)
ADX_THR = 25                # Require ADX >= 24–26
MIN_MA_DIST_BPS = 10        # Min MA distance in basis points (bps)

# Stop/trailing tuning
ATR_MULT = 2.4              # 2.2–2.5x ATR
BREAKEVEN_R = 1.0           # Move stop to entry at +1R
TRAIL_START_R = 1.5         # Start trailing at +1.5R

# -------------------- Data loading --------------------

def _parse_datetime_col(df: pd.DataFrame) -> pd.DataFrame:
    # Detect a datetime column robustly
    for col in df.columns:
        lc = str(col).lower()
        if lc in ("time", "timestamp", "datetime", "date"):
            dt = pd.to_datetime(df[col], utc=True, errors="coerce")
            if dt.notna().sum() > 0:
                df = df.copy()
                df.index = dt.dt.tz_convert(None)
                return df
    # If no explicit time column, assume index is time
    idx = pd.to_datetime(df.index, utc=True, errors="coerce").tz_convert(None)
    df = df.copy()
    df.index = idx
    return df

def load_dataset(symbol: str, data_dir: Path, timeframe: str) -> Optional[pd.DataFrame]:
    """Load CSV dataset if available. Return OHLC DataFrame indexed by time (UTC-naïve)."""
    # Try a few common filename variants (ETHUSD vs ETH-USD)
    variants = {symbol}
    if "-" in symbol:
        variants.add(symbol.replace("-", ""))
    else:
        if symbol.endswith("USD"):
            variants.add(symbol[:-3] + "-USD")

    candidates = []
    for s in variants:
        candidates.append(data_dir / f"dataset_{s}_{timeframe}.csv")
    # Also support date-ranged filenames: dataset_SYMBOL_*_TIMEFRAME.csv
    for s in variants:
        for p in data_dir.glob(f"dataset_{s}_*_{timeframe}.csv"):
            candidates.append(p)
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df = _parse_datetime_col(df)
            cols = {c.lower(): c for c in df.columns}
            need = [
                cols.get("open"), cols.get("high"), cols.get("low"), cols.get("close")
            ]
            if any(c is None for c in need):
                continue
            out = pd.DataFrame(
                {
                    "open": df[need[0]].astype(float),
                    "high": df[need[1]].astype(float),
                    "low": df[need[2]].astype(float),
                    "close": df[need[3]].astype(float),
                },
                index=df.index,
            ).sort_index()
            out = out[~out.index.duplicated(keep="last")]
            return out
    return None


def _yf_candidates(symbol: str) -> list[str]:
    symu = symbol.upper()
    # Preferred mappings
    if symu in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAGUSD"):
        return [symu + "=X"]
    if symu == "XAUUSD":
        # Try spot gold, then COMEX gold futures
        return ["XAUUSD=X", "GC=F"]
    if symu.endswith("USD") and "-" not in symu:
        base = symu[:-3]
        if base in {"BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "DOGE"}:
            return [base + "-USD"]
    return [symu]

def download_yf(symbol: str, start: Optional[str], timeframe: str) -> Optional[pd.DataFrame]:
    if yf is None:
        return None
    for tkr in _yf_candidates(symbol):
        try:
            df = yf.download(tkr, start=start, interval=timeframe, progress=False, auto_adjust=False, actions=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower() for c in df.columns]
            if isinstance(df.index, pd.MultiIndex):
                df.index = df.index.get_level_values(-1)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
            need = [c for c in ("open", "high", "low", "close") if c in df.columns]
            if len(need) < 4:
                continue
            out = df[need].copy().sort_index()
            out = out[~out.index.duplicated(keep="last")]
            return out
        except Exception:
            continue
    return None


# -------------------- Indicators & Edge --------------------

def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["short_ma"] = x["close"].ewm(span=SHORT_EMA, adjust=False).mean()
    x["long_ma"] = x["close"].ewm(span=LONG_EMA, adjust=False).mean()
    x["trend_ma"] = x["close"].ewm(span=LONG_EMA, adjust=False).mean()
    x["prev_short_ma"] = x["short_ma"].shift(1)
    x["atr"] = compute_atr_wilder(x, 14)
    x["adx"] = calculate_adx(x, 14)
    x["rsi"] = calculate_rsi(x["close"], 14)
    # Breakout helpers
    L = 20
    x["highL"] = x["high"].rolling(L).max()
    x["prev_high20"] = x["highL"].shift(1)
    x["atr_median20"] = x["atr"].rolling(20).median()
    # ADX slope
    x["adx_slope"] = x["adx"] - x["adx"].shift(3)
    # ATR percent
    x["atr_pct"] = (x["atr"] / x["close"]).replace([np.inf, -np.inf], np.nan) * 100.0
    return x.dropna()


# -------------------- Backtest structures --------------------

@dataclass
class Position:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    entry_atr: float
    initial_risk: float  # dollars risk at entry
    partial_taken: bool = False
    last_stop: Optional[float] = None


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    qty: float
    pnl: float
    r_multiple: float
    reason: str


# -------------------- Backtest engine --------------------

def is_session_open(symbol: str, ts: pd.Timestamp) -> bool:
    s = symbol.upper()
    h = ts.hour
    wd = ts.weekday()  # Mon=0 ... Sun=6
    if s == "USDJPY":
        # Tokyo (00:00–06:00) & overlap (12:00–16:00) UTC
        return (0 <= h <= 6) or (12 <= h <= 16)
    if s == "XAUUSD":
        # London/NY (07:00–17:00) UTC
        return 7 <= h <= 17
    if s == "ETHUSD" or s == "ETH-USD":
        # Avoid Fri 22:00 – Sun 22:00 UTC
        if (wd == 4 and h >= 22) or (wd == 5) or (wd == 6 and h < 22):
            return False
        return True
    # Fallback window
    return (ACTIVE_HOUR_START <= h <= ACTIVE_HOUR_END)

def _atr_pct_threshold(symbol: str, default_thr: float) -> float:
    s = symbol.upper()
    # Symbol-normalized defaults (can be overridden globally via CLI)
    if default_thr != ATR_PCT_THR:
        # User provided override; honor it for all symbols
        return default_thr
    if s == "USDJPY":
        return 0.06  # ~6 bps for 15m
    if s == "XAUUSD":
        return 0.25
    if s in ("ETHUSD", "ETH-USD"):
        return 0.40
    return default_thr

def _load_news_calendar(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "time" not in df.columns:
            return None
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None)
        # Optional columns: currency/symbol, importance
        if "tags" not in df.columns:
            df["tags"] = "USD"  # default to USD
        return df.dropna(subset=["time"]).copy()
    except Exception:
        return None

def is_news_blackout(ts: pd.Timestamp, symbol: str, news_df: Optional[pd.DataFrame], window_min: int = 10) -> bool:
    if news_df is None:
        return False
    s = symbol.upper()
    # Map symbol to relevant tags
    tags = set()
    if s == "USDJPY":
        tags = {"USD", "JPY"}
    elif s == "XAUUSD":
        tags = {"USD", "XAU", "GOLD"}
    else:
        return False  # no blackout for others by default
    t0 = ts - pd.Timedelta(minutes=window_min)
    t1 = ts + pd.Timedelta(minutes=window_min)
    sub = news_df[(news_df["time"] >= t0) & (news_df["time"] <= t1)]
    if sub.empty:
        return False
    # Match by tag overlap if available
    if "tags" in sub.columns:
        return any((set(str(x).upper().replace(" ", "").split("/")) & tags) for x in sub["tags"].tolist())
    return True


def simulate_portfolio(
    data: Dict[str, pd.DataFrame],
    capital: float,
    risk_pct: float,
    timeframe_seconds: int,
    ml_enable: bool = False,
    ml_model: Optional[str] = None,
    ml_threshold: float = 0.6,
    # Per-symbol ML overrides
    ml_eth_enable: bool = True,
    ml_eth_threshold: float = 0.28,
    ml_usdjpy_enable: bool = False,
    ml_xauusd_enable: bool = False,
    # News blackout (optional)
    news_df: Optional[pd.DataFrame] = None,
    news_window_min: int = 10,
    # Pyramiding (optional)
    pyramid_enable: bool = False,
    pyramid_step_risk: float = 0.0015,
    pyramid_max_total_risk: float = 0.006,
    # Filters and stops
    adx_thr: float = ADX_THR,
    atr_pct_thr: float = ATR_PCT_THR,
    min_ma_dist_bps: float = MIN_MA_DIST_BPS,
    atr_mult: float = ATR_MULT,
    daily_stop_pct: float = DAILY_STOP_PCT,
) -> Tuple[pd.Series, List[Trade]]:
    # Precompute indicators
    ind: Dict[str, pd.DataFrame] = {s: build_indicators(df) for s, df in data.items()}
    # Build a unified timeline (union of timestamps)
    all_times = sorted(set().union(*[set(df.index) for df in ind.values()]))

    cash = float(capital)
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []
    equity_curve: List[float] = []
    equity_index: List[pd.Timestamp] = []

    # Per-symbol edge streaks
    buy_streak: Dict[str, int] = {s: 0 for s in ind}
    exit_streak: Dict[str, int] = {s: 0 for s in ind}
    trades_today: Dict[str, int] = {s: 0 for s in ind}
    last_day: Optional[datetime.date] = None

    # Daily tracking for hard stop
    day_start_equity: Optional[float] = None
    daily_blocked: bool = False

    for t in all_times:
        # Reset day counters
        if (last_day is None) or (t.date() != last_day):
            for s in ind:
                trades_today[s] = 0
            last_day = t.date()
            day_start_equity = None
            daily_blocked = False

        # Mark-to-market equity
        mtm_value = 0.0
        for s, pos in positions.items():
            df = ind[s]
            if t in df.index:
                price = float(df.loc[t, "close"])  # latest bar close
            else:
                # Use last known price if no bar exactly at t
                price = float(df["close"].loc[:t].iloc[-1])
            mtm_value += pos.qty * price
        equity = cash + mtm_value
        equity_index.append(t)
        equity_curve.append(equity)

        # Initialize day start equity
        if day_start_equity is None:
            day_start_equity = equity

        # Enforce hard daily stop: block new entries for the rest of the day
        if (equity / max(day_start_equity, EPS) - 1.0) <= -daily_stop_pct:
            daily_blocked = True

        # Manage each symbol independently
        for s, df in ind.items():
            if t not in df.index:
                continue
            if len(df.index) < 2 or df.index.get_loc(t) == 0:
                continue
            i = df.index.get_loc(t)
            row = df.iloc[i]
            prev = df.iloc[i-1]
            price = float(row["close"])

            # Manage open position
            if s in positions:
                pos = positions[s]
                # Compute R multiple vs initial stop distance
                initial_r = pos.entry_atr * atr_mult
                r_mult = (price - pos.entry_price) / max(initial_r, EPS)

                # Before trail start, keep protective stop and move to breakeven at +1R
                base_stop = pos.entry_price - initial_r
                if pos.last_stop is None:
                    pos.last_stop = base_stop
                else:
                    pos.last_stop = max(pos.last_stop, base_stop)

                if r_mult >= BREAKEVEN_R:
                    pos.last_stop = max(pos.last_stop or -1e99, pos.entry_price)

                # Start trailing from +1.5R using highest high since entry
                if r_mult >= TRAIL_START_R:
                    window_since_entry = df.loc[pos.entry_time:t]
                    highest = float(window_since_entry["high"].max()) if not window_since_entry.empty else pos.entry_price
                    trail_dist = atr_mult * float(row["atr"])  # using current ATR
                    stop = highest - trail_dist
                    pos.last_stop = max(pos.last_stop or -1e99, stop)

                # Partial profit at +2R (active hours only)
                if (not pos.partial_taken) and (r_mult >= 2.0) and is_session_open(s, t):
                    # Use session gate for partials
                    if not is_session_open(s, t):
                        pass
                    else:
                        # Close 50%
                        exit_price = price
                        qty_close = 0.5 * pos.qty
                        pnl = qty_close * (exit_price - pos.entry_price)
                        cash += pnl + qty_close * exit_price
                        pos.qty -= qty_close
                        pos.partial_taken = True
                        trades.append(Trade(
                            symbol=s,
                            entry_time=pos.entry_time,
                            entry_price=pos.entry_price,
                            exit_time=t,
                            exit_price=exit_price,
                            qty=qty_close,
                            pnl=pnl,
                            r_multiple=r_mult,
                            reason="Partial+2R"
                        ))

                # Stop hit any time
                if price <= (pos.last_stop or -1e99):
                    exit_price = price
                    pnl = pos.qty * (exit_price - pos.entry_price)
                    cash += pnl + pos.qty * exit_price
                    trades.append(Trade(
                        symbol=s,
                        entry_time=pos.entry_time,
                        entry_price=pos.entry_price,
                        exit_time=t,
                        exit_price=exit_price,
                        qty=pos.qty,
                        pnl=pnl,
                        r_multiple=r_mult,
                        reason="TrailingStop"
                    ))
                    del positions[s]
                    continue

                # Max duration: close at or beyond +1 day during active hours
                if (t - pos.entry_time) >= timedelta(minutes=MAX_DURATION_MIN) and is_session_open(s, t):
                    exit_price = price
                    pnl = pos.qty * (exit_price - pos.entry_price)
                    cash += pnl + pos.qty * exit_price
                    trades.append(Trade(
                        symbol=s,
                        entry_time=pos.entry_time,
                        entry_price=pos.entry_price,
                        exit_time=t,
                        exit_price=exit_price,
                        qty=pos.qty,
                        pnl=pnl,
                        r_multiple=r_mult,
                        reason="MaxDuration"
                    ))
                    del positions[s]
                    continue

                # Opportunistic exit on edge deterioration (active hours only)
                edge = compute_edge_features_and_score(df.iloc[: i + 1], row, prev, 0.0, 20.0)
                if edge.score <= EDGE_EXIT_SCORE:
                    exit_streak[s] += 1
                else:
                    exit_streak[s] = 0
                if is_session_open(s, t) and exit_streak[s] >= EDGE_EXIT_CONFIRM_BARS:
                    exit_price = price
                    pnl = pos.qty * (exit_price - pos.entry_price)
                    cash += pnl + pos.qty * exit_price
                    trades.append(Trade(
                        symbol=s,
                        entry_time=pos.entry_time,
                        entry_price=pos.entry_price,
                        exit_time=t,
                        exit_price=exit_price,
                        qty=pos.qty,
                        pnl=pnl,
                        r_multiple=r_mult,
                        reason="EdgeExit"
                    ))
                    del positions[s]
                    continue

            # Entry logic (only during symbol session, not news blackout, and if no open position)
            if s not in positions and is_session_open(s, t) and not daily_blocked and not is_news_blackout(t, s, news_df, news_window_min):
                # Daily trade cap per symbol
                if trades_today[s] >= MAX_TRADES_PER_DAY:
                    continue

                # Concurrency risk cap
                # Current used risk = sum of initial risks / equity
                total_risk_used = sum(((p.initial_risk + getattr(p, "added_risk", 0.0)) for p in positions.values())) / max(equity, EPS)
                if total_risk_used >= MAX_TOTAL_RISK:
                    continue

                # Filters
                uptrend = price > row["trend_ma"]
                adx_ok = row["adx"] >= adx_thr
                atr_thr = _atr_pct_threshold(s, atr_pct_thr)
                atr_ok = row.get("atr_pct", np.nan) >= atr_thr
                # Min MA distance in bps
                ma_dist_bps_calc = abs(row["short_ma"] - row["long_ma"]) / max(price, EPS) * 10000.0
                ma_ok = ma_dist_bps_calc >= min_ma_dist_bps

                edge = compute_edge_features_and_score(df.iloc[: i + 1], row, prev, 0.0, 20.0)

                # ML gating (optional)
                ml_ok = True
                s_up = s.upper()
                do_ml = False
                thr = ml_threshold
                if s_up in ("ETHUSD", "ETH-USD") and ml_eth_enable:
                    do_ml = True
                    thr = ml_eth_threshold
                elif s_up == "USDJPY" and ml_usdjpy_enable:
                    do_ml = True
                elif s_up == "XAUUSD" and ml_xauusd_enable:
                    do_ml = True
                if ml_enable and do_ml and ml_model:
                    try:
                        prob = predict_entry_prob(row, ml_model)
                        ml_ok = prob >= thr
                    except Exception:
                        ml_ok = False

                if edge.score >= EDGE_BUY_SCORE and uptrend and adx_ok and atr_ok and ma_ok and ml_ok:
                    buy_streak[s] += 1
                else:
                    buy_streak[s] = 0

                if buy_streak[s] >= EDGE_CONFIRM_BARS:
                    # Sizing by ATR distance (no lot conversion; treats PnL in USD terms)
                    atr = float(row["atr"]) or 1e-6
                    risk_dollars = equity * risk_pct
                    risk_per_unit = atr_mult * atr
                    qty = risk_dollars / max(risk_per_unit, EPS)
                    if qty <= 0:
                        continue

                    entry_price = price
                    cash -= qty * entry_price
                    positions[s] = Position(
                        symbol=s,
                        entry_time=t,
                        entry_price=entry_price,
                        qty=qty,
                        entry_atr=atr,
                        initial_risk=risk_dollars,
                    )
                    trades_today[s] += 1
                    buy_streak[s] = 0

            # Micro-pyramiding: add risk after +1R when allowed
            if pyramid_enable and (s in positions) and is_session_open(s, t) and not daily_blocked and not is_news_blackout(t, s, news_df, news_window_min):
                pos = positions[s]
                initial_r = pos.entry_atr * atr_mult
                r_mult = (price - pos.entry_price) / max(initial_r, EPS)
                s_up = s.upper()
                # Per-symbol total cap
                current_sym_risk = pos.initial_risk + getattr(pos, "added_risk", 0.0)
                sym_cap = pyramid_max_total_risk * equity
                if r_mult >= 1.0 and current_sym_risk < sym_cap:
                    # Respect concurrent risk cap
                    total_risk_used = sum(((p.initial_risk + getattr(p, "added_risk", 0.0)) for p in positions.values())) / max(equity, EPS)
                    if total_risk_used >= MAX_TOTAL_RISK:
                        pass
                    else:
                        step_risk = min(pyramid_step_risk * equity, sym_cap - current_sym_risk)
                        risk_per_unit = atr_mult * float(row["atr"])
                        qty_add = step_risk / max(risk_per_unit, EPS)
                        if qty_add > 0:
                            # Execute add
                            cash -= qty_add * price
                            pos.qty += qty_add
                            pos.added_risk = getattr(pos, "added_risk", 0.0) + step_risk
                            trades.append(Trade(
                                symbol=s,
                                entry_time=t,
                                entry_price=price,
                                exit_time=t,  # placeholder; this row denotes add
                                exit_price=price,
                                qty=qty_add,
                                pnl=0.0,
                                r_multiple=r_mult,
                                reason="PyramidAdd"
                            ))

    # Close remaining positions at last known price
    if positions:
        t = all_times[-1]
        for s, pos in list(positions.items()):
            df = ind[s]
            price = float(df["close"].loc[:t].iloc[-1])
            pnl = pos.qty * (price - pos.entry_price)
            cash += pnl + pos.qty * price
            trades.append(Trade(
                symbol=s,
                entry_time=pos.entry_time,
                entry_price=pos.entry_price,
                exit_time=t,
                exit_price=price,
                qty=pos.qty,
                pnl=pnl,
                r_multiple=(price - pos.entry_price) / max(pos.entry_atr * ATR_MULT, EPS),
                reason="EndOfBacktest"
            ))
            del positions[s]

    equity_series = pd.Series(equity_curve, index=pd.Index(equity_index, name="time"))
    return equity_series, trades


# -------------------- Reporting --------------------

def summarize(equity: pd.Series, trades: List[Trade]) -> Dict[str, float]:
    if equity.empty:
        return {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0, "trades": 0, "win_rate": 0.0, "pf": 0.0}
    ret = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100
    running_max = equity.cummax()
    dd = ((equity - running_max) / running_max).min() * 100
    rets = equity.pct_change().dropna()
    sharpe = float(np.sqrt(252 * 24 * 4) * rets.mean() / (rets.std(ddof=0) + EPS)) if len(rets) else 0.0  # 15m ~ 4*24 bars/day
    closed = trades
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl < 0]
    win_rate = 100.0 * len(wins) / max(len(closed), 1)
    pf = (sum(t.pnl for t in wins) / max(sum(-t.pnl for t in losses), EPS)) if losses else float("inf")
    return {"return_pct": ret, "max_dd_pct": float(dd), "sharpe": sharpe, "trades": len(closed), "win_rate": win_rate, "pf": pf}


def save_reports(output_root: Path, symbol: str, equity: pd.Series, trades: List[Trade]) -> None:
    out_dir = output_root / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    equity.to_csv(out_dir / "equity_curve.csv", header=["equity"])
    if trades:
        df = pd.DataFrame([{k: getattr(t, k) for k in t.__dataclass_fields__.keys()} for t in trades])
        df.to_csv(out_dir / "trades.csv", index=False)


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="FXIFY Phase 1 Backtester (15m)")
    ap.add_argument("--symbols", default="XAUUSD,USDJPY,ETHUSD")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--capital", type=float, default=15000.0)
    ap.add_argument("--risk-pct", type=float, default=RISK_PCT_DEFAULT)
    ap.add_argument("--data-dir", default="backtests/data")
    ap.add_argument("--output", default="backtests/reports/fxify_phase1_15m")
    # ML gating
    ap.add_argument("--ml-enable", action="store_true", help="Enable ML gating master switch")
    ap.add_argument("--ml-model", default=str(ROOT_DIR / "ml" / "models" / "mlp_model.pkl"))
    ap.add_argument("--ml-threshold", type=float, default=0.6)
    ap.add_argument("--ml-eth-enable", action="store_true", default=True)
    ap.add_argument("--ml-eth-threshold", type=float, default=0.28)
    ap.add_argument("--ml-usdjpy-enable", action="store_true", default=False)
    ap.add_argument("--ml-xauusd-enable", action="store_true", default=False)
    ap.add_argument("--start", default=None, help="Optional start date for yfinance fallback")
    # Filters and stops
    ap.add_argument("--adx-threshold", type=float, default=ADX_THR)
    ap.add_argument("--atr-pct-threshold", type=float, default=ATR_PCT_THR)
    ap.add_argument("--min-ma-dist-bps", type=float, default=MIN_MA_DIST_BPS)
    ap.add_argument("--atr-mult", type=float, default=ATR_MULT)
    ap.add_argument("--daily-stop-pct", type=float, default=DAILY_STOP_PCT)
    # News calendar
    ap.add_argument("--news-calendar", default="backtests/data/tier1_news.csv")
    ap.add_argument("--news-window", type=int, default=10)
    # Pyramiding
    ap.add_argument("--pyramid-enable", action="store_true")
    ap.add_argument("--pyramid-step-risk", type=float, default=0.0015)
    ap.add_argument("--pyramid-max-total-risk", type=float, default=0.006)
    args = ap.parse_args()

    tf = args.timeframe.lower()
    if tf.endswith("m"):
        minutes = int(tf[:-1])
        seconds_per_bar = minutes * 60
    elif tf.endswith("h"):
        seconds_per_bar = int(tf[:-1]) * 3600
    else:
        seconds_per_bar = 15 * 60

    data_dir = Path(args.data_dir)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load datasets or fallback to yfinance
    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    data: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        ds = load_dataset(s, data_dir, tf)
        if ds is None:
            ds = download_yf(s, args.start, tf)
        if ds is None:
            print(f"❌ No data for {s} ({tf}). Provide CSV in {data_dir}.")
            continue
        data[s] = ds

    if not data:
        print("No datasets loaded. Exiting.")
        sys.exit(1)

    news_df = _load_news_calendar(Path(args.news_calendar))

    equity, trades = simulate_portfolio(
        data,
        capital=args.capital,
        risk_pct=args.risk_pct,
        timeframe_seconds=seconds_per_bar,
        ml_enable=args.ml_enable,
        ml_model=args.ml_model,
        ml_threshold=args.ml_threshold,
        ml_eth_enable=bool(args.ml_eth_enable),
        ml_eth_threshold=float(args.ml_eth_threshold),
        ml_usdjpy_enable=bool(args.ml_usdjpy_enable),
        ml_xauusd_enable=bool(args.ml_xauusd_enable),
        news_df=news_df,
        news_window_min=int(args.news_window),
        pyramid_enable=bool(args.pyramid_enable),
        pyramid_step_risk=float(args.pyramid_step_risk),
        pyramid_max_total_risk=float(args.pyramid_max_total_risk),
        adx_thr=float(args.adx_threshold),
        atr_pct_thr=float(args.atr_pct_threshold),
        min_ma_dist_bps=float(args.min_ma_dist_bps),
        atr_mult=float(args.atr_mult),
        daily_stop_pct=float(args.daily_stop_pct),
    )

    # Report
    summary = summarize(equity, trades)
    print("\n" + "="*80)
    print("FXIFY PHASE 1 (15m) - PORTFOLIO SUMMARY")
    print("="*80)
    print(f"Symbols: {', '.join(data.keys())}")
    print(f"Capital: ${args.capital:,.2f} | Risk/Trade: {args.risk_pct*100:.2f}% | Max Concurrent Risk: {MAX_TOTAL_RISK*100:.2f}%")
    print(f"Return: {summary['return_pct']:+.2f}% | MaxDD: {summary['max_dd_pct']:.2f}% | Sharpe: {summary['sharpe']:.2f} | Trades: {summary['trades']} | WinRate: {summary['win_rate']:.1f}% | PF: {summary['pf']:.2f}")

    # Save per-symbol reports by re-simulating individually for detail (faster than tracking during multi-sim)
    for s, df in data.items():
        es, trs = simulate_portfolio(
            {s: df},
            capital=args.capital,
            risk_pct=args.risk_pct,
            timeframe_seconds=seconds_per_bar,
            ml_enable=args.ml_enable,
            ml_model=args.ml_model,
            ml_threshold=args.ml_threshold,
            ml_eth_enable=bool(args.ml_eth_enable),
            ml_eth_threshold=float(args.ml_eth_threshold),
            ml_usdjpy_enable=bool(args.ml_usdjpy_enable),
            ml_xauusd_enable=bool(args.ml_xauusd_enable),
            news_df=news_df,
            news_window_min=int(args.news_window),
            pyramid_enable=bool(args.pyramid_enable),
            pyramid_step_risk=float(args.pyramid_step_risk),
            pyramid_max_total_risk=float(args.pyramid_max_total_risk),
            adx_thr=float(args.adx_threshold),
            atr_pct_thr=float(args.atr_pct_threshold),
            min_ma_dist_bps=float(args.min_ma_dist_bps),
            atr_mult=float(args.atr_mult),
            daily_stop_pct=float(args.daily_stop_pct),
        )
        save_reports(output_root, s, es, trs)


if __name__ == "__main__":
    main()
