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
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import pytz

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# Ensure project root on path for strategy imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Strategy utilities
from strategy.indicators import compute_atr_wilder, calculate_adx, calculate_rsi, calculate_macd
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
ATR_PCT_THR = 0.40          # Legacy single-sided threshold (% units)
ADX_THR = 25                # Require ADX >= 24–26
MIN_MA_DIST_BPS = 10        # Min MA distance in basis points (bps)
# New banded ATR filter and MA normalization defaults (align with live bot)
ATR_PCT_MIN = 0.04          # 0.04%
ATR_PCT_MAX = 0.20          # 0.20%
ADAPTIVE_MABPS_ENABLE = False
ADAPTIVE_MABPS_COEFF = 0.35
ADAPTIVE_MABPS_FLOOR_BPS = 1.0
MA_NORM_MIN_OFFHOURS = 0.6
OFFHOURS_STRICT_ADX_MIN = 28

# Stop/trailing tuning
ATR_MULT = 2.4              # 2.2–2.5x ATR
BREAKEVEN_R = 1.0           # Move stop to entry at +1R
TRAIL_START_R = 1.5         # Start trailing at +1.5R

# -------------------- Scalper defaults --------------------
SCALP_ENABLE_DEFAULT = False
SCALP_TF_DEFAULT = "5m"
SCALP_ENTRY_LOGIC_DEFAULT = "RSI+MACD"  # or "RSI+Momentum"
SCALP_RISK_PCT_DEFAULT = 0.0015
SCALP_TP_PIPS_DEFAULT = 10.0
SCALP_SL_PIPS_DEFAULT = 15.0
SCALP_RSI_ENTRY_LONG_DEFAULT = 35
SCALP_RSI_EXIT_LONG_DEFAULT = 60
SCALP_RSI_ENTRY_SHORT_DEFAULT = 65
SCALP_RSI_EXIT_SHORT_DEFAULT = 40
SCALP_MOMENTUM_THRESHOLD_DEFAULT = 0.2
SCALP_SESSION_FILTER_DEFAULT = None  # "Asia+London", "London+NY"
SCALP_COOLDOWN_SEC_DEFAULT = 120

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
        # Also allow broker suffix between symbol and underscore (e.g., dataset_EURUSD.sim_...)
        for p in data_dir.glob(f"dataset_{s}*_{timeframe}.csv"):
            if p not in candidates:
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


def load_lower_tf_dataset(symbol: str, data_dir: Path, timeframe: str) -> Optional[pd.DataFrame]:
    """Load a lower-timeframe CSV (e.g., 5m or 1m) for the given symbol if present."""
    tf = timeframe.lower()
    if tf not in ("1m", "5m"):
        return None
    # Accept same mapping as load_dataset but for lower tf
    variants = {symbol}
    if "+" in symbol:
        pass
    if "-" in symbol:
        variants.add(symbol.replace("-", ""))
    else:
        if symbol.endswith("USD"):
            variants.add(symbol[:-3] + "-USD")
    candidates = []
    for s in variants:
        candidates.append(data_dir / f"dataset_{s}_{tf}.csv")
    for s in variants:
        for p in data_dir.glob(f"dataset_{s}_*_{tf}.csv"):
            candidates.append(p)
        for p in data_dir.glob(f"dataset_{s}*_{tf}.csv"):
            if p not in candidates:
                candidates.append(p)
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df = _parse_datetime_col(df)
            cols = {c.lower(): c for c in df.columns}
            need = [cols.get("open"), cols.get("high"), cols.get("low"), cols.get("close")]
            if any(c is None for c in need):
                continue
            out = pd.DataFrame({
                "open": df[need[0]].astype(float),
                "high": df[need[1]].astype(float),
                "low": df[need[2]].astype(float),
                "close": df[need[3]].astype(float),
            }, index=df.index).sort_index()
            out = out[~out.index.duplicated(keep="last")]
            # If a spread column exists in the source, carry as spread_pips with heuristic conversion
            if "spread" in cols:
                raw = pd.to_numeric(df[cols["spread"]], errors="coerce")
                pv = pip_value(symbol)
                # If spread seems in price units, convert to pips
                if raw.median(skipna=True) > (10 * pv):
                    raw = raw / pv
                out["spread_pips"] = raw
            return out
    return None


def pip_value(symbol: str) -> float:
    s = symbol.upper()
    if "JPY" in s:
        return 0.01
    if "XAU" in s or "XAG" in s or "GOLD" in s:
        return 0.01
    return 0.0001


def _session_ok_scalp(session_filter: Optional[str], ts: pd.Timestamp) -> bool:
    if not session_filter:
        return True
    h = ts.hour
    f = session_filter.replace(" ", "").lower()
    if f == "asia+london":
        return (0 <= h <= 9) or (7 <= h <= 16)
    if f == "london+ny":
        return 7 <= h <= 20
    return True


def simulate_scalper_for_symbol(
    symbol: str,
    lower_df: pd.DataFrame,
    capital: float,
    timeframe: str,
    entry_logic: str,
    risk_pct: float,
    tp_pips: float,
    sl_pips: float,
    rsi_entry_long: int = SCALP_RSI_ENTRY_LONG_DEFAULT,
    rsi_exit_long: int = SCALP_RSI_EXIT_LONG_DEFAULT,
    rsi_entry_short: int = SCALP_RSI_ENTRY_SHORT_DEFAULT,
    rsi_exit_short: int = SCALP_RSI_EXIT_SHORT_DEFAULT,
    momentum_threshold: float = SCALP_MOMENTUM_THRESHOLD_DEFAULT,
    session_filter: Optional[str] = SCALP_SESSION_FILTER_DEFAULT,
    cooldown_sec: int = SCALP_COOLDOWN_SEC_DEFAULT,
    max_spread_pips: Optional[float] = None,
) -> Tuple[pd.Series, List[Trade]]:
    df = lower_df.copy()
    # Indicators
    rsi = calculate_rsi(df["close"], 14)
    macd, macd_sig, macd_hist = calculate_macd(df["close"])  # EWM-based
    df["rsi"] = rsi
    df["macd"] = macd
    df["macd_sig"] = macd_sig

    # Build spread series if gate is requested
    spread_series: Optional[pd.Series] = None
    if max_spread_pips is not None:
        if "spread_pips" in df.columns:
            spread_series = pd.to_numeric(df["spread_pips"], errors="coerce")
        else:
            # Estimate from ATR in price and convert to pips
            try:
                spread_series = _estimate_spread_pips(symbol, df, cap=float(max_spread_pips))
            except Exception:
                spread_series = None

    eq = [capital]
    idx = [df.index[0]]
    cash = float(capital)
    open_dir: Optional[str] = None
    entry_px = 0.0
    units = 0.0
    next_allowed_ts: Optional[pd.Timestamp] = None
    trades: List[Trade] = []
    pip = pip_value(symbol)

    for i in range(1, len(df)):
        ts = df.index[i]
        prev_ts = df.index[i-1]
        # advance equity marking
        if open_dir is None:
            eq.append(cash)
        else:
            # mark-to-market unrealized
            px = float(df["close"].iloc[i])
            pnl = (px - entry_px) * units if open_dir == "long" else (entry_px - px) * units
            eq.append(cash + pnl)
        idx.append(ts)

        # enforce cooldown between entries
        if next_allowed_ts is not None and ts < next_allowed_ts:
            continue

        # skip if session filter blocks
        if not _session_ok_scalp(session_filter, ts):
            continue

        # spread-aware entry gate (entries only)
        if max_spread_pips is not None and spread_series is not None and open_dir is None:
            try:
                if ts in spread_series.index:
                    sp = float(spread_series.loc[ts])
                else:
                    prevs = spread_series.loc[:ts]
                    sp = float(prevs.iloc[-1]) if not prevs.empty else np.nan
                if np.isfinite(sp) and sp > float(max_spread_pips):
                    continue
            except Exception:
                pass

        # manage exit for open trade by TP/SL first
        if open_dir is not None:
            hi = float(df["high"].iloc[i])
            lo = float(df["low"].iloc[i])
            tp_px = entry_px + tp_pips * pip if open_dir == "long" else entry_px - tp_pips * pip
            sl_px = entry_px - sl_pips * pip if open_dir == "long" else entry_px + sl_pips * pip
            hit_tp = hi >= tp_px if open_dir == "long" else lo <= tp_px
            hit_sl = lo <= sl_px if open_dir == "long" else hi >= sl_px
            exit_price = None
            reason = None
            # Assume SL before TP if both hit in same bar (conservative)
            if hit_sl:
                exit_price = sl_px
                reason = "scalp_sl"
            elif hit_tp:
                exit_price = tp_px
                reason = "scalp_tp"
            else:
                # Optional RSI exit
                r = float(df["rsi"].iloc[i])
                if (open_dir == "long" and r >= rsi_exit_long) or (open_dir == "short" and r <= rsi_exit_short):
                    exit_price = float(df["close"].iloc[i])
                    reason = "scalp_rsi_exit"
            if exit_price is not None:
                pnl = (exit_price - entry_px) * units if open_dir == "long" else (entry_px - exit_price) * units
                cash += pnl
                trades.append(Trade(
                    symbol=symbol,
                    entry_time=prev_ts,
                    entry_price=entry_px,
                    exit_time=ts,
                    exit_price=exit_price,
                    qty=units,
                    pnl=pnl,
                    r_multiple=pnl / (risk_pct * capital + EPS),
                    reason=reason,
                ))
                open_dir = None
                entry_px = 0.0
                units = 0.0
                next_allowed_ts = ts + pd.Timedelta(seconds=cooldown_sec)
                continue

        # entry signals if flat
        r_now = float(df["rsi"].iloc[i])
        macd_up = (df["macd"].iloc[i-1] <= df["macd_sig"].iloc[i-1]) and (df["macd"].iloc[i] > df["macd_sig"].iloc[i])
        macd_dn = (df["macd"].iloc[i-1] >= df["macd_sig"].iloc[i-1]) and (df["macd"].iloc[i] < df["macd_sig"].iloc[i])
        # naive momentum proxy
        look = df["close"].iloc[max(0, i-6):i]
        mom = 0.0
        if len(look) >= 3:
            mean = float(look.mean())
            stdv = float(look.std()) + 1e-9
            mom = (float(df["close"].iloc[i]) - mean) / stdv

        long_sig = False
        short_sig = False
        if entry_logic.upper() == "RSI+MOMENTUM":
            long_sig = (r_now <= rsi_entry_long) and (mom >= momentum_threshold)
            short_sig = (r_now >= rsi_entry_short) and (mom <= -momentum_threshold)
        else:
            long_sig = (r_now <= rsi_entry_long) and macd_up
            short_sig = (r_now >= rsi_entry_short) and macd_dn

        if long_sig or short_sig:
            entry_px = float(df["close"].iloc[i])
            # risk-based units: dollars risk / price risk per unit
            stop_dist = sl_pips * pip
            risk_amt = capital * risk_pct
            units = 0.0 if stop_dist <= 0 else risk_amt / stop_dist
            if units <= 0:
                continue
            open_dir = "long" if long_sig else "short"

    # finalize equity series
    eq_series = pd.Series(eq, index=idx).sort_index()
    return eq_series, trades


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
            # Note: We only pass start here; optional sim window slicing can be applied later
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
    layer: str = "core"
    # Entry context (for diagnostics)
    entry_adx: Optional[float] = None
    entry_atr_pct: Optional[float] = None
    entry_ma_bps: Optional[float] = None
    entry_edge_score: Optional[float] = None
    entry_spread_pips: Optional[float] = None


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
    # Diagnostics
    layer: str = "core"
    entry_adx: Optional[float] = None
    entry_atr_pct: Optional[float] = None
    entry_ma_bps: Optional[float] = None
    entry_edge_score: Optional[float] = None
    entry_spread_pips: Optional[float] = None


# -------------------- Backtest engine --------------------

def is_session_open(symbol: str, ts: pd.Timestamp) -> bool:
    # Normalize broker variants like EURUSD.sim -> EURUSD
    s = symbol.upper().split('.')[0]
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
    s = symbol.upper().split('.')[0]
    # Symbol-normalized defaults (can be overridden globally via CLI)
    if default_thr != ATR_PCT_THR:
        # User provided override; honor it for all symbols
        return default_thr
    if s == "USDJPY":
        return 0.06  # ~6 bps for 15m
    if s == "EURUSD":
        return 0.04  # ~4 bps for 15m
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
    atr_pct_thr: float = ATR_PCT_THR,  # legacy; if atr_pct_min/max provided, those take precedence
    min_ma_dist_bps: float = MIN_MA_DIST_BPS,
    # New gates
    atr_pct_min: float = ATR_PCT_MIN,
    atr_pct_max: float = ATR_PCT_MAX,
    adaptive_mabps_enable: bool = ADAPTIVE_MABPS_ENABLE,
    adaptive_mabps_coeff: float = ADAPTIVE_MABPS_COEFF,
    adaptive_mabps_floor_bps: float = ADAPTIVE_MABPS_FLOOR_BPS,
    ma_norm_min_offhours: float = MA_NORM_MIN_OFFHOURS,
    offhours_strict_adx_min: float = OFFHOURS_STRICT_ADX_MIN,
    atr_mult: float = ATR_MULT,
    daily_stop_pct: float = DAILY_STOP_PCT,
    # FTMO/prop-style controls
    day_reset_tz: str = "UTC",
    max_loss_pct: Optional[float] = None,
    # Off-hours/always-active controls
    always_active: bool = False,
    edge_buy_score: float = EDGE_BUY_SCORE,
    offhours_edge_buy_score: Optional[float] = None,
    offhours_adx_thr: Optional[float] = None,
    offhours_atr_pct_thr: Optional[float] = None,
    # Session-specific tweaks
    usdjpy_asia_ma_bps: Optional[float] = None,
) -> Tuple[pd.Series, List[Trade]]:
    # Precompute indicators
    ind: Dict[str, pd.DataFrame] = {s: build_indicators(df) for s, df in data.items()}
    # Precompute simple spread estimates (pips) for diagnostics
    spread_series: Dict[str, pd.Series] = {}
    try:
        for s, df in data.items():
            spread_series[s] = _estimate_spread_pips(s, df)
    except Exception:
        spread_series = {}
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
    overall_breached: bool = False

    # Helper to compute day key in reset TZ
    try:
        reset_tz = pytz.timezone(day_reset_tz)
    except Exception:
        reset_tz = pytz.UTC
    def day_key(ts: pd.Timestamp) -> datetime.date:
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        return ts.tz_convert(reset_tz).date()

    for t in all_times:
        # Reset day counters
        if (last_day is None) or (day_key(t) != last_day):
            for s in ind:
                trades_today[s] = 0
            last_day = day_key(t)
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

        # Enforce overall max loss (from initial capital)
        if (max_loss_pct is not None) and ((equity / max(capital, EPS) - 1.0) <= -max_loss_pct):
            overall_breached = True
            # Liquidate all positions and end simulation
            for s, pos in list(positions.items()):
                df = ind[s]
                if t in df.index:
                    price = float(df.loc[t, 'close'])
                else:
                    price = float(df['close'].loc[:t].iloc[-1])
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
                    r_multiple=(price - pos.entry_price) / max(pos.entry_atr * atr_mult, EPS),
                    reason="OverallMaxLossStop"
                ))
                del positions[s]
            # After breach, stop processing further entries/exits
            break

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
                            reason="Partial+2R",
                            layer=pos.layer,
                            entry_adx=pos.entry_adx,
                            entry_atr_pct=pos.entry_atr_pct,
                            entry_ma_bps=pos.entry_ma_bps,
                            entry_edge_score=pos.entry_edge_score,
                            entry_spread_pips=pos.entry_spread_pips,
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
                        reason="TrailingStop",
                        layer=pos.layer,
                        entry_adx=pos.entry_adx,
                        entry_atr_pct=pos.entry_atr_pct,
                        entry_ma_bps=pos.entry_ma_bps,
                        entry_edge_score=pos.entry_edge_score,
                        entry_spread_pips=pos.entry_spread_pips,
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
                        reason="MaxDuration",
                        layer=pos.layer,
                        entry_adx=pos.entry_adx,
                        entry_atr_pct=pos.entry_atr_pct,
                        entry_ma_bps=pos.entry_ma_bps,
                        entry_edge_score=pos.entry_edge_score,
                        entry_spread_pips=pos.entry_spread_pips,
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
                        reason="EdgeExit",
                        layer=pos.layer,
                        entry_adx=pos.entry_adx,
                        entry_atr_pct=pos.entry_atr_pct,
                        entry_ma_bps=pos.entry_ma_bps,
                        entry_edge_score=pos.entry_edge_score,
                        entry_spread_pips=pos.entry_spread_pips,
                    ))
                    del positions[s]
                    continue

            # Entry logic: during symbol session (or off-hours if always_active), not news blackout, and if no open position
            if overall_breached:
                continue
            session_ok = is_session_open(s, t)
            allowed = session_ok or bool(always_active)
            if s not in positions and allowed and not daily_blocked and not is_news_blackout(t, s, news_df, news_window_min):
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
                # Choose thresholds based on session vs off-hours
                eff_adx_thr = adx_thr
                # ATR threshold handling: prefer band if provided, else legacy per-symbol thr
                eff_atr_pct_thr = _atr_pct_threshold(s, atr_pct_thr)
                eff_edge_buy_score = edge_buy_score
                if not session_ok and always_active:
                    if offhours_adx_thr is not None:
                        eff_adx_thr = float(offhours_adx_thr)
                    # Strict off-hours ADX minimum
                    eff_adx_thr = max(eff_adx_thr, offhours_strict_adx_min)
                    if offhours_atr_pct_thr is not None:
                        eff_atr_pct_thr = float(offhours_atr_pct_thr)
                    if offhours_edge_buy_score is not None:
                        eff_edge_buy_score = float(offhours_edge_buy_score)

                adx_ok = row["adx"] >= eff_adx_thr
                # ATR band check if provided, else fallback to legacy single threshold
                atr_val = float(row.get("atr_pct", np.nan))
                if np.isfinite(atr_pct_min) and np.isfinite(atr_pct_max):
                    atr_ok = (atr_val >= atr_pct_min) and (atr_val <= atr_pct_max)
                else:
                    atr_thr = eff_atr_pct_thr
                    atr_ok = atr_val >= atr_thr
                # Min MA distance in bps (with optional Asia-session easing for USDJPY) and dynamic floor
                ma_dist_bps_calc = abs(row["short_ma"] - row["long_ma"]) / max(price, EPS) * 10000.0
                # Dynamic floor from ATR% (convert to bps via *100)
                dyn_floor = max(adaptive_mabps_floor_bps, adaptive_mabps_coeff * (atr_val * 100.0))
                eff_min_ma_bps = max(min_ma_dist_bps, dyn_floor) if adaptive_mabps_enable else min_ma_dist_bps
                if s.upper().split('.')[0] == "USDJPY":
                    h = t.hour
                    if 0 <= h <= 6 and usdjpy_asia_ma_bps is not None:
                        try:
                            eff_min_ma_bps = float(usdjpy_asia_ma_bps)
                        except Exception:
                            eff_min_ma_bps = min_ma_dist_bps
                ma_norm = ma_dist_bps_calc / max(dyn_floor, EPS)
                if (not session_ok) and always_active:
                    # Off-hours acceptance path similar to live: normalized MA or raw floor
                    ma_ok = (ma_norm >= ma_norm_min_offhours) or (ma_dist_bps_calc >= eff_min_ma_bps)
                else:
                    ma_ok = ma_dist_bps_calc >= eff_min_ma_bps

                edge = compute_edge_features_and_score(df.iloc[: i + 1], row, prev, 0.0, 20.0)

                # ML gating (optional)
                ml_ok = True
                s_up = s.upper().split('.')[0]
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

                if edge.score >= eff_edge_buy_score and uptrend and adx_ok and atr_ok and ma_ok and ml_ok:
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
                    # Entry diagnostics
                    atr_pct_val = float(row.get("atr_pct", np.nan))
                    ma_bps_val = abs(row["short_ma"] - row["long_ma"]) / max(price, EPS) * 10000.0
                    edge_now = compute_edge_features_and_score(ind[s].iloc[: i + 1], row, prev, 0.0, 20.0)
                    sp_est = None
                    try:
                        ss = spread_series.get(s)
                        if ss is not None:
                            sp_est = float(ss.loc[t]) if t in ss.index else float(ss.loc[:t].iloc[-1])
                    except Exception:
                        sp_est = None
                    positions[s] = Position(
                        symbol=s,
                        entry_time=t,
                        entry_price=entry_price,
                        qty=qty,
                        entry_atr=atr,
                        initial_risk=risk_dollars,
                        layer="core",
                        entry_adx=float(row.get("adx", np.nan)),
                        entry_atr_pct=atr_pct_val,
                        entry_ma_bps=ma_bps_val,
                        entry_edge_score=float(edge_now.score),
                        entry_spread_pips=(sp_est if (sp_est is not None and np.isfinite(sp_est)) else None),
                    )
                    trades_today[s] += 1
                    buy_streak[s] = 0

            # Micro-pyramiding: add risk after +1R when allowed
            if pyramid_enable and (s in positions) and is_session_open(s, t) and not daily_blocked and not is_news_blackout(t, s, news_df, news_window_min):
                pos = positions[s]
                initial_r = pos.entry_atr * atr_mult
                r_mult = (price - pos.entry_price) / max(initial_r, EPS)
                s_up = s.upper().split('.')[0]
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
                                reason="PyramidAdd",
                                layer="core",
                                entry_adx=float(row.get("adx", np.nan)),
                                entry_atr_pct=float(row.get("atr_pct", np.nan)),
                                entry_ma_bps=abs(row["short_ma"] - row["long_ma"]) / max(price, EPS) * 10000.0,
                                entry_edge_score=float(edge.score),
                                entry_spread_pips=(float(spread_series.get(s).loc[t]) if (spread_series.get(s) is not None and t in spread_series.get(s).index) else None)
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
                reason="EndOfBacktest",
                layer=pos.layer,
                entry_adx=pos.entry_adx,
                entry_atr_pct=pos.entry_atr_pct,
                entry_ma_bps=pos.entry_ma_bps,
                entry_edge_score=pos.entry_edge_score,
                entry_spread_pips=pos.entry_spread_pips,
            ))
            del positions[s]

    equity_series = pd.Series(equity_curve, index=pd.Index(equity_index, name="time"))
    return equity_series, trades


def _resolve_profile_defaults(profile_path: Optional[str]) -> Tuple[Optional[dict], Optional[dict], Optional[float]]:
    """Load profile JSON if available and return (pairs_cfg, scalper_defaults_by_symbol, global_max_concurrent_risk)."""
    if not profile_path:
        return None, None, None
    p = Path(profile_path)
    if not p.exists():
        return None, None, None
    try:
        js = json.load(open(p, "r", encoding="utf-8"))
        pairs = js.get("pairs", {})
        global_mcr = None
        if isinstance(js.get("global"), dict):
            global_mcr = js["global"].get("max_concurrent_risk")
        scalper_defaults = {}
        for sym, cfg in pairs.items():
            sm = cfg.get("scalp_module") or {}
            scalper_defaults[sym.upper()] = {
                "enabled": bool(sm.get("enabled", False)),
                "timeframe": sm.get("timeframe", SCALP_TF_DEFAULT),
                "entry_logic": sm.get("entry_logic", SCALP_ENTRY_LOGIC_DEFAULT),
                "rsi_entry_long": int(sm.get("rsi_entry_long", SCALP_RSI_ENTRY_LONG_DEFAULT)),
                "rsi_exit_long": int(sm.get("rsi_exit_long", SCALP_RSI_EXIT_LONG_DEFAULT)),
                "rsi_entry_short": int(sm.get("rsi_entry_short", SCALP_RSI_ENTRY_SHORT_DEFAULT)),
                "rsi_exit_short": int(sm.get("rsi_exit_short", SCALP_RSI_EXIT_SHORT_DEFAULT)),
                "momentum_threshold": float(sm.get("momentum_threshold", SCALP_MOMENTUM_THRESHOLD_DEFAULT)),
                "tp_pips": float(sm.get("tp_pips", SCALP_TP_PIPS_DEFAULT)),
                "sl_pips": float(sm.get("sl_pips", SCALP_SL_PIPS_DEFAULT)),
                "risk_per_trade": float(sm.get("risk_per_trade", SCALP_RISK_PCT_DEFAULT)),
                "session_filter": sm.get("session_filter", SCALP_SESSION_FILTER_DEFAULT),
            }
            if "max_spread_pips" in cfg:
                scalper_defaults[sym.upper()]["max_spread_pips"] = float(cfg["max_spread_pips"])
            if "risk_per_trade" in cfg:
                scalper_defaults[sym.upper()]["core_risk_per_trade"] = float(cfg["risk_per_trade"])
        return pairs, scalper_defaults, (float(global_mcr) if global_mcr is not None else None)
    except Exception:
        return None, None, None


def _estimate_spread_pips(symbol: str, df: pd.DataFrame, k: float = 0.15, floor_fx: float = 0.05, floor_xau: float = 0.5, cap: Optional[float] = None) -> pd.Series:
    """Model spread in pips from ATR as a fallback when no spread data is available.
    spread_pips = clamp(k * ATR_pips, floor, cap)
    """
    pip = pip_value(symbol)
    tmp = df.copy()
    tmp["atr"] = compute_atr_wilder(tmp, 14)
    atr_pips = (tmp["atr"] / max(pip, EPS)).astype(float)
    floor = floor_xau if ("XAU" in symbol.upper() or "GOLD" in symbol.upper()) else floor_fx
    sp = k * atr_pips
    sp = sp.clip(lower=floor)
    if cap is not None and np.isfinite(cap):
        sp = sp.clip(upper=float(cap))
    sp.name = "spread_pips_est"
    return sp


def simulate_portfolio_combined(
    core_data: Dict[str, pd.DataFrame],
    lower_data: Dict[str, pd.DataFrame],
    capital: float,
    core_risk_default: float,
    max_total_risk: float,
    # core gates
    adx_thr: float = ADX_THR,
    atr_pct_thr: float = ATR_PCT_THR,
    min_ma_dist_bps: float = MIN_MA_DIST_BPS,
    atr_pct_min: float = ATR_PCT_MIN,
    atr_pct_max: float = ATR_PCT_MAX,
    adaptive_mabps_enable: bool = ADAPTIVE_MABPS_ENABLE,
    adaptive_mabps_coeff: float = ADAPTIVE_MABPS_COEFF,
    adaptive_mabps_floor_bps: float = ADAPTIVE_MABPS_FLOOR_BPS,
    ma_norm_min_offhours: float = MA_NORM_MIN_OFFHOURS,
    offhours_strict_adx_min: float = OFFHOURS_STRICT_ADX_MIN,
    atr_mult: float = ATR_MULT,
    # global controls
    daily_stop_pct: float = DAILY_STOP_PCT,
    day_reset_tz: str = "UTC",
    max_loss_pct: Optional[float] = None,
    always_active: bool = False,
    edge_buy_score: float = EDGE_BUY_SCORE,
    offhours_edge_buy_score: Optional[float] = None,
    offhours_adx_thr: Optional[float] = None,
    offhours_atr_pct_thr: Optional[float] = None,
    usdjpy_asia_ma_bps: Optional[float] = None,
    news_df: Optional[pd.DataFrame] = None,
    news_window_min: int = 10,
    # per-symbol overrides
    core_risk_by_symbol: Optional[Dict[str, float]] = None,
    scalper_cfg: Optional[Dict[str, dict]] = None,
) -> Tuple[pd.Series, List[Trade]]:
    """Integrated simulation of 15m core and lower-TF scalper with shared concurrent risk cap."""
    core_ind: Dict[str, pd.DataFrame] = {s: build_indicators(df) for s, df in core_data.items()}

    # Union of all timestamps (core + scalper)
    all_times = set()
    for df in core_ind.values():
        all_times |= set(df.index)
    for df in lower_data.values():
        all_times |= set(df.index)
    all_times = sorted(all_times)

    cash = float(capital)
    positions: Dict[Tuple[str, str], Position] = {}
    trades: List[Trade] = []
    equity_curve: List[float] = []
    equity_index: List[pd.Timestamp] = []

    buy_streak: Dict[str, int] = {s: 0 for s in core_ind}
    exit_streak: Dict[str, int] = {s: 0 for s in core_ind}
    trades_today: Dict[str, int] = {s: 0 for s in core_ind}
    last_day: Optional[datetime.date] = None
    day_start_equity: Optional[float] = None
    daily_blocked: bool = False
    overall_breached: bool = False

    # scalper state
    scalper_open_dir: Dict[str, Optional[str]] = {s: None for s in lower_data}
    scalper_entry_px: Dict[str, float] = {s: 0.0 for s in lower_data}
    scalper_units: Dict[str, float] = {s: 0.0 for s in lower_data}
    scalper_next_ok: Dict[str, Optional[pd.Timestamp]] = {s: None for s in lower_data}

    # Helper TZ reset
    try:
        reset_tz = pytz.timezone(day_reset_tz)
    except Exception:
        reset_tz = pytz.UTC

    def day_key(ts: pd.Timestamp) -> datetime.date:
        if ts.tzinfo is None:
            ts_local = ts.tz_localize('UTC')
        else:
            ts_local = ts
        return ts_local.tz_convert(reset_tz).date()

    # Spread series for scalper
    spread_series: Dict[str, pd.Series] = {}
    for s, df in lower_data.items():
        if "spread_pips" in df.columns:
            ss = pd.to_numeric(df["spread_pips"], errors="coerce")
            spread_series[s] = ss
        else:
            cap = None
            if scalper_cfg and s.upper() in scalper_cfg and "max_spread_pips" in scalper_cfg[s.upper()]:
                cap = float(scalper_cfg[s.upper()]["max_spread_pips"])
            spread_series[s] = _estimate_spread_pips(s, df, cap=cap)

    for t in all_times:
        # Day reset
        if (last_day is None) or (day_key(t) != last_day):
            for s in core_ind:
                trades_today[s] = 0
            last_day = day_key(t)
            day_start_equity = None
            daily_blocked = False

        # Mark-to-market equity
        mtm_value = 0.0
        for (layer, s), pos in positions.items():
            src_df = core_ind.get(s) if layer == "core" else lower_data.get(s)
            if src_df is None or src_df.empty:
                continue
            if t in src_df.index:
                price = float(src_df.loc[t, "close"])
            else:
                before = src_df.loc[:t]
                if before.empty:
                    continue
                price = float(before["close"].iloc[-1])
            mtm_value += pos.qty * price
        equity = cash + mtm_value
        equity_index.append(t)
        equity_curve.append(equity)

        if day_start_equity is None:
            day_start_equity = equity
        if (equity / max(day_start_equity, EPS) - 1.0) <= -daily_stop_pct:
            daily_blocked = True
        if (max_loss_pct is not None) and ((equity / max(capital, EPS) - 1.0) <= -max_loss_pct):
            overall_breached = True
            for (layer, s), pos in list(positions.items()):
                src_df = core_ind.get(s) if layer == "core" else lower_data.get(s)
                if src_df is None or src_df.empty:
                    continue
                if t in src_df.index:
                    price = float(src_df.loc[t, 'close'])
                else:
                    price = float(src_df['close'].loc[:t].iloc[-1])
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
                    r_multiple=0.0,
                    reason="OverallMaxLossStop"
                ))
                del positions[(layer, s)]
            break

        # Core layer
        for s, df in core_ind.items():
            if t not in df.index:
                continue
            if len(df.index) < 2 or df.index.get_loc(t) == 0:
                continue
            i = df.index.get_loc(t)
            row = df.iloc[i]
            prev = df.iloc[i-1]
            price = float(row["close"])
            key = ("core", s)
            if key in positions:
                pos = positions[key]
                initial_r = pos.entry_atr * atr_mult
                r_mult = (price - pos.entry_price) / max(initial_r, EPS)
                base_stop = pos.entry_price - initial_r
                pos.last_stop = max((pos.last_stop or -1e99), base_stop)
                if r_mult >= BREAKEVEN_R:
                    pos.last_stop = max(pos.last_stop or -1e99, pos.entry_price)
                if r_mult >= TRAIL_START_R:
                    window_since_entry = df.loc[pos.entry_time:t]
                    highest = float(window_since_entry["high"].max()) if not window_since_entry.empty else pos.entry_price
                    trail_dist = atr_mult * float(row["atr"])
                    stop = highest - trail_dist
                    pos.last_stop = max(pos.last_stop or -1e99, stop)
                if (not pos.partial_taken) and (r_mult >= 2.0) and is_session_open(s, t):
                    exit_price = price
                    qty_close = 0.5 * pos.qty
                    pnl = qty_close * (exit_price - pos.entry_price)
                    cash += pnl + qty_close * exit_price
                    pos.qty -= qty_close
                    pos.partial_taken = True
                    trades.append(Trade(
                        symbol=s, entry_time=pos.entry_time, entry_price=pos.entry_price,
                        exit_time=t, exit_price=exit_price, qty=qty_close, pnl=pnl,
                        r_multiple=r_mult, reason="Partial+2R",
                        layer=pos.layer,
                        entry_adx=pos.entry_adx,
                        entry_atr_pct=pos.entry_atr_pct,
                        entry_ma_bps=pos.entry_ma_bps,
                        entry_edge_score=pos.entry_edge_score,
                        entry_spread_pips=pos.entry_spread_pips,
                    ))
                if price <= (pos.last_stop or -1e99):
                    exit_price = price
                    pnl = pos.qty * (exit_price - pos.entry_price)
                    cash += pnl + pos.qty * exit_price
                    trades.append(Trade(
                        symbol=s, entry_time=pos.entry_time, entry_price=pos.entry_price,
                        exit_time=t, exit_price=exit_price, qty=pos.qty, pnl=pnl,
                        r_multiple=r_mult, reason="TrailingStop",
                        layer=pos.layer,
                        entry_adx=pos.entry_adx,
                        entry_atr_pct=pos.entry_atr_pct,
                        entry_ma_bps=pos.entry_ma_bps,
                        entry_edge_score=pos.entry_edge_score,
                        entry_spread_pips=pos.entry_spread_pips,
                    ))
                    del positions[key]
                    continue
                if (t - pos.entry_time) >= timedelta(minutes=MAX_DURATION_MIN) and is_session_open(s, t):
                    exit_price = price
                    pnl = pos.qty * (exit_price - pos.entry_price)
                    cash += pnl + pos.qty * exit_price
                    trades.append(Trade(
                        symbol=s, entry_time=pos.entry_time, entry_price=pos.entry_price,
                        exit_time=t, exit_price=exit_price, qty=pos.qty, pnl=pnl,
                        r_multiple=r_mult, reason="MaxDuration",
                        layer=pos.layer,
                        entry_adx=pos.entry_adx,
                        entry_atr_pct=pos.entry_atr_pct,
                        entry_ma_bps=pos.entry_ma_bps,
                        entry_edge_score=pos.entry_edge_score,
                        entry_spread_pips=pos.entry_spread_pips,
                    ))
                    del positions[key]
                    continue
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
                        symbol=s, entry_time=pos.entry_time, entry_price=pos.entry_price,
                        exit_time=t, exit_price=exit_price, qty=pos.qty, pnl=pnl,
                        r_multiple=r_mult, reason="EdgeExit",
                        layer=pos.layer,
                        entry_adx=pos.entry_adx,
                        entry_atr_pct=pos.entry_atr_pct,
                        entry_ma_bps=pos.entry_ma_bps,
                        entry_edge_score=pos.entry_edge_score,
                        entry_spread_pips=pos.entry_spread_pips,
                    ))
                    del positions[key]
                    continue

            if overall_breached:
                continue
            session_ok = is_session_open(s, t)
            allowed = session_ok or bool(always_active)
            if ("core", s) not in positions and allowed and not daily_blocked and not is_news_blackout(t, s, news_df, news_window_min):
                if trades_today[s] >= MAX_TRADES_PER_DAY:
                    continue
                total_risk_used = sum((p.initial_risk + getattr(p, "added_risk", 0.0) for p in positions.values())) / max(equity, EPS)
                if total_risk_used >= max_total_risk:
                    continue
                uptrend = price > row["trend_ma"]
                eff_adx_thr = adx_thr
                eff_atr_pct_thr = _atr_pct_threshold(s, atr_pct_thr)
                eff_edge_buy_score = edge_buy_score
                if not session_ok and always_active:
                    if offhours_adx_thr is not None:
                        eff_adx_thr = float(offhours_adx_thr)
                    eff_adx_thr = max(eff_adx_thr, offhours_strict_adx_min)
                    if offhours_atr_pct_thr is not None:
                        eff_atr_pct_thr = float(offhours_atr_pct_thr)
                    if offhours_edge_buy_score is not None:
                        eff_edge_buy_score = float(offhours_edge_buy_score)
                adx_ok = row["adx"] >= eff_adx_thr
                atr_val = float(row.get("atr_pct", np.nan))
                if np.isfinite(atr_pct_min) and np.isfinite(atr_pct_max):
                    atr_ok = (atr_val >= atr_pct_min) and (atr_val <= atr_pct_max)
                else:
                    atr_ok = atr_val >= eff_atr_pct_thr
                ma_dist_bps_calc = abs(row["short_ma"] - row["long_ma"]) / max(price, EPS) * 10000.0
                dyn_floor = max(adaptive_mabps_floor_bps, adaptive_mabps_coeff * (atr_val * 100.0))
                eff_min_ma_bps = max(min_ma_dist_bps, dyn_floor) if adaptive_mabps_enable else min_ma_dist_bps
                if s.upper().split('.')[0] == "USDJPY":
                    h = t.hour
                    if 0 <= h <= 6 and usdjpy_asia_ma_bps is not None:
                        try:
                            eff_min_ma_bps = float(usdjpy_asia_ma_bps)
                        except Exception:
                            eff_min_ma_bps = min_ma_dist_bps
                ma_norm = ma_dist_bps_calc / max(dyn_floor, EPS)
                if (not session_ok) and always_active:
                    ma_ok = (ma_norm >= ma_norm_min_offhours) or (ma_dist_bps_calc >= eff_min_ma_bps)
                else:
                    ma_ok = ma_dist_bps_calc >= eff_min_ma_bps
                edge = compute_edge_features_and_score(df.iloc[: i + 1], row, prev, 0.0, 20.0)
                if edge.score >= eff_edge_buy_score and uptrend and adx_ok and atr_ok and ma_ok:
                    buy_streak[s] += 1
                else:
                    buy_streak[s] = 0
                if buy_streak[s] >= EDGE_CONFIRM_BARS:
                    atrv = float(row["atr"]) or 1e-6
                    risk_pct_use = core_risk_by_symbol.get(s.upper(), core_risk_default) if core_risk_by_symbol else core_risk_default
                    risk_dollars = equity * risk_pct_use
                    qty = risk_dollars / max(atr_mult * atrv, EPS)
                    if qty <= 0:
                        continue
                    entry_price = price
                    cash -= qty * entry_price
                    # Entry diagnostics (core layer)
                    atr_pct_val = float(row.get("atr_pct", np.nan))
                    ma_bps_val = abs(row["short_ma"] - row["long_ma"]) / max(price, EPS) * 10000.0
                    edge_now = compute_edge_features_and_score(df.iloc[: i + 1], row, prev, 0.0, 20.0)
                    sp_est = None
                    try:
                        # Estimate using core_data which includes OHLC
                        sp_series = _estimate_spread_pips(s, core_data[s]) if (s in core_data) else None
                        if sp_series is not None:
                            sp_est = float(sp_series.loc[t]) if t in sp_series.index else float(sp_series.loc[:t].iloc[-1])
                    except Exception:
                        sp_est = None
                    positions[("core", s)] = Position(
                        symbol=s,
                        entry_time=t,
                        entry_price=entry_price,
                        qty=qty,
                        entry_atr=atrv,
                        initial_risk=risk_dollars,
                        layer="core",
                        entry_adx=float(row.get("adx", np.nan)),
                        entry_atr_pct=atr_pct_val,
                        entry_ma_bps=ma_bps_val,
                        entry_edge_score=float(edge_now.score),
                        entry_spread_pips=(sp_est if (sp_est is not None and np.isfinite(sp_est)) else None),
                    )
                    trades_today[s] += 1
                    buy_streak[s] = 0

        # Scalper layer
        for s, ldf in lower_data.items():
            if t not in ldf.index:
                continue
            if scalper_cfg and not scalper_cfg.get(s.upper(), {}).get("enabled", False):
                continue
            i = ldf.index.get_loc(t)
            if i == 0:
                continue
            if "rsi" not in ldf.columns:
                ldf.loc[:, "rsi"] = calculate_rsi(ldf["close"], 14)
            if ("macd" not in ldf.columns) or ("macd_sig" not in ldf.columns):
                macd, macd_sig, _ = calculate_macd(ldf["close"])
                ldf.loc[:, "macd"] = macd
                ldf.loc[:, "macd_sig"] = macd_sig
            row = ldf.iloc[i]
            prev = ldf.iloc[i-1]
            price = float(row["close"])
            pip = pip_value(s)
            cfg = scalper_cfg.get(s.upper(), {}) if scalper_cfg else {}
            entry_logic = str(cfg.get("entry_logic", SCALP_ENTRY_LOGIC_DEFAULT)).upper()
            rsi_entry_long = int(cfg.get("rsi_entry_long", SCALP_RSI_ENTRY_LONG_DEFAULT))
            rsi_exit_long = int(cfg.get("rsi_exit_long", SCALP_RSI_EXIT_LONG_DEFAULT))
            rsi_entry_short = int(cfg.get("rsi_entry_short", SCALP_RSI_ENTRY_SHORT_DEFAULT))
            rsi_exit_short = int(cfg.get("rsi_exit_short", SCALP_RSI_EXIT_SHORT_DEFAULT))
            momentum_threshold = float(cfg.get("momentum_threshold", SCALP_MOMENTUM_THRESHOLD_DEFAULT))
            tp_pips = float(cfg.get("tp_pips", SCALP_TP_PIPS_DEFAULT))
            sl_pips = float(cfg.get("sl_pips", SCALP_SL_PIPS_DEFAULT))
            risk_pct_sc = float(cfg.get("risk_per_trade", SCALP_RISK_PCT_DEFAULT))
            session_filter = cfg.get("session_filter", SCALP_SESSION_FILTER_DEFAULT)
            cooldown_sec = int(cfg.get("cooldown_sec", SCALP_COOLDOWN_SEC_DEFAULT))
            max_spread_pips = cfg.get("max_spread_pips", None)

            # manage open
            if scalper_open_dir.get(s):
                open_dir = scalper_open_dir[s]
                entry_px = scalper_entry_px[s]
                units = scalper_units[s]
                hi = float(row["high"])
                lo = float(row["low"])
                tp_px = entry_px + tp_pips * pip if open_dir == "long" else entry_px - tp_pips * pip
                sl_px = entry_px - sl_pips * pip if open_dir == "long" else entry_px + sl_pips * pip
                hit_sl = lo <= sl_px if open_dir == "long" else hi >= sl_px
                hit_tp = hi >= tp_px if open_dir == "long" else lo <= tp_px
                exit_price = None
                reason = None
                if hit_sl:
                    exit_price = sl_px
                    reason = "scalp_sl"
                elif hit_tp:
                    exit_price = tp_px
                    reason = "scalp_tp"
                else:
                    r_now = float(row["rsi"]) if np.isfinite(row.get("rsi", np.nan)) else float(prev.get("rsi", 50))
                    if (open_dir == "long" and r_now >= rsi_exit_long) or (open_dir == "short" and r_now <= rsi_exit_short):
                        exit_price = price
                        reason = "scalp_rsi_exit"
                if exit_price is not None:
                    pnl = (exit_price - entry_px) * units if open_dir == "long" else (entry_px - exit_price) * units
                    cash += pnl + units * exit_price
                    trades.append(Trade(
                        symbol=s, entry_time=ldf.index[i-1], entry_price=entry_px,
                        exit_time=t, exit_price=exit_price, qty=units, pnl=pnl,
                        r_multiple=pnl / max(risk_pct_sc * equity, EPS), reason=reason,
                        layer="scalper") )
                    scalper_open_dir[s] = None
                    scalper_entry_px[s] = 0.0
                    scalper_units[s] = 0.0
                    scalper_next_ok[s] = t + pd.Timedelta(seconds=cooldown_sec)
                    if ("scalper", s) in positions:
                        del positions[("scalper", s)]
                    continue

            # flat: entries
            if overall_breached or daily_blocked:
                continue
            if scalper_next_ok.get(s) is not None and t < scalper_next_ok[s]:
                continue
            if not _session_ok_scalp(session_filter, t):
                continue
            # spread gate
            sp_pips = None
            ss = spread_series.get(s)
            if ss is not None:
                if t in ss.index:
                    val = ss.loc[t]
                else:
                    prevs = ss.loc[:t]
                    val = prevs.iloc[-1] if not prevs.empty else np.nan
                sp_pips = float(val) if np.isfinite(val) else None
            if (sp_pips is not None) and (max_spread_pips is not None):
                try:
                    if sp_pips > float(max_spread_pips):
                        continue
                except Exception:
                    pass

            r_now = float(row.get("rsi", np.nan))
            if not np.isfinite(r_now):
                r_now = float(prev.get("rsi", 50))
            macd_up = (float(ldf["macd"].iloc[i-1]) <= float(ldf["macd_sig"].iloc[i-1])) and (float(ldf["macd"][i]) > float(ldf["macd_sig"][i]))
            macd_dn = (float(ldf["macd"].iloc[i-1]) >= float(ldf["macd_sig"].iloc[i-1])) and (float(ldf["macd"][i]) < float(ldf["macd_sig"][i]))
            look = ldf["close"].iloc[max(0, i-6):i]
            mom = 0.0
            if len(look) >= 3:
                mean = float(look.mean())
                stdv = float(look.std()) + 1e-9
                mom = (price - mean) / stdv
            long_sig = False
            short_sig = False
            if entry_logic == "RSI+MOMENTUM":
                long_sig = (r_now <= rsi_entry_long) and (mom >= momentum_threshold)
                short_sig = (r_now >= rsi_entry_short) and (mom <= -momentum_threshold)
            else:
                long_sig = (r_now <= rsi_entry_long) and macd_up
                short_sig = (r_now >= rsi_entry_short) and macd_dn

            if long_sig or short_sig:
                total_risk_used = sum((p.initial_risk + getattr(p, "added_risk", 0.0) for p in positions.values())) / max(equity, EPS)
                if total_risk_used >= max_total_risk:
                    continue
                stop_dist = sl_pips * pip
                risk_dollars = equity * risk_pct_sc
                units = 0.0 if stop_dist <= 0 else risk_dollars / stop_dist
                if units <= 0:
                    continue
                entry_px = price
                positions[("scalper", s)] = Position(symbol=s, entry_time=t, entry_price=entry_px, qty=units, entry_atr=sl_pips * pip, initial_risk=risk_dollars, layer="scalper")
                scalper_open_dir[s] = "long" if long_sig else "short"
                scalper_entry_px[s] = entry_px
                scalper_units[s] = units
                cash -= units * entry_px

    # Liquidate remaining
    if positions:
        t = all_times[-1]
        for (layer, s), pos in list(positions.items()):
            src_df = core_ind.get(s) if layer == "core" else lower_data.get(s)
            if src_df is None or src_df.empty:
                continue
            price = float(src_df["close"].loc[:t].iloc[-1])
            pnl = pos.qty * (price - pos.entry_price)
            cash += pnl + pos.qty * price
            trades.append(Trade(symbol=s, entry_time=pos.entry_time, entry_price=pos.entry_price, exit_time=t, exit_price=price, qty=pos.qty, pnl=pnl, r_multiple=0.0, reason="EndOfBacktest", layer=pos.layer, entry_adx=pos.entry_adx, entry_atr_pct=pos.entry_atr_pct, entry_ma_bps=pos.entry_ma_bps, entry_edge_score=pos.entry_edge_score, entry_spread_pips=pos.entry_spread_pips))
            del positions[(layer, s)]

    equity_series = pd.Series(equity_curve, index=pd.Index(equity_index, name="time"))
    return equity_series, trades


# -------------------- Reporting --------------------

def summarize(equity: pd.Series, trades: List[Trade]) -> Dict[str, float]:
    if equity.empty:
        return {
            "return_pct": 0.0,
            "max_dd_pct": 0.0,
            "sharpe": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "pf": 0.0,
            "trades_excl_adds": 0,
            "win_rate_excl_adds": 0.0,
            "pf_excl_adds": 0.0,
        }
    ret = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100
    running_max = equity.cummax()
    dd = ((equity - running_max) / running_max).min() * 100
    rets = equity.pct_change().dropna()
    # 15m ~ 4*24 bars/day; for other TFs, this scaling is approximate but consistent across runs
    sharpe = float(np.sqrt(252 * 24 * 4) * rets.mean() / (rets.std(ddof=0) + EPS)) if len(rets) else 0.0

    closed = trades
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl < 0]
    win_rate = 100.0 * len(wins) / max(len(closed), 1)
    pf = (sum(t.pnl for t in wins) / max(sum(-t.pnl for t in losses), EPS)) if losses else float("inf")

    # Exclude pyramid adds (zero-PnL bookkeeping) for a more interpretable win rate
    closed_excl_adds = [t for t in closed if t.reason != "PyramidAdd"]
    wins_excl = [t for t in closed_excl_adds if t.pnl > 0]
    losses_excl = [t for t in closed_excl_adds if t.pnl < 0]
    win_rate_excl = 100.0 * len(wins_excl) / max(len(closed_excl_adds), 1)
    pf_excl = (sum(t.pnl for t in wins_excl) / max(sum(-t.pnl for t in losses_excl), EPS)) if losses_excl else float("inf")

    return {
        "return_pct": ret,
        "max_dd_pct": float(dd),
        "sharpe": sharpe,
        "trades": len(closed),
        "win_rate": win_rate,
        "pf": pf,
        "trades_excl_adds": len(closed_excl_adds),
        "win_rate_excl_adds": win_rate_excl,
        "pf_excl_adds": pf_excl,
    }


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
    # Profile defaults and global risk cap
    ap.add_argument("--profile-json", default="configs/live/profiles/ftmo_hybrid_v1.0.json")
    ap.add_argument("--use-profile-scalper-defaults", action="store_true", default=True)
    ap.add_argument("--use-profile-core-risk", action="store_true", default=True)
    ap.add_argument("--max-total-risk", type=float, default=None, help="Override max concurrent risk across all positions (if omitted, uses profile or default)")
    # ML gating
    ap.add_argument("--ml-enable", action="store_true", help="Enable ML gating master switch")
    ap.add_argument("--ml-model", default=str(ROOT_DIR / "ml" / "models" / "mlp_model.pkl"))
    ap.add_argument("--ml-threshold", type=float, default=0.6)
    ap.add_argument("--ml-eth-enable", action="store_true", default=True)
    ap.add_argument("--ml-eth-threshold", type=float, default=0.28)
    ap.add_argument("--ml-usdjpy-enable", action="store_true", default=False)
    ap.add_argument("--ml-xauusd-enable", action="store_true", default=False)
    ap.add_argument("--start", default=None, help="Optional start date for yfinance fallback")
    # Optional simulation window (slice after loading data)
    ap.add_argument("--sim-start", default=None, help="Optional simulation start (YYYY-MM-DD or timestamp)")
    ap.add_argument("--sim-end", default=None, help="Optional simulation end (YYYY-MM-DD or timestamp)")
    # Filters and stops
    ap.add_argument("--adx-threshold", type=float, default=ADX_THR)
    ap.add_argument("--atr-pct-threshold", type=float, default=ATR_PCT_THR)
    ap.add_argument("--min-ma-dist-bps", type=float, default=MIN_MA_DIST_BPS)
    ap.add_argument("--atr-mult", type=float, default=ATR_MULT)
    ap.add_argument("--daily-stop-pct", type=float, default=DAILY_STOP_PCT)
    # New live-aligned gates
    ap.add_argument("--atr-pct-min", type=float, default=ATR_PCT_MIN)
    ap.add_argument("--atr-pct-max", type=float, default=ATR_PCT_MAX)
    ap.add_argument("--adaptive-mabps-enable", action="store_true", default=ADAPTIVE_MABPS_ENABLE)
    ap.add_argument("--adaptive-mabps-coeff", type=float, default=ADAPTIVE_MABPS_COEFF)
    ap.add_argument("--adaptive-mabps-floor-bps", type=float, default=ADAPTIVE_MABPS_FLOOR_BPS)
    ap.add_argument("--ma-norm-min-offhours", type=float, default=MA_NORM_MIN_OFFHOURS)
    ap.add_argument("--offhours-strict-adx-min", type=float, default=OFFHOURS_STRICT_ADX_MIN)
    # News calendar
    ap.add_argument("--news-calendar", default="backtests/data/tier1_news.csv")
    ap.add_argument("--news-window", type=int, default=10)
    # FTMO/prop-style controls
    ap.add_argument("--day-reset-tz", default="UTC", help="Timezone name for daily reset (e.g., Europe/Prague for FTMO)")
    ap.add_argument("--max-loss-pct", type=float, default=None, help="Overall max loss from initial capital; liquidates and stops if breached")
    ap.add_argument("--ftmo", action="store_true", help="Shortcut: set daily-stop-pct=0.05, day-reset-tz=Europe/Prague, max-loss-pct=0.10")
    # Pyramiding
    ap.add_argument("--pyramid-enable", action="store_true")
    ap.add_argument("--pyramid-step-risk", type=float, default=0.0015)
    ap.add_argument("--pyramid-max-total-risk", type=float, default=0.006)
    # Off-hours / Always-active controls (to simulate live behavior)
    ap.add_argument("--always-active", action="store_true", help="Allow entries off-hours; applies off-hours thresholds if provided")
    ap.add_argument("--offhours-edge-buy-score", type=float, default=None, help="Edge score threshold off-hours (default: same as EDGE_BUY_SCORE)")
    ap.add_argument("--offhours-adx-threshold", type=float, default=None, help="ADX threshold off-hours (default: same as --adx-threshold)")
    ap.add_argument("--offhours-atr-pct-threshold", type=float, default=None, help="ATR%% threshold off-hours (default: same as --atr-pct-threshold)")
    # Session-specific tweaks
    ap.add_argument("--usdjpy-asia-ma-bps", type=float, default=None, help="USDJPY Asia (00-06 UTC) min MA distance bps override")
    # Scalper overlay
    ap.add_argument("--scalp-enable", action="store_true", default=SCALP_ENABLE_DEFAULT)
    ap.add_argument("--scalp-timeframe", default=SCALP_TF_DEFAULT)
    ap.add_argument("--scalp-entry-logic", default=SCALP_ENTRY_LOGIC_DEFAULT)
    ap.add_argument("--scalp-risk-pct", type=float, default=SCALP_RISK_PCT_DEFAULT)
    ap.add_argument("--scalp-tp-pips", type=float, default=SCALP_TP_PIPS_DEFAULT)
    ap.add_argument("--scalp-sl-pips", type=float, default=SCALP_SL_PIPS_DEFAULT)
    ap.add_argument("--scalp-rsi-entry-long", type=int, default=SCALP_RSI_ENTRY_LONG_DEFAULT)
    ap.add_argument("--scalp-rsi-exit-long", type=int, default=SCALP_RSI_EXIT_LONG_DEFAULT)
    ap.add_argument("--scalp-rsi-entry-short", type=int, default=SCALP_RSI_ENTRY_SHORT_DEFAULT)
    ap.add_argument("--scalp-rsi-exit-short", type=int, default=SCALP_RSI_EXIT_SHORT_DEFAULT)
    ap.add_argument("--scalp-momentum-threshold", type=float, default=SCALP_MOMENTUM_THRESHOLD_DEFAULT)
    ap.add_argument("--scalp-session-filter", default=SCALP_SESSION_FILTER_DEFAULT)
    ap.add_argument("--scalp-cooldown-sec", type=int, default=SCALP_COOLDOWN_SEC_DEFAULT)
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

    # Apply optional simulation window slice
    sim_start = pd.to_datetime(args.sim_start, utc=True, errors="coerce").tz_convert(None) if args.sim_start else None
    sim_end = pd.to_datetime(args.sim_end, utc=True, errors="coerce").tz_convert(None) if args.sim_end else None
    if sim_start is not None or sim_end is not None:
        sliced: Dict[str, pd.DataFrame] = {}
        for s, df in data.items():
            df2 = df.copy()
            if sim_start is not None:
                df2 = df2[df2.index >= sim_start]
            if sim_end is not None:
                df2 = df2[df2.index <= sim_end]
            if df2.empty:
                print(f"⚠️ After slicing, no data remains for {s}; skipping.")
                continue
            sliced[s] = df2
        data = sliced

    if not data:
        print("No datasets loaded. Exiting.")
        sys.exit(1)

    news_df = _load_news_calendar(Path(args.news_calendar))

    # FTMO shortcut mapping
    day_reset_tz = args.day_reset_tz
    daily_stop_pct = float(args.daily_stop_pct)
    max_loss_pct = args.max_loss_pct
    if args.ftmo:
        day_reset_tz = "Europe/Prague"
        daily_stop_pct = 0.05
        max_loss_pct = 0.10 if max_loss_pct is None else max_loss_pct

    # Profile-based defaults and max concurrent risk
    pairs_cfg, scalper_defaults, profile_max_mcr = _resolve_profile_defaults(args.profile_json)
    runtime_max_total_risk = float(args.max_total_risk) if args.max_total_risk is not None else (float(profile_max_mcr) if profile_max_mcr is not None else float(MAX_TOTAL_RISK))
    core_risk_by_symbol: Dict[str, float] = {}
    if args.use_profile_core_risk and pairs_cfg:
        for s in list(data.keys()):
            cfg = pairs_cfg.get(s.upper()) if pairs_cfg else None
            if cfg and ("risk_per_trade" in cfg):
                try:
                    core_risk_by_symbol[s.upper()] = float(cfg["risk_per_trade"])
                except Exception:
                    pass

    # Decide integrated (core+scalper) vs core-only
    any_scalper_enabled = bool(args.scalp_enable)
    if not any_scalper_enabled and scalper_defaults:
        any_scalper_enabled = any((scalper_defaults.get(s, {}).get("enabled", False) for s in list(data.keys())))

    if any_scalper_enabled:
        lower: Dict[str, pd.DataFrame] = {}
        scalper_cfg: Dict[str, dict] = {}
        for s in list(data.keys()):
            s_up = s.upper()
            cfg = scalper_defaults.get(s_up, {}) if (args.use_profile_scalper_defaults and scalper_defaults) else {}
            if args.scalp_enable and not cfg.get("enabled", False):
                cfg = {
                    "enabled": True,
                    "timeframe": args.scalp_timeframe,
                    "entry_logic": args.scalp_entry_logic,
                    "rsi_entry_long": args.scalp_rsi_entry_long,
                    "rsi_exit_long": args.scalp_rsi_exit_long,
                    "rsi_entry_short": args.scalp_rsi_entry_short,
                    "rsi_exit_short": args.scalp_rsi_exit_short,
                    "momentum_threshold": args.scalp_momentum_threshold,
                    "tp_pips": args.scalp_tp_pips,
                    "sl_pips": args.scalp_sl_pips,
                    "risk_per_trade": args.scalp_risk_pct,
                    "session_filter": args.scalp_session_filter,
                    "cooldown_sec": args.scalp_cooldown_sec,
                }
            if cfg.get("enabled", False):
                tf_low = str(cfg.get("timeframe", SCALP_TF_DEFAULT))
                ldf = load_lower_tf_dataset(s_up, data_dir, tf_low)
                if ldf is not None and not ldf.empty:
                    if sim_start is not None:
                        ldf = ldf[ldf.index >= sim_start]
                    if sim_end is not None:
                        ldf = ldf[ldf.index <= sim_end]
                    if not ldf.empty:
                        lower[s_up] = ldf
                        scalper_cfg[s_up] = cfg

        equity, trades = simulate_portfolio_combined(
            core_data=data,
            lower_data=lower,
            capital=float(args.capital),
            core_risk_default=float(args.risk_pct),
            max_total_risk=float(runtime_max_total_risk),
            adx_thr=float(args.adx_threshold),
            atr_pct_thr=float(args.atr_pct_threshold),
            min_ma_dist_bps=float(args.min_ma_dist_bps),
            atr_pct_min=float(args.atr_pct_min),
            atr_pct_max=float(args.atr_pct_max),
            adaptive_mabps_enable=bool(args.adaptive_mabps_enable),
            adaptive_mabps_coeff=float(args.adaptive_mabps_coeff),
            adaptive_mabps_floor_bps=float(args.adaptive_mabps_floor_bps),
            ma_norm_min_offhours=float(args.ma_norm_min_offhours),
            offhours_strict_adx_min=float(args.offhours_strict_adx_min),
            atr_mult=float(args.atr_mult),
            daily_stop_pct=float(daily_stop_pct),
            day_reset_tz=day_reset_tz,
            max_loss_pct=(float(max_loss_pct) if max_loss_pct is not None else None),
            always_active=bool(args.always_active),
            edge_buy_score=float(EDGE_BUY_SCORE),
            offhours_edge_buy_score=(float(args.offhours_edge_buy_score) if args.offhours_edge_buy_score is not None else None),
            offhours_adx_thr=(float(args.offhours_adx_threshold) if args.offhours_adx_threshold is not None else None),
            offhours_atr_pct_thr=(float(args.offhours_atr_pct_threshold) if args.offhours_atr_pct_threshold is not None else None),
            usdjpy_asia_ma_bps=(float(args.usdjpy_asia_ma_bps) if args.usdjpy_asia_ma_bps is not None else None),
            news_df=news_df,
            news_window_min=int(args.news_window),
            core_risk_by_symbol=core_risk_by_symbol,
            scalper_cfg=scalper_cfg,
        )
    else:
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
            atr_pct_min=float(args.atr_pct_min),
            atr_pct_max=float(args.atr_pct_max),
            adaptive_mabps_enable=bool(args.adaptive_mabps_enable),
            adaptive_mabps_coeff=float(args.adaptive_mabps_coeff),
            adaptive_mabps_floor_bps=float(args.adaptive_mabps_floor_bps),
            ma_norm_min_offhours=float(args.ma_norm_min_offhours),
            offhours_strict_adx_min=float(args.offhours_strict_adx_min),
            atr_mult=float(args.atr_mult),
            daily_stop_pct=float(args.daily_stop_pct),
            day_reset_tz=day_reset_tz,
            max_loss_pct=(float(max_loss_pct) if max_loss_pct is not None else None),
            always_active=bool(args.always_active),
            edge_buy_score=float(EDGE_BUY_SCORE),
            offhours_edge_buy_score=(float(args.offhours_edge_buy_score) if args.offhours_edge_buy_score is not None else None),
            offhours_adx_thr=(float(args.offhours_adx_threshold) if args.offhours_adx_threshold is not None else None),
            offhours_atr_pct_thr=(float(args.offhours_atr_pct_threshold) if args.offhours_atr_pct_threshold is not None else None),
            usdjpy_asia_ma_bps=(float(args.usdjpy_asia_ma_bps) if args.usdjpy_asia_ma_bps is not None else None),
        )

    # Report
    summary = summarize(equity, trades)
    print("\n" + "="*80)
    print("FXIFY PHASE 1 (15m) - PORTFOLIO SUMMARY")
    print("="*80)
    print(f"Symbols: {', '.join(data.keys())}")
    try:
        runtime_max_total_risk  # noqa: F401
    except NameError:
        runtime_max_total_risk = MAX_TOTAL_RISK
    print(f"Capital: ${args.capital:,.2f} | Risk/Trade (core default): {args.risk_pct*100:.2f}% | Max Concurrent Risk: {float(runtime_max_total_risk)*100:.2f}%")
    print(f"Return: {summary['return_pct']:+.2f}% | MaxDD: {summary['max_dd_pct']:.2f}% | Sharpe: {summary['sharpe']:.2f} | Trades: {summary['trades']} | WinRate: {summary['win_rate']:.1f}% | PF: {summary['pf']:.2f}")
    # Also report a cleaner win rate excluding pyramid adds (zero-PnL rows)
    print(f"WinRate (excl adds): {summary['win_rate_excl_adds']:.1f}% | Trades (excl adds): {summary['trades_excl_adds']} | PF (excl adds): {summary['pf_excl_adds']:.2f}")

    # Save per-symbol reports by re-simulating individually for detail (faster than tracking during multi-sim).
    # If scalper overlay is enabled via profile or CLI, also simulate scalper on lower TF and save combined equity and charts.
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
            atr_pct_min=float(args.atr_pct_min),
            atr_pct_max=float(args.atr_pct_max),
            adaptive_mabps_enable=bool(args.adaptive_mabps_enable),
            adaptive_mabps_coeff=float(args.adaptive_mabps_coeff),
            adaptive_mabps_floor_bps=float(args.adaptive_mabps_floor_bps),
            ma_norm_min_offhours=float(args.ma_norm_min_offhours),
            offhours_strict_adx_min=float(args.offhours_strict_adx_min),
            atr_mult=float(args.atr_mult),
            daily_stop_pct=daily_stop_pct,
            day_reset_tz=day_reset_tz,
            max_loss_pct=(float(max_loss_pct) if max_loss_pct is not None else None),
            always_active=bool(args.always_active),
            edge_buy_score=float(EDGE_BUY_SCORE),
            offhours_edge_buy_score=(float(args.offhours_edge_buy_score) if args.offhours_edge_buy_score is not None else None),
            offhours_adx_thr=(float(args.offhours_adx_threshold) if args.offhours_adx_threshold is not None else None),
            offhours_atr_pct_thr=(float(args.offhours_atr_pct_threshold) if args.offhours_atr_pct_threshold is not None else None),
            usdjpy_asia_ma_bps=(float(args.usdjpy_asia_ma_bps) if args.usdjpy_asia_ma_bps is not None else None),
        )
        # Optional scalper overlay per symbol
        have_cfg = None
        if 'scalper_defaults' in locals() and scalper_defaults:
            have_cfg = scalper_defaults.get(s.upper(), {})
        want_overlay = bool(args.scalp_enable or (have_cfg and have_cfg.get("enabled", False)))
        if want_overlay:
            low = load_lower_tf_dataset(s, data_dir, args.scalp_timeframe)
            if low is not None:
                # Slice to sim window
                if args.sim_start or args.sim_end:
                    if args.sim_start:
                        low = low[low.index >= sim_start]
                    if args.sim_end:
                        low = low[low.index <= sim_end]
                if not low.empty:
                    # Pull config from profile if available otherwise use CLI
                    cfg = have_cfg if (have_cfg and have_cfg.get("enabled", False)) else {
                        "enabled": True,
                        "timeframe": args.scalp_timeframe,
                        "entry_logic": args.scalp_entry_logic,
                        "rsi_entry_long": args.scalp_rsi_entry_long,
                        "rsi_exit_long": args.scalp_rsi_exit_long,
                        "rsi_entry_short": args.scalp_rsi_entry_short,
                        "rsi_exit_short": args.scalp_rsi_exit_short,
                        "momentum_threshold": args.scalp_momentum_threshold,
                        "tp_pips": args.scalp_tp_pips,
                        "sl_pips": args.scalp_sl_pips,
                        "risk_per_trade": args.scalp_risk_pct,
                        "session_filter": args.scalp_session_filter,
                        "cooldown_sec": args.scalp_cooldown_sec,
                    }
                    se, strs = simulate_scalper_for_symbol(
                        s,
                        low,
                        capital=args.capital,
                        timeframe=str(cfg.get("timeframe", args.scalp_timeframe)),
                        entry_logic=str(cfg.get("entry_logic", args.scalp_entry_logic)),
                        risk_pct=float(cfg.get("risk_per_trade", args.scalp_risk_pct)),
                        tp_pips=float(cfg.get("tp_pips", args.scalp_tp_pips)),
                        sl_pips=float(cfg.get("sl_pips", args.scalp_sl_pips)),
                        rsi_entry_long=int(cfg.get("rsi_entry_long", args.scalp_rsi_entry_long)),
                        rsi_exit_long=int(cfg.get("rsi_exit_long", args.scalp_rsi_exit_long)),
                        rsi_entry_short=int(cfg.get("rsi_entry_short", args.scalp_rsi_entry_short)),
                        rsi_exit_short=int(cfg.get("rsi_exit_short", args.scalp_rsi_exit_short)),
                        momentum_threshold=float(cfg.get("momentum_threshold", args.scalp_momentum_threshold)),
                        session_filter=(str(cfg.get("session_filter", args.scalp_session_filter)) if (cfg.get("session_filter", args.scalp_session_filter)) else None),
                        cooldown_sec=int(cfg.get("cooldown_sec", args.scalp_cooldown_sec)),
                        max_spread_pips=(float(cfg.get("max_spread_pips")) if (cfg.get("max_spread_pips") is not None) else None),
                    )
                    # Combine: align to union of timestamps and sum MTM values (approximate overlay)
                    union_idx = sorted(set(es.index) | set(se.index))
                    es2 = es.reindex(union_idx).ffill()
                    se2 = se.reindex(union_idx).ffill()
                    combined = (es2 + (se2 - float(args.capital)))  # add scalper PnL over capital baseline
                    # Save detail
                    out_dir = output_root / s
                    out_dir.mkdir(parents=True, exist_ok=True)
                    combined.to_csv(out_dir / "equity_curve_combined.csv", header=["equity"], index_label="time")
                    if strs:
                        df_sc = pd.DataFrame([{k: getattr(t, k) for k in t.__dataclass_fields__.keys()} for t in strs])
                        df_sc.to_csv(out_dir / "scalp_trades.csv", index=False)
                    # Charts if matplotlib is available
                    if _HAS_MPL:
                        try:
                            dd = combined / combined.cummax() - 1.0
                            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                            ax[0].plot(combined.index, combined.values, label=f"{s} combined")
                            ax[0].set_title(f"{s} Combined Equity")
                            ax[0].grid(True, alpha=0.3)
                            ax[1].plot(dd.index, dd.values * 100.0, color="red", label="Drawdown %")
                            ax[1].set_title("Drawdown (%)")
                            ax[1].grid(True, alpha=0.3)
                            plt.tight_layout()
                            fig.savefig(out_dir / "equity_drawdown_combined.png", dpi=120)
                            plt.close(fig)
                        except Exception:
                            pass
        save_reports(output_root, s, es, trs)

    # Portfolio-level chart
    if _HAS_MPL and not equity.empty:
        try:
            dd_port = equity / equity.cummax() - 1.0
            fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            ax[0].plot(equity.index, equity.values, label="Portfolio Combined")
            ax[0].set_title("Portfolio Combined Equity")
            ax[0].grid(True, alpha=0.3)
            ax[1].plot(dd_port.index, dd_port.values * 100.0, color="red", label="Drawdown %")
            ax[1].set_title("Drawdown (%)")
            ax[1].grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(output_root / "portfolio_equity_drawdown.png", dpi=130)
            plt.close(fig)
        except Exception:
            pass


if __name__ == "__main__":
    main()
