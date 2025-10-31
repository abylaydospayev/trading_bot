"""
FXIFY Forex Backtester (Router Parity)

Adds a simple regime controller to better mirror live bot behavior:
- R0: Flat
- R1: Trend (edge-driven entries)
- R2: Mean-Reversion (sigma z-score entries)

Includes session and volatility-floor gates, plus the dynamic slippage model.
Note: Live spread percentile gates (p20/p30) cannot be replicated from yfinance
because intraday spread data is unavailable; an optional pair-specific spread hard cap
can be enforced for stricter simulations.
"""
from __future__ import annotations

import os
import sys
import argparse
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# Project root for imports
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install yfinance: pip install yfinance")

from strategy.indicators import compute_atr_wilder, calculate_adx, calculate_rsi
from strategy.edge import compute_edge_features_and_score, EdgeResult

# Base backtester utilities for costs and dynamic slippage
from backtests.backtest_forex_fxify import (
    get_pip_value,
    compute_slippage_pips,
)


# ===================== CONFIG =====================

# FXIFY limits
FXIFY_MAX_DAILY_LOSS_PCT = 0.05  # 5%
FXIFY_MAX_TOTAL_LOSS_PCT = 0.10  # 10%

# Risk parameters
RISK_PCT_BASE = 0.01
RISK_PCT_STRONG = 0.02
ADX_RISK_THR = 25
ATR_PCT_RISK_THR = 0.04

# Strategy thresholds (align with live defaults approximately)
EDGE_BUY_SCORE = 60
EDGE_CONFIRM_BARS = 2
EDGE_EXIT_SCORE = 10
EDGE_EXIT_CONFIRM_BARS = 2

P = {
    "short": 10,
    "long": 25,
    "trend": 200,
    "atr_period": 14,
    "trailing_stop_atr_mult": 2.4,  # tightened
    "adx_threshold": 27,
    "vix_spike_threshold": 20.0,
}

INDICATOR = {
    "rsi_period": 14,
    "adx_period": 14,
    "breakout_lookback": 20,
    "adx_slope_lookback": 3,
}

# MR engine params (approx from live)
MR_Z_ENTRY = 1.2
MR_Z_EXIT = 0.3
MR_ADX_MAX = 24.0
MR_TIME_STOP_BARS = 8

# ATR% band
ATR_PCT_MIN = 0.04
ATR_PCT_MAX = 0.20

# Costs
SPREAD_PIPS_DEFAULT = 1.0

# R1 quality gates (approximate live logic)
MIN_MA_DIST_BPS = 1.5
ADAPTIVE_MABPS_COEFF = 0.35  # bps per ATR%*100
ADAPTIVE_MABPS_FLOOR_BPS = 1.0
EPS_BPS = 0.05


# ===================== HELPERS =====================

def download_forex(symbol: str, start: str, interval: str) -> pd.DataFrame:
    # Primary: SYMBOL=X form
    def _primary(sym: str) -> str:
        if "=" not in sym and "/" not in sym:
            return f"{sym}=X"
        return sym
    base = symbol.upper()
    candidates = [_primary(base)]
    if base.startswith("XAU") or "GOLD" in base:
        for alt in ("XAU=X", "GC=F"):
            if alt not in candidates:
                candidates.append(alt)
    last_err = None
    df = None
    for yf_symbol in candidates:
        try:
            df = yf.download(yf_symbol, start=start, interval=interval, progress=False,
                             auto_adjust=False, actions=False)
        except Exception as e:
            last_err = e
            df = None
        if df is not None and not df.empty:
            break
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} {interval} (tried {candidates}); last error: {last_err}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(-1)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df.columns = [str(c).lower().strip() for c in df.columns]
    req = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if len(req) < 4:
        raise RuntimeError(f"Missing OHLC columns. Available: {df.columns.tolist()}")
    return df[req].dropna()


def bars_per_year(interval: str) -> int:
    ival = interval.lower()
    if ival in ("1m", "1min", "1min"):
        return 252 * 24 * 60
    if ival in ("5m", "5min"):
        return 252 * 24 * 12
    if ival in ("15m", "15min"):
        return 252 * 24 * 4
    if ival in ("1h", "60m"):
        return 252 * 24
    if ival in ("4h",):
        return 252 * 6
    if ival in ("1d", "1day"):
        return 252
    return 252 * 24 * 4


def annualize_sigma_percent(sigma_price: float, price_now: float, interval: str) -> float:
    try:
        if sigma_price <= 0 or price_now <= 0:
            return 0.0
        n = bars_per_year(interval)
        return (sigma_price / price_now) * 100.0 * np.sqrt(n)
    except Exception:
        return 0.0


def pair_vol_floor_percent(symbol: str) -> float:
    up = symbol.upper()
    if up.startswith("EURUSD"):
        return 0.03
    if "JPY" in up:
        return 0.02
    if up.startswith("XAU"):
        return 0.04
    return 0.03


def pair_spread_hard_cap(symbol: str) -> float:
    up = symbol.upper()
    if up.startswith("EURUSD"):
        return 0.35
    if up.endswith("JPY") or "JPY" in up:
        return 0.8
    if up.startswith("XAU"):
        return 2.0
    return 1.5


def is_active_hour_utc(ts: pd.Timestamp, symbol: str) -> bool:
    h = int(ts.hour)
    s = symbol.upper()
    if "USDJPY" in s:
        return (0 <= h <= 6) or (12 <= h <= 16)
    if "EURUSD" in s:
        return 6 <= h <= 16
    if "XAUUSD" in s or "XAU" in s or "GOLD" in s:
        return 7 <= h <= 17
    return 6 <= h <= 16


def compute_indicators_router(bars: pd.DataFrame, interval: str) -> pd.DataFrame:
    df = bars.copy()
    # MAs
    df["short_ma"] = df["close"].ewm(span=P["short"], adjust=False).mean()
    df["long_ma"] = df["close"].ewm(span=P["long"], adjust=False).mean()
    df["trend_ma"] = df["close"].ewm(span=P["trend"], adjust=False).mean()

    # Indicators
    df["atr"] = compute_atr_wilder(df, P["atr_period"])
    df["adx"] = calculate_adx(df, INDICATOR["adx_period"])
    df["rsi"] = calculate_rsi(df["close"], INDICATOR["rsi_period"])

    # ATR%
    df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan) * 100.0

    # EWMA stats for MR
    ew_mean = df["close"].ewm(span=60, adjust=False).mean()
    ew_std = df["close"].ewm(span=60, adjust=False).std(bias=False)
    df["mr_ewm_mean"] = ew_mean
    df["mr_ewm_sigma"] = ew_std
    df["mr_z"] = (df["close"] - ew_mean) / ew_std.replace({0.0: np.nan})

    # ADX slope (1-bar delta helpful for trend health)
    df["prev_adx"] = df["adx"].shift(1)
    df["adx_delta_1"] = df["adx"] - df["prev_adx"]
    # Keep RSI too for epsilon gating
    # already present as df["rsi"]

    return df.dropna()


# ===================== BACKTEST ENGINE =====================

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    entry_atr: float
    position_size: float
    entry_equity: float
    regime: str
    edge_score: int
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    r_multiple: Optional[float] = None
    reason: str = ""


def backtest_router(
    bars: pd.DataFrame,
    symbol: str,
    interval: str,
    initial_capital: float = 10000.0,
    atr_min: float = ATR_PCT_MIN,
    mr_z_entry: float = MR_Z_ENTRY,
    # New tunables
    r1_adx_enter: float = 24.0,
    r1_mabps_enter: float = 2.0,
    r1_adx_stay: float = 22.0,
    r1_mabps_stay: float = 1.8,
    r1_adx_gate: float = P["adx_threshold"],
    edge_buy_score: int = EDGE_BUY_SCORE,
    edge_confirm_bars: int = EDGE_CONFIRM_BARS,
    edge_exit_score: int = EDGE_EXIT_SCORE,
    mr_z_exit: float = MR_Z_EXIT,
    mr_time_stop_bars: int = MR_TIME_STOP_BARS,
    enforce_spread_cap: bool = False,
) -> tuple[pd.Series, List[Trade]]:
    df = compute_indicators_router(bars, interval)
    pip_value = get_pip_value(symbol)
    spread_pips = SPREAD_PIPS_DEFAULT
    spread_cost = spread_pips * pip_value

    equity = initial_capital
    initial_balance = initial_capital
    position: Optional[Trade] = None
    equity_curve: List[float] = []
    trades: List[Trade] = []

    # Router state
    regime = "R0"
    regime_cooldown = 0
    r1_bars_in = 0
    r2_bars_in = 0
    r1_enter_buf: List[bool] = []
    r1_fail_streak = 0

    edge_buy_streak = 0
    edge_exit_streak = 0
    last_trade_idx = -999

    # Daily/Total limits tracking
    cur_day = None
    day_start_equity = equity
    day_trades = 0
    # Hard stop flags
    halted_total = False

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = float(row["close"])

        # Day roll
        d = row.name.date()
        if cur_day != d:
            cur_day = d
            day_start_equity = equity
            day_trades = 0

        # FXIFY limit checks
        daily_loss_pct = (day_start_equity - equity) / day_start_equity if day_start_equity > 0 else 0.0
        if daily_loss_pct >= FXIFY_MAX_DAILY_LOSS_PCT:
            # flat and skip until next day
            if position:
                # Close at bid (minus spread and slippage)
                slip_pips = compute_slippage_pips(symbol, float(row.get("atr_pct", 0.05)))
                exit_price = price - spread_cost - slip_pips * pip_value
                pnl = (exit_price - position.entry_price) * position.position_size * 100000
                equity += pnl
                position.exit_time = row.name
                position.exit_price = exit_price
                position.pnl = pnl
                position.pnl_pct = (pnl / position.entry_equity) * 100
                position.reason = "FXIFY Daily Loss"
                trades.append(position)
                position = None
            # Clamp equity to daily floor to simulate prop liquidation at threshold
            daily_floor = day_start_equity * (1.0 - FXIFY_MAX_DAILY_LOSS_PCT)
            if equity < daily_floor:
                equity = daily_floor
            equity_curve.append(equity)
            continue

        total_loss_pct = (initial_balance - equity) / initial_balance if initial_balance > 0 else 0.0
        if total_loss_pct >= FXIFY_MAX_TOTAL_LOSS_PCT:
            if position:
                slip_pips = compute_slippage_pips(symbol, float(row.get("atr_pct", 0.05)))
                exit_price = price - spread_cost - slip_pips * pip_value
                pnl = (exit_price - position.entry_price) * position.position_size * 100000
                equity += pnl
                position.exit_time = row.name
                position.exit_price = exit_price
                position.pnl = pnl
                position.pnl_pct = (pnl / position.entry_equity) * 100
                position.reason = "FXIFY Total Loss"
                trades.append(position)
                position = None
            # Clamp equity to total-loss floor and halt further trading
            total_floor = initial_balance * (1.0 - FXIFY_MAX_TOTAL_LOSS_PCT)
            if equity < total_floor:
                equity = total_floor
            halted_total = True
            equity_curve.append(equity)
            continue

        # If total halted previously, keep flat
        if halted_total:
            equity_curve.append(equity)
            continue

        # Final safety: never record equity below total-loss floor
        total_floor = initial_balance * (1.0 - FXIFY_MAX_TOTAL_LOSS_PCT)
        if equity < total_floor:
            equity = total_floor
            halted_total = True

        # Hard gates: session window and vol floor
        hard_gate = False
        if not is_active_hour_utc(row.name.tz_localize('UTC'), symbol):
            hard_gate = True
        sigma = float(row.get("mr_ewm_sigma", 0.0) or 0.0)
        sigma_ann = annualize_sigma_percent(sigma, float(row.get("close", 0.0) or 0.0), interval)
        if sigma_ann < pair_vol_floor_percent(symbol):
            hard_gate = True
        # Spread hard cap (approx only; yfinance has no live spread). Optional.
        if enforce_spread_cap and (SPREAD_PIPS_DEFAULT > pair_spread_hard_cap(symbol)):
            hard_gate = True

        # Regime signals
        adx_val = float(row.get("adx", 0.0) or 0.0)
        short_ma = float(row.get("short_ma", price))
        long_ma = float(row.get("long_ma", price))
        ma_bps = abs(short_ma - long_ma) / max(price, 1e-12) * 10000.0
        z_now = float(row.get("mr_z", 0.0) or 0.0)
        atr_pct_val = float(row.get("atr_pct", 0.0) or 0.0)

        is_xau = symbol.upper().startswith("XAU") or "GOLD" in symbol.upper()
        atr_min_fx = 0.05
        atr_min_xau = 0.08
        atr_min_used = atr_min_xau if is_xau else atr_min_fx

        r1_ok_enter = (adx_val >= r1_adx_enter) and (ma_bps >= r1_mabps_enter) and (atr_pct_val >= atr_min_used)
        r1_ok_stay = (adx_val >= r1_adx_stay) and (ma_bps >= r1_mabps_stay)
        r2_ok_enter = (adx_val <= r1_adx_stay) and (abs(z_now) >= abs(mr_z_entry))
        r2_ok_stay = (adx_val <= r1_adx_enter) and (abs(z_now) > abs(mr_z_exit))

        # Update R1 enter confirmation buffer (configurable bars)
        r1_enter_buf.append(bool(r1_ok_enter))
        r1_enter_buf[:] = r1_enter_buf[-edge_confirm_bars:]
        r1_confirm = len(r1_enter_buf) >= edge_confirm_bars and all(r1_enter_buf)

        # Regime transitions
        if hard_gate:
            regime = "R0"
            regime_cooldown = 0
            r1_bars_in = 0
            r2_bars_in = 0
        else:
            if regime_cooldown > 0:
                regime_cooldown -= 1
            if regime == "R0":
                if regime_cooldown == 0 and r1_confirm:
                    regime = "R1"
                    regime_cooldown = 2
                    r1_bars_in = 0
                    r1_fail_streak = 0
                elif regime_cooldown == 0 and r2_ok_enter:
                    regime = "R2"
                    regime_cooldown = 2
                    r2_bars_in = 0
            elif regime == "R1":
                r1_bars_in += 1
                if r1_bars_in >= 16:
                    regime = "R0"
                    regime_cooldown = 0
                    r1_bars_in = 0
                elif not r1_ok_stay:
                    r1_fail_streak += 1
                    if r1_fail_streak >= 3:
                        if r2_ok_enter:
                            regime = "R2"
                            regime_cooldown = 2
                            r2_bars_in = 0
                        else:
                            regime = "R0"
                            regime_cooldown = 0
                else:
                    if r1_fail_streak > 0:
                        r1_fail_streak = 0
            elif regime == "R2":
                r2_bars_in += 1
                if adx_val > 24.0:
                    if r1_ok_enter:
                        regime = "R1"
                        regime_cooldown = 2
                        r1_bars_in = 0
                        r1_fail_streak = 0
                    else:
                        regime = "R0"
                        regime_cooldown = 0
                        r2_bars_in = 0
                elif not r2_ok_stay:
                    regime = "R0"
                    regime_cooldown = 0
                    r2_bars_in = 0
                elif r2_bars_in >= mr_time_stop_bars:
                    regime = "R0"
                    regime_cooldown = 0
                    r2_bars_in = 0

        # POSITION MANAGEMENT
        if position is not None:
            # Basic trailing using ATR like base backtester
            init_risk = position.entry_atr * P["trailing_stop_atr_mult"]
            r_mult = (price - position.entry_price) / max(init_risk, 1e-12)
            trail_mult = P["trailing_stop_atr_mult"]
            if r_mult >= 1.0:
                trail_mult = 4.0
            trailing_stop = position.entry_price + r_mult * 0  # anchor on entry; compute absolute below
            trailing_stop = price - trail_mult * float(row.get("atr", 0.0))

            stop_hit = price <= trailing_stop
            # MR exit: if trade opened under R2, close when |z| <= MR_Z_EXIT
            z_now = float(row.get("mr_z", 0.0) or 0.0)
            mr_exit = (position.regime == "R2") and (abs(z_now) <= mr_z_exit)

            # Edge deterioration exit (similar to base)
            edge: EdgeResult = compute_edge_features_and_score(df.iloc[:i+1], row, prev, 0.0, P["vix_spike_threshold"])
            if edge.score <= edge_exit_score:
                edge_exit_streak += 1
            else:
                edge_exit_streak = 0
            edge_exit = edge_exit_streak >= EDGE_EXIT_CONFIRM_BARS

            if stop_hit or edge_exit or mr_exit:
                slip_pips = compute_slippage_pips(symbol, float(row.get("atr_pct", 0.05)))
                exit_price = price - spread_cost - slip_pips * pip_value
                pnl = (exit_price - position.entry_price) * position.position_size * 100000
                equity += pnl
                position.exit_time = row.name
                position.exit_price = exit_price
                position.pnl = pnl
                position.pnl_pct = (pnl / position.entry_equity) * 100
                position.r_multiple = r_mult
                if stop_hit:
                    position.reason = "Trailing Stop"
                elif mr_exit:
                    position.reason = "MR z-exit"
                else:
                    position.reason = "Edge Exit"
                trades.append(position)
                position = None
                edge_exit_streak = 0
                last_trade_idx = i

        # ENTRY LOGIC
        else:
            # cooldown: at least 1 bar after last trade
            if i - last_trade_idx < 1:
                equity_curve.append(equity)
                continue
            # daily trade cap
            if day_trades >= 5:
                equity_curve.append(equity)
                continue
            # ATR band
            atr_pct_val = float(row.get("atr_pct", 0.0) or 0.0)
            if not (atr_min <= atr_pct_val <= ATR_PCT_MAX):
                equity_curve.append(equity)
                continue

            entry_signal = False
            regime_for_entry = regime
            if regime_for_entry == "R2":
                # MR: enter on z-threshold with ADX ceiling
                if adx_val <= MR_ADX_MAX and (z_now <= -abs(mr_z_entry) or z_now >= abs(mr_z_entry)):
                    entry_signal = True
            elif regime_for_entry == "R1":
                # Quality gates: ADX and MA-bps with adaptive floor and epsilon path
                used_adx_thr = float(r1_adx_gate)  # configurable entry ADX gate
                adx_ok = adx_val >= used_adx_thr
                ma_gap_raw = abs(short_ma - long_ma)
                ma_bps_now = ma_gap_raw / max(price, 1e-12) * 10000.0
                atr_pct_now = float(row.get("atr_pct", 0.0) or 0.0)
                dyn_floor = max(ADAPTIVE_MABPS_FLOOR_BPS, ADAPTIVE_MABPS_COEFF * (atr_pct_now * 100.0))
                used_mabps_thr = max(MIN_MA_DIST_BPS, dyn_floor)
                d_adx = float(row.get("adx_delta_1", 0.0) or 0.0)
                rsi_val = float(row.get("rsi", 0.0) or 0.0)
                ma_ok_base = ma_bps_now >= used_mabps_thr
                ma_ok_eps = ((ma_bps_now + EPS_BPS) >= used_mabps_thr) and (d_adx >= 0.5) and (rsi_val >= 48.0)
                ma_ok = ma_ok_base or ma_ok_eps
                # Log gates snapshot for analysis
                try:
                    out_dir = Path(ROOT_DIR) / "backtests" / "reports"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    sym_clean = symbol.replace("/", "").replace("=", "")
                    gates_path = out_dir / f"router_gates_{sym_clean}_{interval}.csv"
                    write_header = not gates_path.exists()
                    with open(gates_path, mode="a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        if write_header:
                            w.writerow([
                                "ts","symbol","interval","regime",
                                "price","adx","adx_thr","adx_ok",
                                "ma_bps","mabps_thr","ma_ok","ma_norm",
                                "atr_pct","edge_score"
                            ])
                        ma_norm = ma_bps_now / max(dyn_floor, 1e-12)
                        # defer edge score until computed; write 0 if not available
                        edge_score_val = 0
                        try:
                            edge_score_val = int(edge.score)  # if defined later
                        except Exception:
                            edge_score_val = 0
                        w.writerow([
                            row.name.isoformat(), symbol, interval, regime_for_entry,
                            round(price, 6), round(adx_val, 3), round(used_adx_thr, 3), bool(adx_ok),
                            round(ma_bps_now, 3), round(used_mabps_thr, 3), bool(ma_ok), round(ma_norm, 3),
                            round(atr_pct_now, 6), edge_score_val
                        ])
                except Exception:
                    pass
                # If quality gates fail, reset streak and skip edge accumulation
                if not (adx_ok and ma_ok):
                    edge_buy_streak = 0
                else:
                    edge: EdgeResult = compute_edge_features_and_score(df.iloc[:i+1], row, prev, 0.0, P["vix_spike_threshold"])
                    if edge.score >= edge_buy_score:
                        edge_buy_streak += 1
                    else:
                        edge_buy_streak = 0
                entry_signal = (adx_ok and ma_ok and (edge_buy_streak >= edge_confirm_bars))

            if entry_signal:
                # Risk sizing similar to base
                atr = float(row.get("atr", 0.0) or 0.0)
                risk_pct = RISK_PCT_BASE
                if adx_val > ADX_RISK_THR and atr_pct_val > ATR_PCT_RISK_THR:
                    risk_pct = RISK_PCT_STRONG
                risk_dollars = equity * risk_pct

                # Entry price with costs
                slip_pips = compute_slippage_pips(symbol, float(row.get("atr_pct", 0.05)))
                entry_price = price + spread_cost + slip_pips * pip_value
                stop_distance = P["trailing_stop_atr_mult"] * atr
                stop_pips = max(1e-12, stop_distance / max(pip_value, 1e-12))
                lot_size = risk_dollars / (stop_pips * pip_value * 100000)
                lot_size = max(0.01, round(lot_size, 2))

                position = Trade(
                    entry_time=row.name,
                    entry_price=entry_price,
                    entry_atr=atr,
                    position_size=lot_size,
                    entry_equity=equity,
                    regime=regime_for_entry,
                    edge_score=int(edge.score if 'edge' in locals() else 0),
                )
                day_trades += 1
                edge_buy_streak = 0

        equity_curve.append(equity)

    # Close any remaining position at last price (minus spread only)
    if position is not None:
        last_price = float(df.iloc[-1]["close"])
        exit_price = last_price - spread_cost
        pnl = (exit_price - position.entry_price) * position.position_size * 100000
        equity += pnl
        position.exit_time = df.iloc[-1].name
        position.exit_price = exit_price
        position.pnl = pnl
        position.pnl_pct = (pnl / position.entry_equity) * 100
        position.reason = "End of backtest"
        trades.append(position)

    equity_series = pd.Series(equity_curve, index=df.index[1:])
    return equity_series, trades


# ===================== ANALYSIS/IO =====================

def analyze_results(equity: pd.Series, trades: List[Trade], initial_capital: float, symbol: str):
    print("\n" + "="*80)
    print(f"BACKTEST RESULTS (Router): {symbol}")
    print("="*80)
    if len(equity) == 0:
        print("No equity data")
        return
    final_equity = float(equity.iloc[-1])
    total_return = (final_equity / initial_capital - 1.0) * 100.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min() * 100.0)
    rets = equity.pct_change().fillna(0.0)
    sharpe = (np.sqrt(252) * rets.mean() / (rets.std() + 1e-12)) if len(rets) > 1 else 0.0
    print(f"  Initial: ${initial_capital:,.2f}")
    print(f"  Final:   ${final_equity:,.2f}")
    print(f"  Return:  {total_return:+.2f}%")
    print(f"  MaxDD:   {max_dd:.2f}%")
    print(f"  Sharpe:  {sharpe:.2f}")
    # PF
    if trades:
        wins = sum([max(0.0, float(t.pnl or 0.0)) for t in trades])
        losses = sum([max(0.0, -float(t.pnl or 0.0)) for t in trades])
        pf = (wins / losses) if losses > 1e-12 else (float('inf') if wins > 0 else 1.0)
        print(f"  PF:      {pf:.2f}")
        print(f"  Trades:  {len(trades)}")


def save_results(equity: pd.Series, trades: List[Trade], symbol: str, interval: str, out_dir: Path):
    out_path = out_dir / f"router_{symbol.replace('=','').replace('/','_')}_{interval}"
    out_path.mkdir(parents=True, exist_ok=True)
    if len(equity) > 0:
        equity.to_csv(out_path / "equity_curve.csv", header=["equity"])
    if trades:
        df = pd.DataFrame([{
            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "entry_atr": t.entry_atr,
            "position_size": t.position_size,
            "entry_equity": t.entry_equity,
            "regime": t.regime,
            "edge_score": t.edge_score,
            "exit_time": t.exit_time,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "r_multiple": t.r_multiple,
            "reason": t.reason,
        } for t in trades])
        df.to_csv(out_path / "trades.csv", index=False)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="FXIFY Forex Backtester (Router Parity)")
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--start", default="2025-09-01")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--out", default="backtests/reports/forex_fxify_router")
    ap.add_argument("--atr-min", type=float, default=ATR_PCT_MIN, help="ATR%% lower band to allow entries")
    ap.add_argument("--mr-z-entry", type=float, default=MR_Z_ENTRY, help="MR z-score entry threshold (abs)")
    # New tunables
    ap.add_argument("--r1-adx-enter", type=float, default=24.0)
    ap.add_argument("--r1-mabps-enter", type=float, default=2.0)
    ap.add_argument("--r1-adx-stay", type=float, default=22.0)
    ap.add_argument("--r1-mabps-stay", type=float, default=1.8)
    ap.add_argument("--r1-adx-gate", type=float, default=P["adx_threshold"], help="R1 entry ADX quality gate threshold")
    ap.add_argument("--edge-buy-score", type=int, default=EDGE_BUY_SCORE)
    ap.add_argument("--edge-confirm-bars", type=int, default=EDGE_CONFIRM_BARS)
    ap.add_argument("--edge-exit-score", type=int, default=EDGE_EXIT_SCORE)
    ap.add_argument("--mr-z-exit", type=float, default=MR_Z_EXIT)
    ap.add_argument("--mr-time-stop", type=int, default=MR_TIME_STOP_BARS)
    ap.add_argument("--enforce-spread-cap", action="store_true", help="Enforce optional pair-specific spread hard cap gate")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    bars = download_forex(args.symbol, args.start, args.interval)
    equity, trades = backtest_router(
        bars, args.symbol, args.interval, args.capital,
        atr_min=args.atr_min,
        mr_z_entry=args.mr_z_entry,
        r1_adx_enter=args.r1_adx_enter,
        r1_mabps_enter=args.r1_mabps_enter,
    r1_adx_stay=args.r1_adx_stay,
        r1_mabps_stay=args.r1_mabps_stay,
    r1_adx_gate=args.r1_adx_gate,
        edge_buy_score=args.edge_buy_score,
        edge_confirm_bars=args.edge_confirm_bars,
        edge_exit_score=args.edge_exit_score,
        mr_z_exit=args.mr_z_exit,
        mr_time_stop_bars=args.mr_time_stop,
        enforce_spread_cap=args.enforce_spread_cap,
    )
    analyze_results(equity, trades, args.capital, args.symbol)
    save_results(equity, trades, args.symbol, args.interval, Path(args.out))


if __name__ == "__main__":
    main()
