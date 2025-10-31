"""Scalper module skeleton for RSI+MACD and RSI+Momentum logic.

This module provides a minimal, decoupled interface for future integration
into live/trading_bot_fxify.py and the backtester. For now, functions return
signals based on provided data but are NOT wired into the main bot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from .indicators import calculate_rsi, calculate_macd


@dataclass
class ScalpParams:
    timeframe: str  # "1m" | "5m"
    entry_logic: str  # "RSI+MACD" | "RSI+Momentum"
    rsi_entry_long: int = 35
    rsi_exit_long: int = 60
    rsi_entry_short: int = 65
    rsi_exit_short: int = 40
    macd_signal_cross: bool = True
    momentum_threshold: float = 0.2
    tp_pips: float = 10.0
    sl_pips: float = 15.0
    risk_per_trade: float = 0.001
    session_filter: Optional[str] = None  # e.g., "Asia+London", "London+NY"


def _session_ok(session_filter: Optional[str], ts: pd.Timestamp) -> bool:
    if not session_filter:
        return True
    h = ts.hour
    flt = session_filter.replace(" ", "").lower()
    if flt == "asia+london":
        return (0 <= h <= 9) or (7 <= h <= 16)
    if flt == "london+ny":
        return 7 <= h <= 20
    return True


def rsi_macd_signal(df: pd.DataFrame, params: ScalpParams) -> Tuple[bool, Optional[str], Optional[str]]:
    """Return (has_signal, direction, reason) using RSI + MACD cross as a micro-edge.

    DataFrame must have columns: open, high, low, close, time index (UTC-naive).
    """
    if df.empty:
        return False, None, None
    ts = df.index[-1]
    if not _session_ok(params.session_filter, ts):
        return False, None, None
    rsi = calculate_rsi(df["close"], 14)
    macd, signal, hist = calculate_macd(df["close"])  # standard defaults
    rsi_now = float(rsi.iloc[-1])
    macd_cross_up = (macd.iloc[-2] <= signal.iloc[-2]) and (macd.iloc[-1] > signal.iloc[-1])
    macd_cross_dn = (macd.iloc[-2] >= signal.iloc[-2]) and (macd.iloc[-1] < signal.iloc[-1])
    if rsi_now <= params.rsi_entry_long and macd_cross_up:
        return True, "long", f"RSI<= {params.rsi_entry_long} & MACD cross up"
    if rsi_now >= params.rsi_entry_short and macd_cross_dn:
        return True, "short", f"RSI>= {params.rsi_entry_short} & MACD cross down"
    return False, None, None


def rsi_momentum_signal(df: pd.DataFrame, params: ScalpParams) -> Tuple[bool, Optional[str], Optional[str]]:
    """Return (has_signal, direction, reason) using RSI gate + simple momentum filter.

    Momentum is approximated as last close change vs. mean of prior N bars.
    """
    if df.empty:
        return False, None, None
    ts = df.index[-1]
    if not _session_ok(params.session_filter, ts):
        return False, None, None
    rsi = calculate_rsi(df["close"], 14)
    rsi_now = float(rsi.iloc[-1])
    # Simple momentum proxy
    last = float(df["close"].iloc[-1])
    look = df["close"].iloc[-6:-1]
    if look.empty:
        return False, None, None
    mom = (last - float(look.mean())) / (float(look.std()) + 1e-9)
    if rsi_now <= params.rsi_entry_long and mom >= params.momentum_threshold:
        return True, "long", f"RSI<= {params.rsi_entry_long} & momentum z={mom:.2f}"
    if rsi_now >= params.rsi_entry_short and mom <= -params.momentum_threshold:
        return True, "short", f"RSI>= {params.rsi_entry_short} & momentum z={mom:.2f}"
    return False, None, None
