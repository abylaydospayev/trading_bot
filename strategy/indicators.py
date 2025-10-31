"""Indicator utilities for the trading strategy backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
MIN_ATR = 0.01


def _wilder_ewm(series: pd.Series, period: int) -> pd.Series:
    """Return Wilder's exponential moving average."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def compute_atr_wilder(bars: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    high_low = bars["high"] - bars["low"]
    high_close = (bars["high"] - bars["close"].shift()).abs()
    low_close = (bars["low"] - bars["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    tr = tr.fillna(0.0)
    atr = _wilder_ewm(tr, period)
    return atr.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def calculate_adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index with Wilder smoothing."""
    high = bars["high"]
    low = bars["low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(0.0, index=bars.index)
    minus_dm = pd.Series(0.0, index=bars.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    atr = compute_atr_wilder(bars, period).clip(lower=MIN_ATR)

    plus_di = 100 * (_wilder_ewm(plus_dm, period) / atr)
    minus_di = 100 * (_wilder_ewm(minus_dm, period) / atr)

    denom = (plus_di + minus_di).replace(0, EPS)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx = _wilder_ewm(dx, period)
    return adx.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _wilder_ewm(gain, period)
    avg_loss = _wilder_ewm(loss, period).replace(0, EPS)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator.

    Returns (macd, signal_line, histogram) where:
    - macd = EMA(fast) - EMA(slow)
    - signal_line = EMA(macd, signal)
    - histogram = macd - signal_line
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = (ema_fast - ema_slow).fillna(0.0)
    signal_line = macd.ewm(span=signal, adjust=False).mean().fillna(0.0)
    hist = (macd - signal_line).fillna(0.0)
    return macd, signal_line, hist
