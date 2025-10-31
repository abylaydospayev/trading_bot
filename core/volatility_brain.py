from __future__ import annotations

from typing import Tuple
import pandas as pd
import numpy as np


def ewma_stats(prices: pd.Series, span_mean: int = 60, span_std: int = 60) -> Tuple[float, float, float]:
    """
    Compute EWMA mean and EWMA std for the given price series and return
    (mean_latest, sigma_latest, z_latest) where z = (price - mean)/sigma.

    - Uses pandas ewm for both mean and std.
    - If sigma is zero/NaN, z is returned as 0.0.
    """
    if prices is None or len(prices) == 0:
        return (np.nan, np.nan, 0.0)
    s = pd.Series(prices).astype(float)
    mean = s.ewm(span=max(1, int(span_mean)), adjust=False).mean()
    std = s.ewm(span=max(1, int(span_std)), adjust=False).std(bias=False)
    m_latest = float(mean.iloc[-1])
    sigma_latest = float(std.iloc[-1]) if float(std.iloc[-1]) > 0 else np.nan
    price_latest = float(s.iloc[-1])
    if not np.isfinite(sigma_latest) or sigma_latest <= 0:
        z = 0.0
    else:
        z = (price_latest - m_latest) / sigma_latest
    return (m_latest, sigma_latest, float(z))


def rolling_stats(prices: pd.Series, lookback: int = 200) -> Tuple[float, float, float]:
    """
    Rolling-mean and rolling-std alternative.
    Returns (mean_latest, sigma_latest, z_latest).
    """
    if prices is None or len(prices) == 0:
        return (np.nan, np.nan, 0.0)
    s = pd.Series(prices).astype(float)
    if len(s) < max(5, int(lookback)):
        s = s.copy()
        # pad by repeating first value to satisfy window requirements
        pad_count = max(0, int(lookback) - len(s))
        if pad_count > 0:
            s = pd.concat([pd.Series([s.iloc[0]] * pad_count), s], ignore_index=True)
    mean = s.rolling(window=max(5, int(lookback))).mean()
    std = s.rolling(window=max(5, int(lookback))).std(ddof=0)
    m_latest = float(mean.iloc[-1])
    sigma_latest = float(std.iloc[-1]) if float(std.iloc[-1]) > 0 else np.nan
    price_latest = float(s.iloc[-1])
    if not np.isfinite(sigma_latest) or sigma_latest <= 0:
        z = 0.0
    else:
        z = (price_latest - m_latest) / sigma_latest
    return (m_latest, sigma_latest, float(z))
