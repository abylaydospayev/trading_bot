from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import pandas as pd

from core.volatility_brain import ewma_stats, rolling_stats


@dataclass
class MREngineParams:
    # z-score thresholds
    z_entry: float = 1.2
    z_exit: float = 0.3
    # ADX ceiling for mean-reversion eligibility (avoid strong-trend regimes)
    adx_max: float = 24.0
    # Volatility estimator config
    mode: Literal["ewma", "rolling"] = "ewma"
    span_mean: int = 60
    span_std: int = 60
    lookback: int = 200
    # Time stop in bars from entry
    time_stop_bars: int = 16


def compute_signal(prices: pd.Series, adx_latest: float, params: MREngineParams) -> Tuple[str, float, float, float]:
    """
    Compute MR signal using z-score against an EWMA/rolling mean and std.

    Returns: (action, z, mean, sigma)
      - action: "long" | "short" | "flat"
    """
    if prices is None or len(prices) < 10:
        return ("flat", 0.0, float('nan'), float('nan'))

    if params.mode == "rolling":
        m, sigma, z = rolling_stats(prices, params.lookback)
    else:
        m, sigma, z = ewma_stats(prices, params.span_mean, params.span_std)

    if not pd.notna(sigma) or sigma <= 0:
        return ("flat", 0.0, m, sigma if pd.notna(sigma) else float('nan'))

    # Regime filter
    if float(adx_latest) > float(params.adx_max):
        return ("flat", float(z), m, sigma)

    action = "flat"
    if z <= -abs(params.z_entry):
        action = "long"
    elif z >= abs(params.z_entry):
        action = "short"

    return (action, float(z), m, float(sigma))
