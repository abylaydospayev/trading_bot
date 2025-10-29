"""Edge score computation mirroring the live trading bot logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .indicators import MIN_ATR


@dataclass
class EdgeResult:
    score: int
    reasons: List[str]
    data: pd.Series


def compute_edge_features_and_score(
    bars: pd.DataFrame,
    latest: pd.Series,
    prev: pd.Series,
    vix_pct_change: float,
    vix_spike_threshold: float,
) -> EdgeResult:
    """Replicate the discretionary edge score logic."""
    score = 0
    reasons: List[str] = []

    if latest["close"] > latest["trend_ma"]:
        score += 20
        reasons.append("Uptrend")

    if latest["short_ma"] > latest["long_ma"] and (
        latest["short_ma"] - prev["short_ma"]
    ) > 1e-6:
        score += 20
        reasons.append("Momentum up")

    if latest["adx"] > 25:
        score += 15
        reasons.append("ADX>25")
    if latest.get("adx_slope", 0) > 0:
        score += 10
        reasons.append("ADX rising")

    rsi = latest.get("rsi", np.nan)
    if pd.notna(rsi) and 50 <= rsi <= 70:
        score += 10
        reasons.append("RSI 50-70")

    prev_high20 = latest.get("prev_high20", np.nan)
    if pd.notna(prev_high20) and latest["close"] > prev_high20:
        score += 10
        reasons.append("Breakout>20H")

    atr_median20 = latest.get("atr_median20", np.nan)
    if pd.notna(atr_median20) and latest["atr"] > atr_median20:
        score += 5
        reasons.append("ATR expansion")

    if vix_pct_change > vix_spike_threshold:
        score -= 15
        reasons.append("VIX spike penalty")

    score = int(max(0, min(100, score)))
    # Do not mutate the input Series to avoid pandas setitem overhead; return metadata only
    return EdgeResult(score=score, reasons=reasons, data=latest)
