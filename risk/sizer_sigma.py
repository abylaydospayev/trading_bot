from __future__ import annotations

from typing import Optional


def pip_size_from_symbol(symbol: str) -> float:
    up = symbol.upper()
    if "JPY" in up:
        return 0.01
    if "XAU" in up or "XAG" in up:
        return 0.01
    return 0.0001


def sigma_to_stop_pips(symbol: str, sigma_price: float, k_sigma: float, price_now: Optional[float] = None) -> float:
    """
    Convert sigma in price units to a stop distance in pips.
    Assumes stop_distance_price = k_sigma * sigma_price.

    pip = 0.0001 for most FX, 0.01 for JPY pairs and metals.
    """
    pip = pip_size_from_symbol(symbol)
    stop_price = max(0.0, float(k_sigma) * float(sigma_price))
    return stop_price / max(pip, 1e-12)
