#!/usr/bin/env python3
"""
FTMO Trading Bot launcher

This thin wrapper sets FTMO-default environment variables, then imports the
existing trading bot implementation and runs it. Keeping env overrides here
lets you use the same core bot while switching prop rules easily.
"""
from __future__ import annotations

import os
import sys

# Ensure project root on sys.path so `live` is importable when running this file directly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _set_default_env(key: str, value: str) -> None:
    if os.getenv(key) is None:
        os.environ[key] = value


def configure_ftmo_env() -> None:
    # FTMO defaults
    _set_default_env("DAY_RESET_TZ", "Europe/Prague")
    _set_default_env("FXIFY_MAX_DAILY_LOSS_PCT", "0.05")  # reuse var name from core bot
    _set_default_env("FXIFY_MAX_TOTAL_LOSS_PCT", "0.10")  # reuse var name from core bot
    _set_default_env("FXIFY_PROFIT_TARGET_PCT", "0.10")   # FTMO phase 1 target 10%
    _set_default_env("FTMO_MODE", "true")                 # flag to adjust banners/logging
    # Use a distinct state file to avoid mixing contexts
    _set_default_env("STATE_FILE", "ftmo_bot_state.json")

    # Sensible live defaults (can be overridden in env file)
    _set_default_env("TIMEFRAME", "15Min")
    _set_default_env("RISK_PCT", "0.003")  # 0.30% per trade
    _set_default_env("TRAIL_ATR_MULT", "2.4")
    _set_default_env("EDGE_BUY_SCORE", "60")
    _set_default_env("ADX_THRESHOLD", "25")


def main() -> None:
    configure_ftmo_env()
    # Import after env is set so module-level config picks up values
    from live.trading_bot_fxify import main as core_main
    core_main()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Ensure non-zero exit on fatal
        print(f"[FTMO] Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
