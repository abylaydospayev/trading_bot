#!/usr/bin/env python3
"""
FXIFY Forex Trading Bot - Adapted from Crypto/Equity Strategy
- Platform: MetaTrader 5 (MT5)
- Prop Firm: FXIFY rules compliance
- Timeframe: Configurable (default 15Min)
- Strategy: Same edge-based + technical indicators
- Risk Management: Adapted for FXIFY drawdown rules
- Position Sizing: Based on account risk percentage

FXIFY Rules:
- Max Daily Loss: 5% of initial balance
- Max Total Loss: 10% of initial balance  
- Consistency Rule: No single trade > 40% of total profit
- Minimum Trading Days: 4-5 days typically
- Profit Target: 8-10% depending on challenge phase
"""
from __future__ import annotations

import os
import csv
import sys
import time
import math
import json
import logging
import signal
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import pytz
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
# Robust MetaTrader5 import (handles module name variations)
try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    try:
        import metatrader5 as mt5  # type: ignore
    except Exception as _e:
        raise ImportError("MetaTrader5/metatrader5 module not installed. pip install MetaTrader5")

from dotenv import load_dotenv

# Ensure project root on sys.path for `strategy` imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from strategy.indicators import compute_atr_wilder, calculate_adx, calculate_rsi, MIN_ATR
from strategy.edge import compute_edge_features_and_score, EdgeResult
from ml.infer import predict_entry_prob

# ===================== CONFIG =====================
load_dotenv()

# --- MT5 Connection ---
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", "")  # Optional path to terminal.exe

# --- Trading Parameters ---
SYMBOL = os.getenv("SYMBOL", "EURUSD")  # Forex pairs: EURUSD, GBPUSD, XAUUSD, etc.
TIMEFRAME_STR = os.getenv("TIMEFRAME", "15Min")  # 1Min, 5Min, 15Min, 1H, 4H, 1D
MAGIC_NUMBER = int(os.getenv("MAGIC_NUMBER", "234567"))  # Unique identifier for bot trades

# MT5 Timeframe mapping
TIMEFRAME_MAP = {
    "1Min": mt5.TIMEFRAME_M1,
    "5Min": mt5.TIMEFRAME_M5,
    "15Min": mt5.TIMEFRAME_M15,
    "30Min": mt5.TIMEFRAME_M30,
    "1H": mt5.TIMEFRAME_H1,
    "4H": mt5.TIMEFRAME_H4,
    "1D": mt5.TIMEFRAME_D1,
}
TIMEFRAME = TIMEFRAME_MAP.get(TIMEFRAME_STR, mt5.TIMEFRAME_M15)

def _seconds_per_bar(tf_str: str) -> int:
    """Convert timeframe string to seconds"""
    mapping = {
        "1Min": 60,
        "5Min": 5 * 60,
        "15Min": 15 * 60,
        "30Min": 30 * 60,
        "1H": 60 * 60,
        "4H": 4 * 60 * 60,
        "1D": 24 * 60 * 60,
    }
    return mapping.get(tf_str, 15 * 60)

SECONDS_PER_BAR = _seconds_per_bar(TIMEFRAME_STR)

# --- FXIFY Risk Parameters ---
# FXIFY rules: typically 5% max daily loss, 10% max total loss
FXIFY_MAX_DAILY_LOSS_PCT = float(os.getenv("FXIFY_MAX_DAILY_LOSS_PCT", "0.05"))  # 5%
FXIFY_MAX_TOTAL_LOSS_PCT = float(os.getenv("FXIFY_MAX_TOTAL_LOSS_PCT", "0.10"))  # 10%
FXIFY_PROFIT_TARGET_PCT = float(os.getenv("FXIFY_PROFIT_TARGET_PCT", "0.08"))   # 8%
FXIFY_MAX_SINGLE_TRADE_PROFIT_PCT = float(os.getenv("FXIFY_MAX_SINGLE_TRADE_PROFIT_PCT", "0.40"))  # 40% of total profit

# --- Position Sizing ---
# Default lowered per request: 0.3% per trade (tune via env RISK_PCT)
PORTFOLIO_RISK_PCT = float(os.getenv("RISK_PCT", "0.003"))
RISK_PCT_BASE = float(os.getenv("RISK_PCT_BASE", "0.01"))
RISK_PCT_STRONG = float(os.getenv("RISK_PCT_STRONG", "0.02"))  # Max 2% on strong setups
SLIPPAGE_PIPS = float(os.getenv("SLIPPAGE_PIPS", "1"))  # Expected slippage in pips

# --- Entry/Exit Controls ---
OPPORTUNISTIC_MODE = os.getenv("OPPORTUNISTIC_MODE", "true").lower() == "true"
EDGE_BUY_SCORE = int(os.getenv("EDGE_BUY_SCORE", "60"))
EDGE_EXIT_SCORE = int(os.getenv("EDGE_EXIT_SCORE", "10"))
EDGE_CONFIRM_BARS = int(os.getenv("EDGE_CONFIRM_BARS", "2"))
EDGE_EXIT_CONFIRM_BARS = int(os.getenv("EDGE_EXIT_CONFIRM_BARS", "2"))

# --- Trading Controls ---
MIN_COOLDOWN_MIN = int(os.getenv("MIN_COOLDOWN_MIN", "15"))  # Longer cooldown for forex
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "5"))  # Conservative for prop firm
MIN_EDGE_ATR_PCT = float(os.getenv("MIN_EDGE_ATR_PCT", "0.03"))

# --- Circuit Breaker ---
CB_ENABLED = os.getenv("CB_ENABLED", "true").lower() == "true"
CB_MAX_LOSS_STREAK = int(os.getenv("CB_MAX_LOSS_STREAK", "2"))
CB_COOLDOWN_MIN = int(os.getenv("CB_COOLDOWN_MIN", "120"))

# --- Indicator Parameters ---
INDICATOR = {
    "rsi_period": int(os.getenv("RSI_PERIOD", "14")),
    "adx_period": int(os.getenv("ADX_PERIOD", "14")),
    "breakout_lookback": int(os.getenv("BREAKOUT_LOOKBACK", "20")),
    "adx_slope_lookback": int(os.getenv("ADX_SLOPE_LOOKBACK", "3")),
}

P = {
    "short": int(os.getenv("SHORT_MA", "10")),
    "long": int(os.getenv("LONG_MA", "25")),
    "trend": int(os.getenv("TREND_MA", "200")),
    "atr_period": int(os.getenv("ATR_PERIOD", "14")),
    # Default tightened: 2.4x ATR
    "trailing_stop_atr_mult": float(os.getenv("TRAIL_ATR_MULT", "2.4")),
    "use_ema": os.getenv("USE_EMA", "true").lower() == "true",
    "adx_threshold": int(os.getenv("ADX_THRESHOLD", "27")),
    "winner_run_atr_mult_widen": float(os.getenv("WINNER_WIDEN_MULT", "1.2")),
    "winner_run_threshold_atr": float(os.getenv("WINNER_THRESHOLD_ATR", "2.0")),
    "partial_profit_threshold_atr": float(os.getenv("PARTIAL_ATR", "3.0")),
    "vix_spike_threshold": float(os.getenv("VIX_SPIKE_THRESHOLD", "20.0")),
}

# Adaptive trailing
TRAIL_WIDEN_AFTER_R = float(os.getenv("TRAIL_WIDEN_AFTER_R", "1.0"))
TRAIL_ABS_WIDEN_TO = float(os.getenv("TRAIL_ABS_WIDEN_TO", "4.0"))

# Adaptive risk
ADX_RISK_THR = float(os.getenv("ADX_RISK_THR", "25"))
ATR_PCT_RISK_THR = float(os.getenv("ATR_PCT_RISK_THR", "0.04"))

# --- File paths ---
STATE_FILE = os.getenv("STATE_FILE", "fxify_bot_state.json")
LOG_DIR = os.getenv("LOG_DIR", "logs")

# --- Optional filters ---
ATR_REGIME_WINDOW = int(os.getenv("ATR_REGIME_WINDOW", "0"))
ATR_REGIME_PCT = float(os.getenv("ATR_REGIME_PCT", "0"))
# Lower default MA-distance gate to permit entries in sideways volatility
# You can override via env MIN_MA_DIST_BPS; recommended range: 1.0–1.5 bps
MIN_MA_DIST_BPS = float(os.getenv("MIN_MA_DIST_BPS", "1.5"))
# Optional adaptive MA-distance threshold (dynamic floor based on ATR%)
ADAPTIVE_MABPS_ENABLE = os.getenv("ADAPTIVE_MABPS_ENABLE", "false").lower() == "true"
ADAPTIVE_MABPS_COEFF = float(os.getenv("ADAPTIVE_MABPS_COEFF", "0.35"))  # coefficient for (ATR in bps)
# Lower adaptive floor to 1.0 bps to match desired 1.0–1.5 operating band
ADAPTIVE_MABPS_FLOOR_BPS = float(os.getenv("ADAPTIVE_MABPS_FLOOR_BPS", "1.0"))  # minimum floor in bps

# --- 24/7 trading toggles (off-hours safeguards)
ALWAYS_ACTIVE = os.getenv("ALWAYS_ACTIVE", "false").lower() == "true"
OFFHOURS_ADX_MIN = float(os.getenv("OFFHOURS_ADX_MIN", "20"))
OFFHOURS_SPREAD_EURUSD = float(os.getenv("OFFHOURS_SPREAD_EURUSD", "0.3"))
OFFHOURS_SPREAD_JPY = float(os.getenv("OFFHOURS_SPREAD_JPY", "0.5"))
# Optional: stricter/looser off-hours edge buy score (default to regular score)
OFFHOURS_EDGE_BUY_SCORE = int(os.getenv("OFFHOURS_EDGE_BUY_SCORE", str(EDGE_BUY_SCORE)))
MTF_ENABLE = os.getenv("MTF_ENABLE", "false").lower() == "true"
MTF_TF_STR = os.getenv("MTF_TF", "1H")
MTF_TF = TIMEFRAME_MAP.get(MTF_TF_STR, mt5.TIMEFRAME_H1)
MTF_TREND_MA = int(os.getenv("MTF_TREND_MA", "200"))
MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "1.5"))  # Tighter default
ADX_SLOPE_MIN = float(os.getenv("ADX_SLOPE_MIN", "0"))

# --- Active trading window (UTC)
ACTIVE_HOUR_START = int(os.getenv("ACTIVE_HOUR_START", "6"))   # 06:00 UTC
ACTIVE_HOUR_END = int(os.getenv("ACTIVE_HOUR_END", "15"))      # 15:00 UTC
# Hard daily stop (-3%): block new entries for remainder of day
DAILY_STOP_PCT = float(os.getenv("DAILY_STOP_PCT", str(FXIFY_MAX_DAILY_LOSS_PCT)))

# FTMO Day reset timezone (CET/CEST)
DAY_RESET_TZ = os.getenv("DAY_RESET_TZ", "Europe/Prague")

# Pyramiding (live)
CONCURRENT_RISK_CAP = float(os.getenv("CONCURRENT_RISK_CAP", "0.01"))
PYRAMID_ENABLE = os.getenv("PYRAMID_ENABLE", "false").lower() == "true"
PYRAMID_STEP_RISK = float(os.getenv("PYRAMID_STEP_RISK", "0.0012"))
PYRAMID_MAX_TOTAL_RISK = float(os.getenv("PYRAMID_MAX_TOTAL_RISK", "0.006"))

# --- Misc ---
EPS = 1e-12
LOOP_ONCE = os.getenv("LOOP_ONCE", "false").lower() == "true"

# --- ML Gating (optional) ---
ML_ENABLE = os.getenv("ML_ENABLE", "false").lower() == "true"
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", os.path.join(ROOT_DIR, "ml", "models", "mlp_model.pkl"))
ML_PROB_THR = float(os.getenv("ML_PROB_THR", "0.6"))

# --- Volatility bands and MA normalization ---
# Require ATR%% within [min,max] band to avoid dead/stormy markets
ATR_PCT_MIN = float(os.getenv("ATR_PCT_MIN", "0.04"))  # 0.04%
ATR_PCT_MAX = float(os.getenv("ATR_PCT_MAX", "0.20"))  # 0.20%
# Off-hours: allow entries if MA distance normalized to dynamic floor is adequate
MA_NORM_MIN_OFFHOURS = float(os.getenv("MA_NORM_MIN_OFFHOURS", "0.6"))
OFFHOURS_STRICT_ADX_MIN = float(os.getenv("OFFHOURS_STRICT_ADX_MIN", "28"))
OFFHOURS_STRICT_MA_NORM = float(os.getenv("OFFHOURS_STRICT_MA_NORM", "0.8"))

# --- Entry sizing and staging ---
FIRST_ENTRY_PCT = float(os.getenv("FIRST_ENTRY_PCT", "0.65"))
STAGE2_ENTRY_PCT = float(os.getenv("STAGE2_ENTRY_PCT", "0.35"))
ADD_SPREAD_BLOCK_PIPS = float(os.getenv("ADD_SPREAD_BLOCK_PIPS", "0.6"))
ADX_ENTRY_BASE = float(os.getenv("ADX_ENTRY_BASE", "18"))
ATR_SIZE_REF = float(os.getenv("ATR_SIZE_REF", "0.08"))  # 0.08% reference

# --- Defense rules ---
EDGE_DROP_TRIM = float(os.getenv("EDGE_DROP_TRIM", "15"))  # trim if edge drops by this within early window
ADX_DROP_TRIM = float(os.getenv("ADX_DROP_TRIM", "3"))     # trim if ADX drops by this within early window
EARLY_WINDOW_BARS = int(os.getenv("EARLY_WINDOW_BARS", "2"))
TRIM_EARLY_FRACTION = float(os.getenv("TRIM_EARLY_FRACTION", "0.5"))
ADX_FLOOR_EXIT = float(os.getenv("ADX_FLOOR_EXIT", "18"))
DD_FUSE_FRAC = float(os.getenv("DD_FUSE_FRAC", "0.6"))     # trim if loss exceeds 60% of initial risk
DD_TRIM_FRAC = float(os.getenv("DD_TRIM_FRAC", "0.4"))      # trim 40% on fuse

# --- JPY caps and stop/BE/trail tweaks ---
MAX_JPY_LOTS = float(os.getenv("MAX_JPY_LOTS", "1.2"))
BE_PROMOTE_R = float(os.getenv("BE_PROMOTE_R", "0.75"))
BE_PADDING_ATR = float(os.getenv("BE_PADDING_ATR", "0.1"))
TRAIL_AFTER_R = float(os.getenv("TRAIL_AFTER_R", "1.5"))
TRAIL_ATR_MULT_LATE = float(os.getenv("TRAIL_ATR_MULT_LATE", "1.0"))
FAILED_BREAKOUT_RSI = float(os.getenv("FAILED_BREAKOUT_RSI", "52.0"))

# --- Mid-bar management (heartbeat) ---
MGMT_ENABLE = os.getenv("MGMT_ENABLE", "true").lower() == "true"
MGMT_HEARTBEAT_SEC = int(os.getenv("MGMT_HEARTBEAT_SEC", "5"))

# ===================== LOGGING =====================

def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "fxify_trading_bot.log")
    logger = logging.getLogger("FXIFYBot")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    # Try to ensure UTF-8 to avoid console encode errors on Windows
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    ch = logging.StreamHandler(stream=sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()

# ===================== SIGNALS =====================

def _graceful_shutdown(signum, frame):
    logger.info(f"Received signal {signum}. Shutting down gracefully.")
    mt5.shutdown()
    sys.exit(0)

signal.signal(signal.SIGTERM, _graceful_shutdown)
signal.signal(signal.SIGINT, _graceful_shutdown)

# ===================== STATE =====================

def get_default_state() -> dict:
    return {
        "in_position": False,
        "position_qty": 0.0,
        "position_ticket": None,
        "entry_price": 0.0,
        "entry_atr": 0.0,
        "trade_high_price": 0.0,
        "trade_low_price": 0.0,
        "entry_time": None,
        "partial_profit_taken": False,
        "edge_buy_streak": 0,
        "edge_exit_streak": 0,
        "last_trade_ts": None,
        "trades_today": 0,
        "trades_day_str": None,
        "loss_streak": 0,
        "circuit_breaker_until": None,
        "last_stop_price": None,
        "initial_balance": None,
        "day_start_equity": None,
        "high_water_mark": None,
        "total_profit": 0.0,
    }

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                st = json.load(f)
                logger.info(f"Loaded state: {st}")
                return st
        except Exception as e:
            logger.critical(f"State load error: {e}")
            raise
    return get_default_state()

def save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.critical(f"State save error: {e}")
        raise

# ===================== TIME HELPERS =====================

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _day_key_iso(ts_iso: Optional[str]) -> Optional[datetime.date]:
    if not ts_iso:
        return None
    try:
        tz = pytz.timezone(DAY_RESET_TZ)
    except Exception:
        tz = pytz.UTC
    dt = datetime.fromisoformat(ts_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).date()

def _reset_day_counters_if_needed(state: dict) -> None:
    now_iso = _utcnow_iso()
    prev_iso = state.get("trades_day_str")
    prev_day = _day_key_iso(prev_iso)
    try:
        tz = pytz.timezone(DAY_RESET_TZ)
    except Exception:
        tz = pytz.UTC
    now_dt = datetime.now(timezone.utc).astimezone(tz)
    now_day = now_dt.date()
    if prev_day != now_day:
        state["trades_today"] = 0
        state["trades_day_str"] = now_iso
        account = mt5.account_info()
        if account:
            state["day_start_equity"] = account.equity
        save_state(state)

def _cooldown_over(state: dict, minutes: int) -> bool:
    last = state.get("last_trade_ts")
    if not last:
        return True
    return datetime.now(timezone.utc) - datetime.fromisoformat(last) >= timedelta(
        minutes=minutes
    )

def _sleep_to_next_bar(seconds_per_bar: int = SECONDS_PER_BAR) -> None:
    now = datetime.now(timezone.utc).timestamp()
    next_slot = math.ceil(now / seconds_per_bar) * seconds_per_bar
    sleep_s = max(5, int(next_slot - now) + 1)
    logger.info(f"Sleeping ~{sleep_s//60}m {sleep_s%60}s to next {TIMEFRAME_STR} bar...")
    time.sleep(sleep_s)

def _is_active_hour_utc(ts: Optional[datetime] = None, symbol: Optional[str] = None) -> bool:
    now = ts or datetime.now(timezone.utc)
    h = now.hour
    wd = now.weekday()
    s = (symbol or SYMBOL).upper()
    # Match common broker suffixes (e.g., .SIM) by substring
    if "USDJPY" in s:
        return (0 <= h <= 6) or (12 <= h <= 16)
    if "EURUSD" in s:
        return 6 <= h <= 16
    if "XAUUSD" in s:
        return 7 <= h <= 17
    if s in ("ETHUSD", "ETH-USD") or "ETH" in s:
        # Avoid Fri 22:00–Sun 22:00 UTC
        if (wd == 4 and h >= 22) or (wd == 5) or (wd == 6 and h < 22):
            return False
        return True
    return ACTIVE_HOUR_START <= h <= ACTIVE_HOUR_END

# ===================== MT5 UTILITIES =====================

def initialize_mt5() -> bool:
    """Initialize MT5 connection"""
    if MT5_PATH:
        if not mt5.initialize(MT5_PATH):
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False
    else:
        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False
    
    # Login
    if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
        if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
        logger.info(f"MT5 connected: Login={MT5_LOGIN}, Server={MT5_SERVER}")
    else:
        logger.info("MT5 initialized without explicit login (using terminal's account)")
    
    return True

def get_symbol_info(symbol: str) -> Optional[mt5.SymbolInfo]:
    """Get symbol information"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        # Try to find a broker variant (e.g., USDJPY.sim)
        variants = mt5.symbols_get(f"*{symbol}*") or []
        if variants:
            alt = variants[0].name
            logger.info(f"Using broker symbol variant: {alt}")
            symbol = alt
            symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return None
    # Update global symbol if variant was used
    global SYMBOL
    SYMBOL = symbol
    return symbol_info

# ===================== ORDER HELPERS =====================

def _order_send_with_fallback(request: Dict, symbol: str) -> Optional[mt5.TradeResult]:
    """Send order with filling mode fallbacks to handle broker-specific requirements.

    Tries RETURN first, then FOK, then IOC. Returns the first successful result
    (retcode == TRADE_RETCODE_DONE) or the last result if all fail.
    """
    fill_modes = [
        getattr(mt5, "ORDER_FILLING_RETURN", None),
        getattr(mt5, "ORDER_FILLING_FOK", None),
        getattr(mt5, "ORDER_FILLING_IOC", None),
    ]
    # Filter out Nones and keep unique order
    tried = []
    for fm in fill_modes:
        if fm is None or fm in tried:
            continue
        tried.append(fm)
        req = dict(request)
        req["type_filling"] = fm
        try:
            result = mt5.order_send(req)
        except Exception as e:
            logger.error(f"order_send exception for {symbol} with filling={fm}: {e}")
            continue
        if result is None:
            logger.error(f"Order send failed (None) for {symbol} with filling={fm}: {mt5.last_error()}")
            continue
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return result
        # Unsupported filling mode? 10030
        if result.retcode == getattr(mt5, "TRADE_RETCODE_INVALID_ORDER", 10030) or result.retcode == 10030:
            logger.warning(f"Filling mode unsupported for {symbol}, tried={fm}; retcode={result.retcode}. Trying next...")
            continue
        # Other error, but break to report
        logger.error(f"Order failed for {symbol}: retcode={result.retcode}, comment={getattr(result,'comment','')}, filling={fm}")
        return result
    return result if 'result' in locals() else None

# --- Volume helpers ---
def round_down_to_step(vol: float, step: float) -> float:
    try:
        return math.floor(float(vol) / float(step)) * float(step)
    except Exception:
        return 0.0

def get_pip_value(symbol: str, symbol_info: mt5.SymbolInfo) -> float:
    """Return the pip size in price units (not account currency).

    Example:
    - EURUSD: 1 pip = 0.0001
    - USDJPY: 1 pip = 0.01
    - XAUUSD: treat 1 pip = 0.01
    """
    up = symbol.upper()
    if "JPY" in up:
        return 0.01
    if "XAU" in up or "XAG" in up:
        return 0.01
    return 0.0001

def calculate_lot_size(
    symbol: str,
    symbol_info: mt5.SymbolInfo,
    account_equity: float,
    risk_pct: float,
    stop_distance_pips: float
) -> float:
    """
    Calculate lot size based on risk percentage and stop distance.
    
    For forex: Lot size = (Account Equity × Risk %) / (Stop Distance in Pips × Pip Value × Contract Size)
    """
    if stop_distance_pips <= 0:
        return 0.0
    
    risk_amount = account_equity * risk_pct
    pip_size = get_pip_value(symbol, symbol_info)

    # Value per pip for 1 lot in ACCOUNT CURRENCY (prefer MT5-provided tick values)
    value_per_pip_usd = None
    try:
        tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
        tick_value = float(getattr(symbol_info, "trade_tick_value", 0.0) or 0.0)
        if tick_size > 0 and tick_value > 0:
            # If 1 tick is worth `tick_value` in account currency per 1 lot,
            # then 1 pip is worth (pip_size / tick_size) * tick_value
            value_per_pip_usd = (pip_size / tick_size) * tick_value
    except Exception:
        value_per_pip_usd = None

    if not value_per_pip_usd or value_per_pip_usd <= 0:
        # Fallback: convert quote-currency pip value to USD using current price
        # This handles cases where tick_value metadata is missing or zero
        tick = mt5.symbol_info_tick(symbol)
        price = float(getattr(tick, "bid", 0.0) or getattr(tick, "last", 0.0) or getattr(tick, "ask", 0.0) or 0.0)
        contract_size = float(getattr(symbol_info, "trade_contract_size", 100000.0) or 100000.0)
        # Value per pip for 1 lot in quote currency is pip_size * contract_size
        value_per_pip_quote = pip_size * contract_size
        # If account currency is USD and quote is not USD, approximate conversion by dividing by price
        if price > 0:
            value_per_pip_usd = value_per_pip_quote / price
        else:
            # As a very last resort, assume 1:1 (will under/over-size; should rarely trigger)
            value_per_pip_usd = value_per_pip_quote

    # Calculate lot size from risk budget and stop distance (in pips)
    lot_size = risk_amount / max((stop_distance_pips * value_per_pip_usd), 1e-12)
    
    # Round to symbol's volume step
    lot_step = symbol_info.volume_step
    lot_size = round(lot_size / lot_step) * lot_step
    
    # Apply limits
    lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
    
    return lot_size

def get_current_spread_pips(symbol: str, symbol_info: mt5.SymbolInfo) -> float:
    """Get current spread in pips"""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 0.0
    spread = tick.ask - tick.bid
    pip_value = get_pip_value(symbol, symbol_info)
    return spread / pip_value

def _symbol_spread_limit(symbol: str) -> float:
    up = symbol.upper()
    base = MAX_SPREAD_PIPS
    if "JPY" in up:
        return min(base, 1.0)
    if up.startswith("EURUSD"):
        return min(base, 1.2)
    return base

# ===================== INDICATORS =====================

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def fetch_bars_and_indicators(symbol: str, timeframe: int, count: int = 500) -> Optional[pd.Series]:
    """
    Fetch bars from MT5 and compute all indicators.
    Returns latest bar as Series with all indicators.
    """
    try:
        # Fetch bars
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to fetch bars for {symbol}")
            return None
        
        # Convert to DataFrame
        bars = pd.DataFrame(rates)
        bars['time'] = pd.to_datetime(bars['time'], unit='s')
        bars.set_index('time', inplace=True)
        
        # Rename columns to match strategy
        bars.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'real_volume': 'real_volume'
        }, inplace=True)
        
        # Compute moving averages
        if P["use_ema"]:
            bars["short_ma"] = _ema(bars["close"], P["short"])
            bars["long_ma"] = _ema(bars["close"], P["long"])
            bars["trend_ma"] = _ema(bars["close"], P["trend"])
        else:
            bars["short_ma"] = bars["close"].rolling(P["short"]).mean()
            bars["long_ma"] = bars["close"].rolling(P["long"]).mean()
            bars["trend_ma"] = bars["close"].rolling(P["trend"]).mean()
        
        # Compute ATR, ADX, RSI
        bars["atr"] = compute_atr_wilder(bars, P["atr_period"])
        bars["adx"] = calculate_adx(bars, INDICATOR["adx_period"])
        bars["rsi"] = calculate_rsi(bars["close"], INDICATOR["rsi_period"])
        
        # Breakout indicators
        L = INDICATOR["breakout_lookback"]
        bars["highL"] = bars["high"].rolling(L).max()
        bars["prev_high20"] = bars["highL"].shift(1)
        bars["atr_median20"] = bars["atr"].rolling(20).median()
        
        # ADX slope
        K = INDICATOR["adx_slope_lookback"]
        bars["adx_slope"] = bars["adx"] - bars["adx"].shift(K)
        
        # ATR percentage
        bars["atr_pct"] = (bars["atr"] / bars["close"]).replace([np.inf, -np.inf], np.nan) * 100.0
        
        # ATR regime filter (optional)
        if ATR_REGIME_WINDOW > 0 and ATR_REGIME_PCT > 0:
            win = int(ATR_REGIME_WINDOW)
            pct = float(ATR_REGIME_PCT)
            def _pctile(x: pd.Series) -> float:
                x = x.dropna()
                if len(x) == 0:
                    return np.nan
                return float(np.percentile(x, pct))
            bars["atr_pct_thresh"] = (
                bars["atr_pct"].rolling(win, min_periods=max(10, win // 5))
                .apply(_pctile, raw=False).shift(1)
            )
        
        # MTF trend alignment (optional)
        if MTF_ENABLE:
            try:
                mtf_rates = mt5.copy_rates_from_pos(symbol, MTF_TF, 0, MTF_TREND_MA + 50)
                if mtf_rates is not None and len(mtf_rates) > 0:
                    mtf_df = pd.DataFrame(mtf_rates)
                    mtf_df['time'] = pd.to_datetime(mtf_df['time'], unit='s')
                    mtf_df.set_index('time', inplace=True)
                    mtf_trend = _ema(mtf_df['close'], MTF_TREND_MA)
                    bars["mtf_trend_ma"] = mtf_trend.reindex(bars.index, method="ffill")
            except Exception as e:
                logger.warning(f"MTF computation failed: {e}")
        
        # Get latest and previous bars
        latest = bars.iloc[-1].copy()
        prev = bars.iloc[-2]
        
        latest["prev_short_ma"] = prev["short_ma"]
        latest["vix_pct_change"] = 0.0  # No VIX for forex
        latest["bar_time"] = str(latest.name)
        
        # Compute edge score
        edge: EdgeResult = compute_edge_features_and_score(
            bars, latest, prev, 0.0, P["vix_spike_threshold"]
        )
        latest["edge_score"] = edge.score
        latest["edge_reasons"] = ", ".join(edge.reasons)
        
        return latest
        
    except Exception as e:
        logger.error(f"fetch_bars_and_indicators error: {e}")
        return None

# ===================== FXIFY COMPLIANCE =====================

def check_fxify_limits(state: dict, account: mt5.AccountInfo) -> Tuple[bool, str, str]:
    """
    Check if trading is allowed according to FXIFY rules.
    Returns (allowed, reason)
    """
    # Initialize initial balance if not set
    if state.get("initial_balance") is None:
        state["initial_balance"] = account.balance
        state["high_water_mark"] = account.balance
        save_state(state)
    
    initial_balance = state["initial_balance"]
    equity = account.equity
    
    # Check max total drawdown
    max_total_loss = initial_balance * FXIFY_MAX_TOTAL_LOSS_PCT
    total_drawdown = initial_balance - equity
    if total_drawdown >= max_total_loss:
        return False, f"Max total loss reached: ${total_drawdown:.2f} >= ${max_total_loss:.2f}", "OVERALL"
    
    # Check max daily drawdown
    if state.get("day_start_equity") is None:
        state["day_start_equity"] = equity
        save_state(state)
    
    day_start = state["day_start_equity"]
    max_daily_loss = day_start * FXIFY_MAX_DAILY_LOSS_PCT
    daily_drawdown = day_start - equity
    if daily_drawdown >= max_daily_loss:
        return False, f"Max daily loss reached: ${daily_drawdown:.2f} >= ${max_daily_loss:.2f}", "DAILY"
    
    # Check profit target (if reached, bot could notify but continue trading)
    profit_target = initial_balance * FXIFY_PROFIT_TARGET_PCT
    current_profit = equity - initial_balance
    if current_profit >= profit_target:
        logger.info(f"✅ PROFIT TARGET REACHED: ${current_profit:.2f} >= ${profit_target:.2f}")
    
    return True, "OK", "OK"

# ===================== TRADING BOT =====================

class FXIFYTradingBot:
    def __init__(self):
        if not initialize_mt5():
            raise RuntimeError("Failed to initialize MT5")
        
        self.symbol = SYMBOL
        self.symbol_info = get_symbol_info(SYMBOL)
        if self.symbol_info is None:
            raise RuntimeError(f"Symbol {SYMBOL} not available")
        
        self.state = load_state()
        self.account = mt5.account_info()
        if self.account is None:
            raise RuntimeError("Failed to get account info")
        
        # Adjust banner depending on FTMO wrapper
        _mode = "FTMO" if str(os.getenv("FTMO_MODE", "")).lower() == "true" else "FXIFY"
        logger.info(f"🤖 {_mode} Trading Bot initialized")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Timeframe: {TIMEFRAME_STR}")
        logger.info(f"Account: {self.account.login}, Balance: ${self.account.balance:.2f}")
        logger.info(f"Risk per trade: {PORTFOLIO_RISK_PCT*100:.2f}%")
        logger.info(f"Max daily loss: {FXIFY_MAX_DAILY_LOSS_PCT*100:.1f}%, Max total loss: {FXIFY_MAX_TOTAL_LOSS_PCT*100:.1f}%")
        try:
            logger.info(
                f"Symbol spec: digits={self.symbol_info.digits}, point={self.symbol_info.point}, contract_size={self.symbol_info.trade_contract_size}"
            )
        except Exception:
            pass
        # Broker-first state reconciliation on startup
        try:
            pos = self.get_position()
            if pos is not None:
                # Rebuild state from broker position
                self.state["in_position"] = True
                self.state["position_ticket"] = pos.ticket
                self.state["position_qty"] = float(getattr(pos, "volume", 0.0) or 0.0)
                self.state["entry_price"] = float(getattr(pos, "price_open", 0.0) or 0.0)
                # If entry ATR missing, fetch a bar to initialize it
                if float(self.state.get("entry_atr", 0.0) or 0.0) <= 0.0:
                    _latest = fetch_bars_and_indicators(self.symbol, TIMEFRAME)
                    if _latest is not None:
                        self.state["entry_atr"] = float(_latest.get("atr", 0.0) or 0.0)
                # Initialize trade high/low around entry if missing
                ep = float(self.state.get("entry_price", 0.0) or 0.0)
                if float(self.state.get("trade_high_price", 0.0) or 0.0) <= 0.0:
                    self.state["trade_high_price"] = ep
                if float(self.state.get("trade_low_price", 0.0) or 0.0) <= 0.0:
                    self.state["trade_low_price"] = ep
                logger.info(
                    f"Reconciled from broker: ticket={pos.ticket}, price_open={self.state['entry_price']:.5f}, volume={self.state['position_qty']}"
                )
                save_state(self.state)
            else:
                # Broker flat: clear any stale local state
                if self.state.get("in_position"):
                    logger.warning("Broker shows flat but state had in_position=True; clearing local state")
                self.state["in_position"] = False
                self.state["position_qty"] = 0.0
                self.state["position_ticket"] = None
                self.state["entry_price"] = 0.0
                self.state["entry_atr"] = 0.0
                self.state["trade_high_price"] = 0.0
                self.state["trade_low_price"] = 0.0
                save_state(self.state)
        except Exception as _e:
            logger.warning(f"Startup reconciliation skipped: {_e}")
        # Clean up any stray pending orders for this symbol/magic
        try:
            self.cleanup_pending_orders()
        except Exception as _e:
            logger.debug(f"Pending order cleanup skipped: {_e}")
        # Pre-flight: surface common MT5 toggles that block trading
        try:
            ti = mt5.terminal_info()
            if ti and hasattr(ti, "trade_allowed") and not ti.trade_allowed:
                logger.warning("AutoTrading is disabled in MT5. Enable the Algo Trading button (Ctrl+E) and check Tools > Options > Expert Advisors > 'Allow automated trading'.")
            ai = mt5.account_info()
            if ai and hasattr(ai, "trade_allowed") and not ai.trade_allowed:
                logger.warning("Trading not allowed on this account (trade_allowed=false). Check account permissions and market hours.")
        except Exception:
            pass
    
        # In-memory spread history for percentile-based gating
        self._spread_hist: list[tuple[float, float]] = []  # list of (epoch_seconds, spread_pips)

    # ----- Margin helpers -----
    def _fit_volume_to_margin(self, volume: float, order_type: int, price: float) -> float:
        """Reduce volume by one step until required margin fits free margin, or min volume.

        Returns the adjusted volume (>= volume_min) that fits, or the current minimum if nothing fits.
        """
        try:
            acc = self.get_account_info()
            if acc is None:
                return volume
            free_margin = float(getattr(acc, "margin_free", 0.0) or getattr(acc, "free_margin", 0.0) or 0.0)
            step = max(self.symbol_info.volume_step, EPS)
            vol = max(self.symbol_info.volume_min, min(volume, self.symbol_info.volume_max))
            # headroom buffer
            while vol >= self.symbol_info.volume_min:
                try:
                    req_margin = mt5.order_calc_margin(order_type, self.symbol, vol, price)
                except Exception:
                    req_margin = None
                if req_margin is None or req_margin <= 0:
                    break
                if req_margin <= free_margin * 0.98:
                    break
                vol = max(self.symbol_info.volume_min, vol - step)
                if abs(vol - self.symbol_info.volume_min) < 1e-9:
                    break
            return vol
        except Exception:
            return volume

    def _current_bar_key(self) -> str:
        """Return ISO timestamp for the current bar start (UTC) to guard once-per-bar actions."""
        now_ts = datetime.now(timezone.utc).timestamp()
        start_ts = math.floor(now_ts / SECONDS_PER_BAR) * SECONDS_PER_BAR
        return datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()

    def get_account_info(self) -> Optional[mt5.AccountInfo]:
        """Refresh account info"""
        self.account = mt5.account_info()
        return self.account
    
    def get_position(self) -> Optional[mt5.TradePosition]:
        """Get current position for this symbol"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None or len(positions) == 0:
            return None
        # Filter by magic number
        for pos in positions:
            if pos.magic == MAGIC_NUMBER:
                return pos
        return None

    def cleanup_pending_orders(self) -> None:
        """Cancel any pending orders for this symbol placed by this bot (by magic)."""
        try:
            orders = mt5.orders_get(symbol=self.symbol)
        except Exception:
            orders = None
        if not orders:
            return
        removed = 0
        for o in orders:
            try:
                if hasattr(o, "magic") and o.magic != MAGIC_NUMBER:
                    continue
                req = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": o.ticket,
                    "symbol": self.symbol,
                    "magic": MAGIC_NUMBER,
                    "comment": "cleanup_pending",
                }
                res = mt5.order_send(req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    removed += 1
                else:
                    logger.debug(f"Pending order remove failed: ticket={o.ticket}, retcode={getattr(res,'retcode',None)}")
            except Exception as _e:
                logger.debug(f"Error removing pending order: {_e}")
        if removed:
            logger.info(f"Cleaned up {removed} pending order(s)")

    def _update_spread_history(self, spread_pips: float, window_minutes: int = 60) -> Optional[float]:
        """Track spread history and return the p20 threshold over the recent window in pips.

        Returns None until enough samples (>= 5) are collected in the window.
        """
        try:
            now_ts = time.time()
            self._spread_hist.append((now_ts, float(spread_pips)))
            # Drop samples older than window
            cutoff = now_ts - window_minutes * 60
            self._spread_hist = [(t, s) for (t, s) in self._spread_hist if t >= cutoff]
            vals = [s for (_, s) in self._spread_hist]
            if len(vals) < 5:
                return None
            return float(np.percentile(vals, 20.0))
        except Exception:
            return None
    
    def close_position(self, position: mt5.TradePosition, reason: str = "") -> bool:
        """Close position"""
        logger.info(f"🔄 Closing position: {reason}")
        
        # Determine order type (opposite of position)
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        # Prepare close request
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error("Failed to get tick for close")
            return False
        
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": f"close_{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = _order_send_with_fallback(request, self.symbol)
        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: retcode={result.retcode}, comment={getattr(result,'comment','')}")
            return False
        
        # Calculate P&L
        pnl = position.profit
        logger.info(f"✅ Position closed: P&L=${pnl:.2f}, Price={price:.5f}")
        
        # Update state
        self.state["in_position"] = False
        self.state["position_qty"] = 0.0
        self.state["position_ticket"] = None
        self.state["entry_price"] = 0.0
        self.state["trade_high_price"] = 0.0
        self.state["trade_low_price"] = 0.0
        self.state["partial_profit_taken"] = False
        self.state["last_trade_ts"] = _utcnow_iso()
        
        # Update profit tracking
        self.state["total_profit"] = self.state.get("total_profit", 0.0) + pnl
        
        # Track win/loss streak
        if pnl < 0:
            self.state["loss_streak"] = self.state.get("loss_streak", 0) + 1
            logger.warning(f"Loss streak: {self.state['loss_streak']}")
        else:
            self.state["loss_streak"] = 0
        
        save_state(self.state)
        return True

    def partial_close(self, position: mt5.TradePosition, fraction: float, reason: str = "") -> bool:
        """Close a fraction of the current position volume."""
        fraction = max(0.0, min(fraction, 1.0))
        if fraction <= 0:
            return False
        # Refresh latest position to avoid using stale volume
        pos = self.get_position() or position
        step = max(self.symbol_info.volume_step, EPS)
        # Floor to step and cap to <= position volume
        target = min(float(pos.volume) * fraction, float(pos.volume))
        vol = max(step, round_down_to_step(target, step))
        # Ensure within [min, pos.volume]
        vol = min(float(pos.volume), max(self.symbol_info.volume_min, vol))
        # Guard against edge cases where floor lands at == pos.volume but broker rejects
        if vol >= float(pos.volume):
            vol = max(self.symbol_info.volume_min, math.floor((float(pos.volume) - step) / step) * step)
        if vol <= 0 or vol > float(pos.volume):
            logger.debug(f"Partial close skipped: computed vol={vol} for pos={pos.volume}")
            return False
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        tries = 0
        while tries < 3 and vol >= self.symbol_info.volume_min and vol <= float(pos.volume):
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": vol,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": MAGIC_NUMBER,
                "comment": f"partial_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = _order_send_with_fallback(req, self.symbol)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✂️ Partial close {vol:.2f} lots for {reason}")
                # Update state qty from broker to avoid drift
                try:
                    latest_pos = self.get_position()
                    if latest_pos:
                        self.state["position_qty"] = float(latest_pos.volume)
                    else:
                        self.state["position_qty"] = max(0.0, float(self.state.get("position_qty", 0.0)) - vol)
                    save_state(self.state)
                except Exception:
                    pass
                return True
            # Handle volume-too-large by stepping down and retrying
            rc = getattr(res, 'retcode', None)
            if rc in (10038,):  # Volume to be closed exceeds the position volume
                vol = max(self.symbol_info.volume_min, vol - step)
                tries += 1
                continue
            logger.warning(f"Partial close failed: {getattr(res,'comment','unknown')} (retcode={rc})")
            break
        return False
    
    def open_position(self, direction: str, latest: pd.Series) -> bool:
        """
        Open a position (BUY or SELL)
        """
        account = self.get_account_info()
        if account is None:
            return False
        
        # Calculate stop loss distance
        atr = latest["atr"]
        stop_distance_atr = P["trailing_stop_atr_mult"] * atr
        pip_value = get_pip_value(self.symbol, self.symbol_info)
        stop_distance_pips = stop_distance_atr / pip_value
        
        # Adaptive risk sizing
        # Base risk
        risk_pct = RISK_PCT_BASE
        if latest["adx"] > ADX_RISK_THR and latest["atr_pct"] > ATR_PCT_RISK_THR:
            risk_pct = RISK_PCT_STRONG
        # Structure quality multiplier
        try:
            adx_term = max(min((float(latest["adx"]) - ADX_ENTRY_BASE) / 10.0, 1.0), 0.25)
            atr_term = max(min(float(latest["atr_pct"]) / max(ATR_SIZE_REF, EPS), 1.0), 0.5)
            size_multiplier = adx_term * atr_term
        except Exception:
            size_multiplier = 1.0
        risk_pct = risk_pct * size_multiplier
        logger.info(f"Sizing: base={RISK_PCT_BASE*100:.2f}% -> adj={risk_pct*100:.2f}% (mult={size_multiplier:.2f})")
        
        # Calculate intended total lot size
        intended_lots = calculate_lot_size(
            self.symbol,
            self.symbol_info,
            account.equity,
            risk_pct,
            stop_distance_pips
        )
        # JPY cap unless unusually tight stops (<= 0.8 ATR)
        if "JPY" in self.symbol.upper() and P["trailing_stop_atr_mult"] > 0.8:
            intended_lots = min(intended_lots, MAX_JPY_LOTS)

        if intended_lots < self.symbol_info.volume_min:
            logger.warning(f"Lot size {intended_lots} below minimum {self.symbol_info.volume_min}")
            return False
        # Two-stage entry sizing
        lot_step = self.symbol_info.volume_step
        stage1_lots = max(self.symbol_info.volume_min, round((intended_lots * FIRST_ENTRY_PCT) / lot_step) * lot_step)
        stage2_lots = max(0.0, intended_lots - stage1_lots)
        
        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error("Failed to get tick")
            return False
        
        # Determine order type and price
        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            stop_loss = price - stop_distance_atr
            take_profit = 0.0  # We'll use trailing stop
        else:  # SELL
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            stop_loss = price + stop_distance_atr
            take_profit = 0.0
        
        # Prepare order request (stage 1)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": stage1_lots,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": f"edge_{latest['edge_score']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        logger.info(f"📈 Opening {direction} position: Size={stage1_lots} lots (of {intended_lots:.2f}), Entry={price:.5f}, SL={stop_loss:.5f}")
        
        result = _order_send_with_fallback(request, self.symbol)
        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: retcode={result.retcode}, comment={getattr(result,'comment','')}")
            return False
        
        # Resolve actual executed price (result.price may be 0 with some filling modes)
        exec_price = float(getattr(result, "price", 0.0) or 0.0)
        # Poll briefly for the position to appear with a non-zero price_open
        pos_after = None
        if exec_price <= 0.0:
            for _ in range(10):  # up to ~2s
                pos_after = self.get_position()
                if pos_after is not None and float(getattr(pos_after, "price_open", 0.0) or 0.0) > 0.0:
                    exec_price = float(pos_after.price_open)
                    break
                time.sleep(0.2)
        else:
            pos_after = self.get_position()
        # If still zero, fallback to request price
        if exec_price <= 0.0:
            exec_price = float(price)
        logger.info(f"✅ Position opened: Ticket={result.order}, Price={exec_price:.5f}")
        
        # Update state
        self.state["in_position"] = True
        self.state["position_qty"] = stage1_lots
        self.state["position_ticket"] = result.order
        self.state["entry_price"] = exec_price
        self.state["entry_atr"] = atr
        self.state["trade_high_price"] = exec_price
        self.state["trade_low_price"] = exec_price
        self.state["entry_time"] = _utcnow_iso()
        self.state["trades_today"] = self.state.get("trades_today", 0) + 1
        self.state["last_stop_price"] = stop_loss
        # Track initial risk dollars for pyramiding caps
        self.state["initial_risk"] = account.equity * risk_pct
        self.state["added_risk"] = 0.0
        # Save entry metrics for defense rules and stage-2 logic
        try:
            self.state["entry_adx"] = float(latest["adx"])  
            self.state["entry_edge"] = float(latest.get("edge_score", 0.0))
            self.state["entry_bar_high"] = float(latest.get("high", exec_price))
            self.state["entry_bar_low"] = float(latest.get("low", exec_price))
        except Exception:
            pass
        # Stage-2 tracking
        self.state["pending_stage2"] = stage2_lots > 0
        self.state["stage2_lots"] = stage2_lots
        self.state["stage2_done"] = False

        save_state(self.state)
        return True

    def maybe_add_pyramid(self, position: mt5.TradePosition, latest: pd.Series) -> None:
        """Add micro position after +1R, respecting caps."""
        if not PYRAMID_ENABLE:
            return
        # Compute current R multiple
        entry_price = self.state.get("entry_price", 0.0)
        entry_atr = self.state.get("entry_atr", 0.0)
        initial_r = entry_atr * P["trailing_stop_atr_mult"]
        price = latest["close"]
        if position.type == mt5.POSITION_TYPE_BUY:
            r_mult = (price - entry_price) / max(initial_r, EPS)
        else:
            r_mult = (entry_price - price) / max(initial_r, EPS)
        if r_mult < 1.0:
            return
        # Spread-aware block for adds
        try:
            sp = get_current_spread_pips(self.symbol, self.symbol_info)
            if sp > ADD_SPREAD_BLOCK_PIPS:
                logger.info(f"Add blocked by spread {sp:.2f} > {ADD_SPREAD_BLOCK_PIPS:.2f}")
                return
        except Exception:
            pass
        # Spread and session checks already enforced elsewhere for entries; reuse here implicitly
        account = self.get_account_info()
        if account is None:
            return
        equity = account.equity
        # Determine headroom
        current_sym_risk = float(self.state.get("initial_risk", 0.0)) + float(self.state.get("added_risk", 0.0))
        sym_cap = PYRAMID_MAX_TOTAL_RISK * equity
        conc_cap = CONCURRENT_RISK_CAP * equity
        headroom = min(sym_cap - current_sym_risk, conc_cap - current_sym_risk)
        step_risk = min(PYRAMID_STEP_RISK * equity, headroom)
        if step_risk <= 0:
            return
        # Convert step risk to quantity using the same risk sizing logic as entries
        atr = latest["atr"]
        pip_size = get_pip_value(self.symbol, self.symbol_info)
        stop_distance_pips = (P["trailing_stop_atr_mult"] * atr) / max(pip_size, EPS)
        # Use calculate_lot_size by expressing step_risk as a fraction of equity
        risk_pct_for_step = step_risk / max(equity, EPS)
        qty_add = calculate_lot_size(
            self.symbol,
            self.symbol_info,
            equity,
            risk_pct_for_step,
            stop_distance_pips
        )
        # Respect volume step/min/max
        lot_step = self.symbol_info.volume_step
        qty_add = round(qty_add / lot_step) * lot_step
        qty_add = max(self.symbol_info.volume_min, min(qty_add, self.symbol_info.volume_max))
        # Optional: cap JPY adds per-order as well for safety
        if "JPY" in self.symbol.upper() and P["trailing_stop_atr_mult"] > 0.8:
            qty_add = min(qty_add, MAX_JPY_LOTS)
        if qty_add <= 0:
            return
        # Send add order in same direction
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        # Attempt add with fallback: step down on 'No money' (10019)
        tries = 0
        base_comment = "pyramid_add"
        while tries < 3 and qty_add >= self.symbol_info.volume_min:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self._fit_volume_to_margin(qty_add, order_type, price),
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": MAGIC_NUMBER,
                "comment": base_comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = _order_send_with_fallback(request, self.symbol)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"➕ Pyramid add: {request['volume']} lots at {getattr(result, 'price', price):.5f}")
                self.state["added_risk"] = float(self.state.get("added_risk", 0.0)) + step_risk
                self.state["position_qty"] = float(self.state.get("position_qty", 0.0)) + float(request['volume'])
                save_state(self.state)
                break
            rc = getattr(result, 'retcode', None)
            cm = getattr(result, 'comment', '') if result else ''
            if rc == 10019 or (isinstance(cm, str) and cm.lower().startswith("no money")):
                # step down one volume_step and retry
                prev = qty_add
                qty_add = max(self.symbol_info.volume_min, qty_add - lot_step)
                logger.warning(f"Pyramid add 'No money' — stepping down volume {prev:.2f} -> {qty_add:.2f} and retrying ({tries+1}/3)")
                tries += 1
                continue
            # other error: log once and stop
            logger.warning(f"Pyramid add failed: {cm if cm else 'unknown'} (retcode={rc})")
            break
    
    def update_trailing_stop(self, position: mt5.TradePosition, latest: pd.Series) -> None:
        """Update trailing stop loss"""
        current_price = latest["close"]
        atr = latest["atr"]
        entry_price = self.state["entry_price"]
        entry_atr = self.state["entry_atr"]
        
        # Calculate current R-multiple vs initial risk distance
        initial_r = entry_atr * P["trailing_stop_atr_mult"]
        if position.type == mt5.POSITION_TYPE_BUY:
            r_mult = (current_price - entry_price) / max(initial_r, EPS)
        else:
            r_mult = (entry_price - current_price) / max(initial_r, EPS)

        # Promote to quasi-breakeven earlier
        if r_mult >= BE_PROMOTE_R:
            pad = BE_PADDING_ATR * atr
            be_stop = entry_price - pad if position.type == mt5.POSITION_TYPE_BUY else entry_price + pad
            # If spread is wide, keep padded stop
            try:
                sp = get_current_spread_pips(self.symbol, self.symbol_info)
            except Exception:
                sp = 0.0
            if sp < (ADD_SPREAD_BLOCK_PIPS - 0.2):  # if spread normal, allow tighter
                be_stop = entry_price
            if (position.type == mt5.POSITION_TYPE_BUY and position.sl < be_stop) or (
                position.type == mt5.POSITION_TYPE_SELL and position.sl > be_stop
            ):
                self._modify_position_sl(position, be_stop)

        # Start trailing after configured R
        if r_mult >= TRAIL_AFTER_R:
            trail_mult = TRAIL_ATR_MULT_LATE
            trail_distance = trail_mult * atr
            if position.type == mt5.POSITION_TYPE_BUY:
                new_stop = current_price - trail_distance
                if new_stop > position.sl:
                    self._modify_position_sl(position, new_stop)
            else:
                new_stop = current_price + trail_distance
                if new_stop < position.sl:
                    self._modify_position_sl(position, new_stop)
    
    def _modify_position_sl(self, position: mt5.TradePosition, new_sl: float) -> bool:
        """Modify position stop loss"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": position.ticket,
            "sl": new_sl,
            "tp": position.tp,
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"🔄 Stop loss updated: {new_sl:.5f}")
            self.state["last_stop_price"] = new_sl
            save_state(self.state)
            return True
        else:
            logger.warning(f"Failed to update SL: {result.comment if result else 'unknown'}")
            return False
    
    def run_iteration(self) -> None:
        """Single iteration of the trading loop"""
        logger.info("=" * 80)
        logger.info(f"🔄 Starting iteration at {datetime.now()}")
        # Remove any stale pending orders each loop for reliability
        try:
            self.cleanup_pending_orders()
        except Exception:
            pass
        
        # Reset daily counters if needed
        _reset_day_counters_if_needed(self.state)
        
        # Get account info
        account = self.get_account_info()
        if account is None:
            logger.error("Failed to get account info")
            return
        
        # Check FXIFY compliance
        allowed, reason, code = check_fxify_limits(self.state, account)
        if not allowed:
            logger.error(f"❌ Trading stopped: {reason}")
            # Close any open positions
            position = self.get_position()
            if position:
                self.close_position(position, "FXIFY_LIMIT")
            if code == "OVERALL":
                logger.error("Max total loss breached. Halting bot.")
                mt5.shutdown()
                sys.exit(0)
            return

        # Daily stop handled via check_fxify_limits (uses FXIFY_MAX_DAILY_LOSS_PCT and DAY_RESET_TZ)
        
        # Check circuit breaker
        if CB_ENABLED and self.state.get("circuit_breaker_until"):
            cb_until = datetime.fromisoformat(self.state["circuit_breaker_until"])
            if datetime.now(timezone.utc) < cb_until:
                logger.warning(f"Circuit breaker active until {cb_until}")
                return
        
        # Fetch bars and indicators
        latest = fetch_bars_and_indicators(self.symbol, TIMEFRAME)
        if latest is None:
            logger.error("Failed to fetch bars")
            return
        
        logger.info(f"📊 Price: {latest['close']:.5f}, ATR: {latest['atr']:.5f}, ADX: {latest['adx']:.1f}, RSI: {latest['rsi']:.1f}")
        logger.info(f"🎯 Edge Score: {latest['edge_score']}/100 - {latest['edge_reasons']}")
        
        # Determine if we are in off-hours; if ALWAYS_ACTIVE, bypass off-hours gates
        is_active_now = _is_active_hour_utc(symbol=self.symbol)
        offhours = False
        if not is_active_now and not ALWAYS_ACTIVE:
            offhours = True
            logger.info("Outside active hours; applying off-hours safeguards")

        # Check spread (tighten caps in off-hours)
        spread_pips = get_current_spread_pips(self.symbol, self.symbol_info)
        logger.info(f"Spread: {spread_pips:.2f} pips")
        limit = _symbol_spread_limit(self.symbol)
        if offhours:
            up = self.symbol.upper()
            if "JPY" in up:
                limit = min(limit, OFFHOURS_SPREAD_JPY)
            elif up.startswith("EURUSD"):
                limit = min(limit, OFFHOURS_SPREAD_EURUSD)
            logger.info(f"Off-hours spread cap: {limit:.2f} pips")
        # Allow equality to pass to avoid false blocks at exact cap
        if limit > 0 and spread_pips > limit:
            logger.warning(f"Spread too high: {spread_pips:.2f} > {limit:.2f} pips")
            return
        # Percentile gate over last ~60 minutes (entry-only)
        block_entries_due_to_percentile = False
        p20 = self._update_spread_history(spread_pips, window_minutes=60)
        if p20 is not None and spread_pips > p20:
            logger.info(f"Spread percentile gate: {spread_pips:.2f} pips > p20={p20:.2f} pips; entries will be skipped this bar")
            block_entries_due_to_percentile = True
        
        # Get current position
        position = self.get_position()
        
        # POSITION MANAGEMENT
        if position:
            # Friendly position HUD: direction and lots
            try:
                dir_str = "BUY" if position.type == mt5.POSITION_TYPE_BUY else "SELL"
            except Exception:
                dir_str = str(position.type)
            logger.info(f"📌 In position: {dir_str} {position.volume:.2f} lots, P&L: ${position.profit:.2f}")
            # Repair zero entry_price if needed
            try:
                if float(self.state.get("entry_price", 0.0) or 0.0) <= 0.0 and float(getattr(position, "price_open", 0.0) or 0.0) > 0.0:
                    self.state["entry_price"] = float(position.price_open)
                    logger.info(f"Repaired missing entry_price from MT5 position: {self.state['entry_price']:.5f}")
                    save_state(self.state)
            except Exception:
                pass
            
            # Stage-2 add if pending and confirmation met (ADX rising, close > entry bar high, spread ok)
            try:
                if bool(self.state.get("pending_stage2")) and not bool(self.state.get("stage2_done")):
                    stage2_lots = float(self.state.get("stage2_lots", 0.0) or 0.0)
                    entry_adx = float(self.state.get("entry_adx", 0.0) or 0.0)
                    entry_high = float(self.state.get("entry_bar_high", 0.0) or 0.0)
                    adx_now = float(latest.get("adx", 0.0) or 0.0)
                    close_now = float(latest.get("close", 0.0) or 0.0)
                    sp_now = get_current_spread_pips(self.symbol, self.symbol_info)
                    rsi_now = float(latest.get("rsi", 0.0) or 0.0)
                    # Softer add gate: allow add when spread <= min(0.5, p35) for JPY, else use configured add cap
                    base_add_cap = 0.5 if "JPY" in self.symbol.upper() else ADD_SPREAD_BLOCK_PIPS
                    p35 = None
                    try:
                        now_ts = time.time()
                        vals = [s for (t, s) in self._spread_hist if (now_ts - t) <= 60*60]
                        if len(vals) >= 5:
                            p35 = float(np.percentile(vals, 35.0))
                    except Exception:
                        p35 = None
                    spread_thr = base_add_cap if p35 is None else min(base_add_cap, p35)
                    if stage2_lots > 0 and adx_now >= entry_adx and close_now > entry_high and rsi_now >= 55.0 and sp_now <= spread_thr:
                        tick = mt5.symbol_info_tick(self.symbol)
                        if tick is not None:
                            order_type = mt5.ORDER_TYPE_BUY if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_SELL
                            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
                            # Margin preflight
                            adj_lots = self._fit_volume_to_margin(stage2_lots, order_type, price)
                            if adj_lots < self.symbol_info.volume_min:
                                logger.warning("Stage-2 add skipped: insufficient margin for minimum volume")
                                raise Exception("stage2_margin_skip")
                            req = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": self.symbol,
                                "volume": adj_lots,
                                "type": order_type,
                                "price": price,
                                "deviation": 20,
                                "magic": MAGIC_NUMBER,
                                "comment": "stage2_add",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_IOC,
                            }
                            res = _order_send_with_fallback(req, self.symbol)
                            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                logger.info(f"➕ Stage-2 add {adj_lots:.2f} lots confirmed")
                                self.state["position_qty"] = float(self.state.get("position_qty", 0.0)) + adj_lots
                                self.state["stage2_done"] = True
                                self.state["pending_stage2"] = False
                                save_state(self.state)
            except Exception as _e:
                logger.debug(f"Stage-2 add check failed: {_e}")

            # Early defense rules
            try:
                entry_ts = datetime.fromisoformat(self.state.get("entry_time")) if self.state.get("entry_time") else None
                if entry_ts is not None:
                    age_sec = (datetime.now(timezone.utc) - entry_ts).total_seconds()
                    early_sec = EARLY_WINDOW_BARS * SECONDS_PER_BAR
                    entry_edge = float(self.state.get("entry_edge", 0.0) or 0.0)
                    entry_adx = float(self.state.get("entry_adx", 0.0) or 0.0)
                    edge_now = float(latest.get("edge_score", 0.0) or 0.0)
                    adx_now = float(latest.get("adx", 0.0) or 0.0)
                    # One-trim-per-bar guard
                    bar_key = self._current_bar_key()
                    last_trim_bar = self.state.get("last_trim_bar")
                    if age_sec <= early_sec:
                        if (edge_now <= entry_edge - EDGE_DROP_TRIM) or (adx_now < entry_adx - ADX_DROP_TRIM):
                            if last_trim_bar != bar_key and self.partial_close(position, TRIM_EARLY_FRACTION, "early_defense"):
                                self.state["last_trim_bar"] = bar_key
                                self.state["last_trim_ts"] = _utcnow_iso()
                                save_state(self.state)
                    # Failed breakout: next bar closes below entry bar low and RSI < threshold
                    entry_low = float(self.state.get("entry_bar_low", 0.0) or 0.0)
                    if age_sec <= SECONDS_PER_BAR * 1.5 and latest.get("close", 0.0) < entry_low and float(latest.get("rsi", 100.0)) < FAILED_BREAKOUT_RSI:
                        self.close_position(position, "failed_breakout")
                        return
            except Exception as _e:
                logger.debug(f"Defense checks failed: {_e}")

            # ADX floor exit after consecutive bars
            try:
                adx_now = float(latest.get("adx", 100.0))
                st = int(self.state.get("adx_below_floor_streak", 0) or 0)
                if adx_now < ADX_FLOOR_EXIT:
                    st += 1
                else:
                    st = 0
                self.state["adx_below_floor_streak"] = st
                if st >= 2:
                    self.close_position(position, "adx_floor_exit")
                    return
            except Exception:
                pass

            # Drawdown fuse based on initial risk dollars
            try:
                init_risk = float(self.state.get("initial_risk", 0.0) or 0.0)
                unrealized_loss = max(0.0, -float(position.profit))
                if init_risk > 0 and unrealized_loss >= DD_FUSE_FRAC * init_risk:
                    self.partial_close(position, DD_TRIM_FRAC, "dd_fuse")
            except Exception:
                pass

            # Update trailing stop
            self.update_trailing_stop(position, latest)
            # Consider pyramiding
            self.maybe_add_pyramid(position, latest)
            
            # Check for opportunistic exit
            if OPPORTUNISTIC_MODE:
                if latest["edge_score"] <= EDGE_EXIT_SCORE:
                    self.state["edge_exit_streak"] = self.state.get("edge_exit_streak", 0) + 1
                else:
                    self.state["edge_exit_streak"] = 0
                
                if self.state["edge_exit_streak"] >= EDGE_EXIT_CONFIRM_BARS:
                    logger.info(f"Edge deterioration confirmed: {self.state['edge_exit_streak']} bars")
                    self.close_position(position, "EDGE_EXIT")
                    return
        
        # ENTRY LOGIC
        else:
            logger.info("📭 No position")
            # Reconcile stale state: if state says in_position but MT5 shows flat, clear state
            if self.state.get("in_position"):
                logger.warning("State indicated in_position=True but no MT5 position found; clearing stale state")
                self.state["in_position"] = False
                self.state["position_qty"] = 0.0
                self.state["position_ticket"] = None
                self.state["entry_price"] = 0.0
                self.state["entry_time"] = None
                self.state["last_stop_price"] = None
                save_state(self.state)

            # Respect active trading hours, unless ALWAYS_ACTIVE is enabled
            if not is_active_now and not ALWAYS_ACTIVE:
                logger.info(f"Outside active hours (UTC {ACTIVE_HOUR_START:02d}-{ACTIVE_HOUR_END:02d}); entries paused")
                return
            
            # Check daily trade limit
            if self.state.get("trades_today", 0) >= MAX_TRADES_PER_DAY:
                logger.warning(f"Daily trade limit reached: {MAX_TRADES_PER_DAY}")
                return
            
            # Spread percentile gate (entry-only)
            if block_entries_due_to_percentile:
                logger.info("Entry blocked by spread percentile gate for this bar")
                return

            # Check cooldown
            if not _cooldown_over(self.state, MIN_COOLDOWN_MIN):
                logger.info(f"Cooldown period active ({MIN_COOLDOWN_MIN} min)")
                return
            
            # Check ATR regime filter
            if ATR_REGIME_WINDOW > 0 and ATR_REGIME_PCT > 0:
                atr_pct = latest.get("atr_pct", 0)
                atr_thresh = latest.get("atr_pct_thresh", 0)
                if pd.notna(atr_thresh) and atr_pct < atr_thresh:
                    logger.info(f"ATR regime filter: {atr_pct:.3f}% < {atr_thresh:.3f}%")
                    return
            
            # ATR band filter
            if latest.get("atr_pct") is None:
                logger.info("ATR% not available; skipping entry")
                return
            if latest["atr_pct"] < ATR_PCT_MIN or latest["atr_pct"] > ATR_PCT_MAX:
                if latest["atr_pct"] < ATR_PCT_MIN:
                    logger.info(f"ATR% too low: {latest['atr_pct']:.3f}% < {ATR_PCT_MIN:.3f}%")
                else:
                    logger.info(f"ATR% too high: {latest['atr_pct']:.3f}% > {ATR_PCT_MAX:.3f}%")
                return
            
            # Opportunistic entry with quality gates (ADX/MA/ATR)
            if OPPORTUNISTIC_MODE:
                # Quality gates
                adx_val = float(latest.get("adx", 0))
                used_adx_thr = max(P["adx_threshold"], OFFHOURS_ADX_MIN) if offhours else P["adx_threshold"]
                adx_ok = adx_val >= used_adx_thr
                short_ma = float(latest.get("short_ma", 0))
                long_ma = float(latest.get("long_ma", 0))
                price_now = float(max(latest.get("close", 1), EPS))
                ma_gap_raw = abs(short_ma - long_ma)
                ma_dist_bps = ma_gap_raw / price_now * 10000.0
                # Adaptive threshold selection (read enable flag at runtime to honor env overrides)
                adaptive_on = os.getenv("ADAPTIVE_MABPS_ENABLE", "false").lower() == "true"
                # atr_val is in percent (e.g., 0.05%), convert to bps by * 100 for dynamic floor calc
                atr_pct_val = float(latest.get("atr_pct", 0))
                dyn_floor = max(ADAPTIVE_MABPS_FLOOR_BPS, ADAPTIVE_MABPS_COEFF * (atr_pct_val * 100.0))
                if adaptive_on:
                    used_mabps_thr = max(MIN_MA_DIST_BPS, dyn_floor)
                else:
                    used_mabps_thr = MIN_MA_DIST_BPS
                # MA normalization relative to dynamic floor
                ma_norm = ma_dist_bps / max(dyn_floor, EPS)
                if offhours:
                    # Off-hours acceptance: either normalized MA distance is adequate or raw floor met
                    ma_ok = (ma_norm >= MA_NORM_MIN_OFFHOURS) or (ma_dist_bps >= used_mabps_thr)
                else:
                    ma_ok = ma_dist_bps >= used_mabps_thr
                atr_val = atr_pct_val
                atr_ok = (atr_val >= ATR_PCT_MIN) and (atr_val <= ATR_PCT_MAX)
                # Strict off-hours exception path: require stronger conditions
                if offhours:
                    strict_ok = True
                    if adx_val < max(used_adx_thr, OFFHOURS_STRICT_ADX_MIN):
                        logger.info(f"Off-hours strict: ADX {adx_val:.1f} < {max(used_adx_thr, OFFHOURS_STRICT_ADX_MIN)}")
                        strict_ok = False
                    if p20 is None or spread_pips > p20:
                        logger.info(
                            f"Off-hours strict: Spread {spread_pips:.2f} pips not <= p20 {p20:.2f} pips" if p20 is not None else "Off-hours strict: insufficient spread history for p20 gate"
                        )
                        strict_ok = False
                    if ma_norm < OFFHOURS_STRICT_MA_NORM:
                        logger.info(f"Off-hours strict: ma_norm {ma_norm:.2f} < {OFFHOURS_STRICT_MA_NORM}")
                        strict_ok = False
                else:
                    strict_ok = True
                gates_log = (
                    f"Gates: ADX {adx_val:.1f}>={used_adx_thr}:{adx_ok} | "
                    f"MAbps {ma_dist_bps:.2f}>={used_mabps_thr:.2f}:{ma_ok} (ma_norm={ma_norm:.2f}, short={short_ma:.5f}, long={long_ma:.5f}, gap={ma_gap_raw:.5f}) | "
                    f"ATR% band {ATR_PCT_MIN:.2f}%–{ATR_PCT_MAX:.2f}%: {atr_val:.3f}% -> {atr_ok}"
                )
                if not ALWAYS_ACTIVE:
                    gates_log += f" | Off-hours strict OK: {strict_ok}"
                logger.info(gates_log)
                # Write gates CSV for offline analysis
                try:
                    os.makedirs(LOG_DIR, exist_ok=True)
                    sym_sanitized = str(self.symbol).replace("/", "")
                    csv_path = os.path.join(LOG_DIR, f"gates_{sym_sanitized}.csv")
                    write_header = not os.path.exists(csv_path)
                    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        if write_header:
                            w.writerow([
                                "ts", "symbol", "price", "adx", "adx_thr", "adx_ok",
                                "ma_bps", "ma_thr", "ma_ok", "ma_norm", "atr_pct", "atr_min", "atr_max", "atr_ok", "edge_score"
                            ])
                        ts = None
                        try:
                            ts = latest.name.isoformat()
                        except Exception:
                            ts = _utcnow_iso()
                        w.writerow([
                            ts, self.symbol, price_now, round(adx_val, 2), used_adx_thr, adx_ok,
                            round(ma_dist_bps, 3), round(used_mabps_thr, 3), ma_ok, round(ma_norm, 3),
                            round(atr_val, 3), round(ATR_PCT_MIN, 3), round(ATR_PCT_MAX, 3), atr_ok, int(latest.get("edge_score", 0))
                        ])
                except Exception as _e:
                    logger.debug(f"gates CSV write failed: {_e}")
                if not (adx_ok and ma_ok and atr_ok and strict_ok):
                    self.state["edge_buy_streak"] = 0
                    # Log the actual MA threshold used (adaptive or static) for clarity
                    logger.info(
                        f"Quality gates blocked entry: ADX>={used_adx_thr}:{adx_ok}, "
                        f"MAbps>={used_mabps_thr}:{ma_ok}, ATR% band [{ATR_PCT_MIN},{ATR_PCT_MAX}]:{atr_ok}, Off-hours strict:{strict_ok}"
                    )
                    return
                # Use a potentially different buy score threshold during off-hours
                used_edge_buy_score = OFFHOURS_EDGE_BUY_SCORE if offhours else EDGE_BUY_SCORE
                if latest["edge_score"] >= used_edge_buy_score:
                    self.state["edge_buy_streak"] = self.state.get("edge_buy_streak", 0) + 1
                else:
                    self.state["edge_buy_streak"] = 0
                
                if self.state["edge_buy_streak"] >= EDGE_CONFIRM_BARS:
                    logger.info(f"✅ Edge entry signal confirmed: {self.state['edge_buy_streak']} bars")
                    # Optional ML gating
                    if ML_ENABLE:
                        try:
                            prob = predict_entry_prob(latest, ML_MODEL_PATH)
                            logger.info(f"ML gate prob={prob:.3f} (thr={ML_PROB_THR:.2f})")
                            if prob < ML_PROB_THR:
                                logger.info("ML gate blocked entry")
                                return
                        except Exception as e:
                            logger.warning(f"ML gating failed: {e}")
                            return
                    # Determine direction based on trend
                    direction = "BUY" if latest["close"] > latest["trend_ma"] else "SELL"
                    self.open_position(direction, latest)
                    self.state["edge_buy_streak"] = 0
                    return
            
            # Classic entry (golden cross + uptrend)
            else:
                uptrend = latest["close"] > latest["trend_ma"]
                golden = latest["short_ma"] > latest["long_ma"]
                adx_ok = latest["adx"] > P["adx_threshold"]
                
                if uptrend and golden and adx_ok:
                    logger.info("✅ Classic entry signal: Golden cross + uptrend + ADX")
                    # Optional ML gating
                    if ML_ENABLE:
                        try:
                            prob = predict_entry_prob(latest, ML_MODEL_PATH)
                            logger.info(f"ML gate prob={prob:.3f} (thr={ML_PROB_THR:.2f})")
                            if prob < ML_PROB_THR:
                                logger.info("ML gate blocked entry")
                                return
                        except Exception as e:
                            logger.warning(f"ML gating failed: {e}")
                            return
                    direction = "BUY" if uptrend else "SELL"
                    self.open_position(direction, latest)
                    return
        
        logger.info("✅ Iteration complete")
    
    def run(self) -> None:
        """Main trading loop"""
        _mode = "FTMO" if str(os.getenv("FTMO_MODE", "")).lower() == "true" else "FXIFY"
        logger.info(f"🚀 Starting {_mode} Trading Bot")
        logger.info(f"Symbol: {self.symbol}, Timeframe: {TIMEFRAME_STR}")
        logger.info(f"Opportunistic mode: {OPPORTUNISTIC_MODE}")
        logger.info(f"Edge buy score: {EDGE_BUY_SCORE}, Exit score: {EDGE_EXIT_SCORE}")
        # Surface session and off-hours configuration for clarity
        logger.info(
            f"Session window UTC {ACTIVE_HOUR_START:02d}-{ACTIVE_HOUR_END:02d}; ALWAYS_ACTIVE={ALWAYS_ACTIVE}"
        )
        if ALWAYS_ACTIVE:
            logger.info("Off-hours safeguards bypassed (ALWAYS_ACTIVE=True); gates still logged for diagnostics.")
            if OFFHOURS_EDGE_BUY_SCORE != EDGE_BUY_SCORE:
                logger.info(
                    f"Off-hours edge buy score configured: {OFFHOURS_EDGE_BUY_SCORE} (regular: {EDGE_BUY_SCORE})"
                )
        else:
            logger.info(
                f"Off-hours safeguards: ADX_MIN={OFFHOURS_ADX_MIN}, SPREAD_CAP (EURUSD)={OFFHOURS_SPREAD_EURUSD} pips, (JPY)={OFFHOURS_SPREAD_JPY} pips"
            )
        
        try:
            while True:
                try:
                    self.run_iteration()
                except Exception as e:
                    logger.error(f"Iteration error: {e}", exc_info=True)
                
                if LOOP_ONCE:
                    logger.info("LOOP_ONCE=true, exiting")
                    break
                
                self._sleep_with_management()
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            logger.info("Shutting down MT5")
            mt5.shutdown()

    def _sleep_with_management(self) -> None:
        """Sleep until next bar, running lightweight mid-bar risk management at a heartbeat interval."""
        now = datetime.now(timezone.utc).timestamp()
        next_slot = math.ceil(now / SECONDS_PER_BAR) * SECONDS_PER_BAR
        remain = int(max(5, next_slot - now) + 1)
        # Log once for visibility (matches previous behavior)
        logger.info(f"Sleeping ~{remain//60}m {remain%60}s to next {TIMEFRAME_STR} bar...")
        hb = max(2, MGMT_HEARTBEAT_SEC)
        while remain > 0:
            # Run mid-bar management if enabled
            if MGMT_ENABLE:
                try:
                    self.manage_midbar()
                except Exception as _e:
                    logger.debug(f"mid-bar management error: {_e}")
            sleep_s = min(hb, remain)
            time.sleep(sleep_s)
            now = datetime.now(timezone.utc).timestamp()
            remain = int(max(0, next_slot - now))

    def manage_midbar(self) -> None:
        """Lightweight management between bars: fuse trims and BE/trailing using entry ATR."""
        position = self.get_position()
        if not position:
            return
        # Refresh spread-aware BE padding and trailing using entry ATR (lightweight)
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return
            price = tick.bid if position.type == mt5.POSITION_TYPE_SELL else tick.ask
            entry_price = float(self.state.get("entry_price", 0.0) or 0.0)
            entry_atr = float(self.state.get("entry_atr", 0.0) or 0.0)
            if entry_price <= 0 or entry_atr <= 0:
                return
            initial_r = entry_atr * P["trailing_stop_atr_mult"]
            r_mult = (price - entry_price) / max(initial_r, EPS) if position.type == mt5.POSITION_TYPE_BUY else (entry_price - price) / max(initial_r, EPS)
            # DD fuse check (mid-bar)
            try:
                init_risk = float(self.state.get("initial_risk", 0.0) or 0.0)
                unrealized_loss = max(0.0, -float(position.profit))
                # One-trim-per-bar guard
                bar_key = self._current_bar_key()
                last_trim_bar = self.state.get("last_trim_bar")
                if init_risk > 0 and unrealized_loss >= DD_FUSE_FRAC * init_risk and last_trim_bar != bar_key:
                    if self.partial_close(position, DD_TRIM_FRAC, "dd_fuse_midbar"):
                        self.state["last_trim_bar"] = bar_key
                        self.state["last_trim_ts"] = _utcnow_iso()
                        save_state(self.state)
            except Exception:
                pass
            # Early window: mid-bar defense if adverse move exceeds threshold
            try:
                entry_time = self.state.get("entry_time")
                if entry_time:
                    et = datetime.fromisoformat(entry_time)
                    age_sec2 = (datetime.now(timezone.utc) - et).total_seconds()
                    if age_sec2 <= EARLY_WINDOW_BARS * SECONDS_PER_BAR:
                        # If trade moves against us by more than 0.4R in early window, trim once per bar
                        if r_mult <= -0.4:
                            bar_key2 = self._current_bar_key()
                            if self.state.get("last_trim_bar") != bar_key2:
                                if self.partial_close(position, TRIM_EARLY_FRACTION, "early_defense_midbar"):
                                    self.state["last_trim_bar"] = bar_key2
                                    self.state["last_trim_ts"] = _utcnow_iso()
                                    save_state(self.state)
            except Exception:
                pass
            # BE promote and simple trailing using entry ATR
            if r_mult >= BE_PROMOTE_R:
                pad = BE_PADDING_ATR * entry_atr
                be_stop = entry_price - pad if position.type == mt5.POSITION_TYPE_BUY else entry_price + pad
                try:
                    sp = get_current_spread_pips(self.symbol, self.symbol_info)
                except Exception:
                    sp = 0.0
                if sp < (ADD_SPREAD_BLOCK_PIPS - 0.2):
                    be_stop = entry_price
                if (position.type == mt5.POSITION_TYPE_BUY and position.sl < be_stop) or (position.type == mt5.POSITION_TYPE_SELL and position.sl > be_stop):
                    self._modify_position_sl(position, be_stop)
            if r_mult >= TRAIL_AFTER_R:
                trail_distance = TRAIL_ATR_MULT_LATE * entry_atr
                if position.type == mt5.POSITION_TYPE_BUY:
                    new_stop = price - trail_distance
                    if new_stop > position.sl:
                        self._modify_position_sl(position, new_stop)
                else:
                    new_stop = price + trail_distance
                    if new_stop < position.sl:
                        self._modify_position_sl(position, new_stop)
        except Exception as _e:
            logger.debug(f"manage_midbar error: {_e}")

# ===================== MAIN =====================

def main():
    try:
        bot = FXIFYTradingBot()
        bot.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        mt5.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()
