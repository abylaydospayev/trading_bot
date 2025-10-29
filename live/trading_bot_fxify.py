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
import sys
import time
import math
import json
import logging
import signal
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

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
    "adx_threshold": int(os.getenv("ADX_THRESHOLD", "20")),
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
MIN_MA_DIST_BPS = float(os.getenv("MIN_MA_DIST_BPS", "0"))
MTF_ENABLE = os.getenv("MTF_ENABLE", "false").lower() == "true"
MTF_TF_STR = os.getenv("MTF_TF", "1H")
MTF_TF = TIMEFRAME_MAP.get(MTF_TF_STR, mt5.TIMEFRAME_H1)
MTF_TREND_MA = int(os.getenv("MTF_TREND_MA", "200"))
MAX_SPREAD_PIPS = float(os.getenv("MAX_SPREAD_PIPS", "3"))  # Max spread in pips
ADX_SLOPE_MIN = float(os.getenv("ADX_SLOPE_MIN", "0"))

# --- Active trading window (UTC)
ACTIVE_HOUR_START = int(os.getenv("ACTIVE_HOUR_START", "6"))   # 06:00 UTC
ACTIVE_HOUR_END = int(os.getenv("ACTIVE_HOUR_END", "15"))      # 15:00 UTC
# Hard daily stop (-3%): block new entries for remainder of day
DAILY_STOP_PCT = float(os.getenv("DAILY_STOP_PCT", "0.03"))

# --- Misc ---
EPS = 1e-12
LOOP_ONCE = os.getenv("LOOP_ONCE", "false").lower() == "true"

# --- ML Gating (optional) ---
ML_ENABLE = os.getenv("ML_ENABLE", "false").lower() == "true"
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", os.path.join(ROOT_DIR, "ml", "models", "mlp_model.pkl"))
ML_PROB_THR = float(os.getenv("ML_PROB_THR", "0.6"))

# ===================== LOGGING =====================

def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "fxify_trading_bot.log")
    logger = logging.getLogger("FXIFYBot")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    ch = logging.StreamHandler()
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

def _same_day(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return datetime.fromisoformat(a).date() == datetime.fromisoformat(b).date()

def _reset_day_counters_if_needed(state: dict) -> None:
    now = _utcnow_iso()
    if not _same_day(state.get("trades_day_str"), now):
        state["trades_today"] = 0
        state["trades_day_str"] = now
        # Reset day_start_equity for daily drawdown tracking
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
    if s == "USDJPY":
        return (0 <= h <= 6) or (12 <= h <= 16)
    if s == "XAUUSD":
        return 7 <= h <= 17
    if s in ("ETHUSD", "ETH-USD"):
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
        logger.error(f"Symbol {symbol} not found")
        return None
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return None
    return symbol_info

def get_pip_value(symbol: str, symbol_info: mt5.SymbolInfo) -> float:
    """Calculate pip value for position sizing"""
    # For most forex pairs, 1 pip = 0.0001 (or 0.01 for JPY pairs)
    if "JPY" in symbol:
        return 0.01
    elif "XAU" in symbol or "XAG" in symbol:  # Gold/Silver
        return 0.01
    else:
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
    pip_value = get_pip_value(symbol, symbol_info)
    
    # Contract size (typically 100,000 for standard lot)
    contract_size = symbol_info.trade_contract_size
    
    # Value per pip for 1 lot
    value_per_pip = pip_value * contract_size
    
    # Calculate lot size
    lot_size = risk_amount / (stop_distance_pips * value_per_pip)
    
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

def check_fxify_limits(state: dict, account: mt5.AccountInfo) -> Tuple[bool, str]:
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
        return False, f"Max total loss reached: ${total_drawdown:.2f} >= ${max_total_loss:.2f}"
    
    # Check max daily drawdown
    if state.get("day_start_equity") is None:
        state["day_start_equity"] = equity
        save_state(state)
    
    day_start = state["day_start_equity"]
    max_daily_loss = day_start * FXIFY_MAX_DAILY_LOSS_PCT
    daily_drawdown = day_start - equity
    if daily_drawdown >= max_daily_loss:
        return False, f"Max daily loss reached: ${daily_drawdown:.2f} >= ${max_daily_loss:.2f}"
    
    # Check profit target (if reached, bot could notify but continue trading)
    profit_target = initial_balance * FXIFY_PROFIT_TARGET_PCT
    current_profit = equity - initial_balance
    if current_profit >= profit_target:
        logger.info(f"✅ PROFIT TARGET REACHED: ${current_profit:.2f} >= ${profit_target:.2f}")
    
    return True, "OK"

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
        
        logger.info(f"🤖 FXIFY Trading Bot initialized")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Timeframe: {TIMEFRAME_STR}")
        logger.info(f"Account: {self.account.login}, Balance: ${self.account.balance:.2f}")
        logger.info(f"Risk per trade: {PORTFOLIO_RISK_PCT*100:.2f}%")
        logger.info(f"Max daily loss: {FXIFY_MAX_DAILY_LOSS_PCT*100:.1f}%, Max total loss: {FXIFY_MAX_TOTAL_LOSS_PCT*100:.1f}%")
    
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
        
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: retcode={result.retcode}, comment={result.comment}")
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
        risk_pct = RISK_PCT_BASE
        if latest["adx"] > ADX_RISK_THR and latest["atr_pct"] > ATR_PCT_RISK_THR:
            risk_pct = RISK_PCT_STRONG
            logger.info(f"Strong setup: using {risk_pct*100:.2f}% risk")
        
        # Calculate lot size
        lot_size = calculate_lot_size(
            self.symbol,
            self.symbol_info,
            account.equity,
            risk_pct,
            stop_distance_pips
        )
        
        if lot_size < self.symbol_info.volume_min:
            logger.warning(f"Lot size {lot_size} below minimum {self.symbol_info.volume_min}")
            return False
        
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
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
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
        
        logger.info(f"📈 Opening {direction} position: Size={lot_size} lots, Entry={price:.5f}, SL={stop_loss:.5f}")
        
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: retcode={result.retcode}, comment={result.comment}")
            return False
        
        logger.info(f"✅ Position opened: Ticket={result.order}, Price={result.price:.5f}")
        
        # Update state
        self.state["in_position"] = True
        self.state["position_qty"] = lot_size
        self.state["position_ticket"] = result.order
        self.state["entry_price"] = result.price
        self.state["entry_atr"] = atr
        self.state["trade_high_price"] = result.price if direction == "BUY" else result.price
        self.state["trade_low_price"] = result.price if direction == "SELL" else result.price
        self.state["entry_time"] = _utcnow_iso()
        self.state["trades_today"] = self.state.get("trades_today", 0) + 1
        self.state["last_stop_price"] = stop_loss
        
        save_state(self.state)
        return True
    
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

        # Break-even at +1R (BUY only for simplicity)
        if position.type == mt5.POSITION_TYPE_BUY and r_mult >= 1.0:
            be_stop = entry_price
            if position.sl < be_stop:
                self._modify_position_sl(position, be_stop)

        # Start trailing after +1.5R
        if r_mult >= 1.5:
            trail_mult = P["trailing_stop_atr_mult"]
            if r_mult >= TRAIL_WIDEN_AFTER_R:
                trail_mult = TRAIL_ABS_WIDEN_TO
                logger.info(f"Trailing stop widened to {trail_mult}x ATR after +{r_mult:.1f}R")
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
        
        # Reset daily counters if needed
        _reset_day_counters_if_needed(self.state)
        
        # Get account info
        account = self.get_account_info()
        if account is None:
            logger.error("Failed to get account info")
            return
        
        # Check FXIFY compliance
        allowed, reason = check_fxify_limits(self.state, account)
        if not allowed:
            logger.error(f"❌ Trading stopped: {reason}")
            # Close any open positions
            position = self.get_position()
            if position:
                self.close_position(position, "FXIFY_LIMIT")
            return

        # Hard daily stop at -DAILY_STOP_PCT: block new entries for rest of day
        day_start = self.state.get("day_start_equity") or account.equity
        if day_start > 0:
            dd_pct = (day_start - account.equity) / day_start
            if dd_pct >= DAILY_STOP_PCT:
                logger.warning(f"Daily stop reached ({dd_pct*100:.2f}% >= {DAILY_STOP_PCT*100:.2f}%). Entries paused until UTC midnight.")
                # Manage existing position only; no new entries
                position = self.get_position()
                if position:
                    # Continue to manage position stops/exits below
                    pass
                else:
                    return
        
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
        
        # Check spread
        spread_pips = get_current_spread_pips(self.symbol, self.symbol_info)
        logger.info(f"Spread: {spread_pips:.1f} pips")
        if MAX_SPREAD_PIPS > 0 and spread_pips > MAX_SPREAD_PIPS:
            logger.warning(f"Spread too high: {spread_pips:.1f} > {MAX_SPREAD_PIPS:.1f} pips")
            return
        
        # Get current position
        position = self.get_position()
        
        # POSITION MANAGEMENT
        if position:
            logger.info(f"📌 In position: {position.type}, P&L: ${position.profit:.2f}")
            
            # Update trailing stop
            self.update_trailing_stop(position, latest)
            
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

            # Respect active trading hours for new entries
            if not _is_active_hour_utc(symbol=self.symbol):
                logger.info(f"Outside active hours (UTC {ACTIVE_HOUR_START:02d}-{ACTIVE_HOUR_END:02d}); entries paused")
                return
            
            # Check daily trade limit
            if self.state.get("trades_today", 0) >= MAX_TRADES_PER_DAY:
                logger.warning(f"Daily trade limit reached: {MAX_TRADES_PER_DAY}")
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
            
            # Check minimum ATR
            if latest["atr_pct"] < MIN_EDGE_ATR_PCT:
                logger.info(f"ATR too low: {latest['atr_pct']:.3f}% < {MIN_EDGE_ATR_PCT:.3f}%")
                return
            
            # Opportunistic entry
            if OPPORTUNISTIC_MODE:
                if latest["edge_score"] >= EDGE_BUY_SCORE:
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
        logger.info("🚀 Starting FXIFY Trading Bot")
        logger.info(f"Symbol: {self.symbol}, Timeframe: {TIMEFRAME_STR}")
        logger.info(f"Opportunistic mode: {OPPORTUNISTIC_MODE}")
        logger.info(f"Edge buy score: {EDGE_BUY_SCORE}, Exit score: {EDGE_EXIT_SCORE}")
        
        try:
            while True:
                try:
                    self.run_iteration()
                except Exception as e:
                    logger.error(f"Iteration error: {e}", exc_info=True)
                
                if LOOP_ONCE:
                    logger.info("LOOP_ONCE=true, exiting")
                    break
                
                _sleep_to_next_bar()
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            logger.info("Shutting down MT5")
            mt5.shutdown()

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
