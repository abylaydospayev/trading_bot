#!/usr/bin/env python3
"""
Continuous Trading Bot (Alpaca Paper by default)
- Default cadence: 15m, aligns sleep to next bar.
- Sizing = RISK_PCT of account equity (max loss if initial stop is hit), incl. exit costs.
- Indicators: EMA(short/long/trend), ATR(Wilder), ADX(Wilder), RSI.
- Entry: opportunistic Edge score (with confirmation) OR classic golden-cross + uptrend + ADX.
- Exit: adaptive trailing stop (ratchet), optional partial profits, opportunistic deterioration.
- Protections: opening calm (equities only), cooldown, daily trade cap, circuit breaker.
- Logging: rotating file + console; state persisted to bot_state.json.

This implementation reuses shared indicator and edge logic from the strategy/ package
and removes the stop "lock-in" floor, matching the backtester behavior.
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
from typing import Optional

import numpy as np
import pandas as pd

from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

# Ensure project root on sys.path for `strategy` imports when running as a script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from strategy.indicators import compute_atr_wilder, calculate_adx, calculate_rsi, MIN_ATR
from strategy.edge import compute_edge_features_and_score, EdgeResult

# ===================== CONFIG =====================
load_dotenv()  # optional .env support

BASE_URL = os.getenv("BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"

# --- Instrument & timeframe ---
RAW_SYMBOL = os.getenv("SYMBOL", "BTCUSD")  # SPY / QQQ / BTCUSD (no dash)
TIMEFRAME = os.getenv("TIMEFRAME", "15Min")  # '1Min' / '15Min' / '1Day'


def _normalize_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    if s.endswith("USD") and ("/" in s or "-" in s):
        s = s.replace("/", "").replace("-", "")
    return s


SYMBOL = _normalize_symbol(RAW_SYMBOL)


def _seconds_per_bar(tf: str) -> int:
    t = (tf or "").lower()
    # Common variations
    if t in ("1min", "1m"):  # 1 minute
        return 60
    if t in ("15min", "15m"):
        return 15 * 60
    if t in ("1day", "1d", "day", "d"):
        return 24 * 60 * 60
    # Try to parse like '5min', '30min'
    try:
        if t.endswith("min"):
            return int(t[:-3]) * 60
    except Exception:
        pass
    return 15 * 60


SECONDS_PER_BAR = _seconds_per_bar(TIMEFRAME)

# --- Risk/position sizing ---
PORTFOLIO_RISK_PCT = float(os.getenv("RISK_PCT", "0.05"))  # default 5%
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "2"))  # 0.02% one-way
SPREAD_BPS = float(os.getenv("SPREAD_BPS", "1"))  # 0.01% one-way

# --- Entry/exit controls ---
OPPORTUNISTIC_MODE = os.getenv("OPPORTUNISTIC_MODE", "true").lower() == "true"
EDGE_BUY_SCORE = int(os.getenv("EDGE_BUY_SCORE", "60"))
EDGE_EXIT_SCORE = int(os.getenv("EDGE_EXIT_SCORE", "10"))
EDGE_CONFIRM_BARS = int(os.getenv("EDGE_CONFIRM_BARS", "2"))
EDGE_EXIT_CONFIRM_BARS = int(os.getenv("EDGE_EXIT_CONFIRM_BARS", "2"))

# --- Intraday protections ---
MIN_COOLDOWN_MIN = int(os.getenv("MIN_COOLDOWN_MIN", "1"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "200"))
MIN_EDGE_ATR_PCT = float(os.getenv("MIN_EDGE_ATR_PCT", "0.03"))

# --- Opening calm (equities only) ---
USE_OPENING_CALM = os.getenv("USE_OPENING_CALM", "true").lower() == "true"
OPENING_CALM_MIN = int(os.getenv("OPENING_CALM_MIN", "30"))

# --- Fractional shares ---
ALLOW_FRACTIONAL = os.getenv("ALLOW_FRACTIONAL", "true").lower() == "true"
QTY_DECIMALS = int(os.getenv("QTY_DECIMALS", "3"))
MIN_FRACTIONAL_QTY = float(os.getenv("MIN_FRACTIONAL_QTY", "0.001"))

# --- Circuit breaker ---
CB_ENABLED = os.getenv("CB_ENABLED", "true").lower() == "true"
CB_MAX_LOSS_STREAK = int(os.getenv("CB_MAX_LOSS_STREAK", "2"))
CB_COOLDOWN_MIN = int(os.getenv("CB_COOLDOWN_MIN", "90"))

# --- Orders ---
ORDER_POLL_INTERVAL_SEC = float(os.getenv("ORDER_POLL_INTERVAL_SEC", "0.5"))
ORDER_FILL_TIMEOUT_SEC = float(os.getenv("ORDER_FILL_TIMEOUT_SEC", "60"))

# --- Indicators/params ---
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
    "trailing_stop_atr_mult": float(os.getenv("TRAIL_ATR_MULT", "3.0")),
    "use_ema": os.getenv("USE_EMA", "true").lower() == "true",
    "adx_threshold": int(os.getenv("ADX_THRESHOLD", "20")),
    "winner_run_atr_mult_widen": float(os.getenv("WINNER_WIDEN_MULT", "1.2")),
    "winner_run_threshold_atr": float(os.getenv("WINNER_THRESHOLD_ATR", "2.0")),
    "partial_profit_threshold_atr": float(os.getenv("PARTIAL_ATR", "3.0")),
    "vix_spike_threshold": float(os.getenv("VIX_SPIKE_THRESHOLD", "20.0")),
}

# Adaptive trailing: widen to absolute multiple after +R gain
TRAIL_WIDEN_AFTER_R = float(os.getenv("TRAIL_WIDEN_AFTER_R", "1.0"))  # after +1R
TRAIL_ABS_WIDEN_TO = float(os.getenv("TRAIL_ABS_WIDEN_TO", "4.0"))     # widen to 4x ATR

# Adaptive risk sizing
RISK_PCT_BASE = float(os.getenv("RISK_PCT_BASE", os.getenv("RISK_PCT", "0.05")))
RISK_PCT_STRONG = float(os.getenv("RISK_PCT_STRONG", "0.15"))
ADX_RISK_THR = float(os.getenv("ADX_RISK_THR", "25"))
ATR_PCT_RISK_THR = float(os.getenv("ATR_PCT_RISK_THR", "0.04"))

USE_VIX = os.getenv("USE_VIX", "false").lower() == "true"
STATE_FILE = os.getenv("STATE_FILE", "bot_state.json")
LOG_DIR = os.getenv("LOG_DIR", "logs")

# --- Optional entry filters (match backtester)
# Enable a rolling ATR%% regime filter: require current ATR%% >= rolling percentile over a window
ATR_REGIME_WINDOW = int(os.getenv("ATR_REGIME_WINDOW", "0"))     # e.g., 200
ATR_REGIME_PCT = float(os.getenv("ATR_REGIME_PCT", "0"))          # e.g., 30 (percentile)
# Require a minimum distance between short and long MAs, in basis points relative to price
MIN_MA_DIST_BPS = float(os.getenv("MIN_MA_DIST_BPS", "0"))        # e.g., 5

# Higher-timeframe alignment (optional)
MTF_ENABLE = os.getenv("MTF_ENABLE", "false").lower() == "true"
MTF_TF = os.getenv("MTF_TF", "1H")  # e.g., 1H or 4H
MTF_TREND_MA = int(os.getenv("MTF_TREND_MA", "200"))

# Market quality guards
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "0"))  # e.g., 5 bps; 0 disables
ADX_SLOPE_MIN = float(os.getenv("ADX_SLOPE_MIN", "0"))     # require ADX slope >= this (0 disables)
MAX_DAILY_DD_PCT = float(os.getenv("MAX_DAILY_DD_PCT", "0"))  # e.g., 0.025 (2.5%); 0 disables

# ML gating
ML_ENABLE = os.getenv("ML_ENABLE", "false").lower() == "true"
# Allow per-asset overrides, e.g., ML_MODEL_PATH_BTCUSD, ML_PROB_THR_ETHUSD
ML_MODEL_PATH = os.getenv(
    f"ML_MODEL_PATH_{SYMBOL}",
    os.getenv("ML_MODEL_PATH", os.path.join(ROOT_DIR, "ml", "models", "mlp_model.pkl"))
)
ML_PROB_THR = float(os.getenv(f"ML_PROB_THR_{SYMBOL}", os.getenv("ML_PROB_THR", "0.6")))

# --- Numerics ---
EPS = 1e-12

# Smoke test toggle: process one iteration then exit
LOOP_ONCE = os.getenv("LOOP_ONCE", "false").lower() == "true"

# ===================== LOGGING =====================

def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "trading_bot.log")
    logger = logging.getLogger("TradingBot")
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
    sys.exit(0)


signal.signal(signal.SIGTERM, _graceful_shutdown)
signal.signal(signal.SIGINT, _graceful_shutdown)

# ===================== STATE =====================


def get_default_state() -> dict:
    return {
        "in_position": False,
        "position_qty": 0.0,
        "entry_price": 0.0,
        "entry_atr": 0.0,
        "trade_high_price": 0.0,
        "entry_time": None,
        "partial_profit_taken": False,
        "edge_buy_streak": 0,
        "edge_exit_streak": 0,
        "last_trade_ts": None,
        "trades_today": 0,
        "trades_day_str": None,
        "stop_order_id": None,
        "loss_streak": 0,
        "circuit_breaker_until": None,
        "last_stop_price": None,
        "day_start_equity": None,
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
    logger.info(
        f"Sleeping ~{sleep_s//60}m {sleep_s%60}s to next {TIMEFRAME} bar..."
    )
    time.sleep(sleep_s)


# ===================== TECH + EDGE =====================


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _to_crypto_data_symbol(sym: str) -> str:
    """Convert trading ticker (BTCUSD) → data symbol (BTC/USD). Leave already-slashed symbols alone."""
    s = (sym or "").upper().strip()
    if "/" in s:
        return s
    if s.endswith("USD"):
        return f"{s[:-3]}/USD"
    if s.endswith("USDT"):
        return f"{s[:-4]}/USDT"
    if len(s) > 6:
        return f"{s[:-4]}/{s[-4:]}"
    return s


def fetch_bars_and_indicators_unified(api, symbol: str, timeframe: str, is_crypto: bool) -> Optional[pd.Series]:
    """
    Pull bars via Alpaca (crypto or equities), compute indicators, return latest row.
    Handles MultiIndex frames and missing volume for crypto.
    """
    try:
        limit = P["trend"] + 120
        # Fetch
        if is_crypto:
            data_symbol = _to_crypto_data_symbol(symbol)
            bars = None
            try:
                try:
                    bars = api.get_crypto_bars([data_symbol], timeframe, limit=limit, exchanges=["CBSE", "ERSX"]).df
                except TypeError:
                    bars = api.get_crypto_bars([data_symbol], timeframe, limit=limit).df
            except Exception:
                try:
                    bars = api.get_crypto_bars(data_symbol, timeframe, limit=limit, exchanges=["CBSE", "ERSX"]).df
                except TypeError:
                    bars = api.get_crypto_bars(data_symbol, timeframe, limit=limit).df
        else:
            bars = api.get_bars(symbol, timeframe, limit=limit).df

        if bars is None or bars.empty:
            logger.warning("Empty bars from API")
            return None

        # Normalize index/columns
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.copy()
            bars.index = pd.to_datetime(bars.index.get_level_values(-1))
        else:
            bars = bars.copy()
            bars.index = pd.to_datetime(bars.index)
        bars.columns = [str(c).lower() for c in bars.columns]
        need = {"open", "high", "low", "close"}
        missing = need - set(bars.columns)
        if missing:
            logger.error(f"Missing cols from bars: {missing}")
            return None
        if "volume" not in bars.columns:
            bars["volume"] = 0.0

        # Indicators
        if P["use_ema"]:
            bars["short_ma"] = _ema(bars["close"], P["short"])
            bars["long_ma"] = _ema(bars["close"], P["long"])
            bars["trend_ma"] = _ema(bars["close"], P["trend"])
        else:
            bars["short_ma"] = bars["close"].rolling(P["short"]).mean()
            bars["long_ma"] = bars["close"].rolling(P["long"]).mean()
            bars["trend_ma"] = bars["close"].rolling(P["trend"]).mean()

        bars["atr"] = compute_atr_wilder(bars, P["atr_period"])  # Wilder ATR
        bars["adx"] = calculate_adx(bars, INDICATOR["adx_period"])  # Wilder ADX
        bars["rsi"] = calculate_rsi(bars["close"], INDICATOR["rsi_period"])  # Wilder RSI

        L = INDICATOR["breakout_lookback"]
        bars["highL"] = bars["high"].rolling(L).max()
        bars["prev_high20"] = bars["highL"].shift(1)
        bars["atr_median20"] = bars["atr"].rolling(20).median()

        K = INDICATOR["adx_slope_lookback"]
        bars["adx_slope"] = bars["adx"] - bars["adx"].shift(K)

        # ATR% series and optional rolling regime threshold (shifted to avoid lookahead)
        bars["atr_pct"] = (bars["atr"] / bars["close"]).replace([np.inf, -np.inf], np.nan) * 100.0
        if ATR_REGIME_WINDOW > 0 and ATR_REGIME_PCT > 0:
            win = int(ATR_REGIME_WINDOW)
            pct = float(ATR_REGIME_PCT)
            def _pctile(x: pd.Series) -> float:
                x = x.dropna()
                if len(x) == 0:
                    return np.nan
                return float(np.percentile(x, pct))
            bars["atr_pct_thresh"] = (
                bars["atr_pct"].rolling(win, min_periods=max(10, win // 5)).apply(_pctile, raw=False).shift(1)
            )

        # Additional ML feature parity: time-of-day cyclical features and ATR regime percentiles
        try:
            hours = bars.index.hour.astype(float)
            bars["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            bars["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
            bars["is_active_hours"] = ((hours >= 12) & (hours <= 22)).astype(float)
        except Exception:
            pass
        try:
            rollw = int(ATR_REGIME_WINDOW) if int(ATR_REGIME_WINDOW) > 0 else 200
            def _pctile40(x: pd.Series) -> float:
                x = x.dropna()
                if len(x) == 0:
                    return np.nan
                return float(np.percentile(x, 40.0))
            def _pctile60(x: pd.Series) -> float:
                x = x.dropna()
                if len(x) == 0:
                    return np.nan
                return float(np.percentile(x, 60.0))
            bars["atr_pct_thr40"] = bars["atr_pct"].rolling(rollw, min_periods=max(10, rollw // 5)).apply(_pctile40, raw=False).shift(1)
            bars["atr_pct_thr60"] = bars["atr_pct"].rolling(rollw, min_periods=max(10, rollw // 5)).apply(_pctile60, raw=False).shift(1)
            bars["atr_above_40"] = (bars["atr_pct"] >= bars["atr_pct_thr40"]).astype(float)
            bars["atr_above_60"] = (bars["atr_pct"] >= bars["atr_pct_thr60"]).astype(float)
        except Exception as e:
            logger.debug(f"ATR regime feature build failed: {e}")

        # Optional: MTF trend alignment (resample to higher TF, EMA trend)
        if MTF_ENABLE:
            try:
                htf_close = bars["close"].resample(str(MTF_TF).lower()).last()
                htf_trend = htf_close.ewm(span=MTF_TREND_MA, adjust=False).mean()
                bars["mtf_trend_ma"] = htf_trend.reindex(bars.index, method="ffill")
            except Exception as e:
                logger.warning(f"MTF computation failed: {e}")

        # VIX penalty (disabled by default to avoid optional dependency conflicts)
        vix_pct_change = 0.0
        if USE_VIX:
            try:
                import yfinance as yf  # lazy import to keep optional
                vix = yf.download("^VIX", period="10d", interval="1d", progress=False)
                if not vix.empty and len(vix) >= 2:
                    v_latest = float(vix["Close"].iloc[-1])
                    v_prev = float(vix["Close"].iloc[-2])
                    vix_pct_change = ((v_latest - v_prev) / max(v_prev, EPS)) * 100.0
            except Exception as e:
                logger.warning(f"VIX fetch failed: {e}")

        latest = bars.iloc[-1].copy()
        prev = bars.iloc[-2]
        latest["prev_short_ma"] = prev["short_ma"]
        latest["vix_pct_change"] = vix_pct_change
        latest["bar_time"] = str(latest.name)

        # Use shared edge logic
        edge: EdgeResult = compute_edge_features_and_score(
            bars, latest, prev, vix_pct_change, P["vix_spike_threshold"]
        )
        latest["edge_score"] = edge.score
        latest["edge_reasons"] = ", ".join(edge.reasons)
        return latest

    except Exception as e:
        logger.error(f"fetch_bars_and_indicators_unified error: {e}")
        return None


# ===================== TRADING BOT =====================


class TradingBot:
    def __init__(self):
        if not ALPACA_API_KEY or not ALPACA_API_SECRET:
            raise ValueError("Set ALPACA_API_KEY and ALPACA_API_SECRET env vars.")
        # Safety: require explicit LIVE_TRADING=true when not using paper endpoint
        if ("paper-api.alpaca.markets" not in str(BASE_URL).lower()) and (not LIVE_TRADING):
            raise RuntimeError(
                "Refusing to run against non-paper BASE_URL without LIVE_TRADING=true."
            )
        self.api = tradeapi.REST(
            ALPACA_API_KEY, ALPACA_API_SECRET, base_url=BASE_URL, api_version="v2"
        )
        self.symbol = SYMBOL
        self.state = load_state()

        # Validate symbol & detect crypto
        try:
            asset = self.api.get_asset(self.symbol)
            if not asset.tradable:
                raise RuntimeError(
                    f"Symbol {self.symbol} exists but is not tradable on your account."
                )
            atype = str(getattr(asset, "asset_class", "")).lower()
            self._is_crypto = (
                "crypto" in atype
                or self.symbol.endswith("USD")
                or self.symbol.endswith("USDT")
            )
            logger.info(
                f"Asset class={getattr(asset,'asset_class',None)} | is_crypto={self._is_crypto}"
            )
        except APIError as e:
            msg = str(e).lower()
            if (
                "not found" in msg
                or "symbol not found" in msg
                or "no day symbol" in msg
            ):
                if self.symbol.upper().endswith("USD"):
                    raise RuntimeError(
                        f"Alpaca can’t find {RAW_SYMBOL}. Try SYMBOL=BTCUSD (no dash/slash) "
                        f"and ensure crypto trading is enabled for your Alpaca account."
                    )
                raise RuntimeError(
                    f"Alpaca can’t find {RAW_SYMBOL}. Did you mistype the ticker?"
                )
            raise

        logger.info("Trading bot initialized.")

    # ---------- market quality ----------
    def _get_spread_bps(self) -> Optional[float]:
        try:
            if getattr(self, "_is_crypto", False):
                data_symbol = _to_crypto_data_symbol(self.symbol)
                try:
                    q = self._with_retries(self.api.get_crypto_latest_quote, data_symbol, exchanges=["CBSE", "ERSX"])
                except TypeError:
                    q = self._with_retries(self.api.get_crypto_latest_quote, data_symbol)
            else:
                q = self._with_retries(self.api.get_latest_quote, self.symbol)
            bid = float(getattr(q, "bid_price", 0.0) or 0.0)
            ask = float(getattr(q, "ask_price", 0.0) or 0.0)
            if bid > 0 and ask > 0 and ask >= bid:
                mid = (ask + bid) / 2.0
                return ((ask - bid) / mid) * 10000.0
        except Exception as e:
            logger.debug(f"Spread fetch failed, skipping guard: {e}")
        return None

    # ---------- retry wrapper ----------
    def _with_retries(self, fn, *args, retries: int = 3, **kwargs):
        for i in range(retries):
            try:
                return fn(*args, **kwargs)
            except APIError as e:
                logger.warning(f"{getattr(fn, '__name__', 'fn')} APIError: {e} ({i+1}/{retries})")
                time.sleep(min(60, 2 ** (i + 1)))
            except Exception as e:
                logger.warning(
                    f"{getattr(fn, '__name__', 'fn')} transient error: {e} ({i+1}/{retries})"
                )
                time.sleep(min(60, 2 ** (i + 1)))
        raise RuntimeError(f"{getattr(fn, '__name__', 'fn')} failed after retries")

    # ---------- stops ----------
    def _cancel_stop_if_any(self):
        soid = self.state.get("stop_order_id")
        if not soid:
            return
        try:
            self._with_retries(self.api.cancel_order, soid)
            logger.info(f"Canceled stop id={soid}")
        except Exception as e:
            logger.warning(f"Failed cancel stop {soid}: {e}")
        self.state["stop_order_id"] = None
        save_state(self.state)

    def _replace_stop(self, qty, stop_price):
        self._cancel_stop_if_any()
        qty = round(float(qty), QTY_DECIMALS) if ALLOW_FRACTIONAL else int(qty)
        if (ALLOW_FRACTIONAL and qty < MIN_FRACTIONAL_QTY) or (
            not ALLOW_FRACTIONAL and qty <= 0
        ):
            logger.warning("Skip placing stop: qty too small.")
            return
        if not math.isfinite(stop_price) or float(stop_price) <= 0:
            logger.warning("Skip placing stop: invalid price.")
            return
        prev_stop = self.state.get("last_stop_price")
        if prev_stop:
            stop_price = max(float(prev_stop), float(stop_price))  # ratchet only tighter
        try:
            order = self._with_retries(
                self.api.submit_order,
                symbol=self.symbol,
                qty=qty,
                side="sell",
                type="stop",
                time_in_force="gtc",
                stop_price=round(float(stop_price), 2),
                client_order_id=f"protstop-{int(time.time())}",
            )
            self.state["stop_order_id"] = order.id
            self.state["last_stop_price"] = float(stop_price)
            save_state(self.state)
            logger.info(f"Placed/updated stop {qty} @ ${float(stop_price):.2f}")
        except Exception as e:
            logger.warning(f"_replace_stop error: {e}")

    # ---------- equity risk sizing ----------
    def _size_from_risk(self, ref_price, atr, equity, buying_power, risk_pct: Optional[float] = None):
        atr = max(float(atr), MIN_ATR)
        stop_price = ref_price - (P["trailing_stop_atr_mult"] * atr)
        risk_per_share = ref_price - stop_price
        exit_cost_ps = ((SLIPPAGE_BPS + SPREAD_BPS) / 10000.0) * ref_price
        effective_risk_ps = risk_per_share + exit_cost_ps
        if effective_risk_ps <= 0:
            return 0.0
        rp = float(risk_pct) if risk_pct is not None else PORTFOLIO_RISK_PCT
        dollars_to_risk = equity * rp
        qty_from_risk = dollars_to_risk / max(effective_risk_ps, EPS)
        qty_affordable = buying_power / ref_price
        return max(0.0, min(qty_from_risk, qty_affordable))

    # ---------- reconcile ----------
    def check_and_sync_position(self) -> bool:
        try:
            pos = self.api.get_position(self.symbol)
            qty = float(pos.qty)
            if qty > 0:
                self.state["in_position"] = True
                self.state["position_qty"] = qty
                self.state["entry_price"] = float(
                    self.state.get("entry_price") or pos.avg_entry_price
                )
                self.state["trade_high_price"] = float(
                    self.state.get("trade_high_price") or pos.current_price
                )
                if not self.state.get("entry_atr"):
                    logger.warning(
                        "entry_atr unknown; will use current ATR on the fly when needed."
                    )
                save_state(self.state)
                return True
            else:
                self.state = get_default_state()
                save_state(self.state)
                return True
        except APIError as e:
            if "position does not exist" in str(e).lower():
                self.state = get_default_state()
                save_state(self.state)
                return True
            logger.error(f"check_and_sync_position API error: {e}")
            return False
        except Exception as e:
            logger.error(f"check_and_sync_position error: {e}")
            return False

    def _update_day_start_equity(self):
        """Set or reset day_start_equity at UTC day boundary."""
        try:
            now_day = datetime.now(timezone.utc).date()
            marker = self.state.get("trades_day_str")
            # Ensure trades_day_str is the same day marker we already use
            if not _same_day(marker, _utcnow_iso()):
                # Refresh account equity as the new day baseline
                acct = self._with_retries(self.api.get_account)
                self.state["day_start_equity"] = float(getattr(acct, "equity", 0.0) or 0.0)
                save_state(self.state)
        except Exception as e:
            logger.debug(f"_update_day_start_equity failed: {e}")

    # ---------- open / close ----------
    def open_position(self, ref_price: float, atr: float, risk_pct: Optional[float] = None):
        account = self._with_retries(self.api.get_account)
        equity = float(account.equity)
        buying_power = float(account.buying_power)

        qty = self._size_from_risk(ref_price, atr, equity, buying_power, risk_pct=risk_pct)
        if ALLOW_FRACTIONAL:
            qty = round(qty, QTY_DECIMALS)
            if qty < MIN_FRACTIONAL_QTY:
                logger.info("Qty below min frac. Skip.")
                return
        else:
            qty = int(qty)
            if qty <= 0:
                logger.info("Qty 0. Skip.")
                return

        eff_risk_pct = float(risk_pct) if risk_pct is not None else PORTFOLIO_RISK_PCT
        logger.info(
            f"BUY SIZING — Equity ${equity:.2f}, Risk {eff_risk_pct*100:.1f}%, Qty {qty}"
        )

        order = self._with_retries(
            self.api.submit_order,
            symbol=self.symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
        )
        entry_price = None
        entry_time = None
        deadline = time.time() + ORDER_FILL_TIMEOUT_SEC
        while time.time() < deadline:
            os_ = self._with_retries(self.api.get_order, order.id)
            if os_.status == "filled" and os_.filled_avg_price:
                entry_price = float(os_.filled_avg_price)
                entry_time = (
                    os_.filled_at.isoformat() if getattr(os_, "filled_at", None) else _utcnow_iso()
                )
                break
            if os_.status in ("canceled", "expired", "rejected"):
                logger.critical(f"BUY {order.id} ended {os_.status}. Reconciling.")
                self.check_and_sync_position()
                raise RuntimeError(f"Entry failed: {os_.status}")
            time.sleep(ORDER_POLL_INTERVAL_SEC)
        if entry_price is None:
            try:
                self._with_retries(self.api.cancel_order, order.id)
                time.sleep(1)
            finally:
                self.check_and_sync_position()
            raise RuntimeError("Buy order did not fill in time.")

        self.state["in_position"] = True
        self.state["position_qty"] = float(qty)
        self.state["entry_price"] = entry_price
        the_atr = max(float(atr), MIN_ATR)
        self.state["trade_high_price"] = entry_price
        self.state["entry_atr"] = the_atr
        self.state["entry_time"] = entry_time
        self.state["partial_profit_taken"] = False
        self.state["edge_buy_streak"] = 0
        self.state["last_trade_ts"] = _utcnow_iso()
        self.state["trades_today"] = int(self.state.get("trades_today", 0)) + 1
        save_state(self.state)

        initial_stop = entry_price - (P["trailing_stop_atr_mult"] * the_atr)
        exit_cost_ps = ((SLIPPAGE_BPS + SPREAD_BPS) / 10000.0) * entry_price
        eff_risk_ps = (entry_price - initial_stop) + exit_cost_ps
        planned_loss = eff_risk_ps * float(qty)
        target_loss = equity * eff_risk_pct
        if planned_loss > target_loss * 1.01:
            extra = (planned_loss - target_loss) / float(qty)
            initial_stop = max(0.01, initial_stop + extra)
            logger.info("Tightened stop to cap risk per trade.")

        self._replace_stop(qty, initial_stop)
        logger.info(
            f"OPENED | Entry ${entry_price:.2f} | ATR {the_atr:.2f} | Qty {qty}"
        )

    def _calc_trailing_stop(
        self,
        prev_stop: float,
        trade_high: float,
        entry_atr: float,
        entry_price: float,
        profit_atr: float,
        adx_value: float,
    ) -> float:
        """
        Trailing stop based on entry ATR. No lock-in floor; only ratchets closer as highs progress.
        """
        entry_atr = max(float(entry_atr or 0.0), MIN_ATR)
        entry_price = float(entry_price or 0.0)
        trade_high = float(trade_high or entry_price)
        prev_stop = float(prev_stop or 0.0)
        adx_value = float(adx_value or 0.0)

        base = P["trailing_stop_atr_mult"]
        mult = base * (
            P["winner_run_atr_mult_widen"]
            if profit_atr > P["winner_run_threshold_atr"]
            else 1.0
        )
        # Adaptive widen after +R: switch to absolute wider multiple
        try:
            if float(profit_atr) >= float(TRAIL_WIDEN_AFTER_R):
                mult = max(float(mult), float(TRAIL_ABS_WIDEN_TO))
        except Exception:
            pass
        # Optionally nudge tighter in very strong trends
        if adx_value >= 30:
            mult = max(base, mult - 0.5)
        candidate = trade_high - mult * entry_atr
        return max(prev_stop, candidate)

    def close_position(self, reason: str, current_price: float, partial_qty: Optional[float] = None):
        is_partial = partial_qty is not None
        if is_partial:
            qty = (
                round(float(self.state.get("position_qty", 0)) / 2.0, QTY_DECIMALS)
                if ALLOW_FRACTIONAL
                else int(float(self.state.get("position_qty", 0)) // 2)
            )
            if (ALLOW_FRACTIONAL and qty < MIN_FRACTIONAL_QTY) or (
                not ALLOW_FRACTIONAL and qty <= 0
            ):
                logger.info("Partial too small. Skip.")
                return
        else:
            qty = float(self.state.get("position_qty", 0))

        logger.info(f"EXIT ({reason}) — {'PARTIAL' if is_partial else 'FULL'} {qty} {self.symbol}")
        if is_partial:
            order = self._with_retries(
                self.api.submit_order,
                symbol=self.symbol,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="day",
            )
        else:
            order = self._with_retries(self.api.close_position, self.symbol)

        deadline = time.time() + ORDER_FILL_TIMEOUT_SEC
        exit_price = None
        while time.time() < deadline:
            os_ = self._with_retries(self.api.get_order, order.id)
            if os_.status == "filled" and os_.filled_avg_price:
                exit_price = float(os_.filled_avg_price)
                break
            if os_.status in ("canceled", "expired", "rejected"):
                logger.critical(f"EXIT {order.id} ended {os_.status}. Reconciling.")
                self.check_and_sync_position()
                raise RuntimeError(f"Exit failed: {os_.status}")
            time.sleep(ORDER_POLL_INTERVAL_SEC)
        if exit_price is None:
            try:
                self._with_retries(self.api.cancel_order, order.id)
                time.sleep(1)
            finally:
                self.check_and_sync_position()
            raise RuntimeError("Exit order did not fill in time.")

        # Reconcile state, then maintain/clear stops
        self.check_and_sync_position()
        self.state["last_trade_ts"] = _utcnow_iso()
        self.state["trades_today"] = int(self.state.get("trades_today", 0)) + 1
        save_state(self.state)

        try:
            remaining = float(self.state.get("position_qty", 0))
            if remaining > 0:
                prev_stop = float(self.state.get("last_stop_price") or 0.0)
                entry_atr = max(float(self.state.get("entry_atr", MIN_ATR)), MIN_ATR)
                entry_px = float(self.state.get("entry_price", 0.0) or 0.0)
                high_p = float(self.state.get("trade_high_price", entry_px))
                prof_atr = (high_p - entry_px) / entry_atr if entry_atr > 0 else 0.0
                latest = fetch_bars_and_indicators_unified(
                    self.api, self.symbol, TIMEFRAME, getattr(self, "_is_crypto", False)
                )
                adx_val = float(latest.get("adx", 0.0)) if latest is not None else 0.0
                new_stop = self._calc_trailing_stop(
                    prev_stop,
                    self.state["trade_high_price"],
                    entry_atr,
                    entry_px,
                    prof_atr,
                    adx_val,
                )
                self._replace_stop(remaining, new_stop)
            else:
                self._cancel_stop_if_any()
        except Exception as e:
            logger.warning(f"Stop maintenance after exit failed: {e}")

        # Circuit breaker on full exits only
        if not is_partial:
            try:
                entry_px = float(self.state.get("entry_price", 0.0) or 0.0)
                pnl = (exit_price - entry_px) * float(qty)
                if pnl < 0:
                    self.state["loss_streak"] = int(self.state.get("loss_streak", 0)) + 1
                    if CB_ENABLED and self.state["loss_streak"] >= CB_MAX_LOSS_STREAK:
                        until = datetime.now(timezone.utc) + timedelta(minutes=CB_COOLDOWN_MIN)
                        self.state["circuit_breaker_until"] = until.isoformat()
                        logger.warning(f"Circuit breaker TRIPPED. Pausing until {until}.")
                        self.state["loss_streak"] = 0
                else:
                    self.state["loss_streak"] = 0
                save_state(self.state)
            except Exception:
                pass

            if not self.state["in_position"]:
                self.state = get_default_state()
                save_state(self.state)

    # ---------- trade logic ----------
    def execute_trade_logic(self, latest: pd.Series):
        if latest is None or latest.isnull().any():
            logger.info("Incomplete latest data — skip.")
            return

        c = float(latest["close"])
        edge = int(latest.get("edge_score", 0))

        # Backfill missing entry ATR once, from current latest ATR
        if self.state["in_position"] and (self.state.get("entry_atr", 0.0) <= 0.0):
            self.state["entry_atr"] = max(float(latest.get("atr", MIN_ATR) or MIN_ATR), MIN_ATR)
            logger.info(
                f"Backfilled entry_atr from latest ATR → {self.state['entry_atr']:.4f}"
            )
            save_state(self.state)

        # In position: manage
        if self.state["in_position"]:
            self.state["trade_high_price"] = float(
                max(self.state.get("trade_high_price", 0.0), c)
            )
            save_state(self.state)

            entry_atr = float(self.state.get("entry_atr", 0.0))
            entry_price = float(self.state.get("entry_price", 0.0))
            profit_atr = (
                c - entry_price
            ) / max(entry_atr, MIN_ATR) if entry_atr > 0 else 0.0

            # partial at configured ATR multiple
            if (not self.state["partial_profit_taken"]) and (
                profit_atr >= P["partial_profit_threshold_atr"]
            ):
                qty_to_sell = float(self.state.get("position_qty", 0)) / 2.0
                self.close_position("Partial Profit", c, partial_qty=qty_to_sell)
                self.check_and_sync_position()
                rem = float(self.state.get("position_qty", 0))
                if rem > 0:
                    prev_stop = float(self.state.get("last_stop_price") or 0.0)
                    new_stop = self._calc_trailing_stop(
                        prev_stop,
                        self.state["trade_high_price"],
                        max(entry_atr, MIN_ATR),
                        entry_price,
                        profit_atr,
                        float(latest.get("adx", 0.0)),
                    )
                    self._replace_stop(rem, new_stop)
                else:
                    self._cancel_stop_if_any()
                self.state["partial_profit_taken"] = True
                save_state(self.state)
                return

            prev_stop = float(self.state.get("last_stop_price") or 0.0)
            adx_val = float(latest.get("adx", 0.0))
            t_stop = self._calc_trailing_stop(
                prev_stop,
                self.state["trade_high_price"],
                max(entry_atr, MIN_ATR),
                entry_price,
                profit_atr,
                adx_val,
            )

            if OPPORTUNISTIC_MODE and edge <= EDGE_EXIT_SCORE:
                self.state["edge_exit_streak"] = int(
                    self.state.get("edge_exit_streak", 0)
                ) + 1
            else:
                self.state["edge_exit_streak"] = 0
            save_state(self.state)

            death_cross = latest["short_ma"] < latest["long_ma"]
            stop_hit = c <= t_stop

            if (
                OPPORTUNISTIC_MODE
                and (self.state["edge_exit_streak"] >= EDGE_EXIT_CONFIRM_BARS)
                and not stop_hit
                and not death_cross
            ):
                qty = float(self.state.get("position_qty", 0))
                if not self.state["partial_profit_taken"] and qty >= 2:
                    self.close_position("Edge deterioration (partial)", c, partial_qty=qty / 2.0)
                    self.check_and_sync_position()
                    rem = float(self.state.get("position_qty", 0))
                    if rem > 0:
                        prev_stop = float(self.state.get("last_stop_price") or 0.0)
                        new_stop = self._calc_trailing_stop(
                            prev_stop,
                            self.state["trade_high_price"],
                            max(entry_atr, MIN_ATR),
                            entry_price,
                            profit_atr,
                            adx_val,
                        )
                        self._replace_stop(rem, new_stop)
                    else:
                        self._cancel_stop_if_any()
                    return
                else:
                    self.close_position("Edge deterioration (full)", c)
                    self._cancel_stop_if_any()
                    return

            if stop_hit or death_cross:
                self.close_position("Trailing stop" if stop_hit else "Death cross", c)
                self._cancel_stop_if_any()
                return

        # Flat: evaluate entry
        else:
            cros_up = latest["short_ma"] > latest["long_ma"]
            uptrend = c > latest["trend_ma"]
            slope_ok = (latest["short_ma"] - latest["prev_short_ma"]) > 1e-6
            adx_value = float(latest.get("adx", 0.0))
            trending = adx_value > P["adx_threshold"]

            if OPPORTUNISTIC_MODE and edge >= EDGE_BUY_SCORE:
                self.state["edge_buy_streak"] = int(
                    self.state.get("edge_buy_streak", 0)
                ) + 1
            else:
                self.state["edge_buy_streak"] = 0
            save_state(self.state)
            confirmed_edge = OPPORTUNISTIC_MODE and (
                self.state["edge_buy_streak"] >= EDGE_CONFIRM_BARS
            )

            # Opening calm — equities only
            if USE_OPENING_CALM and not getattr(self, "_is_crypto", False):
                try:
                    today = datetime.now(timezone.utc).date()
                    cal = self._with_retries(
                        self.api.get_calendar, start=str(today), end=str(today)
                    )
                    if cal:
                        open_dt = cal[0].open
                        if isinstance(open_dt, str):
                            open_dt = datetime.fromisoformat(open_dt)
                        elif not isinstance(open_dt, datetime):
                            open_dt = datetime.combine(
                                open_dt, datetime.min.time(), tzinfo=timezone.utc
                            )
                        if datetime.now(timezone.utc) < (
                            open_dt + timedelta(minutes=OPENING_CALM_MIN)
                        ):
                            logger.info("Opening calm window. Skip entries.")
                            return
                except Exception:
                    pass

            # Circuit breaker
            if CB_ENABLED:
                cb_until = self.state.get("circuit_breaker_until")
                if cb_until and datetime.now(timezone.utc) < datetime.fromisoformat(
                    cb_until
                ):
                    mins_left = int(
                        (
                            datetime.fromisoformat(cb_until) - datetime.now(timezone.utc)
                        ).total_seconds()
                        // 60
                    )
                    logger.info(
                        f"Circuit breaker active ({mins_left}m). Skip entries."
                    )
                    return

            # Daily cap & cooldown
            if self.state.get("trades_today", 0) >= MAX_TRADES_PER_DAY:
                logger.info("Daily trade cap reached.")
                return
            if not _cooldown_over(self.state, MIN_COOLDOWN_MIN):
                logger.info("In cooldown window. Skip entry.")
                return

            # Daily max drawdown guard (UTC day): pause new entries after threshold drop
            if MAX_DAILY_DD_PCT > 0:
                try:
                    self._update_day_start_equity()
                    acct = self._with_retries(self.api.get_account)
                    cur_eq = float(getattr(acct, "equity", 0.0) or 0.0)
                    day_start = float(self.state.get("day_start_equity") or 0.0)
                    if day_start > 0:
                        drop = (cur_eq / day_start) - 1.0
                        if drop <= -float(MAX_DAILY_DD_PCT):
                            logger.warning(
                                f"Daily DD {drop*100:.2f}% <= -{MAX_DAILY_DD_PCT*100:.2f}% — pausing new entries until next day."
                            )
                            return
                except Exception as e:
                    logger.debug(f"Daily DD guard check failed: {e}")

            # Cost-aware guard
            atr_pct = (float(latest["atr"]) / c) * 100.0 if c > 0 else 0.0
            if atr_pct < MIN_EDGE_ATR_PCT:
                logger.info(f"ATR% {atr_pct:.2f} < {MIN_EDGE_ATR_PCT}. Skip entry.")
                return

            # Optional: ADX slope guard
            if ADX_SLOPE_MIN > 0:
                adx_slope = float(latest.get("adx_slope", 0.0))
                if adx_slope < ADX_SLOPE_MIN:
                    logger.info(f"ADX slope {adx_slope:.2f} < {ADX_SLOPE_MIN}. Skip entry.")
                    return

            # Optional: minimum MA distance in bps (short - long) relative to price
            if MIN_MA_DIST_BPS > 0:
                ma_dist_bps = ((float(latest["short_ma"]) - float(latest["long_ma"])) / c) * 10000.0 if c > 0 else 0.0
                if ma_dist_bps < MIN_MA_DIST_BPS:
                    logger.info(f"MA dist {ma_dist_bps:.2f}bps < {MIN_MA_DIST_BPS}bps. Skip entry.")
                    return

            # Optional: ATR% regime filter
            if ATR_REGIME_WINDOW > 0 and ATR_REGIME_PCT > 0:
                thr = latest.get("atr_pct_thresh", np.nan)
                if pd.notna(thr) and atr_pct < float(thr):
                    logger.info(f"ATR% {atr_pct:.2f} < regime thr {float(thr):.2f}. Skip entry.")
                    return

            # Optional: MTF alignment
            if MTF_ENABLE:
                mtf_ma = float(latest.get("mtf_trend_ma", np.nan))
                if not np.isnan(mtf_ma) and not (c > mtf_ma):
                    logger.info("MTF filter: price below higher-TF trend. Skip entry.")
                    return

            # Optional: current spread guard
            if MAX_SPREAD_BPS > 0:
                sp = self._get_spread_bps()
                if sp is not None and sp > MAX_SPREAD_BPS:
                    logger.info(f"Spread {sp:.2f}bps > {MAX_SPREAD_BPS}bps. Skip entry.")
                    return

            if confirmed_edge or (cros_up and uptrend and slope_ok and trending):
                # Optional ML gating
                if ML_ENABLE:
                    try:
                        from ml.infer import predict_entry_prob
                        cur = latest.copy()
                        if pd.isna(cur.get("prev_short_ma", np.nan)):
                            cur["prev_short_ma"] = float(cur.get("short_ma", 0.0))
                        p = predict_entry_prob(cur, ML_MODEL_PATH)
                        if not (np.isfinite(p) and p >= ML_PROB_THR):
                            logger.info(f"ML gating blocked entry (p={p:.2f} < {ML_PROB_THR}).")
                            return
                    except Exception as e:
                        logger.warning(f"ML gating error; proceeding without ML: {e}")
                # Adaptive risk: higher risk in strong regime
                strong_regime = (adx_value >= ADX_RISK_THR) and (atr_pct >= ATR_PCT_RISK_THR)
                risk_now = RISK_PCT_STRONG if strong_regime else RISK_PCT_BASE
                self.open_position(c, float(latest["atr"]), risk_pct=risk_now)
                self.state["edge_buy_streak"] = 0
                save_state(self.state)

    # ---------- main loop ----------
    def run(self):
        logger.info("=" * 60)
        logger.info(
            f"Trading Bot — {self.symbol} @ {TIMEFRAME} | Base: {BASE_URL}"
        )
        logger.info(
            f"Risk/Trade: {PORTFOLIO_RISK_PCT*100:.1f}%  | Opportunistic: {OPPORTUNISTIC_MODE} "
            f"(Buy>={EDGE_BUY_SCORE}, Exit<={EDGE_EXIT_SCORE}; Confirm {EDGE_CONFIRM_BARS}/{EDGE_EXIT_CONFIRM_BARS})"
        )
        logger.info("=" * 60)

        if not self.check_and_sync_position():
            raise RuntimeError("Startup reconciliation failed.")

        iter_count = 0
        while True:
            try:
                # Equities respect market clock; crypto runs 24/7
                if not getattr(self, "_is_crypto", False):
                    clock = self.api.get_clock()
                    if not clock.is_open:
                        logger.info(f"Market CLOSED — Next open: {clock.next_open}")
                        time.sleep(60 * 30)
                        continue

                _reset_day_counters_if_needed(self.state)
                latest = fetch_bars_and_indicators_unified(
                    self.api, self.symbol, TIMEFRAME, getattr(self, "_is_crypto", False)
                )

                if latest is not None:
                    self.execute_trade_logic(latest)
                else:
                    logger.warning("No latest data; skipping.")
                logger.info("-" * 60)
                iter_count += 1
                if LOOP_ONCE and iter_count >= 1:
                    logger.info("LOOP_ONCE=true — exiting after one iteration.")
                    break
                _sleep_to_next_bar(SECONDS_PER_BAR)
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user.")
                break
            except Exception as e:
                logger.critical(f"Main loop error: {e}", exc_info=True)
                time.sleep(5)


# ===================== ENTRY =====================

if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run()
    except Exception as e:
        logger.critical(f"Bot failed to start: {e}", exc_info=True)
        raise
