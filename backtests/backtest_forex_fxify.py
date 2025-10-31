"""
FXIFY Forex Backtester - Using Same Strategy as Crypto Bot

Tests the edge-based + technical strategy on forex pairs using yfinance data.
Simulates FXIFY prop firm rules and conservative risk management.

Strategy:
- Entry: Edge score >= 60 OR (Golden cross + uptrend + ADX)
- Exit: Adaptive trailing stop (3x ATR -> 4x ATR after +1R)
- Position sizing: 1-2% risk per trade based on setup strength
- FXIFY compliance: Max 5% daily loss, 10% total loss

Outputs:
- Console summary (Return, MaxDD, Sharpe, Win Rate, Profit Factor)
- CSV reports under backtests/reports/forex_fxify/
- FXIFY compliance metrics
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root for strategy imports
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install yfinance: pip install yfinance")

from strategy.indicators import compute_atr_wilder, calculate_adx, calculate_rsi
from strategy.edge import compute_edge_features_and_score, EdgeResult

# ===================== CONFIG =====================

# FXIFY limits
FXIFY_MAX_DAILY_LOSS_PCT = 0.05  # 5%
FXIFY_MAX_TOTAL_LOSS_PCT = 0.10  # 10%
FXIFY_PROFIT_TARGET_PCT = 0.08   # 8%

# Risk parameters (conservative for prop firm)
RISK_PCT_BASE = 0.01      # 1% base risk
RISK_PCT_STRONG = 0.02    # 2% on strong setups
ADX_RISK_THR = 25         # ADX threshold for strong setup
ATR_PCT_RISK_THR = 0.04   # ATR% threshold for strong setup

# Strategy parameters
EDGE_BUY_SCORE = 60
EDGE_EXIT_SCORE = 10
EDGE_CONFIRM_BARS = 2
EDGE_EXIT_CONFIRM_BARS = 2

# Indicator parameters
P = {
    "short": 10,
    "long": 25,
    "trend": 200,
    "atr_period": 14,
    "trailing_stop_atr_mult": 3.0,
    "adx_threshold": 20,
    "vix_spike_threshold": 20.0,
}

INDICATOR = {
    "rsi_period": 14,
    "adx_period": 14,
    "breakout_lookback": 20,
    "adx_slope_lookback": 3,
}

# Adaptive trailing
TRAIL_WIDEN_AFTER_R = 1.0
TRAIL_ABS_WIDEN_TO = 4.0

# Trading limits
MAX_TRADES_PER_DAY = 5
MIN_COOLDOWN_BARS = 1  # Minimum bars between trades

# Costs and slippage model
SPREAD_PIPS = 1.0  # Typical spread (baseline)
SLIPPAGE_PIPS = 0.5  # Legacy fixed slippage (used if DYNAMIC_SLIPPAGE=False)
DYNAMIC_SLIPPAGE = True

def _base_slippage_pips(symbol: str) -> float:
    s = symbol.upper()
    if s.startswith("EURUSD"):
        return 0.2
    if "JPY" in s:
        return 0.3
    if s.startswith("XAU") or "GOLD" in s:
        return 0.8
    return 0.25

def _slip_scale(symbol: str) -> float:
    s = symbol.upper()
    if s.startswith("EURUSD"):
        return 2.0
    if "JPY" in s:
        return 2.5
    if s.startswith("XAU") or "GOLD" in s:
        return 8.0
    return 2.0

def compute_slippage_pips(symbol: str, atr_pct: float, spread_pips: float = SPREAD_PIPS) -> float:
    """Volatility-adaptive slippage in pips.

    atr_pct is in percent (e.g., 0.06 means 0.06%).
    Model: base + scale * max(0, atr_pct - 0.05)
    Capped to 0.9 * spread to avoid pathological over-slippage.
    """
    try:
        base = _base_slippage_pips(symbol)
        scale = _slip_scale(symbol)
        vol_excess = max(0.0, float(atr_pct) - 0.05)
        slip = base + scale * vol_excess
        cap = max(0.1, 0.9 * float(spread_pips))
        return float(min(slip, cap))
    except Exception:
        return float(SLIPPAGE_PIPS)

EPS = 1e-12

# ===================== DATA UTILITIES =====================

def download_forex(symbol: str, start: str, interval: str) -> pd.DataFrame:
    """Download forex data from yfinance with simple fallbacks for metals."""
    print(f"Downloading {symbol} from {start} ({interval})...")

    # Primary mapping: spot FX style e.g., EURUSD -> EURUSD=X
    def _primary(sym: str) -> str:
        if "=" not in sym and "/" not in sym:
            return f"{sym}=X"
        return sym

    candidates = []
    base = symbol.upper()
    candidates.append(_primary(base))
    # Metals fallbacks: some environments fail on XAUUSD=X; try XAU=X (per-ounce USD index) and GC=F (COMEX gold futures)
    if base.startswith("XAU") or "GOLD" in base:
        for alt in ("XAU=X", "GC=F"):
            if alt not in candidates:
                candidates.append(alt)

    last_err = None
    for yf_symbol in candidates:
        try:
            df = yf.download(yf_symbol, start=start, interval=interval, progress=False, auto_adjust=False, actions=False)
        except Exception as e:
            last_err = e
            df = None
        if df is not None and not df.empty:
            print(f"  Loaded from {yf_symbol}")
            break
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} {interval} (tried {candidates}); last error: {last_err}")
    
    # Handle MultiIndex columns - yfinance can return (Price, Symbol) structure  
    if isinstance(df.columns, pd.MultiIndex):
        # The first level should be Price type (Open, High, Low, Close, etc.)
        # Get that level
        df.columns = df.columns.get_level_values(0)
    
    # Handle MultiIndex in index
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(-1)
    
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    
    # Standardize column names (handle case variations)
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    # Select OHLC columns
    required_cols = []
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            required_cols.append(col)
    
    if len(required_cols) < 4:
        raise RuntimeError(f"Missing OHLC columns. Available: {df.columns.tolist()}")
    
    df = df[required_cols].dropna()
    print(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    return df

def get_pip_value(symbol: str) -> float:
    """Get pip value for different forex pairs"""
    if "JPY" in symbol:
        return 0.01  # JPY pairs: 1 pip = 0.01
    elif "XAU" in symbol or "GOLD" in symbol:
        return 0.10  # Gold: 1 pip = 0.10
    elif "XAG" in symbol or "SILVER" in symbol:
        return 0.01  # Silver: 1 pip = 0.01
    else:
        return 0.0001  # Standard forex: 1 pip = 0.0001

# ===================== BACKTEST DATA STRUCTURES =====================

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    entry_atr: float
    position_size: float  # In lots
    initial_stop: float
    entry_equity: float
    edge_score: int
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    r_multiple: Optional[float] = None
    reason: str = ""
    max_favorable: float = 0.0
    max_adverse: float = 0.0

@dataclass
class DayStats:
    date: str
    trades: int
    pnl: float
    pnl_pct: float
    equity_start: float
    equity_end: float

# ===================== BACKTEST ENGINE =====================

def compute_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators on bars"""
    df = bars.copy()
    
    # Moving averages
    df["short_ma"] = df["close"].ewm(span=P["short"], adjust=False).mean()
    df["long_ma"] = df["close"].ewm(span=P["long"], adjust=False).mean()
    df["trend_ma"] = df["close"].ewm(span=P["trend"], adjust=False).mean()
    
    # Technical indicators
    df["atr"] = compute_atr_wilder(df, P["atr_period"])
    df["adx"] = calculate_adx(df, INDICATOR["adx_period"])
    df["rsi"] = calculate_rsi(df["close"], INDICATOR["rsi_period"])
    
    # Breakout indicators
    L = INDICATOR["breakout_lookback"]
    df["highL"] = df["high"].rolling(L).max()
    df["prev_high20"] = df["highL"].shift(1)
    df["atr_median20"] = df["atr"].rolling(20).median()
    
    # ADX slope
    K = INDICATOR["adx_slope_lookback"]
    df["adx_slope"] = df["adx"] - df["adx"].shift(K)
    
    # ATR percentage
    df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan) * 100.0
    
    return df.dropna()

def backtest_fxify(
    bars: pd.DataFrame,
    symbol: str,
    initial_capital: float = 10000.0,
    opportunistic_mode: bool = True
) -> tuple[pd.Series, List[Trade], List[DayStats]]:
    """
    Run backtest with FXIFY rules and edge-based strategy
    """
    print(f"\nBacktesting {symbol}...")
    print(f"  Initial capital: ${initial_capital:,.2f}")
    print(f"  Mode: {'Opportunistic (Edge-based)' if opportunistic_mode else 'Classic (Golden Cross)'}")
    print(f"  Bars: {len(bars)}")
    
    # Compute indicators
    df = compute_indicators(bars)
    pip_value = get_pip_value(symbol)
    spread_cost = SPREAD_PIPS * pip_value
    
    # State
    equity = initial_capital
    initial_balance = initial_capital
    high_water_mark = initial_capital
    position = None
    trades: List[Trade] = []
    equity_curve = []
    day_stats: List[DayStats] = []
    
    # Daily tracking
    current_day = None
    day_start_equity = equity
    day_trades = 0
    day_pnl = 0.0
    
    # Streaks
    edge_buy_streak = 0
    edge_exit_streak = 0
    last_trade_idx = -999
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = float(row["close"])
        
        # Day tracking
        trade_date = row.name.date()
        if current_day != trade_date:
            # Save previous day stats
            if current_day is not None:
                day_stats.append(DayStats(
                    date=str(current_day),
                    trades=day_trades,
                    pnl=day_pnl,
                    pnl_pct=(day_pnl / day_start_equity * 100) if day_start_equity > 0 else 0,
                    equity_start=day_start_equity,
                    equity_end=equity
                ))
            # Reset for new day
            current_day = trade_date
            day_start_equity = equity
            day_trades = 0
            day_pnl = 0.0
        
        # Check FXIFY daily loss limit
        daily_loss = day_start_equity - equity
        daily_loss_pct = daily_loss / day_start_equity if day_start_equity > 0 else 0
        if daily_loss_pct >= FXIFY_MAX_DAILY_LOSS_PCT:
            if position:
                # Force close position
                if DYNAMIC_SLIPPAGE:
                    slip_pips = compute_slippage_pips(symbol, float(row.get("atr_pct", 0.05)))
                else:
                    slip_pips = SLIPPAGE_PIPS
                slippage_cost = slip_pips * pip_value
                exit_price = price - spread_cost - slippage_cost
                pnl = (exit_price - position.entry_price) * position.position_size * 100000
                equity += pnl
                position.exit_time = row.name
                position.exit_price = exit_price
                position.pnl = pnl
                position.pnl_pct = (pnl / position.entry_equity) * 100
                position.reason = "FXIFY Daily Loss Limit"
                trades.append(position)
                position = None
            equity_curve.append(equity)
            continue
        
        # Check FXIFY total loss limit
        total_loss = initial_balance - equity
        total_loss_pct = total_loss / initial_balance if initial_balance > 0 else 0
        if total_loss_pct >= FXIFY_MAX_TOTAL_LOSS_PCT:
            if position:
                # Force close
                if DYNAMIC_SLIPPAGE:
                    slip_pips = compute_slippage_pips(symbol, float(row.get("atr_pct", 0.05)))
                else:
                    slip_pips = SLIPPAGE_PIPS
                slippage_cost = slip_pips * pip_value
                exit_price = price - spread_cost - slippage_cost
                pnl = (exit_price - position.entry_price) * position.position_size * 100000
                equity += pnl
                position.exit_time = row.name
                position.exit_price = exit_price
                position.pnl = pnl
                position.pnl_pct = (pnl / position.entry_equity) * 100
                position.reason = "FXIFY Total Loss Limit"
                trades.append(position)
                position = None
            equity_curve.append(equity)
            continue
        
        # Update high water mark
        high_water_mark = max(high_water_mark, equity)
        
        # POSITION MANAGEMENT
        if position:
            # Update MAE/MFE
            position.max_favorable = max(position.max_favorable, price - position.entry_price)
            position.max_adverse = min(position.max_adverse, price - position.entry_price)
            
            # Calculate R-multiple
            initial_risk = position.entry_atr * P["trailing_stop_atr_mult"]
            r_mult = (price - position.entry_price) / initial_risk if initial_risk > 0 else 0
            
            # Adaptive trailing stop
            trail_mult = P["trailing_stop_atr_mult"]
            if r_mult >= TRAIL_WIDEN_AFTER_R:
                trail_mult = TRAIL_ABS_WIDEN_TO
            
            current_atr = float(row["atr"])
            trailing_stop = price - (trail_mult * current_atr)
            
            # Check stop hit
            stop_hit = price <= trailing_stop
            
            # Check edge deterioration
            if opportunistic_mode:
                edge: EdgeResult = compute_edge_features_and_score(
                    df.iloc[:i+1], row, prev, 0.0, P["vix_spike_threshold"]
                )
                if edge.score <= EDGE_EXIT_SCORE:
                    edge_exit_streak += 1
                else:
                    edge_exit_streak = 0
                
                edge_exit = edge_exit_streak >= EDGE_EXIT_CONFIRM_BARS
            else:
                edge_exit = False
            
            # Exit logic
            if stop_hit or edge_exit:
                if DYNAMIC_SLIPPAGE:
                    slip_pips = compute_slippage_pips(symbol, float(row.get("atr_pct", 0.05)))
                else:
                    slip_pips = SLIPPAGE_PIPS
                slippage_cost = slip_pips * pip_value
                exit_price = price - spread_cost - slippage_cost
                pnl = (exit_price - position.entry_price) * position.position_size * 100000
                equity += pnl
                day_pnl += pnl
                
                position.exit_time = row.name
                position.exit_price = exit_price
                position.pnl = pnl
                position.pnl_pct = (pnl / position.entry_equity) * 100
                position.r_multiple = r_mult
                position.reason = "Trailing Stop" if stop_hit else "Edge Exit"
                
                trades.append(position)
                position = None
                edge_exit_streak = 0
                last_trade_idx = i
        
        # ENTRY LOGIC
        else:
            # Cooldown
            if i - last_trade_idx < MIN_COOLDOWN_BARS:
                equity_curve.append(equity)
                continue
            
            # Daily trade limit
            if day_trades >= MAX_TRADES_PER_DAY:
                equity_curve.append(equity)
                continue
            
            # Compute edge score
            edge: EdgeResult = compute_edge_features_and_score(
                df.iloc[:i+1], row, prev, 0.0, P["vix_spike_threshold"]
            )
            
            entry_signal = False
            
            if opportunistic_mode:
                # Edge-based entry
                if edge.score >= EDGE_BUY_SCORE:
                    edge_buy_streak += 1
                else:
                    edge_buy_streak = 0
                
                if edge_buy_streak >= EDGE_CONFIRM_BARS:
                    entry_signal = True
                    edge_buy_streak = 0  # Reset after entry
            else:
                # Classic golden cross entry
                uptrend = price > row["trend_ma"]
                golden = row["short_ma"] > row["long_ma"]
                adx_ok = row["adx"] > P["adx_threshold"]
                entry_signal = uptrend and golden and adx_ok
            
            if entry_signal:
                # Position sizing
                atr = float(row["atr"])
                
                # Adaptive risk
                risk_pct = RISK_PCT_BASE
                if row["adx"] > ADX_RISK_THR and row["atr_pct"] > ATR_PCT_RISK_THR:
                    risk_pct = RISK_PCT_STRONG
                
                # Calculate lot size
                if DYNAMIC_SLIPPAGE:
                    slip_pips = compute_slippage_pips(symbol, float(row.get("atr_pct", 0.05)))
                else:
                    slip_pips = SLIPPAGE_PIPS
                slippage_cost = slip_pips * pip_value
                entry_price = price + spread_cost + slippage_cost
                risk_dollars = equity * risk_pct
                stop_distance = P["trailing_stop_atr_mult"] * atr
                stop_pips = stop_distance / pip_value
                
                # Lot size = risk / (stop_pips * pip_value * contract_size)
                # For simplicity, 1 lot = 100,000 units
                lot_size = risk_dollars / (stop_pips * pip_value * 100000)
                lot_size = max(0.01, round(lot_size, 2))  # Min 0.01 lot
                
                # Create position
                position = Trade(
                    entry_time=row.name,
                    entry_price=entry_price,
                    entry_atr=atr,
                    position_size=lot_size,
                    initial_stop=entry_price - stop_distance,
                    entry_equity=equity,
                    edge_score=edge.score
                )
                
                day_trades += 1
        
        equity_curve.append(equity)
    
    # Close any remaining position
    if position:
        exit_price = df.iloc[-1]["close"] - spread_cost
        pnl = (exit_price - position.entry_price) * position.position_size * 100000
        equity += pnl
        position.exit_time = df.iloc[-1].name
        position.exit_price = exit_price
        position.pnl = pnl
        position.pnl_pct = (pnl / position.entry_equity) * 100
        position.reason = "End of backtest"
        trades.append(position)
    
    # Final day stats
    if current_day is not None:
        day_stats.append(DayStats(
            date=str(current_day),
            trades=day_trades,
            pnl=day_pnl,
            pnl_pct=(day_pnl / day_start_equity * 100) if day_start_equity > 0 else 0,
            equity_start=day_start_equity,
            equity_end=equity
        ))
    
    equity_series = pd.Series(equity_curve, index=df.index[1:])
    return equity_series, trades, day_stats

# ===================== ANALYSIS =====================

def analyze_results(equity: pd.Series, trades: List[Trade], day_stats: List[DayStats], 
                     initial_capital: float, symbol: str):
    """Analyze and print backtest results"""
    print("\n" + "="*80)
    print(f"BACKTEST RESULTS: {symbol}")
    print("="*80)
    
    # Overall performance
    final_equity = equity.iloc[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    rets = equity.pct_change().fillna(0)
    sharpe = (np.sqrt(252) * rets.mean() / (rets.std() + EPS)) if len(rets) > 1 else 0
    
    print(f"\n📊 PERFORMANCE METRICS")
    print(f"  Initial Capital:  ${initial_capital:,.2f}")
    print(f"  Final Equity:     ${final_equity:,.2f}")
    print(f"  Total Return:     {total_return:+.2f}%")
    print(f"  Max Drawdown:     {max_dd:.2f}%")
    print(f"  Sharpe Ratio:     {sharpe:.2f}")
    
    # Trade statistics
    if len(trades) == 0:
        print("\n⚠️  No trades executed")
        return
    
    closed_trades = [t for t in trades if t.exit_time is not None]
    winning_trades = [t for t in closed_trades if (t.pnl or 0) > 0]
    losing_trades = [t for t in closed_trades if (t.pnl or 0) < 0]
    
    win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
    
    avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0
    
    profit_factor = (sum([t.pnl for t in winning_trades]) / 
                     sum([abs(t.pnl) for t in losing_trades])) if losing_trades else float('inf')
    
    avg_r = np.mean([t.r_multiple for t in closed_trades if t.r_multiple is not None])
    
    print(f"\n📈 TRADE STATISTICS")
    print(f"  Total Trades:     {len(closed_trades)}")
    print(f"  Winning Trades:   {len(winning_trades)}")
    print(f"  Losing Trades:    {len(losing_trades)}")
    print(f"  Win Rate:         {win_rate:.1f}%")
    print(f"  Avg Win:          ${avg_win:,.2f}")
    print(f"  Avg Loss:         ${avg_loss:,.2f}")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print(f"  Avg R-Multiple:   {avg_r:.2f}R")
    
    # FXIFY compliance
    max_daily_loss_pct = max([abs(d.pnl_pct) for d in day_stats if d.pnl < 0], default=0)
    total_loss_pct = (initial_capital - equity.min()) / initial_capital * 100
    profit_target_reached = total_return >= (FXIFY_PROFIT_TARGET_PCT * 100)
    
    print(f"\n🏆 FXIFY COMPLIANCE")
    print(f"  Max Daily Loss:   {max_daily_loss_pct:.2f}% (Limit: {FXIFY_MAX_DAILY_LOSS_PCT*100:.1f}%)")
    print(f"  Max Total DD:     {abs(max_dd):.2f}% (Limit: {FXIFY_MAX_TOTAL_LOSS_PCT*100:.1f}%)")
    print(f"  Profit Target:    {total_return:.2f}% (Target: {FXIFY_PROFIT_TARGET_PCT*100:.1f}%)")
    print(f"  Target Reached:   {'✅ YES' if profit_target_reached else '❌ NO'}")
    
    # Best/worst trades
    if closed_trades:
        best_trade = max(closed_trades, key=lambda t: t.pnl or 0)
        worst_trade = min(closed_trades, key=lambda t: t.pnl or 0)
        
        print(f"\n🏅 NOTABLE TRADES")
        print(f"  Best Trade:       ${best_trade.pnl:,.2f} ({best_trade.pnl_pct:+.2f}%)")
        print(f"  Worst Trade:      ${worst_trade.pnl:,.2f} ({worst_trade.pnl_pct:+.2f}%)")
    
    print("\n" + "="*80)

def save_results(equity: pd.Series, trades: List[Trade], day_stats: List[DayStats],
                 symbol: str, interval: str, output_dir: Path):
    """Save backtest results to CSV files"""
    out_path = output_dir / f"{symbol.replace('=','').replace('/','_')}_{interval}"
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Equity curve
    equity.to_csv(out_path / "equity_curve.csv", header=["equity"])
    
    # Trades
    if trades:
        trades_df = pd.DataFrame([{
            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "entry_atr": t.entry_atr,
            "position_size": t.position_size,
            "edge_score": t.edge_score,
            "exit_time": t.exit_time,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "r_multiple": t.r_multiple,
            "reason": t.reason,
        } for t in trades])
        trades_df.to_csv(out_path / "trades.csv", index=False)
    
    # Daily stats
    if day_stats:
        day_df = pd.DataFrame([{
            "date": d.date,
            "trades": d.trades,
            "pnl": d.pnl,
            "pnl_pct": d.pnl_pct,
            "equity_start": d.equity_start,
            "equity_end": d.equity_end,
        } for d in day_stats])
        day_df.to_csv(out_path / "daily_stats.csv", index=False)
    
    print(f"\n💾 Results saved to: {out_path}")

# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description="FXIFY Forex Backtester")
    parser.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY,AUDUSD,XAUUSD",
                        help="Comma-separated forex symbols")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="15m", help="Timeframe (1m,5m,15m,1h,1d)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--mode", default="opportunistic", 
                        choices=["opportunistic", "classic"],
                        help="Entry mode: opportunistic (edge) or classic (golden cross)")
    parser.add_argument("--output", default="backtests/reports/forex_fxify",
                        help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    opportunistic = args.mode == "opportunistic"
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    
    print("="*80)
    print("FXIFY FOREX BACKTESTER")
    print("="*80)
    print(f"Symbols:  {', '.join(symbols)}")
    print(f"Start:    {args.start}")
    print(f"Interval: {args.interval}")
    print(f"Capital:  ${args.capital:,.2f}")
    print(f"Mode:     {args.mode.capitalize()}")
    print("="*80)
    
    all_results = []
    
    for symbol in symbols:
        try:
            # Download data
            bars = download_forex(symbol, args.start, args.interval)
            
            # Run backtest
            equity, trades, day_stats = backtest_fxify(
                bars, symbol, args.capital, opportunistic
            )
            
            # Analyze
            analyze_results(equity, trades, day_stats, args.capital, symbol)
            
            # Save
            save_results(equity, trades, day_stats, symbol, args.interval, output_dir)
            
            # Collect summary
            if len(equity) > 0:
                final_equity = equity.iloc[-1]
                total_return = (final_equity / args.capital - 1) * 100
                all_results.append({
                    "symbol": symbol,
                    "return_pct": total_return,
                    "trades": len(trades),
                })
        
        except Exception as e:
            print(f"\n❌ Error backtesting {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary across all symbols
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY ACROSS ALL SYMBOLS")
        print("="*80)
        for r in all_results:
            print(f"  {r['symbol']:10s}  {r['return_pct']:+7.2f}%  ({r['trades']} trades)")
        print("="*80)

if __name__ == "__main__":
    main()
