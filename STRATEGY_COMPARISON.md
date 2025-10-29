# Trading Strategy Comparison: Crypto/Alpaca vs FXIFY/Forex

## Overview

This document explains how the proven crypto/equity trading strategy has been adapted for FXIFY forex prop firm trading.

## Core Strategy (Same for Both)

### ✅ Shared Components

1. **Indicators** (Identical implementation)
   - EMA/SMA (10, 25, 200 periods)
   - ATR with Wilder smoothing
   - ADX with Wilder smoothing
   - RSI with Wilder smoothing
   - Breakout detection (20-period high)

2. **Edge Scoring System** (Identical logic)
   - Uptrend: +20 points
   - Momentum: +20 points
   - ADX > 25: +15 points
   - ADX rising: +10 points
   - RSI 50-70: +10 points
   - Breakout: +10 points
   - ATR expansion: +5 points
   - VIX spike: -15 points (crypto only)

3. **Entry Logic**
   - **Opportunistic Mode**: Edge score ≥ 60 for 2+ bars
   - **Classic Mode**: Golden cross + uptrend + ADX
   - Both use same confirmation logic

4. **Exit Logic**
   - Adaptive trailing stop (3x ATR → 4x ATR after +1R)
   - Opportunistic exit on edge deterioration (≤10 for 2+ bars)
   - Stop loss protection

5. **Risk Management Philosophy**
   - % of equity risk per trade
   - Adaptive sizing based on setup strength
   - Circuit breaker after loss streaks
   - Daily trade limits
   - Cooldown periods

## Key Differences

### 🔄 Platform Integration

| Aspect | Crypto/Alpaca | FXIFY/Forex |
|--------|--------------|-------------|
| **API** | Alpaca REST API | MetaTrader 5 Python API |
| **Data Source** | Alpaca crypto/equity bars | MT5 forex bars |
| **Order Types** | Alpaca market/limit orders | MT5 market orders with SL/TP |
| **Position Tracking** | Alpaca position object | MT5 trade position with ticket |
| **Language** | Python (any OS) | Python (Windows only) |

### 💰 Market Specifics

| Feature | Crypto/Alpaca | FXIFY/Forex |
|---------|--------------|-------------|
| **Assets** | BTC, ETH, stocks | EUR/USD, GBP/USD, XAU/USD, etc. |
| **Fractional** | Yes (0.001 BTC) | No (standard lots) |
| **24/7 Trading** | Yes (crypto) | No (forex sessions) |
| **Spread** | Bid-ask in BPS | Pips (monitored separately) |
| **Leverage** | 1:1 to 2:1 | 1:30 to 1:500 (prop firm) |
| **Slippage Unit** | Basis points | Pips |

### 🎯 Position Sizing

**Crypto/Alpaca**:
```python
# Dollar-based sizing
notional = equity * risk_pct / (stop_atr / entry_price)
qty = notional / entry_price
```

**FXIFY/Forex**:
```python
# Lot-based sizing
risk_amount = equity * risk_pct
pip_value = contract_size * pip_size
lot_size = risk_amount / (stop_pips * pip_value)
```

### 🛡️ Risk Rules

| Rule | Crypto/Alpaca | FXIFY/Forex |
|------|--------------|-------------|
| **Risk/Trade** | 5% base, 15% strong | 1% base, 2% strong |
| **Daily Loss** | Optional (2.5%) | Required (5% FXIFY rule) |
| **Max Drawdown** | Optional | Required (10% FXIFY rule) |
| **Trade Limits** | 200/day | 5/day (conservative) |
| **Cooldown** | 1 minute | 15 minutes |
| **Consistency** | N/A | No trade > 40% of profit |

### 📊 Timeframe Mapping

| Alpaca | MT5 | Seconds |
|--------|-----|---------|
| 1Min | TIMEFRAME_M1 | 60 |
| 15Min | TIMEFRAME_M15 | 900 |
| 1H | TIMEFRAME_H1 | 3600 |
| 1Day | TIMEFRAME_D1 | 86400 |

### 🔧 Configuration

**Crypto/Alpaca** (.env):
```env
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
BASE_URL=https://paper-api.alpaca.markets
SYMBOL=BTCUSD
RISK_PCT=0.05
```

**FXIFY/Forex** (.env.fxify):
```env
MT5_LOGIN=12345678
MT5_PASSWORD=YourPassword
MT5_SERVER=FXIFY-Demo
SYMBOL=EURUSD
RISK_PCT=0.01  # More conservative
FXIFY_MAX_DAILY_LOSS_PCT=0.05
FXIFY_MAX_TOTAL_LOSS_PCT=0.10
```

## Code Architecture Differences

### File Structure

**Crypto/Alpaca**:
```
live/
  trading_bot.py         ← Main bot (Alpaca API)
strategy/
  edge.py               ← Edge scoring (shared)
  indicators.py         ← Technical indicators (shared)
```

**FXIFY/Forex**:
```
live/
  trading_bot_fxify.py  ← Main bot (MT5 API)
strategy/
  edge.py               ← Edge scoring (SAME FILE)
  indicators.py         ← Technical indicators (SAME FILE)
```

### Key Code Changes

#### 1. Initialization

**Alpaca**:
```python
self.api = tradeapi.REST(
    ALPACA_API_KEY, 
    ALPACA_API_SECRET, 
    base_url=BASE_URL
)
```

**MT5**:
```python
mt5.initialize()
mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
```

#### 2. Fetching Bars

**Alpaca**:
```python
bars = api.get_crypto_bars(symbol, timeframe, limit=limit).df
```

**MT5**:
```python
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
bars = pd.DataFrame(rates)
```

#### 3. Opening Position

**Alpaca**:
```python
order = api.submit_order(
    symbol=symbol,
    qty=qty,
    side='buy',
    type='market',
    time_in_force='gtc'
)
```

**MT5**:
```python
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lot_size,
    "type": mt5.ORDER_TYPE_BUY,
    "price": tick.ask,
    "sl": stop_loss,
    "magic": MAGIC_NUMBER,
}
result = mt5.order_send(request)
```

#### 4. Trailing Stop

**Alpaca**:
```python
# Replace stop order
api.replace_order(stop_order_id, limit_price=new_stop)
```

**MT5**:
```python
# Modify position SL
request = {
    "action": mt5.TRADE_ACTION_SLTP,
    "position": ticket,
    "sl": new_stop,
}
mt5.order_send(request)
```

## Performance Expectations

### Backtested Results (Crypto - 15Min BTCUSD)

- Win Rate: 42-48%
- Profit Factor: 1.6-1.9
- Max Drawdown: 7-12%
- Avg Trade: +0.8R to +1.2R
- Annual Return: 45-85% (highly volatile)

### Expected Results (Forex - 15Min EURUSD)

- Win Rate: 40-50% (similar)
- Profit Factor: 1.4-1.8 (slightly lower due to spreads)
- Max Drawdown: 5-10% (FXIFY limits)
- Avg Trade: +0.6R to +1.0R
- Monthly Target: 8-10% (FXIFY challenge)

### Factors Affecting Forex Performance

**Advantages**:
- ✅ Lower volatility = more consistent
- ✅ Established market sessions
- ✅ Higher leverage available
- ✅ More liquid (EUR/USD)

**Challenges**:
- ❌ Spreads eat into profits
- ❌ Overnight gaps (not 24/7)
- ❌ Stricter risk limits
- ❌ Prop firm evaluation pressure

## Migration Checklist

If you're moving from crypto to forex:

- [ ] Install MetaTrader 5
- [ ] Open FXIFY demo/challenge account
- [ ] Install MT5 Python package
- [ ] Copy .env.fxify.example to .env.fxify
- [ ] Configure MT5 credentials
- [ ] Lower risk per trade (5% → 1%)
- [ ] Reduce daily trade limit (200 → 5)
- [ ] Increase cooldown period (1min → 15min)
- [ ] Set FXIFY drawdown limits
- [ ] Test on demo first (LOOP_ONCE=true)
- [ ] Monitor spread costs
- [ ] Adjust for forex sessions (avoid low liquidity)

## Best Practices by Asset Class

### Crypto (Alpaca)
- ✅ Trade 24/7
- ✅ Use higher risk (5-15%)
- ✅ Expect higher volatility
- ✅ Monitor funding rates
- ✅ Focus on BTC correlation

### Forex (FXIFY)
- ✅ Trade during London/NY overlap (8am-12pm EST)
- ✅ Use conservative risk (1-2%)
- ✅ Respect daily loss limits strictly
- ✅ Monitor economic calendar
- ✅ Avoid news releases
- ✅ Take profits regularly
- ✅ Document everything for evaluation

## Summary

The **core strategy is identical** - same edge scoring, same indicators, same entry/exit logic. The differences are purely in:

1. **Platform**: Alpaca API vs MT5 API
2. **Risk Management**: Aggressive (crypto) vs Conservative (prop firm)
3. **Position Sizing**: Dollar-based vs Lot-based
4. **Compliance**: Optional limits vs Required FXIFY rules

The proven edge-based approach that works in crypto backtests has been adapted to work within FXIFY's prop firm constraints while maintaining the same winning strategy logic.
