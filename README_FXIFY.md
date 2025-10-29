# FXIFY Trading Bot

This bot adapts the proven crypto/equity trading strategy to work with FXIFY prop firm challenges using MetaTrader 5.

## 🎯 Features

- **Platform**: MetaTrader 5 (MT5) integration
- **Strategy**: Same edge-based + technical analysis from backtested crypto bot
- **Risk Management**: FXIFY-compliant risk controls
- **Indicators**: EMA, ATR, ADX, RSI with Wilder smoothing
- **Entry**: Opportunistic edge score or classic golden cross
- **Exit**: Adaptive trailing stop with profit protection
- **Safety**: Daily loss limits, max drawdown, circuit breaker

## 🏆 FXIFY Compliance

The bot enforces FXIFY prop firm rules:
- ✅ Max Daily Loss: 5% (configurable)
- ✅ Max Total Drawdown: 10% (configurable)
- ✅ Profit Target Tracking: 8-10% (configurable)
- ✅ Consistency Rule: No single trade > 40% of profit
- ✅ Conservative position sizing (1-2% risk per trade)

## 📋 Prerequisites

1. **MetaTrader 5** installed on Windows
2. **FXIFY Account** (demo or live challenge)
3. **Python 3.8+** with pip

## 🚀 Installation

### 1. Install Python Dependencies

```powershell
cd c:\Users\abyla\Desktop\trading_bot_live
pip install -r requirements_fxify.txt
```

### 2. Configure Environment

Copy the example configuration:
```powershell
copy .env.fxify.example .env.fxify
```

Edit `.env.fxify` with your settings:
- MT5 login credentials
- Symbol to trade (e.g., EURUSD, GBPUSD, XAUUSD)
- Risk parameters
- FXIFY challenge rules

### 3. MetaTrader 5 Setup

1. Open MetaTrader 5
2. Login to your FXIFY account
3. Enable algorithmic trading: Tools → Options → Expert Advisors → "Allow algorithmic trading"
4. Ensure Python API is accessible (MT5 should be running)

## ▶️ Running the Bot

### Test Mode (Single Iteration)

```powershell
# Set LOOP_ONCE=true in .env.fxify first
python live\trading_bot_fxify.py
```

### Live Trading

```powershell
# Ensure LOOP_ONCE=false in .env.fxify
python live\trading_bot_fxify.py
```

The bot will:
1. Connect to MT5
2. Verify symbol and account
3. Load previous state (if exists)
4. Start trading loop aligned to timeframe

## 📊 Strategy Overview

### Entry Signals

**Opportunistic Mode (Default)**:
- Edge score ≥ 60 for 2+ consecutive bars
- Considers: trend, momentum, ADX, RSI, breakout, ATR expansion
- Adaptive to market conditions

**Classic Mode**:
- Golden cross (short MA > long MA)
- Price above trend MA (200 EMA)
- ADX > 20 (trending market)

### Exit Signals

1. **Trailing Stop**: 3x ATR initially, widens to 4x ATR after +1R gain
2. **Edge Deterioration**: Exit if edge score drops below 10 for 2+ bars
3. **Stop Loss Hit**: Hard stop at entry - (3x ATR)

### Position Sizing

```
Lot Size = (Account Equity × Risk%) / (Stop Distance in Pips × Pip Value × Contract Size)
```

- Base risk: 1%
- Strong setup risk: 2% (when ADX > 25 and ATR% > 4%)
- Rounded to symbol's minimum lot step

## 🛡️ Risk Management

### Daily Loss Protection
- Tracks equity from day start
- Stops trading if daily loss ≥ 5% (configurable)

### Total Drawdown Protection
- Tracks from initial balance
- Stops trading if total drawdown ≥ 10% (configurable)

### Circuit Breaker
- Triggers after 2 consecutive losses
- Pauses trading for 120 minutes
- Prevents revenge trading

### Cooldown Period
- 15 minutes between trades (configurable)
- Prevents overtrading

### Trade Limits
- Max 5 trades per day (configurable)
- Conservative for prop firm rules

## 📁 Files

- `fxify_bot_state.json`: Persistent state (position, equity tracking, counters)
- `logs/fxify_trading_bot.log`: Detailed trading log (rotating, 10MB max)

## 🎛️ Key Configuration Parameters

### Conservative Settings (Recommended for Challenge)
```env
RISK_PCT=0.01          # 1% risk per trade
MAX_TRADES_PER_DAY=5   # 5 trades max
MIN_COOLDOWN_MIN=15    # 15 min cooldown
EDGE_BUY_SCORE=60      # Selective entries
TRAIL_ATR_MULT=3.0     # Tight trailing stop
```

### Aggressive Settings (For Experienced Traders)
```env
RISK_PCT=0.02          # 2% risk per trade
MAX_TRADES_PER_DAY=10  # 10 trades max
MIN_COOLDOWN_MIN=5     # 5 min cooldown
EDGE_BUY_SCORE=50      # More entries
TRAIL_ATR_MULT=4.0     # Wider trailing stop
```

## 📈 Recommended Pairs

### Major Forex Pairs (Low Spread)
- EURUSD - Most liquid
- GBPUSD - Good volatility
- USDJPY - Asian session
- AUDUSD - Commodity currency

### Volatile Pairs (Higher Risk/Reward)
- XAUUSD (Gold) - High ATR
- GBPJPY - High volatility
- EURJPY - Trending

### Timeframes
- **15Min**: Balanced (default)
- **1H**: Swing trading, fewer trades
- **5Min**: Scalping, more trades
- **4H**: Position trading

## ⚠️ Important Notes

### Before Running on Live Challenge
1. ✅ Test on demo account first
2. ✅ Verify MT5 connection works
3. ✅ Check symbol specifications (spreads, swaps)
4. ✅ Understand FXIFY rules for your challenge phase
5. ✅ Monitor first few trades closely
6. ✅ Keep MT5 running (don't close terminal)

### Prop Firm Best Practices
- Start with small position sizes
- Trade during liquid hours (avoid spreads widening)
- Respect daily loss limits strictly
- Document all trades and decisions
- Don't force trades - wait for quality setups
- Take profits regularly (don't be greedy)

### Limitations
- Requires Windows (MT5 Python API limitation)
- MT5 must be running during bot operation
- No VIX integration (forex-specific)
- Single symbol per bot instance

## 🔧 Troubleshooting

### Bot Won't Connect to MT5
- Ensure MT5 is running and logged in
- Check MT5 Python API is enabled
- Verify credentials in `.env.fxify`
- Try specifying `MT5_PATH` to terminal64.exe

### No Trades Being Placed
- Check edge score requirements (lower `EDGE_BUY_SCORE`)
- Verify ATR% is above minimum (`MIN_EDGE_ATR_PCT`)
- Check spread limits (`MAX_SPREAD_PIPS`)
- Review logs for filter rejections

### Positions Not Opening
- Verify account has sufficient margin
- Check symbol is tradable
- Ensure lot size ≥ minimum volume
- Review MT5 error codes in logs

### Daily Loss Limit Hit Prematurely
- Check `day_start_equity` in state file
- Verify FXIFY limits are correct
- Consider resetting state file (carefully)

## 📞 Support

For issues specific to:
- **Strategy Logic**: Review `strategy/edge.py` and `strategy/indicators.py`
- **MT5 Integration**: Check MetaTrader5 Python documentation
- **FXIFY Rules**: Contact FXIFY support for clarification

## 📊 Monitoring

Watch for these log messages:

✅ **Good Signs**:
- `Edge entry signal confirmed`
- `Position opened`
- `Trailing stop updated`
- `PROFIT TARGET REACHED`

⚠️ **Warnings**:
- `Loss streak: X`
- `Spread too high`
- `ATR too low`
- `Circuit breaker active`

❌ **Errors**:
- `Max daily loss reached`
- `Max total loss reached`
- `Failed to get account info`
- `Order send failed`

## 🎯 Expected Performance

Based on backtests (crypto/equity):
- **Win Rate**: 40-50%
- **Profit Factor**: 1.5-2.0
- **Max Drawdown**: 5-10%
- **Avg R-multiple**: 1.2-1.5

*Note: Forex performance may vary. Past performance doesn't guarantee future results.*

## 📝 License & Disclaimer

This bot is for educational and testing purposes. Trading involves risk. Always test on demo accounts first. Not financial advice.
