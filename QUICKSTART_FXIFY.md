# 🚀 FXIFY Quick Start Guide

Get your FXIFY forex trading bot running in 5 minutes!

## Prerequisites

- ✅ Windows PC
- ✅ Python 3.8+ installed
- ✅ MetaTrader 5 installed
- ✅ FXIFY account (demo or challenge)

## Step-by-Step Setup

### 1️⃣ Install Dependencies

Open PowerShell in the project directory and run:

```powershell
.\setup_fxify.ps1
```

This will:
- Check Python installation
- Install required packages
- Create configuration file
- Set up logging directory

### 2️⃣ Configure Your Account

Edit the `.env.fxify` file with your MT5 credentials:

```env
MT5_LOGIN=12345678          # Your MT5 account number
MT5_PASSWORD=YourPassword   # Your MT5 password
MT5_SERVER=FXIFY-Demo       # Your broker server
SYMBOL=EURUSD               # Pair to trade
```

### 3️⃣ Start MetaTrader 5

1. Open MetaTrader 5
2. Login to your FXIFY account
3. Go to: **Tools → Options → Expert Advisors**
4. Check ✅ "Allow algorithmic trading"
5. Keep MT5 running

### 4️⃣ Test Connection

Run the connection test:

```powershell
python test_fxify_connection.py
```

You should see:
```
✅ MT5 initialized
✅ MT5 logged in
✅ Account info retrieved
✅ Symbol EURUSD available
✅ All tests passed!
```

### 5️⃣ Test Run (Dry Run)

Do a single iteration test:

```powershell
# Edit .env.fxify and set: LOOP_ONCE=true
python live\trading_bot_fxify.py
```

Check the logs in `logs/fxify_trading_bot.log` to verify everything works.

### 6️⃣ Go Live

Once verified, start the bot:

```powershell
# Edit .env.fxify and set: LOOP_ONCE=false
python live\trading_bot_fxify.py
```

The bot will now run continuously!

## What the Bot Does

```
┌─────────────────────────────────────────┐
│ Every 15 minutes (configurable):        │
├─────────────────────────────────────────┤
│ 1. Fetch latest price bars from MT5    │
│ 2. Calculate indicators (EMA,ATR,ADX)   │
│ 3. Compute edge score (0-100)           │
│ 4. Check FXIFY risk limits              │
│ 5. Evaluate entry/exit signals          │
│ 6. Place/manage trades via MT5          │
│ 7. Update trailing stops                │
│ 8. Log everything                        │
└─────────────────────────────────────────┘
```

## Key Settings

### Conservative (Recommended for Challenge)

```env
RISK_PCT=0.01               # Risk 1% per trade
MAX_TRADES_PER_DAY=5        # Max 5 trades daily
EDGE_BUY_SCORE=60           # Selective entries
TRAIL_ATR_MULT=3.0          # Tight stops
```

### Moderate

```env
RISK_PCT=0.015              # Risk 1.5% per trade
MAX_TRADES_PER_DAY=8        # Max 8 trades daily
EDGE_BUY_SCORE=55           # More entries
TRAIL_ATR_MULT=3.5          # Balanced stops
```

### Aggressive (⚠️ Higher Risk)

```env
RISK_PCT=0.02               # Risk 2% per trade
MAX_TRADES_PER_DAY=10       # Max 10 trades daily
EDGE_BUY_SCORE=50           # Many entries
TRAIL_ATR_MULT=4.0          # Wider stops
```

## Monitoring

### Live Dashboard (Terminal Output)

```
🔄 Starting iteration at 2025-10-28 14:30:00
📊 Price: 1.08450, ATR: 0.00082, ADX: 28.3, RSI: 58.2
🎯 Edge Score: 65/100 - Uptrend, Momentum up, ADX>25, RSI 50-70
📭 No position
   Spread: 1.2 pips
✅ Edge entry signal confirmed: 2 bars
📈 Opening BUY position: Size=0.1 lots, Entry=1.08451, SL=1.08205
✅ Position opened: Ticket=123456789, Price=1.08451
✅ Iteration complete
```

### Log Files

- **Location**: `logs/fxify_trading_bot.log`
- **Rotation**: 10MB max, 5 backups
- **Format**: Timestamp, level, message

### State File

- **Location**: `fxify_bot_state.json`
- **Contains**: Position info, equity tracking, counters
- **Updated**: After every trade

## Safety Features

### 🛡️ FXIFY Compliance

| Protection | Default | Purpose |
|------------|---------|---------|
| Max Daily Loss | 5% | Stop trading if daily loss reaches limit |
| Max Total Loss | 10% | Shut down if total drawdown hits limit |
| Profit Target | 8% | Track progress toward challenge goal |
| Consistency | 40% | No single trade > 40% of total profit |

### 🔴 Circuit Breaker

After **2 consecutive losses**:
- ⏸️ Pause trading for **120 minutes**
- 📊 Log warning
- 🔄 Auto-resume after cooldown

### ⏱️ Cooldown Period

Between trades:
- ⏱️ Wait **15 minutes** minimum
- 🚫 Prevents overtrading
- 📉 Reduces emotional decisions

### 📊 Daily Limits

- Max **5 trades per day** (default)
- Resets at midnight UTC
- Configurable via `MAX_TRADES_PER_DAY`

## Common Issues

### ❌ "MT5 initialize failed"

**Solution**:
1. Ensure MT5 is installed and running
2. Check MT5 is logged into your account
3. Try specifying `MT5_PATH` in `.env.fxify`:
   ```env
   MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
   ```

### ❌ "Symbol EURUSD not found"

**Solution**:
1. Right-click Market Watch in MT5
2. Select "Show All"
3. Find your symbol and verify it's visible

### ❌ "Order failed: TRADE_RETCODE_NO_MONEY"

**Solution**:
1. Check account balance
2. Reduce `RISK_PCT` (try 0.005 = 0.5%)
3. Verify leverage is sufficient

### ❌ "No trades being placed"

**Solution**:
1. Lower `EDGE_BUY_SCORE` (try 50 instead of 60)
2. Check `MIN_EDGE_ATR_PCT` isn't too high
3. Verify spread isn't exceeding `MAX_SPREAD_PIPS`
4. Review logs for rejection reasons

### ❌ "Max daily loss reached"

**Solution**:
1. Wait until next trading day
2. Review trades in MT5 history
3. Consider lowering `RISK_PCT`
4. Manually edit `fxify_bot_state.json` if needed (carefully!)

## Best Trading Times (UTC)

| Session | Time (UTC) | Pairs | Notes |
|---------|------------|-------|-------|
| **London** | 08:00-17:00 | EUR, GBP | High volume |
| **New York** | 13:00-22:00 | USD | High volume |
| **Overlap** | 13:00-17:00 | All majors | ⭐ Best time |
| **Asian** | 00:00-09:00 | JPY, AUD | Lower volume |

**Avoid**:
- ⚠️ Major news releases (NFP, FOMC, ECB)
- ⚠️ Weekends (spreads widen)
- ⚠️ Holidays (low liquidity)

## Recommended Pairs

### Beginners
- **EURUSD** - Most liquid, tight spreads
- **GBPUSD** - Good volatility, reasonable spreads

### Intermediate
- **USDJPY** - Different behavior, good for diversification
- **AUDUSD** - Commodity-linked, trending

### Advanced
- **XAUUSD (Gold)** - High volatility, larger moves
- **GBPJPY** - Very volatile, higher risk/reward

## Performance Tracking

Monitor these metrics daily:

```python
# In fxify_bot_state.json
{
  "initial_balance": 10000.00,
  "day_start_equity": 10200.00,
  "total_profit": 200.00,           # +2%
  "trades_today": 3,
  "loss_streak": 0,
  "in_position": false
}
```

### MT5 Account Terminal

Track:
- Balance vs Equity
- Total profit
- Win rate
- Average trade duration

## Stopping the Bot

### Graceful Shutdown

Press `Ctrl+C` in the terminal:
```
Received signal 2. Shutting down gracefully.
Shutting down MT5
```

The bot will:
1. ✅ Save current state
2. ✅ Close MT5 connection
3. ✅ Exit cleanly

⚠️ **Do NOT force close** - let it shut down gracefully

### Emergency Stop

If needed:
1. Close all positions manually in MT5
2. Stop the Python script
3. Edit `fxify_bot_state.json`:
   ```json
   {
     "in_position": false,
     "position_ticket": null
   }
   ```

## Next Steps

Once you're comfortable:

1. 📚 Read `README_FXIFY.md` for detailed docs
2. 📊 Review `STRATEGY_COMPARISON.md` to understand the strategy
3. 🎯 Optimize settings for your trading style
4. 📈 Track performance and iterate
5. 💼 Scale up after consistent demo results

## Support & Resources

- **Strategy Files**: `strategy/edge.py`, `strategy/indicators.py`
- **Documentation**: `README_FXIFY.md`
- **Comparison**: `STRATEGY_COMPARISON.md`
- **Logs**: `logs/fxify_trading_bot.log`
- **State**: `fxify_bot_state.json`

## Important Reminders

⚠️ **Always test on demo first**
⚠️ **Never risk more than you can afford to lose**
⚠️ **Keep MT5 running while bot is active**
⚠️ **Monitor daily loss limits**
⚠️ **Review trades regularly**
⚠️ **This is not financial advice**

---

**Good luck with your FXIFY challenge!** 🎯📈

Remember: Prop firm success requires patience, discipline, and risk management. The bot handles execution - you handle strategy decisions and monitoring.
