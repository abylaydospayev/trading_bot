# FTMO Mode — Rules and Mapping

This bot can run in an FTMO-friendly mode using the same 15m strategy and risk controls, with FTMO loss limits enforced.

## Core FTMO Rules (summary)

- Daily loss limit: 5% of initial balance (resets daily at 00:00 CET/CEST)
- Max loss overall: 10% of initial balance (breach at any time)
- Profit target: 10% for Challenge Phase 1 (30 days), 5% for Phase 2 (60 days)
- Min trading days: 5
- News/holding: Generally permitted, but check your specific program conditions

Always verify the current FTMO rules on their site; the above is a practical summary, not legal advice.

## Bot mapping to FTMO

- Daily loss enforcement
  - `DAILY_STOP_PCT=0.05` and `DAY_RESET_TZ=Europe/Prague` (CET/CEST) to match FTMO midnight reset
  - When hit, new entries are blocked for the rest of the FTMO day
- Overall max loss
  - `MAX_LOSS_PCT=0.10` — if breached, the backtester liquidates and stops; live bot should also flatten and halt
- Risk and concurrency
  - Default risk: `RISK_PCT=0.003` (0.30% per trade)
  - Portfolio cap: `CONCURRENT_RISK_CAP=0.010` (1.0%)
- Strategy thresholds (reference improved settings)
  - ADX≥27, MA distance ≥12 bps, ATR_MULT=2.5, BE at +1R, trailing from +1.5R, partial at +2R
- Sessions
  - USDJPY: 00:00–06:00 & 12:00–16:00 UTC
  - XAUUSD: 07:00–17:00 UTC
  - ETHUSD: Avoid Fri 22:00–Sun 22:00 UTC

## How to run (backtest)

- Full window (example)

```powershell
# FTMO 15m portfolio (improved settings + FTMO rules)
.\u007cscripts\run_backtest_v3_ftmo.ps1
```

- Walk-forward / unseen windows

Use `--sim-start/--sim-end` as shown in `scripts/run_backtest_v2_walkforward.ps1` and add `--ftmo` for FTMO rules.

## Live template

- `configs/live/mt5_ftmo.env` — set credentials and risk caps; mirror FTMO rules

```env
DAY_RESET_TZ=Europe/Prague
DAILY_STOP_PCT=0.05
MAX_LOSS_PCT=0.10
RISK_PCT=0.003
CONCURRENT_RISK_CAP=0.010
```

Implementation note: the MT5 live bot should compute the FTMO day key using `Europe/Prague` timezone (CET/CEST) and enforce both daily and overall stops. If you’d like, we can wire these into `live/trading_bot_fxify.py` or create a parallel `live/trading_bot_ftmo.py` adapter.
