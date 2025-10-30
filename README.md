# 15m FX Prop-Trading Bot (MT5) — Backtests + Live with FTMO/FXIFY Guardrails

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20(MT5)-orange.svg)](#)
[![Backtests](https://img.shields.io/badge/backtests-Python%2Fpandas-informational.svg)](#)

Live MT5 bot and aligned backtester for 15‑minute FX trading, designed to comply with FTMO/FXIFY constraints. Includes session and spread‑aware entries, strict risk controls, mid‑bar management, and resilient broker‑state reconciliation.

Highlights
- FTMO/FXIFY guardrails: daily stop, total loss cap, Prague day reset, compliance checks before any trade
- Quality gates: ADX, ATR% band, minimum MA distance with adaptive off‑hours floor, spread caps + percentiles
- Risk‑based sizing with stage‑2 adds and pyramiding, margin‑aware preflight, and JPY per‑order lot caps
- Mid‑bar heartbeat for trims/BE/trailing with a one‑trim‑per‑bar guard; clear, concise logs and CSV outputs

Pairs: optimized for EURUSD and USDJPY on 15m; extendable to others.

> Looking for a step‑by‑step? See QUICKSTART: `QUICKSTART_FXIFY.md`.

## Repository structure

```
backtests/
  fxify_phase1_backtest.py      # 15m backtester mirroring live gates and FTMO/FXIFY constraints
  reports/                      # CSV summaries and sweep results
live/
  trading_bot_fxify.py          # MT5 live bot with spread/session gates and risk controls
scripts/
  run_ftmo_bot_eurusd.ps1       # PowerShell launchers with per‑symbol overrides
  run_ftmo_bot_usdjpy.ps1
strategy/
  indicators.py, edge.py        # indicators + edge scoring utilities
ml/                             # optional; not required for FX bots
```

## Requirements

- Windows with MetaTrader 5 installed and logged in
- Python 3.10+
- Packages: `pip install -r requirements.txt`

## Live trading (MT5)

Recommended: start via the provided PowerShell scripts to pass per‑symbol overrides.

USDJPY
```powershell
$env:ALWAYS_ACTIVE = "true"
$env:ADX_THRESHOLD = "18"
$env:MIN_MA_DIST_BPS = "1.5"
$env:MAX_SPREAD_PIPS = "0.8"
$env:MGMT_ENABLE = "true"
$env:MGMT_HEARTBEAT_SEC = "5"
powershell -ExecutionPolicy Bypass -File scripts\run_ftmo_bot_usdjpy.ps1
```

EURUSD
```powershell
$env:ALWAYS_ACTIVE = "true"
$env:ADX_THRESHOLD = "22"
$env:MIN_MA_DIST_BPS = "1.2"
$env:MAX_SPREAD_PIPS = "0.5"
$env:MGMT_ENABLE = "true"
$env:MGMT_HEARTBEAT_SEC = "5"
powershell -ExecutionPolicy Bypass -File scripts\run_ftmo_bot_eurusd.ps1
```

Notes
- `ALWAYS_ACTIVE=true` bypasses off‑hours strict checks; the startup banner will indicate the bypass.
- Mid‑bar management (`MGMT_ENABLE`) refreshes trims/BE/trailing every `MGMT_HEARTBEAT_SEC` seconds.

## Backtesting

Run a 15m portfolio backtest aligned to live gates:

```powershell
python backtests\fxify_phase1_backtest.py `
  --symbols EURUSD,USDJPY `
  --timeframe 15m `
  --capital 100000 `
  --risk-pct 0.003 `
  --start 2025-09-05 `
  --always-active `
  --adx-threshold 27 `
  --min-ma-dist-bps 1.5 `
  --atr-pct-min 0.04 `
  --atr-pct-max 0.20 `
  --offhours-strict-adx-min 28 `
  --ma-norm-min-offhours 0.6 `
  --adaptive-mabps-enable `
  --adaptive-mabps-coeff 0.35 `
  --adaptive-mabps-floor-bps 1.0 `
  --day-reset-tz Europe/Prague `
  --daily-stop-pct 0.05 `
  --max-loss-pct 0.10
```

Outputs are written to `backtests/reports/` (portfolio and per‑asset summaries).

## Configuration reference

Entries and adds
- ADX threshold: `ADX_THRESHOLD`
- Min MA distance (bps): `MIN_MA_DIST_BPS`
- ATR% band: `ATR_PCT_MIN`, `ATR_PCT_MAX` (backtests via flags)
- Adaptive MA floor (off‑hours): `ADAPTIVE_MABPS_ENABLE`, `ADAPTIVE_MABPS_COEFF`, `ADAPTIVE_MABPS_FLOOR_BPS`
- Spread caps and percentiles: `MAX_SPREAD_PIPS`; entries use p20; adds use up to p35 (equality allowed at cap)

Risk and compliance
- Risk per trade: backtests `--risk-pct`; live uses risk‑based sizing per stop distance
- Daily stop and overall loss caps:
  - Live: `FXIFY_MAX_DAILY_LOSS_PCT`, `FXIFY_MAX_TOTAL_LOSS_PCT` (defaults 0.05 and 0.10)
  - Backtests: `--daily-stop-pct`, `--max-loss-pct`
- Day reset timezone: `DAY_RESET_TZ` (e.g., `Europe/Prague`)

Management
- Mid‑bar management enable: `MGMT_ENABLE`
- Heartbeat seconds: `MGMT_HEARTBEAT_SEC`
- One‑trim‑per‑bar guard enforced automatically

## How it works (brief)

- Indicators: EMA(10/25/200), ADX(14), ATR(14), RSI(14)
- Entry gates: trend and edge alignment, ATR% in band, MA distance with adaptive off‑hours floor, spread cap/percentile
- Staging: stage‑2 add only if price breaks entry bar high/low, ADX now ≥ entry ADX, RSI filter, and spread within cap
- Pyramiding: add sizing uses the same risk‑to‑lots calculator as entries; margin‑aware preflight steps down to fit; JPY per‑order caps
- Defense: BE promotions and ATR trailing (spread‑aware), early adverse‑move trims, failed breakout and ADX floor exits, drawdown fuse
- Heartbeat: mid‑bar loop updates trims and stops every few seconds while ensuring no more than one trim per bar

## Logs and state

- Logs: `logs/` folder; look for startup banner, gate decisions, adds, trims, BE/trail, and margin step‑downs
- State: `ftmo_bot_state.json` and per‑symbol variants (e.g., `ftmo_bot_state_usdjpy.json`)
- CSV outputs: `backtests/reports/*`

## Troubleshooting

- MT5 initialize/login issues: ensure MT5 is running and logged in; if needed, set `MT5_PATH` in your env
- "TRADE_RETCODE_NO_MONEY": live bot auto‑reduces add size via margin preflight; reduce risk or widen stop if persistent
- Partial‑close errors (10038): volumes are auto‑floored to the symbol’s step and re‑tried with step‑down
- Spreads too high: tighten `MAX_SPREAD_PIPS` and/or wait for liquid sessions; entries use p20, adds allow equality at cap

## License

MIT — see `LICENSE`.

## Related docs

- Quick start: `QUICKSTART_FXIFY.md`
- FXIFY‑oriented readme (legacy): `README_FXIFY.md`
