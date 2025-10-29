# General Trading Bot (FX, Gold, Crypto) – Backtesting + Live (MT5)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platform-Backtest%3A%20Win%2FLinux%20%7C%20Live%3A%20Windows-informational.svg)](#)
[![Trading](https://img.shields.io/badge/domain-trading%2Fbacktesting-blue.svg)](#)
[![MT5](https://img.shields.io/badge/execution-MetaTrader5-orange.svg)](#)
[![Data](https://img.shields.io/badge/data-Yahoo%20Finance%20%2B%20CSV-yellowgreen.svg)](#)

High‑confidence 15m portfolio framework with session‑aware entries, tight risk controls, and optional ML gating. Backtests run on any OS; live execution mirrors the rules on Windows via MT5.

Highlights
- Low drawdown profile: per‑trade risk 0.25–0.30%, 1.0% max concurrent, −3% daily stop
- Quality filters that reduce chop (ADX, ATR% by symbol, min MA distance)
- Cleaner exits: break‑even at +1R, ATR trail from +1.5R, partials at +2R; optional micro‑pyramiding to compound winners
- Works out‑of‑the‑box for XAUUSD, USDJPY, ETHUSD; easy to extend to other assets/timeframes

Production‑ready backtesting and live‑trading framework focused on robust risk controls, session‑aware entries, and optional ML gating. Works out‑of‑the‑box for USDJPY, XAUUSD (Gold), ETHUSD on 15m; easily adaptable to other assets/timeframes.

## Features

- Multi‑asset, multi‑timeframe backtests with portfolio risk caps and daily stop
- Indicator edge (EMA 50/200, ADX, ATR, RSI) with quality filters (ADX, ATR%, MA distance)
- Session‑aware entries (symbol‑specific windows) and Tier‑1 news blackout (optional)
- Risk management: per‑trade risk %, max concurrent risk %, hard daily stop
- Exits: break‑even at +1R, trailing from +1.5R (ATR‑based), partial at +2R
- Optional micro‑pyramiding after +1R in small risk steps with per‑symbol caps
- Light ML gating (ETH enabled by default) using provided model interface
- Live trading via MetaTrader 5 (MT5) with the same guardrails

## Quickstart

1) Create and activate a Python 3.10+ virtual environment.
2) Install dependencies:

```
pip install -r requirements.txt
```

3) Backtest a 3‑asset 15m portfolio (uses Yahoo for FX/crypto; GC=F fallback for gold):

```
python backtests\fxify_phase1_backtest.py --symbols XAUUSD,USDJPY,ETHUSD --timeframe 15m --capital 15000 --risk-pct 0.003 --ml-enable --ml-eth-threshold 0.28 --start 2025-09-10
```

Notes:
- Yahoo intraday history is ~60 days. For longer windows, export CSVs from MT5: `backtests/export_mt5_csv.py`.
- Results and per‑asset CSVs are written under `backtests/reports/`.

## Live trading (MT5)

Live trading uses `live/trading_bot_fxify.py` and requires MetaTrader 5 installed and a funded/login account configured. Set environment variables (use `.env`):

```
MT5_LOGIN=...
MT5_PASSWORD=...
MT5_SERVER=...
SYMBOL=XAUUSD              # or USDJPY, ETHUSD
TIMEFRAME=15Min
RISK_PCT=0.003            # 0.3% per trade
TRAIL_ATR_MULT=2.5        # suggested
DAILY_STOP_PCT=0.03       # -3% daily stop

# Optional ML gating (recommended for ETH)
ML_ENABLE=true
ML_MODEL_PATH=ml/models/mlp_model.pkl
ML_PROB_THR=0.28
```

Run the bot:

```
python live\trading_bot_fxify.py
```

## Key configuration

- Risk controls: `--risk-pct`, `--daily-stop-pct`, `--pyramid-*`, `--max-trades-per-day` (code default 6)
- Quality filters: `--adx-threshold`, `--atr-pct-threshold`, `--min-ma-dist-bps`
- Sessions and news: per‑symbol windows are built‑in; use `--news-calendar` CSV to blackout ±10min around Tier‑1 events
- ML gating: `--ml-enable`, per‑symbol overrides `--ml-eth-threshold`, `--ml-usdjpy-enable`, `--ml-xauusd-enable`

## Project layout

```
backtests/
  fxify_phase1_backtest.py       # 15m portfolio backtester with sessions, risk caps, ML gating
  export_mt5_csv.py              # export 15m (or other) bars from MT5 to CSV
  reports/                       # outputs and summaries
live/
  trading_bot_fxify.py           # MT5 live bot with the same guardrails
ml/
  infer.py                       # model loading + predict_entry_prob interface
strategy/
  indicators.py, edge.py         # indicator suite and edge scoring
```

## Architecture

- Strategy: EMA(50/200), ATR(14), ADX(14), RSI(14), plus edge scoring from `strategy/edge.py`.
- Backtester: Simulates per‑symbol positions, respects sessions/news, enforces risk limits, exits with BE/trailing/partials.
- Live: Mirrors backtest logic, adds MT5 execution, spread checks, consistency with daily/total drawdown policies.

## Example: symbol‑specific sessions

- USDJPY: 00:00–06:00 and 12:00–16:00 UTC
- XAUUSD: 07:00–17:00 UTC
- ETHUSD: avoid Fri 22:00–Sun 22:00 UTC

## Results snapshot (15m, last ~6 weeks)

- Baseline (no pyramiding): Return +4.92%, MaxDD −2.27%, PF 1.92
- With pyramiding (+0.2% steps, cap +0.6%/symbol): Return +9.33%, MaxDD −2.27%, PF 3.41
- ATR_MULT sweep: 2.5 > 2.2 on PF and DD; ETH ML thr 0.25–0.30 neutral → default 0.28

For detailed metrics, see `backtests/reports/BACKTEST_RESULTS_SUMMARY.md`.

## Resume snippet

See `docs/RESUME_SNIPPET.md` for a copy‑paste friendly blurb highlighting design, technology, and impact.

## License

MIT — see `LICENSE`.
