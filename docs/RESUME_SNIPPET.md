General Trading Bot (FX/Gold/Crypto) — Backtesting + Live (MT5)

- Built a production‑ready trading framework with portfolio backtesting and live execution (MetaTrader 5), focused on robust risk controls and low drawdowns.
- Strategy: EMA(50/200), ATR(14), ADX(14), RSI(14) with quality filters (ADX, ATR% by symbol, min MA distance) and symbol‑specific sessions.
- Risk: 0.25–0.30% per trade, 1.0% max concurrent risk, −3% hard daily stop; break‑even at +1R, ATR trailing from +1.5R, partial at +2R.
- Optional micro‑pyramiding after +1R (+0.2% steps, max +0.6% per symbol) to compound winners while maintaining portfolio caps.
- ML: Light gating for ETH entries via pluggable model interface to improve win rate without over‑filtering.
- Results (15m, recent ~6 weeks): Baseline PF≈1.9 (MaxDD≈−2.3%); with pyramiding PF≈3.4 at similar DD.
- Tech: Python (pandas, numpy), yfinance/CSV for data, MetaTrader 5 for execution, dotenv‑based config.

Repo highlights: backtester with sessions/news/risk caps, MT5 live bot mirroring backtest logic, export tool for MT5→CSV, and comprehensive README.
