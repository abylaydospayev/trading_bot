# Walk-forward runs with improved settings (two-week windows)
# Settings: ADX>=27, MA>=12bps, ATR_MULT=2.5, ETH ML thr=0.28, pyramiding step=0.0012 cap=0.006

$base = "python backtests/fxify_phase1_backtest.py --symbols XAUUSD,USDJPY,ETHUSD --timeframe 15m --capital 100000 --risk-pct 0.003 --atr-mult 2.5 --adx-threshold 27 --min-ma-dist-bps 12 --ml-enable --ml-model ml/models/mlp_thr005.pkl --ml-eth-threshold 0.28 --pyramid-enable --pyramid-step-risk 0.0012 --pyramid-max-total-risk 0.006"

# W1: 2025-08-31..2025-09-12
Invoke-Expression "$base --start 2025-08-31 --sim-start 2025-08-31 --sim-end 2025-09-12"

# W2: 2025-09-13..2025-09-27
Invoke-Expression "$base --start 2025-09-01 --sim-start 2025-09-13 --sim-end 2025-09-27"

# W3: 2025-09-28..2025-10-12
Invoke-Expression "$base --start 2025-09-01 --sim-start 2025-09-28 --sim-end 2025-10-12"

# W4: 2025-10-13..2025-10-27
Invoke-Expression "$base --start 2025-09-01 --sim-start 2025-10-13 --sim-end 2025-10-27"