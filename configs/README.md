# Configs and Run Scripts

This folder organizes stable versions of backtest and live configurations so you can revert or compare easily.

## Backtest configs

- `backtest/v1_baseline_15m.json`
  - Baseline 15m portfolio (no pyramiding, milder filters)
  - Run: `scripts/run_backtest_v1.ps1`

- `backtest/v2_improved_15m.json`
  - Improved 15m (ADX≥27, MA≥12bps), ETH-only ML gate (thr=0.28), pyramiding step 0.12%
  - Run: `scripts/run_backtest_v2.ps1`
  - Unseen windows:
    - `scripts/run_backtest_v2_unseen_A.ps1` (2025-09-01..2025-09-28)
    - `scripts/run_backtest_v2_unseen_B.ps1` (2025-09-29..2025-10-28)

- `backtest/v3_ftmo_15m.json`
  - FTMO mode (daily loss 5%, overall 10%, Europe/Prague reset) with improved 15m settings
  - Run: `scripts/run_backtest_v3_ftmo.ps1`

Note: The backtester supports `--sim-start/--sim-end` for slicing the simulation window after data load.

## Live configs

- `live/mt5_v2_improved.env` — template for MT5 live credentials and reference to improved thresholds. Adjust environment and run your live bot accordingly.

- `live/mt5_ftmo.env` — template for MT5 live FTMO mode (daily 5%, max loss 10%, CET/CEST reset). Adjust and run.

