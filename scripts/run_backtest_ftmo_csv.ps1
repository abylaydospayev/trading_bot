# Run FTMO-mode 15m backtest using MT5-exported CSVs
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_backtest_ftmo_csv.ps1

$ErrorActionPreference = "Stop"

# Parameters (edit as needed)
$symbols = "EURUSD.sim,USDJPY.sim"
$tf = "15m"
$capital = 100000
$risk = 0.003 # 0.30%
$dataDir = "backtests/data"
$outDir = "backtests/reports/forward_15m_cfg_ftmo_csv"
$simStart = "2025-08-31"
$simEnd   = "2025-10-28"

Write-Host "Running FTMO backtest on $symbols ($tf) ..."
python backtests/fxify_phase1_backtest.py --symbols $symbols --timeframe $tf --capital $capital --risk-pct $risk --data-dir $dataDir --output $outDir --ftmo --sim-start $simStart --sim-end $simEnd
