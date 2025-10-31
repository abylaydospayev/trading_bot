# Launch FTMO bot for USDJPY.sim with its own state and overrides
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_ftmo_bot_usdjpy.ps1

$ErrorActionPreference = "Stop"

# Ensure UTF-8 console
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $env:PYTHONIOENCODING = "utf-8"
} catch {}

# Load base FTMO env
$envFile = "configs/live/mt5_ftmo.env"
if (Test-Path $envFile) {
    Write-Host "Loading $envFile ..."
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^[#\s]') { return }
        $parts = $_.Split('=', 2)
        if ($parts.Length -eq 2) {
            $name = $parts[0].Trim()
            $value = $parts[1].Trim()
            [Environment]::SetEnvironmentVariable($name, $value)
        }
    }
}

# Optional aggressive preset
if ($env:AGGRESSIVE -and $env:AGGRESSIVE.ToLower() -eq "true") {
    $preset = "configs/live/presets/aggressive_usdjpy.env"
    if (Test-Path $preset) {
        Write-Host "AGGRESSIVE=true -> loading preset $preset"
        Get-Content $preset | ForEach-Object {
            if ($_ -match '^[#\s]') { return }
            $parts = $_.Split('=', 2)
            if ($parts.Length -eq 2) {
                $name = $parts[0].Trim()
                $value = $parts[1].Trim()
                [Environment]::SetEnvironmentVariable($name, $value)
            }
        }
    } else {
        Write-Host "AGGRESSIVE preset not found at $preset"
    }
}

# Overrides for USDJPY instance (process env)
$env:SYMBOL = "USDJPY.sim"
$env:MAGIC_NUMBER = "923451"
$env:STATE_FILE = "ftmo_bot_state_usdjpy.json"
# Core thresholds
$env:ADX_THRESHOLD = "18"
$env:MIN_MA_DIST_BPS = "1.5"
$env:MIN_EDGE_ATR_PCT = "0.02"
# Spread caps
$env:MAX_SPREAD_PIPS = "0.8"
# Disable adaptive MA-distance to align with backtested constant threshold
$env:ADAPTIVE_MABPS_ENABLE = "false"
[Environment]::SetEnvironmentVariable("ADAPTIVE_MABPS_COEFF", $null)
[Environment]::SetEnvironmentVariable("ADAPTIVE_MABPS_FLOOR_BPS", $null)
# 24/7 with off-hours safeguards (slightly looser JPY off-hours cap)
$env:ALWAYS_ACTIVE = "true"
$env:OFFHOURS_ADX_MIN = "20"
$env:OFFHOURS_SPREAD_EURUSD = "0.3"
$env:OFFHOURS_SPREAD_JPY = "0.8"
# Risk and pyramiding (leave preset values if AGGRESSIVE=true)
if (-not $env:RISK_PCT) { $env:RISK_PCT = "0.005" }
if (-not $env:RISK_PCT_BASE) { $env:RISK_PCT_BASE = "0.005" }
if (-not $env:RISK_PCT_STRONG) { $env:RISK_PCT_STRONG = "0.005" }
if (-not $env:PYRAMID_ENABLE) { $env:PYRAMID_ENABLE = "true" }
if (-not $env:PYRAMID_STEP_RISK) { $env:PYRAMID_STEP_RISK = "0.0012" }
if (-not $env:PYRAMID_MAX_TOTAL_RISK) { $env:PYRAMID_MAX_TOTAL_RISK = "0.006" }

Write-Host ("USDJPY overrides -> ADX_THRESHOLD={0}, MIN_MA_DIST_BPS={1}, MIN_EDGE_ATR_PCT={2}, MAX_SPREAD_PIPS={3}, ALWAYS_ACTIVE={4}, OFFHOURS_SPREAD_JPY={5}" -f $env:ADX_THRESHOLD, $env:MIN_MA_DIST_BPS, $env:MIN_EDGE_ATR_PCT, $env:MAX_SPREAD_PIPS, $env:ALWAYS_ACTIVE, $env:OFFHOURS_SPREAD_JPY)

python live/trading_bot_ftmo.py
