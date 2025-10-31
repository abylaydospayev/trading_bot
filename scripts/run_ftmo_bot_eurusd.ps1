# Launch FTMO bot for EURUSD.sim with its own state and magic number
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_ftmo_bot_eurusd.ps1

$ErrorActionPreference = "Stop"

# Ensure UTF-8 console
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $env:PYTHONIOENCODING = "utf-8"
} catch {}

# Disabled by default per user decision to run only USDJPY.
# Set ENABLE_EURUSD=true in the environment to re-enable this launcher.
if (-not $env:ENABLE_EURUSD -or $env:ENABLE_EURUSD.ToLower() -ne "true") {
    Write-Host "EURUSD bot is disabled (run_ftmo_bot_eurusd.ps1). Set ENABLE_EURUSD=true to re-enable." -ForegroundColor Yellow
    exit 0
}

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
    $preset = "configs/live/presets/aggressive_eurusd.env"
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

# Overrides for EURUSD instance (use process env so Python sees them and echo reflects)
$env:SYMBOL = "EURUSD.sim"
$env:MAGIC_NUMBER = "923452"
$env:STATE_FILE = "ftmo_bot_state_eurusd.json"
# Tighten EURUSD spread cap per 15m reality
$env:MAX_SPREAD_PIPS = "0.5"
# Session window for EURUSD (UTC)
$env:ACTIVE_HOUR_START = "6"
$env:ACTIVE_HOUR_END = "16"
# Loosen gates for 15m FX
$env:ADX_THRESHOLD = "18"
$env:MIN_MA_DIST_BPS = "1.5"
$env:MIN_EDGE_ATR_PCT = "0.02"
# Keep regular edge score at 60; use the same threshold off-hours (rollback to more selective setup)
$env:EDGE_BUY_SCORE = "60"
# Unset any prior override for off-hours edge so default (60) applies
[Environment]::SetEnvironmentVariable("OFFHOURS_EDGE_BUY_SCORE", $null)
# Disable adaptive MA-distance to match backtested constant threshold behavior
$env:ADAPTIVE_MABPS_ENABLE = "false"
[Environment]::SetEnvironmentVariable("ADAPTIVE_MABPS_COEFF", $null)
[Environment]::SetEnvironmentVariable("ADAPTIVE_MABPS_FLOOR_BPS", $null)
# Enable 24/7 entries with stricter off-hours safeguards
$env:ALWAYS_ACTIVE = "true"
$env:OFFHOURS_ADX_MIN = "20"
$env:OFFHOURS_SPREAD_EURUSD = "0.35"
$env:OFFHOURS_SPREAD_JPY = "0.5"
# Enable pyramiding with conservative steps and caps
$env:PYRAMID_ENABLE = "true"
$env:PYRAMID_STEP_RISK = "0.0010"
$env:PYRAMID_MAX_TOTAL_RISK = "0.006"

# Echo key overrides for visibility
$offEdge = if ($env:OFFHOURS_EDGE_BUY_SCORE) { $env:OFFHOURS_EDGE_BUY_SCORE } else { "(default)" }
Write-Host ("EURUSD overrides -> ADX_THRESHOLD={0}, MIN_MA_DIST_BPS={1}, MIN_EDGE_ATR_PCT={2}, MAX_SPREAD_PIPS={3}, ALWAYS_ACTIVE={4}, OFFHOURS_EDGE_BUY_SCORE={5}" -f $env:ADX_THRESHOLD, $env:MIN_MA_DIST_BPS, $env:MIN_EDGE_ATR_PCT, $env:MAX_SPREAD_PIPS, $env:ALWAYS_ACTIVE, $offEdge)

# Risk per trade (leave preset values if AGGRESSIVE=true)
if (-not $env:RISK_PCT) { $env:RISK_PCT = "0.005" }
if (-not $env:RISK_PCT_BASE) { $env:RISK_PCT_BASE = "0.005" }
if (-not $env:RISK_PCT_STRONG) { $env:RISK_PCT_STRONG = "0.005" }

python live/trading_bot_ftmo.py
