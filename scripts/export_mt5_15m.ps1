# Export 15m bars from MT5 to CSV for XAUUSD, USDJPY, ETHUSD
# Requires MT5 terminal installed and either logged-in terminal or .env.fxify with MT5_LOGIN/MT5_PASSWORD/MT5_SERVER
# Usage: powershell -ExecutionPolicy Bypass -File scripts/export_mt5_15m.ps1

$ErrorActionPreference = "Stop"

# Dates (UTC)
$start = "2025-08-31"
$end   = "2025-10-28"
$tf    = "15m"
$out   = "backtests/data"

# Attempt to load env file if present (for local runs)
if (Test-Path ".env.fxify") {
    Write-Host "Loading .env.fxify into process env..."
    Get-Content .env.fxify | ForEach-Object {
        if ($_ -match '^[#\s]') { return }
        $parts = $_.Split('=', 2)
        if ($parts.Length -eq 2) {
            $name = $parts[0].Trim()
            $value = $parts[1].Trim()
            [Environment]::SetEnvironmentVariable($name, $value)
        }
    }
}

# Export portfolio symbols in one call
$symbols = "XAUUSD,USDJPY,ETHUSD"

Write-Host "Exporting $symbols $tf from $start to $end ..."
python backtests/export_mt5_csv.py --symbols $symbols --timeframe $tf --start $start --end $end --out $out --env-file .env.fxify

Write-Host "Done. Files saved under $out"