# Launch FTMO bot using FTMO env template
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_ftmo_bot.ps1

$ErrorActionPreference = "Stop"

# Ensure UTF-8 console to avoid emoji logging encode errors
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $env:PYTHONIOENCODING = "utf-8"
} catch {}

# Load FTMO env template if present
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

# Optional local override (.env.ftmo)
if (Test-Path ".env.ftmo") {
    Write-Host "Loading .env.ftmo overrides ..."
    Get-Content .env.ftmo | ForEach-Object {
        if ($_ -match '^[#\s]') { return }
        $parts = $_.Split('=', 2)
        if ($parts.Length -eq 2) {
            $name = $parts[0].Trim()
            $value = $parts[1].Trim()
            [Environment]::SetEnvironmentVariable($name, $value)
        }
    }
}

# Run FTMO bot (wrapper sets FTMO defaults)
python live/trading_bot_ftmo.py
