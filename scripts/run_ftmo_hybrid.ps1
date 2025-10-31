# FTMO Hybrid Orchestrator
# Loads configs/live/profiles/ftmo_hybrid_v1.0.json and maps settings to
# environment variables for live/trading_bot_ftmo.py per pair.
# By default runs in WhatIf mode (prints the plan, does not launch).
#
# Usage examples:
#  - Dry run (default):
#      powershell -ExecutionPolicy Bypass -File scripts\run_ftmo_hybrid.ps1
#  - Launch selected pairs (comma-separated):
#      powershell -ExecutionPolicy Bypass -File scripts\run_ftmo_hybrid.ps1 -Pairs "USDJPY,XAUUSD" -WhatIf:$false
#  - Launch USDJPY only using hybrid params:
#      powershell -ExecutionPolicy Bypass -File scripts\run_ftmo_hybrid.ps1 -Pairs "USDJPY" -WhatIf:$false

param(
    [string]$ProfileJsonPath = "configs/live/profiles/ftmo_hybrid_v1.0.json",
    [string]$Pairs = "USDJPY", # default to our chosen single-symbol flow
    [Parameter(ValueFromRemainingArguments = $true)]
    [object]$DryRun = $true
)

$ErrorActionPreference = "Stop"

# Ensure UTF-8 console
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $env:PYTHONIOENCODING = "utf-8"
} catch {}

if (-not (Test-Path $ProfileJsonPath)) {
    Write-Host "Profile JSON not found: $ProfileJsonPath" -ForegroundColor Red
    exit 1
}

# Load base FTMO env if present
$envFile = "configs/live/mt5_ftmo.env"
if (Test-Path $envFile) {
    Write-Host "Loading base env: $envFile"
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

# Read hybrid profile
$profileRaw = Get-Content -Raw -Path $ProfileJsonPath
$hybrid = $profileRaw | ConvertFrom-Json

# Coerce DryRun to boolean (robust against string passing)
function Convert-ToBool($val) {
    if ($val -is [bool]) { return $val }
    if ($val -is [int]) { return [bool]$val }
    if ($val -is [string]) {
        $s = $val.Trim().ToLower()
        if ($s -in @("true","1","yes","y")) { return $true }
        if ($s -in @("false","0","no","n")) { return $false }
    }
    return $true
}
$DryRunFlag = Convert-ToBool $DryRun

# Configure global FTMO/prop guardrails
$session = $hybrid.global.session_window_utc
if ($session -match "^(\d{2})-(\d{2})$") {
    $env:ACTIVE_HOUR_START = [int]$Matches[1]
    $env:ACTIVE_HOUR_END = [int]$Matches[2]
}
$env:FXIFY_MAX_DAILY_LOSS_PCT = ([double]$hybrid.global.risk_limit_daily / 100.0).ToString("0.######")
$env:FXIFY_MAX_TOTAL_LOSS_PCT = ([double]$hybrid.global.risk_limit_total / 100.0).ToString("0.######")
$env:CONCURRENT_RISK_CAP = ([double]$hybrid.global.max_concurrent_risk / 100.0).ToString("0.######")
$env:CB_ENABLED = "true"
$env:CB_MAX_LOSS_STREAK = [string]$hybrid.global.circuit_breaker_loss_streak

$selected = $Pairs.Split(",") | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ -ne "" }

# Map 15m engine params per pair
foreach ($sym in $selected) {
    if (-not $hybrid.pairs.$sym) {
        Write-Host "Pair '$sym' not found in profile; skipping" -ForegroundColor Yellow
        continue
    }
    $p = $hybrid.pairs.$sym

    # Derive timeframe string for our bot
    $tf = [string]$p.timeframe
    switch -Regex ($tf) {
        "^1m$" { $tfBot = "1Min" }
        "^5m$" { $tfBot = "5Min" }
        "^15m$" { $tfBot = "15Min" }
        "^30m$" { $tfBot = "30Min" }
        "^1h$" { $tfBot = "1H" }
        Default { $tfBot = "15Min" }
    }

    $mt5Symbol = $p.mt5_symbol
    if (-not $mt5Symbol) { $mt5Symbol = $sym }
    $map = [ordered]@{
        SYMBOL                 = [string]$mt5Symbol
        TIMEFRAME              = $tfBot
        RISK_PCT               = ([double]$p.risk_per_trade).ToString("0.######")
        ADX_THRESHOLD          = [string]$p.adx_threshold
        MIN_MA_DIST_BPS        = ([double]$p.min_ma_bps).ToString("0.######")
        MIN_EDGE_ATR_PCT       = ([double]$p.min_edge_atr_pct).ToString("0.######")
        MAX_SPREAD_PIPS        = ([double]$p.max_spread_pips).ToString("0.######")
    PYRAMID_ENABLE         = "true"
    PYRAMID_STEP_RISK      = ([double]0.001).ToString("0.######")
    PYRAMID_MAX_TOTAL_RISK = ([double]0.006).ToString("0.######")
        STATE_FILE             = "ftmo_bot_state_$($sym.ToLower()).json"
        MAGIC_NUMBER           = (Get-Random -Minimum 100000 -Maximum 999999)
    }

    # ALWAYS_ACTIVE: prefer profile setting when available, else default to false
    $alwaysActive = $false
    if ($hybrid.global.PSObject.Properties.Name -contains 'always_active') {
        try { $alwaysActive = [bool]$hybrid.global.always_active } catch {}
    }
    $map["ALWAYS_ACTIVE"] = ($alwaysActive).ToString().ToLower()

    # Map scalper module if present
    if ($p.scalp_module -and $p.scalp_module.enabled) {
        $map["SCALP_ENABLE"] = "true"
        $scalpTf = [string]$p.scalp_module.timeframe
        if ($scalpTf -match "^1m$") { $map["SCALP_TF"] = "1Min" }
        elseif ($scalpTf -match "^5m$") { $map["SCALP_TF"] = "5Min" }
        else { $map["SCALP_TF"] = "5Min" }
        $map["SCALP_ENTRY_LOGIC"] = [string]$p.scalp_module.entry_logic
        if ($p.scalp_module.risk_per_trade) { $map["SCALP_RISK_PCT"] = ([double]$p.scalp_module.risk_per_trade).ToString("0.######") }
        if ($p.scalp_module.rsi_entry_long) { $map["SCALP_RSI_ENTRY_LONG"] = [string]$p.scalp_module.rsi_entry_long }
        if ($p.scalp_module.rsi_exit_long) { $map["SCALP_RSI_EXIT_LONG"] = [string]$p.scalp_module.rsi_exit_long }
        if ($p.scalp_module.rsi_entry_short) { $map["SCALP_RSI_ENTRY_SHORT"] = [string]$p.scalp_module.rsi_entry_short }
        if ($p.scalp_module.rsi_exit_short) { $map["SCALP_RSI_EXIT_SHORT"] = [string]$p.scalp_module.rsi_exit_short }
        if ($p.scalp_module.macd_signal_cross -ne $null) { $map["SCALP_MACD_SIGNAL_CROSS"] = ([bool]$p.scalp_module.macd_signal_cross).ToString().ToLower() }
        if ($p.scalp_module.momentum_threshold) { $map["SCALP_MOMENTUM_THRESHOLD"] = ([double]$p.scalp_module.momentum_threshold).ToString("0.######") }
        if ($p.scalp_module.tp_pips) { $map["SCALP_TP_PIPS"] = ([double]$p.scalp_module.tp_pips).ToString("0.######") }
        if ($p.scalp_module.sl_pips) { $map["SCALP_SL_PIPS"] = ([double]$p.scalp_module.sl_pips).ToString("0.######") }
        if ($p.scalp_module.session_filter) { $map["SCALP_SESSION_FILTER"] = [string]$p.scalp_module.session_filter }
    } else {
        $map["SCALP_ENABLE"] = "false"
    }

    # Optional: map additional router/MR and edge tunables if provided in profile
    if ($p.PSObject.Properties.Name -contains 'mr_z_entry') {
        try { $map["MR_Z_ENTRY"] = ([double]$p.mr_z_entry).ToString("0.######") } catch {}
    }
    if ($p.PSObject.Properties.Name -contains 'edge_buy_score') {
        try { $map["EDGE_BUY_SCORE"] = ([int]$p.edge_buy_score) } catch {}
    }
    if ($p.PSObject.Properties.Name -contains 'edge_confirm_bars') {
        try { $map["EDGE_CONFIRM_BARS"] = ([int]$p.edge_confirm_bars) } catch {}
    }

    Write-Host "--- HYBRID PLAN for $sym ---" -ForegroundColor Cyan
    $map.GetEnumerator() | ForEach-Object { Write-Host ("{0}={1}" -f $_.Key, $_.Value) }

    if ($DryRunFlag) {
        Write-Host "(WhatIf) Would launch: python live/trading_bot_ftmo.py for $sym" -ForegroundColor DarkGray
        continue
    }

    # Launch child process with isolated environment
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "python"
    $psi.Arguments = "live/trading_bot_ftmo.py"
    $psi.WorkingDirectory = (Get-Location).Path
    $psi.UseShellExecute = $false

    # Inherit current env first
    foreach ($kv in [System.Environment]::GetEnvironmentVariables().GetEnumerator()) {
        $psi.Environment[$kv.Key] = [string]$kv.Value
    }
    # Apply per-pair overrides
    foreach ($kv in $map.GetEnumerator()) {
        $psi.Environment[$kv.Key] = [string]$kv.Value
    }

    $proc = [System.Diagnostics.Process]::Start($psi)
    if ($proc) {
        Write-Host "Launched $sym (PID=$($proc.Id))" -ForegroundColor Green
    } else {
        Write-Host "Failed to launch $sym" -ForegroundColor Red
    }
}

Write-Host "Note: Scalper modules (RSI+MACD / RSI+Momentum) are defined in strategy/scalp.py but not yet wired into live.\n      Multi-pair global correlation guard is not enforced across processes yet." -ForegroundColor Yellow
