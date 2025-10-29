# FXIFY Trading Bot - Quick Setup Script
# Run this to set up your FXIFY trading bot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  FXIFY Trading Bot - Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Install requirements
Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements_fxify.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Create .env.fxify if it doesn't exist
Write-Host ""
if (-Not (Test-Path ".env.fxify")) {
    Write-Host "Creating .env.fxify configuration file..." -ForegroundColor Yellow
    Copy-Item ".env.fxify.example" ".env.fxify"
    Write-Host "✓ Configuration file created: .env.fxify" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Edit .env.fxify with your MT5 credentials!" -ForegroundColor Yellow
    Write-Host "   - MT5_LOGIN" -ForegroundColor Yellow
    Write-Host "   - MT5_PASSWORD" -ForegroundColor Yellow
    Write-Host "   - MT5_SERVER" -ForegroundColor Yellow
} else {
    Write-Host "✓ Configuration file already exists: .env.fxify" -ForegroundColor Green
}

# Create logs directory
Write-Host ""
Write-Host "Creating logs directory..." -ForegroundColor Yellow
if (-Not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
    Write-Host "✓ Logs directory created" -ForegroundColor Green
} else {
    Write-Host "✓ Logs directory already exists" -ForegroundColor Green
}

# Check MetaTrader 5 installation
Write-Host ""
Write-Host "Checking MetaTrader 5..." -ForegroundColor Yellow
$mt5Paths = @(
    "$env:ProgramFiles\MetaTrader 5\terminal64.exe",
    "${env:ProgramFiles(x86)}\MetaTrader 5\terminal64.exe",
    "$env:APPDATA\MetaQuotes\Terminal\terminal64.exe"
)

$mt5Found = $false
foreach ($path in $mt5Paths) {
    if (Test-Path $path) {
        Write-Host "✓ MT5 found at: $path" -ForegroundColor Green
        $mt5Found = $true
        break
    }
}

if (-Not $mt5Found) {
    Write-Host "⚠️  MT5 not found in common locations" -ForegroundColor Yellow
    Write-Host "   If MT5 is installed, specify MT5_PATH in .env.fxify" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Edit .env.fxify with your MT5 credentials" -ForegroundColor White
Write-Host "2. Ensure MetaTrader 5 is running and logged in" -ForegroundColor White
Write-Host "3. Enable algorithmic trading in MT5 (Tools → Options → Expert Advisors)" -ForegroundColor White
Write-Host "4. Test the bot:" -ForegroundColor White
Write-Host "   python live\trading_bot_fxify.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "For detailed documentation, see: README_FXIFY.md" -ForegroundColor Yellow
Write-Host ""
