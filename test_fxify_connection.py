#!/usr/bin/env python3
"""
FXIFY Bot Testing Script
Run this to test your MT5 connection and configuration before live trading.
"""
import os
import sys
from dotenv import load_dotenv

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def test_mt5_connection():
    """Test MT5 connection"""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("❌ MetaTrader5 module not installed")
        print("   Run: pip install MetaTrader5")
        return False
    
    load_dotenv(".env.fxify")
    
    MT5_PATH = os.getenv("MT5_PATH", "")
    MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER = os.getenv("MT5_SERVER", "")
    
    print("Testing MT5 connection...")
    
    # Initialize
    if MT5_PATH:
        if not mt5.initialize(MT5_PATH):
            print(f"❌ MT5 initialize failed: {mt5.last_error()}")
            return False
    else:
        if not mt5.initialize():
            print(f"❌ MT5 initialize failed: {mt5.last_error()}")
            return False
    
    print("✅ MT5 initialized")
    
    # Login
    if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
        if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
            print(f"❌ MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
        print(f"✅ MT5 logged in: {MT5_LOGIN}@{MT5_SERVER}")
    else:
        print("⚠️  No login credentials provided, using terminal's account")
    
    # Get account info
    account = mt5.account_info()
    if account is None:
        print("❌ Failed to get account info")
        mt5.shutdown()
        return False
    
    print(f"✅ Account info retrieved:")
    print(f"   Login: {account.login}")
    print(f"   Balance: ${account.balance:.2f}")
    print(f"   Equity: ${account.equity:.2f}")
    print(f"   Leverage: 1:{account.leverage}")
    print(f"   Currency: {account.currency}")
    
    mt5.shutdown()
    return True

def test_symbol_access():
    """Test symbol availability"""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return False
    
    load_dotenv(".env.fxify")
    SYMBOL = os.getenv("SYMBOL", "EURUSD")
    
    if not mt5.initialize():
        return False
    
    print(f"\nTesting symbol: {SYMBOL}")
    
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"❌ Symbol {SYMBOL} not found")
        mt5.shutdown()
        return False
    
    if not symbol_info.visible:
        if not mt5.symbol_select(SYMBOL, True):
            print(f"❌ Failed to select {SYMBOL}")
            mt5.shutdown()
            return False
    
    print(f"✅ Symbol {SYMBOL} available:")
    print(f"   Bid: {symbol_info.bid:.5f}")
    print(f"   Ask: {symbol_info.ask:.5f}")
    print(f"   Spread: {symbol_info.spread} points")
    print(f"   Digits: {symbol_info.digits}")
    print(f"   Min volume: {symbol_info.volume_min}")
    print(f"   Max volume: {symbol_info.volume_max}")
    print(f"   Volume step: {symbol_info.volume_step}")
    
    mt5.shutdown()
    return True

def test_strategy_imports():
    """Test strategy module imports"""
    print("\nTesting strategy imports...")
    try:
        from strategy.indicators import compute_atr_wilder, calculate_adx, calculate_rsi
        print("✅ strategy.indicators imported")
    except Exception as e:
        print(f"❌ Failed to import strategy.indicators: {e}")
        return False
    
    try:
        from strategy.edge import compute_edge_features_and_score, EdgeResult
        print("✅ strategy.edge imported")
    except Exception as e:
        print(f"❌ Failed to import strategy.edge: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration"""
    print("\nTesting configuration...")
    load_dotenv(".env.fxify")
    
    required = ["MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"]
    optional = ["SYMBOL", "TIMEFRAME", "RISK_PCT"]
    
    missing = []
    for key in required:
        value = os.getenv(key)
        if not value or (key == "MT5_LOGIN" and value == "0"):
            missing.append(key)
            print(f"❌ {key} not set")
        else:
            if key == "MT5_PASSWORD":
                print(f"✅ {key} = ***")
            else:
                print(f"✅ {key} = {value}")
    
    if missing:
        print(f"\n⚠️  Missing required configuration: {', '.join(missing)}")
        print("   Edit .env.fxify and set these values")
        return False
    
    for key in optional:
        value = os.getenv(key)
        print(f"   {key} = {value if value else '(default)'}")
    
    return True

def main():
    print("=" * 60)
    print("  FXIFY Trading Bot - Connection Test")
    print("=" * 60)
    print()
    
    # Test configuration
    if not test_configuration():
        print("\n❌ Configuration test failed")
        print("   Please check your .env.fxify file")
        sys.exit(1)
    
    # Test strategy imports
    if not test_strategy_imports():
        print("\n❌ Strategy import test failed")
        sys.exit(1)
    
    # Test MT5 connection
    if not test_mt5_connection():
        print("\n❌ MT5 connection test failed")
        print("\nTroubleshooting:")
        print("1. Ensure MetaTrader 5 is installed and running")
        print("2. Check your MT5 credentials in .env.fxify")
        print("3. Verify MT5 allows algorithmic trading")
        print("4. Try specifying MT5_PATH if auto-detection fails")
        sys.exit(1)
    
    # Test symbol access
    if not test_symbol_access():
        print("\n❌ Symbol access test failed")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  ✅ All tests passed!")
    print("=" * 60)
    print("\nYou're ready to run the bot:")
    print("  python live\\trading_bot_fxify.py")
    print()

if __name__ == "__main__":
    main()
