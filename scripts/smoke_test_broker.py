"""Smoke test for broker connectivity.

Simple script to verify broker connectivity and basic functionality.
This is separate from live readiness checks which evaluate trading performance.

Usage:
    python scripts/smoke_test_broker.py --broker alpaca_paper
    python scripts/smoke_test_broker.py --broker alpaca_live
    python scripts/smoke_test_broker.py --broker paper
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import argparse
from loguru import logger


def test_broker(broker_type: str):
    """Test broker connectivity and basic methods."""
    print(f"\n{'='*60}")
    print(f"BROKER SMOKE TEST - {broker_type.upper()}")
    print(f"{'='*60}\n")
    
    try:
        if broker_type == "paper":
            from live.paper_broker import PaperBroker
            broker = PaperBroker(initial_capital=100000.0)
            print("✅ PaperBroker initialized")
        elif broker_type == "alpaca_paper":
            from live.alpaca_broker import AlpacaBroker
            broker = AlpacaBroker(paper=True)
            print("✅ AlpacaBroker (PAPER) initialized")
        elif broker_type == "alpaca_live":
            from live.alpaca_broker import AlpacaBroker
            broker = AlpacaBroker(paper=False)
            print("✅ AlpacaBroker (LIVE) initialized")
        else:
            print(f"❌ Unknown broker type: {broker_type}")
            return False
        
        # Test all critical methods
        print("\nTesting broker methods...")
        
        account_value = broker.get_account_value()
        print(f"  get_account_value(): ${account_value:,.2f}")
        if account_value <= 0:
            print(f"  ❌ Invalid account value: ${account_value:,.2f}")
            return False
        
        cash = broker.get_cash()
        print(f"  get_cash(): ${cash:,.2f}")
        if cash < 0:
            print(f"  ❌ Invalid cash balance: ${cash:,.2f}")
            return False
        
        buying_power = broker.get_buying_power()
        print(f"  get_buying_power(): ${buying_power:,.2f}")
        if buying_power < 0:
            print(f"  ❌ Invalid buying power: ${buying_power:,.2f}")
            return False
        
        positions = broker.get_positions()
        print(f"  get_positions(): {len(positions)} positions")
        for pos in positions[:5]:  # Show first 5
            print(f"    - {pos.get('symbol', 'UNKNOWN')}: {pos.get('quantity', 0)} shares")
        if len(positions) > 5:
            print(f"    ... and {len(positions) - 5} more")
        
        # Test quote for a common symbol
        test_symbol = "SPY"
        quote = broker.get_quote(test_symbol)
        print(f"  get_quote('{test_symbol}'): ${quote.get('last', 0):.2f}")
        if quote.get('last', 0) <= 0:
            print(f"  ⚠️  Warning: Invalid quote for {test_symbol}")
        
        print(f"\n{'='*60}")
        print("✅ ALL TESTS PASSED")
        print(f"{'='*60}\n")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        if "alpaca" in str(e).lower():
            print("   Install with: pip install alpaca-py")
        return False
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        if "ALPACA" in str(e):
            print("   Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test broker connectivity")
    parser.add_argument(
        "--broker",
        choices=["paper", "alpaca_paper", "alpaca_live"],
        default="alpaca_paper",
        help="Broker to test (default: alpaca_paper)"
    )
    args = parser.parse_args()
    
    success = test_broker(args.broker)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

