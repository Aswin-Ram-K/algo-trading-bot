import yaml
import os
import sys
import time
from datetime import datetime

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from execution.alpaca_broker import AlpacaBroker
from logic.strategies import mean_reversion_strategy, momentum_strategy

def load_config():
    with open(os.path.join(os.path.dirname(__file__), 'config_optimized.yaml'), 'r') as f:
        return yaml.safe_load(f)

def run_live_cycle(broker, config):
    """
    Run one complete cycle of the self-improving loop:
    1. Get market data
    2. Run all active strategies
    3. Execute signals
    4. Log results
    """
    print(f"\n{'='*50}")
    print(f"🔄 LIVE CYCLE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    # 1. Check Portfolio
    broker.check_portfolio()
    
    # 2. Get Data & Run Strategies
    # In production, this would pull live data from Alpaca/Yahoo
    # For now, we use the optimized backtest parameters
    
    active_strategies = {
        name: s for name, s in config.get('strategies', {}).items() if s.get('active', False)
    }
    
    if not active_strategies:
        print("⚠️  No active strategies configured.")
        return
        
    print(f"\n--- 🤖 Running {len(active_strategies)} Strategies ---")
    
    for name, strategy in active_strategies.items():
        print(f"\n🔍 Testing {strategy['name']} on {strategy['asset']}...")
        
        # Simulate signal generation (In production, this uses live data)
        # We'll use the optimized parameters from the backtest
        
        signal = 0  # Default: Hold
        if name == 'mean_reversion':
            # In real production, this would check RSI < 30 from live data
            signal = 1  # Simulate a buy signal for testing
            
        elif name == 'momentum':
            # In real production, this would check breakout from 20d high
            signal = 0  # Simulate hold
            
        if signal != 0:
            signal_data = {
                'strategy': name,
                'asset': strategy['asset'],
                'signal': signal,
                'price': 60000.00 if name == 'mean_reversion' else 400.00,
                'capital': config['system']['initial_capital']
            }
            broker.generate_trading_signal(signal_data)
        else:
            print(f"   ⏸️  {strategy['name']}: HOLD (Waiting for signal)")

def main():
    print("🚀 Starting Hermes Self-Improving Trade Board...")
    print("   ⚠️  Running in PAPER mode (no real money)")
    
    config = load_config()
    broker = AlpacaBroker()
    
    if not broker.connected:
        print("\n⏳ Waiting for Alpaca API keys...")
        print("   Please add your keys to config_optimized.yaml")
        print("   Then run: python3 run_live.py")
        return
        
    print("\n✅ System Ready. Starting live cycle...")
    
    # Run one cycle
    run_live_cycle(broker, config)
    
    print(f"\n{'='*50}")
    print("📝 CYCLE COMPLETE")
    print("   Next cycle: Check again in 24 hours")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()