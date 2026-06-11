import yaml
import sys
import os
import pandas as pd
import yfinance as yf
from typing import Dict
import numpy as np

# Add path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtester import BacktesterEngine
from logic.strategies import momentum_strategy, mean_reversion_strategy, commodity_carry_strategy, vol_strategy

def fetch_data(symbol: str, period: str = "10y") -> pd.DataFrame:
    """Fetches historical data."""
    print(f"   📥 Downloading {symbol}...")
    try:
        data = yf.Ticker(symbol).history(period=period)
        if data.empty:
            return None
        # Clean data
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"   ❌ Error fetching {symbol}: {e}")
        return None

def optimize_strategy(name: str, strategy_fn, data: pd.DataFrame, base_params: dict) -> Dict:
    """
    Runs a grid search over parameters to find the best combination.
    """
    print(f"   🔬 Optimizing {name}...")
    
    best_score = -9999
    best_params = base_params
    best_metrics = {}
    
    # Define parameter ranges for optimization
    param_grid = []
    if "Momentum" in name:
        # Vary: Lookback, Volume Multiplier, Stop Loss, Take Profit
        for lb in [10, 20, 30, 40, 50]:
            for vm in [1.2, 1.5, 1.8]:
                for sl in [0.03, 0.05, 0.08]:
                    for tp in [0.05, 0.08, 0.12]:
                        param_grid.append({'lookback': lb, 'volume_mult': vm, 'stop_loss': sl, 'take_profit': tp})
    
    elif "Mean Reversion" in name:
        # Vary: RSI Period, Bollinger Std, Stop Loss, Take Profit
        for rsi in [7, 14, 21]:
            for bb in [1.5, 2.0, 2.5]:
                for sl in [0.03, 0.05, 0.08]:
                    for tp in [0.05, 0.08, 0.12]:
                        param_grid.append({'rsi_period': rsi, 'bb_std': bb, 'stop_loss': sl, 'take_profit': tp})
                        
    elif "Carry" in name or "Commodity" in name:
        # Vary: SMA Lookbacks, Stop Loss
        for sl in [0.03, 0.05, 0.08]:
            for tp in [0.05, 0.08, 0.12]:
                param_grid.append({'stop_loss': sl, 'take_profit': tp})
                
    elif "Volatility" in name:
        # Vary: Lookback, Volume Mult, Stop Loss
        for lb in [5, 10, 15, 20]:
            for vm in [1.5, 2.0, 2.5]:
                for sl in [0.05, 0.08, 0.12]:
                    for tp in [0.08, 0.12, 0.20]:
                        param_grid.append({'lookback': lb, 'volume_mult': vm, 'stop_loss': sl, 'take_profit': tp})

    # Run Optimization (Limit iterations to keep it fast)
    # In a real prod env, we'd use Optuna or Bayesian Optimization
    # For now, brute force over reasonable range
    
    engine = BacktesterEngine(initial_capital=500, risk_per_trade=0.01)
    valid_runs = 0
    total_runs = len(param_grid)
    
    for i, p in enumerate(param_grid):
        if valid_runs > 200: # Stop after finding top 200 valid configs to save time
            break
            
        try:
            metrics = engine.run(strategy_fn, data, p)
            if metrics.get('status') == 'NO_TRADES':
                continue
            
            # Score: Weighted combination of Key Metrics
            score = (
                metrics['total_return'] * 1000 + 
                metrics['win_rate'] * 50 + 
                metrics['profit_factor'] * 20 + 
                metrics['sharpe'] * 5 - 
                (metrics['max_drawdown'] * 100)
            )
            
            if score > best_score:
                best_score = score
                best_params = p
                best_metrics = metrics
                valid_runs += 1
                
        except Exception as e:
            continue

    return best_params, best_metrics, best_score

def main():
    print("=" * 60)
    print("🚀 ALGO TRADING OPTIMIZATION: 5-YEAR BACKTEST")
    print("=" * 60)
    
    # 1. Fetch Data for all assets
    data = {
        "SPY": fetch_data("SPY"),
        "BTC-USD": fetch_data("BTC-USD"),
        "GLD": fetch_data("GLD"), # Gold proxy for carry
        "UVXY": fetch_data("UVXY") # Volatility proxy
    }
    
    # Filter out None
    data = {k: v for k, v in data.items() if v is not None}
    
    # 2. Define Seed Strategies
    strategies = [
        ("Momentum (SPY)", momentum_strategy, data["SPY"], {'lookback': 20, 'volume_mult': 1.5}),
        ("Mean Reversion (BTC)", mean_reversion_strategy, data["BTC-USD"], {'rsi_period': 14, 'bb_std': 2.0}),
        ("Carry (GLD)", commodity_carry_strategy, data["GLD"], {}),
        ("Volatility (UVXY)", vol_strategy, data["UVXY"], {'lookback': 10, 'volume_mult': 2.0})
    ]
    
    results = {}
    
    for name, fn, asset_data, base_p in strategies:
        p, m, s = optimize_strategy(name, fn, asset_data, base_p)
        results[name] = {
            'params': p,
            'metrics': m,
            'score': s
        }
        
        print(f"\n   ✅ {name}")
        print(f"   💰 Final Capital: ${m.get('final_capital', 0):,.2f} ({m.get('total_return', 0)*100:.2f}%)")
        print(f"   🎯 Win Rate: {m.get('win_rate', 0)*100:.1f}%")
        print(f"   📈 Profit Factor: {m.get('profit_factor', 0):.2f}")
        print(f"   📉 Max Drawdown: {m.get('max_drawdown', 0)*100:.2f}%")
        print(f"   📊 Sharpe: {m.get('sharpe', 0):.2f}")
        print(f"   🧩 Best Params: {p}")

    # 3. Generate Config
    print("\n" + "=" * 60)
    print("📝 OPTIMIZED CONFIGURATION READY")
    print("=" * 60)
    
    optimized_config = {
        "system": {
            "name": "Hermes Operator",
            "mode": "LIVE",
            "capital": 500.00,
            "risk_per_trade": 0.01
        },
        "optimized_strategies": results
    }
    
    with open("config_optimized.yaml", "w") as f:
        yaml.dump(optimized_config, f)
        
    print("✅ Saved to config_optimized.yaml")

if __name__ == "__main__":
    main()
