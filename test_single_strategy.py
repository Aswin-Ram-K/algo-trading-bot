import pandas as pd
import yfinance as yf
from logic.strategies import momentum_strategy, mean_reversion_strategy, commodity_carry_strategy, vol_strategy
import numpy as np

# Fetch Data
print("📥 Downloading data...")
df = yf.Ticker("SPY").history(period="5y")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df.dropna(inplace=True)

print(f"📊 Data Shape: {df.shape}")
print(df.head())

# Test Momentum
print("\n🔍 Testing Momentum Strategy on SPY (First 200 days)...")
lookback = 20
params = {'lookback': lookback, 'volume_mult': 1.5}
trades = []

for i in range(20, len(df)):
    window_data = df.iloc[:i+1]
    signal, meta = momentum_strategy(window_data, params)
    
    if signal != 0:
        price = df.iloc[i]['Close']
        print(f"Day {i}: Signal={signal} | Price=${price:.2f} | {meta}")

# Test Mean Reversion
print("\n🔍 Testing Mean Reversion on BTC (First 200 days)...")
btc = yf.Ticker("BTC-USD").history(period="5y")
btc = btc[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
btc.dropna(inplace=True)

params_mr = {'rsi_period': 14, 'bb_std': 2.0}
for i in range(20, len(btc)):
    window_data = btc.iloc[:i+1]
    signal, meta = mean_reversion_strategy(window_data, params_mr)
    
    if signal != 0:
        price = btc.iloc[i]['Close']
        print(f"Day {i}: Signal={signal} | Price=${price:.2f} | {meta}")
