import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Regime gate helper
# ──────────────────────────────────────────────────────────────────────────────

# Map each strategy → the regime(s) in which it is expected to work well.
# HMM regime filter: strategy only fires when current regime is compatible.
_STRATEGY_REGIME_MAP = {
    "MOMENTUM":    {"Bull/Trending"},
    "MEAN_REV":    {"Sideways/Chop"},
    "CARRY":       {"Bull/Trending", "LowVol"},
    "VOLATILITY":  {"HighVol"},
}


def should_fire(strategy_type: str, regime: str, regime_confidence: float,
                confidence_threshold: float = 0.60) -> bool:
    """
    Gate function: should this strategy be allowed to fire given the current
    HMM regime and its confidence?

    Parameters
    ----------
    strategy_type : str
        Strategy identifier (e.g. "MOMENTUM", "MEAN_REV", "CARRY", "VOLATILITY").
    regime : str
        Current regime label from RegimeDetector (e.g. "Bull/Trending",
        "Bear/Declining", "Sideways/Chop", "HighVol", "LowVol").
    regime_confidence : float
        Probability of the dominant regime (0–1).
    confidence_threshold : float, optional
        Minimum confidence to allow any signal (default 0.60).

    Returns
    -------
    bool
        True if the strategy is allowed to fire under the current regime.
    """
    # If regime detection is not confident enough, block all signals
    if regime_confidence < confidence_threshold:
        return False

    allowed_regimes = _STRATEGY_REGIME_MAP.get(strategy_type, set())
    if not allowed_regimes:
        return True  # Unknown strategy — passthrough (conservative)

    return regime in allowed_regimes


# --- Strategy 1: Equity Momentum (Swing Breakout) ---
def momentum_strategy(df: pd.DataFrame, params: dict):
    """
    Buys when SPY/QQQ breaks above the high of the last N days on high volume.
    """
    lookback = int(params.get('lookback', 20))
    volume_mult = float(params.get('volume_mult', 1.2))
    
    if len(df) < lookback:
        return 0, {}

    last_row = df.iloc[-1]
    window = df.tail(lookback + 1) # Include current day in lookback
    
    # Conditions
    highest_high = window['High'].iloc[:-1].max() # High of previous N days
    avg_vol = window['Volume'].iloc[:-1].mean()    # Avg vol of previous N days
    
    if last_row['Close'] > highest_high and last_row['Volume'] > (avg_vol * volume_mult):
        return 1, {'type': 'MOMENTUM', 'signal': 'BUY'}
    
    if last_row['Close'] < window['Low'].iloc[:-1].min():
        return -1, {'type': 'MOMENTUM', 'signal': 'SELL'}
        
    return 0, {}

# --- Strategy 2: Crypto Mean Reversion ---
def mean_reversion_strategy(df: pd.DataFrame, params: dict):
    """
    Buys when RSI is oversold (<30) and price touches the lower Bollinger Band.
    Exits when RSI returns to 70 or middle band.
    """
    rsi_period = int(params.get('rsi_period', 14))
    bb_std = int(params.get('bb_std', 2))
    
    if len(df) < rsi_period + 20:
        return 0, {}

    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    upper_band = rolling_mean + (bb_std * rolling_std)
    lower_band = rolling_mean - (bb_std * rolling_std)
    
    current_rsi = rsi.iloc[-1]
    current_price = df['Close'].iloc[-1]
    lower = lower_band.iloc[-1]
    upper = upper_band.iloc[-1]
    
    if current_rsi < 30 and current_price <= lower:
        return 1, {'type': 'MEAN_REV', 'signal': 'BUY'}
    
    if current_rsi > 70 or current_price >= rolling_mean.iloc[-1]:
        return -1, {'type': 'MEAN_REV', 'signal': 'SELL'}
        
    return 0, {}

# --- Strategy 3: Commodity Carry (Simplified for ETFs) ---
def commodity_carry_strategy(df: pd.DataFrame, params: dict):
    """
    Buys Gold/USO when 50-day SMA > 200-day SMA (Trend is up).
    Exits when 50-day SMA < 200-day SMA.
    """
    if len(df) < 200:
        return 0, {}
        
    sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
    sma_200 = df['Close'].rolling(window=200).mean().iloc[-1]
    
    if sma_50 > sma_200 and sma_50 > df['Close'].iloc[-2]:
        return 1, {'type': 'CARRY', 'signal': 'BUY'}
        
    if sma_50 < sma_200:
        return -1, {'type': 'CARRY', 'signal': 'SELL'}
        
    return 0, {}

# --- Strategy 4: Volatility Regime Shift (UVXY) ---
def vol_strategy(df: pd.DataFrame, params: dict):
    """
    Buys UVXY when it breaks above its 10-day high and Volume is 2x average.
    Fades (Sells) when it closes below its 10-day SMA.
    """
    lookback = int(params.get('lookback', 10))
    volume_mult = float(params.get('volume_mult', 2.0))
    
    last_row = df.iloc[-1]
    
    highest_high = df['High'].tail(lookback).max()
    avg_vol = df['Volume'].tail(lookback).mean()
    
    if last_row['Close'] > highest_high and last_row['Volume'] > (avg_vol * volume_mult):
        return 1, {'type': 'VOLATILITY', 'signal': 'BUY'}
        
    if last_row['Close'] < df['Close'].tail(lookback).mean():
        return -1, {'type': 'VOLATILITY', 'signal': 'SELL'}
        
    return 0, {}
