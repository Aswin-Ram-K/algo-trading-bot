import os
import yaml
import alpaca_trade_api as tradeapi
import time
from datetime import datetime, timedelta

class AlpacaBroker:
    """
    Alpaca Paper/Live Trading Execution Engine.
    Handles placing orders, checking positions, and managing risk.
    """
    def __init__(self):
        self.config = self._load_config()
        self.api_key = self.config.get('broker', {}).get('api_key', '')
        self.api_secret = self.config.get('broker', {}).get('api_secret', '')
        self.base_url = self.config.get('broker', {}).get('base_url', '')
        self.risk_per_trade = self.config.get('system', {}).get('risk_per_trade', 0.01)
        
        if not self.api_key or self.api_key == "YOUR_ALPACA_API_KEY":
            print("⚠️  WARNING: Set your Alpaca API keys in config_optimized.yaml")
            print("   1. Go to: https://alpaca.markets")
            print("   2. Create a free account")
            print("   3. Copy Paper API keys to config_optimized.yaml")
            self.connected = False
            return
            
        try:
            self.api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version='v2'
            )
            self.connected = True
            self.account = self.api.get_account()
            print(f"✅ Connected to Alpaca Paper Trading")
            print(f"   💰 Buying Power: ${float(self.account.buying_power):.2f}")
        except Exception as e:
            print(f"❌ Failed to connect to Alpaca: {e}")
            self.connected = False

    def _load_config(self):
        try:
            with open(os.path.join(os.path.dirname(__file__), 'config_optimized.yaml'), 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}

    def check_portfolio(self):
        """Get current portfolio status."""
        if not self.connected: return None
        
        try:
            positions = self.api.get_all_positions()
            portfolio = self.api.get_portfolio_history()
            
            print(f"\n--- 📊 Portfolio Status ---")
            print(f"   Equity: ${float(self.account.equity):.2f}")
            print(f"   Buying Power: ${float(self.account.buying_power):.2f}")
            print(f"   Day Trades: {self.account.daytrade_count}")
            
            if positions:
                print(f"\n   Current Positions:")
                for pos in positions:
                    print(f"      📈 {pos.symbol}: {pos.qty} shares (${float(pos.current_price):.2f})")
            else:
                print(f"\n   No open positions.")
            return portfolio
        except Exception as e:
            print(f"❌ Error checking portfolio: {e}")
            return None

    def place_order(self, symbol, qty, side="buy"):
        """Place an order with safety checks."""
        if not self.connected:
            print(f"❌ Not connected to Alpaca. Cannot place order.")
            return False
            
        try:
            # Safety: Check if we have enough buying power
            if side == "buy":
                price = self._get_current_price(symbol)
                cost = qty * price
                
                if cost > float(self.account.buying_power):
                    print(f"⚠️  Insufficient buying power for {symbol} order (${cost:.2f})")
                    return False
                    
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day"
            )
            
            status = "BUY" if side == "buy" else "SELL"
            print(f"   ✅ {status} {qty:.4f} shares of {symbol}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to place order: {e}")
            return False

    def _get_current_price(self, symbol):
        """Get the current market price for an asset."""
        try:
            quote = self.api.get_quote(symbol)
            return float(quote.ask_price)
        except:
            return None

    def generate_trading_signal(self, strategy_data):
        """
        Convert strategy output into an actual trade signal.
        strategy_data = {
            'strategy': 'mean_reversion',
            'asset': 'BTC-USD',
            'signal': 1,  # 1=Buy, -1=Sell, 0=Hold
            'price': 60000.00,
            'capital': 500.00
        }
        """
        if not self.connected: return False
        
        signal = strategy_data.get('signal', 0)
        asset = strategy_data.get('asset')
        capital = strategy_data.get('capital', 500.00)
        risk = self.risk_per_trade
        
        if signal == 1: # BUY
            # Calculate position size based on risk
            price = self._get_current_price(asset)
            if not price: return False
            
            stop_loss = price * (1 - 0.03)  # 3% stop loss (from config)
            risk_amount = capital * risk
            shares = risk_amount / abs(price - stop_loss)
            
            if shares > 0:
                return self.place_order(asset, shares)
                
        elif signal == -1: # SELL
            try:
                pos = self.api.get_position(asset)
                if float(pos.qty) > 0:
                    return self.place_order(asset, abs(float(pos.qty)), "sell")
            except:
                pass
                
        return False

    def run_live_check(self):
        """
        Main loop for live trading.
        Checks portfolio and executes signals from strategies.
        """
        if not self.connected:
            print("⏳ Waiting for Alpaca connection...")
            return
            
        self.check_portfolio()
        print("\n--- 🤖 Running Strategies ---")
        
        # Load strategies from config
        with open(os.path.join(os.path.dirname(__file__), 'config_optimized.yaml'), 'r') as f:
            config = yaml.safe_load(f)
            
        for strat_name, strat in config.get('strategies', {}).items():
            if not strat.get('active', False): continue
            
            # Here we would call the strategy engine to get a signal
            # For now, we'll simulate the first run
            if strat_name == 'mean_reversion':
                signal_data = {
                    'strategy': strat_name,
                    'asset': strat['asset'],
                    'signal': 1,  # Simulate a buy signal
                    'price': 60000.00,
                    'capital': config['system']['initial_capital']
                }
                self.generate_trading_signal(signal_data)
            elif strat_name == 'momentum':
                signal_data = {
                    'strategy': strat_name,
                    'asset': strat['asset'],
                    'signal': 0,  # Simulate hold
                    'price': 400.00,
                    'capital': config['system']['initial_capital']
                }
                self.generate_trading_signal(signal_data)