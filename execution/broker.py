import yaml
import os
import logging
from pathlib import Path
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AccountStatus
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('AlpacaRouter')

class AlpacaRouter:
    """Production-grade Alpaca Paper Router."""
    def __init__(self, config_path=None):
        base_dir = Path(__file__).resolve().parent.parent
        self.config_path = config_path or str(base_dir / 'config.yaml')
        self.config = self._load_config()
        self.client = self._init_client()
        self.account = None

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _init_client(self):
        keys = self.config['alpaca_keys']['paper']
        return TradingClient(
            keys['api_key'],
            keys['secret_key'],
            paper=True
        )

    def sync_account(self):
        """Fetches live account state on every call."""
        self.account = self.client.get_account()
        return {
            'status': self.account.status,
            'equity': float(self.account.equity),
            'cash': float(self.account.cash),
            'buying_power': float(self.account.buying_power),
            'positions_count': len(self.client.get_all_positions())
        }

    def place_order(self, ticker: str, qty: float, side: str, 
                    order_type: str = 'market', limit_price: float = None,
                    stop_loss: float = None, take_profit: float = None,
                    trail_percent: float = None):
        """
        Unified order submission. Supports market, limit, trailing stop.
        Returns order dict with id, status, filled_qty.
        """
        try:
            if order_type == 'market':
                order_data = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )
            elif order_type == 'limit':
                if not limit_price:
                    raise ValueError("limit_price required for limit orders")
                order_data = LimitOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.GTC
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            order = self.client.submit_order(order_data)
            
            # Attach risk management if provided
            if stop_loss or take_profit or trail_percent:
                if stop_loss:
                    self.client.submit_order(MarketOrderRequest(
                        symbol=ticker, qty=qty, side=OrderSide.SELL,
                        stop_loss=StopLossRequest(stop_price=stop_loss)))
                if take_profit:
                    self.client.submit_order(MarketOrderRequest(
                        symbol=ticker, qty=qty, side=OrderSide.SELL,
                        take_profit=TakeProfitRequest(limit_price=take_profit)))
                if trail_percent:
                    self.client.submit_order(MarketOrderRequest(
                        symbol=ticker, qty=qty, side=OrderSide.SELL,
                        trailing_percent=trail_percent))

            logger.info(f"✅ ORDER FILLED: {side.upper()} {qty} {ticker} @ market")
            return {
                'status': 'FILLED',
                'order_id': order.id,
                'filled_qty': float(order.filled_qty),
                'avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0
            }

        except Exception as e:
            logger.error(f"❌ ORDER FAILED: {e}")
            return {'status': 'FAILED', 'error': str(e)}

    def get_positions(self):
        """Returns active positions as list of dicts."""
        positions = self.client.get_all_positions()
        return [{
            'symbol': p.symbol,
            'qty': float(p.qty),
            'market_value': float(p.market_value),
            'unrealized_pnl': float(p.unrealized_perunrealized_pl) if hasattr(p, 'unrealized_pl') else 0,
            'avg_entry_price': float(p.avg_entry_price)
        } for p in positions]

    def cancel_all_open_orders(self, ticker: str = None):
        """Cancels pending orders. Optional filter by ticker."""
        try:
            self.client.cancel_orders(cancel_asynchronously=False)
            logger.info("🚫 All open orders cancelled.")
            return True
        except Exception as e:
            logger.error(f"❌ Cancel failed: {e}")
            return False
