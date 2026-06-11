import ccxt
import yfinance as yf
import finnhub
import requests
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

class DataFetcher:
    """Multi-source data pipeline: Yahoo (Mass), Finnhub (Real-time), CCXT (Crypto)."""
    def __init__(self, config):
        self.finnhub_api_key = config.get("data_sources", {}).get("finnhub_api", "DEMO")
        self.finnhub_client = finnhub.Client(api_key=self.finnhub_api_key)

    def get_crypto_realtime(self, symbol="BTC/USD"):
        """Uses CCXT to fetch live orderbook and trade data from Binance."""
        exchange = ccxt.binance()
        orderbook = exchange.fetch_order_book(symbol)
        trades = exchange.fetch_trades(symbol)
        return {"orderbook": orderbook, "trades": trades}

    def get_stock_realtime(self, ticker="SPY"):
        """Uses Finnhub for live quotes (60 req/min limit)."""
        try:
            quote = self.finnhub_client.quote(ticker)
            return quote
        except Exception as e:
            logging.error(f"Finnhub Error for {ticker}: {e}")
            return None

    def get_historical_mass(self, ticker, period="2y"):
        """Uses Yahoo Finance for heavy lifting: 1-year daily/mid-frequency data."""
        return yf.Ticker(ticker).history(period=period)

    def update_all_markets(self):
        """Orchestrates the mass pull and real-time updates."""
        # 1. Mass Historical (Yahoo)
        for t in ["SPY", "AAPL", "BTC-USD"]:
            logging.info(f"Data Pipeline: Yahoo Mass Pull for {t}...")
            df = self.get_historical_mass(t)
            if not df.empty:
                logging.info(f"   ✓ {len(df)} bars loaded.")

        # 2. Real-time (Finnhub/CCXT)
        logging.info("Data Pipeline: Finnhub Real-time for SPY...")
        self.get_stock_realtime("SPY")
        logging.info("Data Pipeline: CCXT Real-time for BTC/USD...")
        self.get_crypto_realtime("BTC/USD")
