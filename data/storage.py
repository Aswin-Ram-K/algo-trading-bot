import sqlite3
import pandas as pd

class DataVault:
    """Local SQLite storage for 2026 historical data."""
    def __init__(self):
        self.conn = sqlite3.connect("algo_data.db")

    def store_ohlcv(self, symbol, df):
        df.to_sql(f"ohlcv_{symbol}", self.conn, if_exists="replace", index=False)

    def store_trade_log(self, trade_data):
        pd.DataFrame([trade_data]).to_sql("trade_log", self.conn, if_exists="append", index=False)

def log_daily_report():
    logging.info("DataVault: Daily trade report logged.")
