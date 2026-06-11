# -*- coding: utf-8 -*-
"""Smoke tests for the entire LOGIC & DATA infrastructure."""

import sys
import warnings
warnings.filterwarnings("ignore")

print("=" * 64)
print("  ALGO-TRADING-BOT — Smoke Test Suite")
print("=" * 64)

passed = 0
failed = 0

def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅  {name}")
        passed += 1
    except Exception as exc:
        print(f"  ❌  {name} → {exc}")
        failed += 1

# ── 1. data/fetcher ──────────────────────────────────────────────────
print("\n[data/fetcher.py]")

def test_import_fetcher():
    from data.fetcher import AlphaVantageFetcher, CCXTCryptoFetcher, MultiAssetFetcher
    av = AlphaVantageFetcher()
    assert av.api_key is not None
    assert hasattr(av, "get_daily")
    ccxt = CCXTCryptoFetcher()
    assert len(ccxt.exchanges) == 3
    mf = MultiAssetFetcher()
    assert mf.av is not None
    assert mf.ccxt is not None

check("AlphaVantageFetcher init", test_import_fetcher)

def test_ccxt_list():
    from data.fetcher import CCXTCryptoFetcher
    ccxt = CCXTCryptoFetcher()
    markets = ccxt.list_markets("binance")
    assert isinstance(markets, list)
    assert len(markets) > 0

check("CCXTCryptoFetcher list markets", test_ccxt_list)

# ── 2. data/storage ─────────────────────────────────────────────────
print("\n[data/storage.py]")

def test_storage():
    from data.storage import TradingDB, seed_demo_data
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = TradingDB(db_path)
        seed_demo_data(db)
        df = db.load_ohlcv("equity", "DEMO")
        assert len(df) > 1000, f"Expected >1000 OHLCV rows, got {len(df)}"
        trades = db.load_trades()
        assert len(trades) > 0
        db.store_metrics("test_strategy", "2026-01-01", win_rate=0.65)
        snap = db.load_metrics("test_strategy", "2026-01-01")
        assert len(snap) > 0
    finally:
        os.unlink(db_path)

check("DB init + seed + store/load metrics", test_storage)

# ── 3. logic/metrics ─────────────────────────────────────────────────
print("\n[logic/metrics.py]")

def test_metrics():
    import numpy as np
    from logic.metrics import TradingMetrics

    pnl = np.random.normal(200, 500, 100).tolist()
    eq = np.cumsum([0] + pnl)
    ret = np.random.normal(0.02, 0.08, 100).tolist()

    tm = TradingMetrics(
        pnl=pnl, equity_curve=eq, returns=ret,
        n_trades=100,
        start_date="2021-01-01", end_date="2026-01-01",
        regime="AI_growth",
        paper_pnl=15000, live_pnl=12000,
    )
    m = tm.compute()
    assert len(m) == 10, f"Expected 10 metrics, got {len(m)}"
    assert 0 <= m["win_rate"] <= 1
    assert m["profit_factor"] >= 0
    assert m["max_drawdown"] >= 0
    assert 0 <= m["regime_fit"] <= 1
    assert 0 <= m["live_decay"] <= 1

check("All 10 metrics compute correctly", test_metrics)

def test_metrics_summary():
    from logic.metrics import TradingMetrics
    import numpy as np
    pnl = np.random.normal(100, 300, 50).tolist()
    eq = np.cumsum([0] + pnl)
    tm = TradingMetrics(pnl=pnl, equity_curve=eq, returns=pnl, n_trades=50)
    s = tm.summary()
    assert "Win Rate" in s
    assert "Max Drawdown" in s

check("Metrics summary string format", test_metrics_summary)

def test_from_trades_df():
    import pandas as pd
    from logic.metrics import TradingMetrics
    df = pd.DataFrame({
        "pnl": np.random.normal(150, 400, 80).tolist(),
        "pnl_pct": np.random.normal(0.02, 0.06, 80).tolist(),
        "timestamp": pd.date_range("2022-01-01", periods=80, freq="D"),
    })
    tm = TradingMetrics.from_trades_df(df)
    m = tm.compute()
    assert len(m) == 10

check("from_trades_df factory", test_from_trades_df)

# ── 4. logic/strategies ──────────────────────────────────────────────
print("\n[logic/strategies.py]")

def test_strategies_import():
    from logic.strategies import (
        CryptoMeanReversion, CommodityCarry, EquityMomentum,
        get_strategy, list_strategies, BaseStrategy,
    )
    assert len(list_strategies()) == 3

check("Strategy imports + registry", test_strategies_import)

def test_crypto_mr():
    import numpy as np, pandas as pd
    from logic.strategies import CryptoMeanReversion
    np.random.seed(42)
    n = 200
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    df = pd.DataFrame({
        "open": prices * 0.998, "high": prices * 1.005,
        "low": prices * 0.995, "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=pd.date_range("2021-01-01", periods=n))
    strat = CryptoMeanReversion()
    sigs = strat.generate_signals(df)
    assert "signal" in sigs.columns
    assert set(sigs["signal"].unique()).issubset({-1, 0, 1})

check("CryptoMeanReversion generate_signals", test_crypto_mr)

def test_commodity_carry():
    import numpy as np, pandas as pd
    from logic.strategies import CommodityCarry
    np.random.seed(42)
    n = 200
    soft = 50 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
    hard = 200 * np.exp(np.cumsum(np.random.normal(0.0001, 0.012, n)))
    df = pd.DataFrame({
        "soft": soft, "hard": hard,
        "spread": soft - hard,
    }, index=pd.date_range("2021-01-01", periods=n))
    strat = CommodityCarry()
    sigs = strat.generate_signals(df)
    assert "signal" in sigs.columns

check("CommodityCarry generate_signals", test_commodity_carry)

def test_equity_momentum():
    import numpy as np, pandas as pd
    from logic.strategies import EquityMomentum
    np.random.seed(42)
    n = 300  # need 200+ for long_ma
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.012, n)))
    df = pd.DataFrame({
        "open": prices * 0.998, "high": prices * 1.005,
        "low": prices * 0.995, "close": prices,
        "volume": np.random.randint(1_000_000, 30_000_000, n),
    }, index=pd.date_range("2021-01-01", periods=n))
    strat = EquityMomentum()
    sigs = strat.generate_signals(df)
    assert "signal" in sigs.columns
    assert "vol_scale" in sigs.columns

check("EquityMomentum generate_signals", test_equity_momentum)

def test_backtest():
    import numpy as np, pandas as pd
    from logic.strategies import EquityMomentum
    np.random.seed(42)
    n = 300
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.012, n)))
    df = pd.DataFrame({
        "open": prices, "high": prices * 1.01,
        "low": prices * 0.99, "close": prices,
        "volume": np.random.randint(1_000_000, 30_000_000, n),
    }, index=pd.date_range("2021-01-01", periods=n))
    strat = EquityMomentum()
    bt = strat.backtest_signals(df)
    assert "total_return" in bt
    assert "num_trades" in bt
    assert "win_rate" in bt

check("Backtest simulation", test_backtest)

def test_get_strategy():
    from logic.strategies import get_strategy
    s = get_strategy("crypto_mean_reversion")
    assert s.name == "crypto_mean_reversion"

check("get_strategy factory", test_get_strategy)

# ── results ──────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print(f"  Results:  {passed} passed,  {failed} failed  out of {passed+failed}")
print("=" * 64)

if failed > 0:
    sys.exit(1)
else:
    print("\n  🎉  All smoke tests passed!")
