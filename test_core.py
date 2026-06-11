#!/usr/bin/env python3
"""
Quick test of the algo-trading-bot core engine.
Validates: config parsing, backtester ingestion, orchestrator lifecycle.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

# Ensure the repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_config():
    """Verify config.yaml parses correctly and has expected keys."""
    import yaml
    with open("config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    # Top-level keys
    assert "system" in cfg, "Missing 'system' key"
    assert "capital" in cfg, "Missing 'capital' key"
    assert "markets" in cfg, "Missing 'markets' key"
    assert "metrics" in cfg, "Missing 'metrics' key"

    # Capital constraints
    cap = cfg["capital"]
    assert cap["min_allocation"] == 300, "min_allocation should be 300"
    assert cap["max_allocation"] == 500, "max_allocation should be 500"
    assert cap["rebalance_interval_days"] == 7, "weekly rebalance expected"

    # Markets
    markets = cfg["markets"]
    assert "stocks" in markets, "Missing stocks market"
    assert "crypto" in markets, "Missing crypto market"
    assert "commodities" in markets, "Missing commodities market"
    assert len(markets["stocks"]["tickers"]) >= 5, "Stocks need 5+ tickers"
    assert len(markets["crypto"]["tickers"]) >= 4, "Crypto needs 4+ tickers"
    assert len(markets["commodities"]["tickers"]) >= 4, "Commodities needs 4+ tickers"

    # 10 metrics
    metrics = cfg["metrics"]
    expected = [
        "momentum", "rsi", "bollinger_bands", "macd",
        "volume_profile", "correlation_matrix", "live_decay",
        "sharpe_ratio", "max_drawdown", "win_rate",
    ]
    for m in expected:
        assert m in metrics, f"Missing metric: {m}"

    # Live Decay (2026 metric) params
    ld = metrics["live_decay"]
    assert "base_decay_rate" in ld["parameters"]
    assert "recapture_rate" in ld["parameters"]
    assert "cap" in ld["parameters"]
    assert "half_life_days" in ld["parameters"]

    print("✓ config.yaml: all checks passed")

def test_backtester_in_out():
    """Verify Backtester ingests a DataFrame and returns a report dict."""
    from core.backtester import Backtester

    # Minimal backtest config
    bt_cfg = {
        "commission_per_trade_bps": 1.0,
        "slippage_bps": 10.0,
        "live_decay_active": True,
        "live_decay": {
            "base_decay_rate": 0.05,
            "recapture_rate": 0.15,
            "cap": 0.90,
            "half_life_days": 14,
        },
    }

    bt = Backtester(bt_cfg, initial_capital=10000.0)

    # Build a tiny synthetic DataFrame
    np.random.seed(42)
    n = 252
    dates = pd.bdate_range("2024-01-01", periods=n)
    returns = pd.Series(np.random.normal(0.0003, 0.015, n))
    prices = (1 + returns).cumprod() * 100
    close = prices.values
    open_col = close * (1 + np.random.uniform(-0.003, 0.003, n))
    high = np.maximum(close, open_col) * 1.002
    low = np.minimum(close, open_col) * 0.998
    volume = np.random.lognormal(15, 1, n).astype(int)
    signal = np.clip(np.random.randn(n), -1, 1)

    df = pd.DataFrame({
        "date": dates,
        "open": open_col,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "signal": signal,
    })

    # Run backtest
    report = bt.run(df, signal_col="signal", close_col="close")

    # Validate report structure
    assert isinstance(report, dict), "report must be a dict"
    assert "initial_capital" in report
    assert "final_capital" in report
    assert "total_return" in report
    assert "total_trades" in report
    assert "sharpe_ratio" in report
    assert "max_drawdown_pct" in report
    assert "win_rate" in report
    assert "profit_factor" in report
    assert "live_decay_final_quality" in report
    assert "bars_backtested" in report

    assert report["initial_capital"] == 10000.0
    assert report["bars_backtested"] == 252
    assert report["total_trades"] >= 0
    assert 0 <= report["win_rate"] <= 1
    assert 0 <= report["live_decay_final_quality"] <= 1

    print("✓ backtester.py: DataFrame → report dict, all keys present")
    print(f"  Final capital : ${report['final_capital']:,.2f}")
    print(f"  Total return  : {report['total_return']:.2%}")
    print(f"  Sharpe ratio  : {report['sharpe_ratio']:.3f}")
    print(f"  Max DD        : {report['max_drawdown_pct']:.2f}%")
    print(f"  Win rate      : {report['win_rate']:.2%}")
    print(f"  Profit factor : {report['profit_factor']:.2f}")
    print(f"  Live decay    : {report['live_decay_final_quality']:.2%}")

def test_backtester_summary():
    """Verify the summary() method works."""
    from core.backtester import Backtester
    bt_cfg = {
        "commission_per_trade_bps": 1.0,
        "slippage_bps": 10.0,
        "live_decay_active": True,
        "live_decay": {"base_decay_rate": 0.05, "recapture_rate": 0.15,
                       "cap": 0.90, "half_life_days": 14},
    }
    bt = Backtester(bt_cfg)
    np.random.seed(99)
    n = 504
    dates = pd.bdate_range("2023-01-01", periods=n)
    returns = pd.Series(np.random.normal(0.0002, 0.012, n))
    close = (1 + returns).cumprod() * 100
    signal = np.clip(np.random.randn(n), -1, 1)
    df = pd.DataFrame({"close": close.values, "signal": signal, "volume": 1_000_000})

    report = bt.run(df, signal_col="signal", close_col="close")
    summary = bt.summary()
    assert "Forge Backtest Report" in summary
    assert "Capital" in summary
    print("✓ backtester summary: renders correctly")

def test_orchestrator():
    """Verify CrestAgent lifecycle."""
    from core.orchestrator import CrestAgent
    agent = CrestAgent("config.yaml")
    status = agent.print_status()
    assert "Crest Agent Status" in status or "CREST AGENT STATUS" in status

    # Run a short backtest
    data = agent.fetch_data(days=60)
    assert len(data) > 0
    first = next(iter(data))
    report = agent.backtest(data[first])
    assert report["bars_backtested"] > 0
    print("✓ orchestrator.py: fetch_data + backtest cycle works")
    print(f"  Tickers fetched : {len(data)}")
    print(f"  Bars per ticker : {report['bars_backtested']}")
    print(f"  Trades          : {report['total_trades']}")

def test_live_decay_model():
    """Verify the Live Decay model state machine."""
    from core.backtester import LiveDecay
    decay = LiveDecay(base_decay_rate=0.05, recapture_rate=0.15, cap=0.90, half_life_days=14)

    # Quality should start near 1.0
    q1 = decay.update(1)
    assert 0.90 < q1 <= 1.0, f"Initial quality should be near 1.0, got {q1}"

    # After many steps without harvesting, quality should decay
    for i in range(2, 50):
        decay.update(i)
    q_after_decay = 1.0 - decay._active_decay
    assert q_after_decay < 0.90, "Decay should reduce quality below 90%"

    # Harvest should recapture
    q_harvest = decay.harvest(50)
    assert q_harvest > q_after_decay, "Harvest should improve quality"

    # Reset
    decay.reset()
    q_reset = decay.update(1)
    assert q_reset == 1.0, "Reset should restore quality to 1.0"

    print("✓ Live Decay model: state machine works correctly")

if __name__ == "__main__":
    print("=" * 60)
    print("  Algo Trading Bot — Core Engine Test Suite")
    print("=" * 60)

    test_config()
    test_backtester_in_out()
    test_backtester_summary()
    test_live_decay_model()
    test_orchestrator()

    print()
    print("=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
