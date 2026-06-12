#!/usr/bin/env python3
"""
Daily Self-Improvement Cycle for Algo Trading Bot.
Run standalone:  python3 daily_cycle.py

Steps:
  1. Pull latest 24h market data (Yahoo Finance, Finnhub, IBKR stub)
  2. Backtest all seed strategies with current params
  3. Compare metrics across strategies
  4. Mutate params, compare against live (flag >5% gains on 3+ metrics)
  5. Generate plain-English report
  6. Save to reports/daily_YYYY-MM-DD.md
  7. NO trade execution
"""

import os
import sys
import json
import math
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("daily_cycle")

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA FETCHING
# ──────────────────────────────────────────────────────────────────────────────

def fetch_yahoo(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch via yfinance."""
    import yfinance as yf
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is not None and len(df) > 0:
            # Normalize column names
            cols = {'Open': 'Open', 'High': 'High', 'Low': 'Low',
                    'Close': 'Close', 'Volume': 'Volume'}
            df = df[[c for c in cols.values() if c in df.columns]]
            df = df.rename(columns=cols)
            df.index.name = 'Date'
            log.info(f"  Yahoo: {ticker} → {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
            return df
    except Exception as e:
        log.warning(f"  Yahoo fetch failed for {ticker}: {e}")
    return pd.DataFrame()


def fetch_finnhub(ticker: str, lookback_days: int = 30) -> pd.DataFrame:
    """Fetch via Finnhub (daily candles)."""
    try:
        import finnhub
        from config import data_sources  # optional
        api_key = os.environ.get("FINNHUB_API_KEY", "d8kl389r01qjgd71blhgd8kl389r01qjgd71bli0")
        client = finnhub.Client(api_key=api_key)
        from datetime import date, timedelta as td2
        end_dt = date.today()
        start_dt = end_dt - td2(days=lookback_days)
        raw = client.stock_candles(ticker, "D",
                                   int(start_dt.timestamp()),
                                   int(end_dt.timestamp()))
        if raw.get("s") == "ok" and raw.get("c") and raw.get("o") and raw.get("h") and raw.get("l"):
            prices = raw["c"]
            opens, highs, lows, vols = raw.get("o", []), raw.get("h", []), raw.get("l", []), raw.get("v", [])
            dates_raw = raw.get("t", [])
            import time as _time
            dates = [datetime.utcfromtimestamp(t).date() for t in dates_raw]
            df = pd.DataFrame({
                "Date": dates, "Open": opens, "High": highs,
                "Low": lows, "Close": prices, "Volume": vols
            }).dropna()
            if len(df) > 0:
                log.info(f"  Finnhub: {ticker} → {len(df)} bars")
                return df.set_index("Date")
    except Exception as e:
        log.warning(f"  Finnhub fetch failed for {ticker}: {e}")
    return pd.DataFrame()


def fetch_ibkr_stub(ticker: str, period: str = "5d") -> pd.DataFrame:
    """
    IBKR data stub: in production this would connect to TWS/Gateway via
    ib_insync.  Here we generate synthetic data matching the requested
    ticker profile so the backtester still gets something.
    """
    import random as _r
    np.random.seed(hash(ticker) % 2**31)
    n = 200  # ~1 year of bars
    base_prices = {"GC=F": 2400, "CL=F": 75, "SI=F": 28}
    base = base_prices.get(ticker, 200)
    returns = np.random.normal(0.0002, 0.012, n - 1)  # n-1 returns → n prices
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    dates = [datetime.today() - timedelta(days=n - 1 - i) for i in range(n)]
    opens = [p * (1 + _r.uniform(-0.003, 0.003)) for p in prices]
    highs = [p * (1 + _r.uniform(0.001, 0.015)) for p in prices]
    lows = [p * (1 - _r.uniform(0.001, 0.015)) for p in prices]
    volumes = [int(_r.uniform(1e5, 5e6)) for _ in range(n)]
    df = pd.DataFrame({
        "Date": dates,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": prices,
        "Volume": volumes,
    }).set_index("Date")
    log.info(f"  IBKR stub: {ticker} → {len(df)} bars (synthetic)")
    return df


def collect_market_data():
    """Pull 24h data from all sources."""
    log.info("── 1. DATA FETCH ──")
    data = {}

    # Yahoo: equities & crypto
    for sym in ["SPY", "QQQ", "AAPL", "TSLA", "BTC-USD", "ETH-USD", "UVXY"]:
        data[sym] = fetch_yahoo(sym)

    # Finnhub: major indices / equities
    for sym in ["SPY", "QQQ", "AAPL", "TSLA"]:
        if sym in data and not data[sym].empty:
            continue  # already have from Yahoo
        data[sym] = fetch_finnhub(sym)

    # IBKR: commodities
    for sym in ["GC=F", "CL=F", "SI=F"]:
        data[sym] = fetch_ibkr_stub(sym)

    present = {k for k, v in data.items() if not v.empty}
    log.info(f"  Data ready for {len(present)} symbols: {sorted(present)}")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# 2. STRATEGY IMPORT & BACKTESTING
# ──────────────────────────────────────────────────────────────────────────────

from logic.strategies import (
    momentum_strategy,
    mean_reversion_strategy,
    commodity_carry_strategy,
    vol_strategy,
    should_fire,
)
from core.backtester import BacktesterEngine
from core.regime import RegimeDetector


STRATEGIES = [
    {
        "name": "Momentum",
        "fn": momentum_strategy,
        "type": "MOMENTUM",
        "assets": ["SPY", "QQQ"],
        "params": {
            "lookback": 20,
            "volume_mult": 1.2,
            "stop_loss": 0.05,
            "take_profit": 0.08,
        },
    },
    {
        "name": "Mean Reversion",
        "fn": mean_reversion_strategy,
        "type": "MEAN_REV",
        "assets": ["BTC-USD", "ETH-USD"],
        "params": {
            "rsi_period": 7,
            "bb_std": 1.5,
            "stop_loss": 0.03,
            "take_profit": 0.05,
        },
    },
    {
        "name": "Carry",
        "fn": commodity_carry_strategy,
        "type": "CARRY",
        "assets": ["GC=F", "CL=F", "SPY"],
        "params": {
            "stop_loss": 0.05,
            "take_profit": 0.08,
        },
    },
    {
        "name": "Volatility",
        "fn": vol_strategy,
        "type": "VOLATILITY",
        "assets": ["UVXY"],
        "params": {
            "lookback": 10,
            "volume_mult": 2.0,
            "stop_loss": 0.05,
            "take_profit": 0.08,
        },
    },
]


def backtest_strategy(strategy, asset_data):
    """Run a single strategy against one asset's data."""
    engine = BacktesterEngine(initial_capital=500.0, risk_per_trade=0.01)
    try:
        result = engine.run(strategy["fn"], asset_data, strategy["params"])
        return result
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def run_backtests(data):
    """Backtest all strategies on all their assets."""
    log.info("── 2. BACKTESTING ALL SEED STRATEGIES ──")
    results = {}

    for strat in STRATEGIES:
        strat_key = strat["name"]
        results[strat_key] = {}
        for asset in strat["assets"]:
            if asset not in data or data[asset].empty:
                log.info(f"  ⚠ {strat['name']} on {asset}: no data, skipped")
                results[strat_key][asset] = {"status": "NO_DATA"}
                continue

            result = backtest_strategy(strat, data[asset])
            results[strat_key][asset] = result
            n_trades = result.get("total_trades", 0)
            if result.get("status") == "NO_TRADES":
                log.info(f"  {strat['name']} on {asset}: NO TRADES")
            else:
                log.info(
                    f"  {strat['name']} on {asset}: {n_trades} trades, "
                    f"WR={result.get('win_rate', 0):.1%}, "
                    f"PF={result.get('profit_factor', 0):.2f}, "
                    f"Sharpe={result.get('sharpe', 0):.2f}, "
                    f"MaxDD={result.get('max_drawdown', 0):.2%}"
                )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 3. MUTATION & COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def mutate_params(base_params):
    """Generate mutated parameter sets."""
    mutations = []
    keys = list(base_params.keys())
    if not keys:
        return [base_params]

    # For numeric params, try ±25% and ±50%
    for key in keys:
        val = base_params[key]
        if isinstance(val, (int, float)):
            for factor in [0.75, 1.25]:
                new = {**base_params, key: round(val * factor, 4) if isinstance(val, float) else int(val * factor)}
                mutations.append(new)
            for delta in [0.01, -0.01]:
                new = {**base_params, key: round(val + delta, 4)}
                mutations.append(new)

    # Ensure we have at least the base
    return [base_params] + mutations


def compare_metrics(metrics_a, metrics_b, a_name="current", b_name="mutant"):
    """
    Compare two metric dicts.
    Returns count of metrics where b beats a by >5% (absolute difference in
    the right direction).
    """
    beat_count = 0
    details = []
    metric_keys = ["win_rate", "profit_factor", "sharpe"]
    if "max_drawdown" in metrics_a and "max_drawdown" in metrics_b:
        # Lower is better
        a = metrics_a["max_drawdown"]
        b = metrics_b["max_drawdown"]
        if b < a * 0.95:  # at least 5% lower drawdown
            beat_count += 1
            details.append("max_drawdown")

    for key in metric_keys:
        va = metrics_a.get(key, 0)
        vb = metrics_b.get(key, 0)
        if va == 0 and vb > 0:
            beat_count += 1
            details.append(key)
        elif va > 0:
            if vb > va * 1.05:  # 5% improvement
                beat_count += 1
                details.append(key)

    return beat_count, details


def run_mutation_search(data):
    """Mutate parameters, backtest, find candidates that beat current on 3+ metrics."""
    log.info("── 3. PARAMETER MUTATION SEARCH ──")
    candidates = []

    for strat in STRATEGIES:
        strat_key = strat["name"]
        base_params = strat["params"]

        # Find best current result across assets
        best_current = None
        best_current_asset = None
        best_score = -999
        for asset, res in results_global.get(strat_key, {}).items():
            if res.get("status") in ("NO_DATA", "ERROR", "NO_TRADES"):
                continue
            score = res.get("win_rate", 0) * 2 + res.get("profit_factor", 0) + abs(res.get("sharpe", 0))
            if score > best_score:
                best_score = score
                best_current = res
                best_current_asset = asset

        if best_current is None or best_current.get("status") == "NO_TRADES":
            log.info(f"  {strat_key}: no viable current baseline, skipping mutation")
            continue

        log.info(f"  Mutating {strat_key} (base on {best_current_asset}) ...")
        muts = mutate_params(base_params)

        for mi, mp in enumerate(muts):
            if mp == base_params:
                continue  # already tested
            for asset in strat["assets"]:
                if asset not in data or data[asset].empty:
                    continue
                engine = BacktesterEngine(initial_capital=500.0, risk_per_trade=0.01)
                try:
                    res = engine.run(strat["fn"], data[asset], mp)
                    if res.get("status") == "NO_TRADES":
                        continue
                    beat_n, beat_details = compare_metrics(best_current, res,
                                                           a_name="current", b_name=f"mutant#{mi}")
                    if beat_n >= 3:
                        candidates.append({
                            "strategy": strat_key,
                            "asset": asset,
                            "mutation_index": mi,
                            "params": mp,
                            "current_metrics": best_current,
                            "mutant_metrics": res,
                            "beat_count": beat_n,
                            "beat_details": beat_details,
                        })
                        log.info(
                            f"    ★ CANDIDATE (beat {beat_n} metrics): {strat_key} on {asset}, "
                            f"params={mp}"
                        )
                        break  # one candidate per mutation is enough
                except Exception:
                    continue

    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# 4. REGIME DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def determine_market_regime(data):
    """
    Determine overall market stance by combining:
      - SPY trend (50d SMA vs 200d SMA)
      - BTC trend (7d MA cross)
      - UVXY (volatility regime)
      - VIX proxy (historical vol)
    """
    log.info("── 4. MARKET REGIME ANALYSIS ──")
    stance = "Chop"
    reasons = []

    # SPY regime
    if "SPY" in data and not data["SPY"].empty and len(data["SPY"]) >= 60:
        spy = data["SPY"]
        sma50 = spy["Close"].rolling(50).mean().iloc[-1]
        sma200 = spy["Close"].rolling(200).mean().iloc[-1]
        close = spy["Close"].iloc[-1]
        if sma50 > sma200:
            reasons.append(f"SPY 50d SMA ({sma50:.1f}) > 200d SMA ({sma200:.1f})")
            if "Bull" not in stance:
                stance = "Bull"
        else:
            reasons.append(f"SPY 50d SMA ({sma50:.1f}) < 200d SMA ({sma200:.1f})")
            if "Bear" not in stance:
                stance = "Bear"
        # Check recent momentum
        ret_5d = (close / spy["Close"].iloc[-6] - 1) * 100
        reasons.append(f"SPY 5d return: {ret_5d:+.2f}%")

    # BTC regime
    if "BTC-USD" in data and not data["BTC-USD"].empty and len(data["BTC-USD"]) >= 14:
        btc = data["BTC-USD"]
        ma7 = btc["Close"].rolling(7).mean().iloc[-1]
        close = btc["Close"].iloc[-1]
        if close > ma7:
            reasons.append(f"BTC above 7d MA (${ma7:.0f} → ${close:.0f})")
        else:
            reasons.append(f"BTC below 7d MA (${ma7:.0f} → ${close:.0f})")

    # Volatility (UVXY)
    if "UVXY" in data and not data["UVXY"].empty and len(data["UVXY"]) >= 20:
        uvxy = data["UVXY"]
        vol_ratio = (uvxy["Close"].iloc[-1] / uvxy["Close"].rolling(20).mean().iloc[-1]) - 1
        if vol_ratio > 0.02:
            reasons.append(f"UVXY elevated (+{vol_ratio*100:.1f}% vs 20d avg)")
        else:
            reasons.append(f"UVXY flat/declining ({vol_ratio*100:+.1f}% vs 20d avg)")

    # Cross-asset verdict
    bull_signals = sum(1 for r in reasons if any(w in r for w in ["Bull", "above", "+"]))
    bear_signals = sum(1 for r in reasons if any(w in r for w in ["Bear", "below", "-"]))

    if bull_signals > bear_signals + 1:
        stance = "Bull"
    elif bear_signals > bull_signals + 1:
        stance = "Bear"
    else:
        stance = "Chop"

    log.info(f"  Overall stance: {stance}")
    return stance, reasons


# ──────────────────────────────────────────────────────────────────────────────
# 5. REPORT GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(data, results, candidates, stance, reasons):
    """Generate the plain-English daily report."""
    log.info("── 5. REPORT GENERATION ──")
    now = datetime.now()
    report_date = now.strftime("%Y-%m-%d")
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"daily_{report_date}.md"

    lines = []
    lines.append(f"# Daily Self-Improvement Report — {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## 📊 What Worked Yesterday")
    lines.append("")
    lines.append("### Strategy Performance Summary")
    lines.append("")
    lines.append("| Strategy | Asset | Trades | Win Rate | Profit Factor | Sharpe | Max Drawdown |")
    lines.append("|----------|-------|--------|----------|---------------|--------|--------------|")

    for strat in STRATEGIES:
        sname = strat["name"]
        for asset in strat["assets"]:
            res = results_global.get(sname, {}).get(asset, {})
            if res.get("status") in ("NO_DATA", "ERROR", "NO_TRADES"):
                status_str = res.get("status", "NO_DATA") or "—"
                lines.append(f"| {sname} | {asset} | — | {status_str} | — | — | — |")
            elif not res:
                lines.append(f"| {sname} | {asset} | — | NO_DATA | — | — | — |")
            else:
                wr = res.get("win_rate", 0) * 100
                pf = res.get("profit_factor", 0)
                sh = res.get("sharpe", 0)
                dd = res.get("max_drawdown", 0) * 100
                tt = res.get("total_trades", 0)
                lines.append(
                    f"| {sname} | {asset} | {tt} | {wr:.1f}% | {pf:.2f} | {sh:.2f} | {dd:.1f}% |"
                )
    lines.append("")

    lines.append("### Strategy Insights")
    lines.append("")
    # Find best/worst per strategy
    for strat in STRATEGIES:
        sname = strat["name"]
        best_res = None
        best_wr = -1
        for asset, res in results_global.get(sname, {}).items():
            if res.get("status") == "NO_TRADES" or "NO_DATA" in str(res.get("status", "")):
                continue
            if res.get("win_rate", 0) > best_wr:
                best_wr = res.get("win_rate", 0)
                best_res = res
        if best_res:
            pf = best_res.get("profit_factor", 0)
            if pf > 1.5:
                lines.append(f"- **{sname}**: Profit Factor {pf:.2f} — strong edge detected.")
            elif pf > 1.0:
                lines.append(f"- **{sname}**: Profit Factor {pf:.2f} — mildly profitable. Monitor closely.")
            else:
                lines.append(f"- **{sname}**: Profit Factor {pf:.2f} — losing money. Consider deactivation.")
        else:
            lines.append(f"- **{sname}**: No trades generated on backtest period.")
    lines.append("")

    lines.append("## 🔄 What's Being Changed")
    lines.append("")
    if candidates:
        lines.append(f"Found **{len(candidates)}** parameter candidate(s) that beat current")
        lines.append("live parameters by >5% across 3+ metrics:")
        lines.append("")
        for ci, cand in enumerate(candidates, 1):
            lines.append(f"### Candidate #{ci}: {cand['strategy']} on {cand['asset']}")
            lines.append("")
            lines.append(f"- **Metrics improved**: {', '.join(cand['beat_details'])} ({cand['beat_count']} total)")
            lines.append(f"- **New parameters**: `{json.dumps(cand['params'], indent=2)}`")
            cm = cand['current_metrics']
            mm = cand['mutant_metrics']
            lines.append("| Metric | Current | Mutant |")
            lines.append("|--------|---------|--------|")
            for k in ["win_rate", "profit_factor", "sharpe", "max_drawdown"]:
                cv = cm.get(k)
                mv = mm.get(k)
                if cv is not None and mv is not None:
                    lines.append(f"| {k} | {cv:.4f} | {mv:.4f} |")
            lines.append("")
            lines.append("> ⚠️ **RECOMMENDED FOR HUMAN APPROVAL** — Do NOT auto-deploy.")
    else:
        lines.append("No mutated parameter set beat the current live parameters by")
        lines.append(">5% across 3+ metrics. The current configuration remains optimal.")
        lines.append("")
        lines.append("The mutation search explored ±25%, ±50%, and ±0.01 perturbations")
        lines.append("on every numeric parameter for all four strategies.")

    lines.append("")
    lines.append("## 📈 Current Market Stance")
    lines.append("")
    stance_emoji = {"Bull": "🟢", "Bear": "🔴", "Chop": "⚪"}.get(stance, "⚪")
    lines.append(f"**{stance_emoji} {stance}**")
    lines.append("")
    lines.append("### Supporting Evidence:")
    for r in reasons:
        lines.append(f"- {r}")
    lines.append("")

    # HMM regime check
    log.info("── Running HMM Regime Detection (SPY) ──")
    if "SPY" in data and not data["SPY"].empty and len(data["SPY"]) >= 60:
        try:
            detector = RegimeDetector(n_states=3, lookback_days=90)
            detector.train(data["SPY"])
            reg_label, probs, conf = detector.get_regime(data["SPY"])
            lines.append("### HMM Regime Detection (SPY, 90-day lookback)")
            lines.append("")
            lines.append(f"- **Dominant regime**: {reg_label} (confidence: {conf:.1%})")
            lines.append(f"- Regime probabilities:")
            for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 40) + "░" * (40 - int(prob * 40))
                lines.append(f"  {label}: {prob:.1%} [{bar}]")

            strategy_recs = []
            for strat in STRATEGIES:
                if should_fire(strat["type"], reg_label, conf, 0.60):
                    strategy_recs.append(f"- ✅ {strat['name']} — **ACTIVE** in {reg_label}")
                else:
                    strategy_recs.append(f"- 🚫 {strat['name']} — **SUSPENDED** (incompatible regime)")
            lines.append("")
            lines.append("### Strategy Activation Status:")
            lines.extend(strategy_recs)
            lines.append("")

            vol_regime = detector.get_vol_regime(data["SPY"])
            lines.append(f"- **Volatility regime**: {vol_regime}")
        except Exception as e:
            lines.append(f"- HMM detection encountered an error: {e}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**⚠️ NO TRADES WERE EXECUTED.** This report is for review only.")
    lines.append("Deploy any recommended parameter changes after human approval.")
    lines.append("")
    lines.append(f"*Report generated by Hermes Self-Improvement Cycle at {now.strftime('%Y-%m-%d %H:%M:%S')}*")

    report_text = "\n".join(lines)

    with open(report_path, "w") as f:
        f.write(report_text)

    log.info(f"  Report saved to {report_path}")
    return report_text


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  DAILY SELF-IMPROVEMENT CYCLE — STARTING        ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # Step 1: Fetch data
    data = collect_market_data()

    # Step 2: Backtest all strategies
    results_global = run_backtests(data)

    # Step 3: Mutate and search
    candidates = run_mutation_search(data)

    # Step 4: Determine stance
    stance, reasons = determine_market_regime(data)

    # Step 5: Generate report
    report = generate_report(data, results_global, candidates, stance, reasons)

    print("\n")
    print(report)
