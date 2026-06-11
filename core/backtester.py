import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

from core.regime import RegimeDetector

class BacktesterEngine:
    """
    Vectorized Backtesting Engine.
    Simulates trading over a period, tracking P&L, Drawdown, and Metrics.
    """
    def __init__(self, initial_capital=500.0, risk_per_trade=0.01):
        self.capital = initial_capital
        self.risk = risk_per_trade
        self.trades = []
        self.equity_curve = []
        self.starting_capital = initial_capital

    def _calculate_position_size(self, price: float, stop_loss: float, capital: float):
        """Risk Management: Calculates shares based on stop loss distance."""
        risk_amount = capital * self.risk
        if price == stop_loss: return 0
        shares = risk_amount / abs(price - stop_loss)
        return shares  # Allow fractional shares for small accounts

    def run(self, strategy_fn, data: pd.DataFrame, params: Dict) -> Dict:
        """
        Runs a strategy function against the data.
        strategy_fn(df, params) -> returns Signal (1=Buy, -1=Sell, 0=Hold)
        """
        capital = self.starting_capital
        position = None  # {'entry_price': float, 'shares': int}

        # Initialize metrics tracking
        self.trades = []
        self.equity_curve = []

        # Add stop loss/take profit to params if missing
        sl_pct = params.get('stop_loss', 0.05)
        tp_pct = params.get('take_profit', 0.08)


        for i in range(50, len(data)): # Start after warmup
            window_data = data.iloc[:i+1]

            # Get Signal from Strategy
            signal, metadata = strategy_fn(window_data, params)

            # Update position state
            if position:
                current_price = data.iloc[i]['Close']
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']

                # Exit Conditions
                if pnl_pct >= -sl_pct: # Stop Loss Hit
                    capital += (position['shares'] * current_price)
                    self.trades.append({
                        'entry': position['entry_price'],
                        'exit': current_price,
                        'pnl_pct': pnl_pct,
                        'days': i - position['start_day']
                    })
                    # print(f"   🔴 SELL {position['shares']:.4f} shares @ ${current_price:.2f} | PnL: {pnl_pct:.2%}")
                    position = None
                elif pnl_pct >= tp_pct: # Take Profit Hit
                    capital += (position['shares'] * current_price)
                    self.trades.append({
                        'entry': position['entry_price'],
                        'exit': current_price,
                        'pnl_pct': pnl_pct,
                        'days': i - position['start_day']
                    })
                    # print(f"   🟢 SELL {position['shares']:.4f} shares @ ${current_price:.2f} | PnL: {pnl_pct:.2%}")
                    position = None

            # Entry Condition
            if signal == 1 and not position:
                price = data.iloc[i]['Close']
                sl_pct = params.get('stop_loss', 0.05)
                tp_pct = params.get('take_profit', 0.08)
                shares = self._calculate_position_size(price, price * (1 - sl_pct), capital)
                if shares > 0:
                    capital -= (shares * price)
                    position = {
                        'entry_price': price,
                        'shares': shares,
                        'start_day': i
                    }
                    # print(f"   🟢 BUY {shares:.4f} shares @ ${price:.2f}")

            # Update Equity Curve (Realized + Unrealized)
            if position:
                current_capital = capital + (position['shares'] * data.iloc[i]['Close'])
            else:
                current_capital = capital

            self.equity_curve.append(current_capital)

        return self.calculate_metrics(capital)

    def run_with_regime_filter(
        self,
        strategy_fn,
        strategy_type: str,
        data: pd.DataFrame,
        params: Dict,
        regime_config: Dict = None,
    ) -> Dict:
        """
        Runs the strategy with HMM regime filtering.

        Slides a 60-day HMM training window backward. For each bar after warmup,
        computes regime probabilities and only allows the strategy to fire when:
          1. The dominant regime is compatible with the strategy type
          2. The dominant regime probability >= confidence_threshold

        Parameters
        ----------
        strategy_fn : callable — the original strategy function (unchanged).
        strategy_type : str — identifier used for regime gating
                         ("MOMENTUM", "MEAN_REV", "CARRY", "VOLATILITY").
        data : pd.DataFrame — full OHLCV history.
        params : dict — strategy parameters.
        regime_config : dict — regime detection settings (overrides defaults).

        Returns
        -------
        dict — same metrics as run(), plus:
            regime_breakdown: {regime_label: {"win_rate": float, "trades": int}}
        """
        if regime_config is None:
            regime_config = {}

        lookback = int(regime_config.get("lookback_days", 90))
        confidence_thresh = float(regime_config.get("confidence_threshold", 0.60))
        warmup_bars = int(regime_config.get("warmup_bars", 60))

        detector = RegimeDetector(
            n_states=regime_config.get("n_states", 3),
            lookback_days=lookback,
            confidence_threshold=confidence_thresh,
        )

        # Train the HMM on the full available window (we retrain periodically
        # in practice, but for a single backtest we train once on the
        # available data)
        detector.train(data)

        # Run the backtest bar by bar with regime gating
        capital = self.starting_capital
        position = None
        self.trades = []
        self.equity_curve = []

        # Track regime-level stats
        regime_trades: Dict[str, List[float]] = {}  # regime → list of pnl_pcts

        for i in range(warmup_bars, len(data)):
            window_data = data.iloc[:i + 1]

            # Get regime for this bar
            if i < len(data) - 1:
                # For backtesting: get regime at historical point i
                regime, probs, conf = detector.get_regime_for_backtest(
                    data, start_idx=i
                )
            else:
                regime, probs, conf = detector.get_regime(data)

            # Get signal from strategy
            signal, metadata = strategy_fn(window_data, params)

            # Apply regime filter
            strategy_label = metadata.get("type", strategy_type) if metadata else strategy_type
            from logic.strategies import should_fire
            if not should_fire(strategy_label, regime, conf, confidence_thresh):
                # Regime doesn't match → suppress the signal
                signal = 0

            # Update position state (same logic as run())
            if position:
                current_price = data.iloc[i]['Close']
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']

                sl_pct = params.get('stop_loss', 0.05)
                tp_pct = params.get('take_profit', 0.08)

                if pnl_pct >= -sl_pct:
                    capital += (position['shares'] * current_price)
                    trade_info = {
                        'entry': position['entry_price'],
                        'exit': current_price,
                        'pnl_pct': pnl_pct,
                        'days': i - position['start_day'],
                    }
                    # Record regime for breakdown
                    if regime not in regime_trades:
                        regime_trades[regime] = []
                    regime_trades[regime].append(pnl_pct)
                    self.trades.append(trade_info)
                    position = None
                elif pnl_pct >= tp_pct:
                    capital += (position['shares'] * current_price)
                    trade_info = {
                        'entry': position['entry_price'],
                        'exit': current_price,
                        'pnl_pct': pnl_pct,
                        'days': i - position['start_day'],
                    }
                    if regime not in regime_trades:
                        regime_trades[regime] = []
                    regime_trades[regime].append(pnl_pct)
                    self.trades.append(trade_info)
                    position = None

            # Entry Condition
            if signal == 1 and not position:
                price = data.iloc[i]['Close']
                sl_pct = params.get('stop_loss', 0.05)
                tp_pct = params.get('take_profit', 0.08)
                shares = self._calculate_position_size(price, price * (1 - sl_pct), capital)
                if shares > 0:
                    capital -= (shares * price)
                    position = {
                        'entry_price': price,
                        'shares': shares,
                        'start_day': i
                    }

            # Update Equity Curve
            if position:
                current_capital = capital + (position['shares'] * data.iloc[i]['Close'])
            else:
                current_capital = capital

            self.equity_curve.append(current_capital)

        metrics = self.calculate_metrics(capital)

        # Add regime breakdown
        regime_breakdown = {}
        for regime_label, pnls in regime_trades.items():
            if pnls:
                wins = [p for p in pnls if p > 0]
                regime_breakdown[regime_label] = {
                    "win_rate": len(wins) / len(pnls),
                    "trades": len(pnls),
                }
            else:
                regime_breakdown[regime_label] = {
                    "win_rate": 0.0,
                    "trades": 0,
                }
        metrics["regime_breakdown"] = regime_breakdown

        return metrics

    def calculate_metrics(self, final_capital: float) -> Dict:
        """Calculates the 10 Success/Failure Metrics."""
        if not self.trades:
            return {'status': 'NO_TRADES', 'final_capital': final_capital, 'total_return': 0.0}

        # 1. Win Rate
        wins = [t for t in self.trades if t['pnl_pct'] > 0]
        win_rate = len(wins) / len(self.trades)

        # 2. Profit Factor
        gross_win = sum(t['pnl_pct'] for t in wins)
        gross_loss = abs(sum(t['pnl_pct'] for t in self.trades if t['pnl_pct'] <= 0))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 999.0

        # 3. Max Drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = np.min(drawdown)

        # 4. Sharpe Ratio (Approximate based on daily returns)
        daily_returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) > 0 else 0

        # 5. Avg Win vs Avg Loss
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in self.trades if t['pnl_pct'] < 0]) if [t for t in self.trades if t['pnl_pct'] < 0] else 0
        win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0

        # 6. Recovery Time (Simple: days to recover from peak)
        peak_cap = self.equity_curve[0]
        max_recovery_days = 0
        in_drawdown = False
        for i, eq in enumerate(self.equity_curve):
            if eq < peak_cap * 0.95:
                in_drawdown = True
            if in_drawdown and eq > peak_cap:
                max_recovery_days = max(max_recovery_days, i - self.equity_curve.index(peak_cap * 0.95))
                in_drawdown = False

        final_pnl_pct = (final_capital - self.starting_capital) / self.starting_capital

        return {
            'final_capital': final_capital,
            'total_return': final_pnl_pct,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'win_loss_ratio': win_loss_ratio,
            'total_trades': len(self.trades)
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helper: walk-forward backtest (used by orchestrator)
# ──────────────────────────────────────────────────────────────────────────────

def run_walk_forward():
    """
    Placeholder for walk-forward parameter optimization.
    Returns a dict of adjusted strategy parameters.
    In production this would run multiple backtests and pick the best params.
    """
    # This is called by orchestrator.weekly_improvement()
    # Returns a minimal dict for now; the real implementation would
    # optimize strategy params + regime config jointly.
    return {
        "mean_reversion": {
            "rsi_period": 7,
            "bb_std": 1.5,
            "stop_loss": 0.03,
            "take_profit": 0.05,
        }
    }
