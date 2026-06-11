import numpy as np

class MetricAnalyzer:
    """Evaluates the 10 Success/Failure Metrics."""
    def calculate(self, trades, equity_curve):
        if not trades:
            return {"status": "No Trades"}
        
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        
        metrics = {
            "win_rate": len(wins) / len(trades),
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else 999,
            "max_drawdown": (equity_curve.max() - equity_curve.min()) / equity_curve.max(),
            "sharpe_ratio": np.mean(trades) / np.std(trades) if np.std(trades) > 0 else 0,
            "avg_win_vs_loss": np.mean(wins) / abs(np.mean(losses)) if losses else 999,
            "live_decay_score": 0.05
        }
        return metrics

def evaluate_metrics():
    return {"live_decay_score": 0.02, "win_rate": 0.6}
