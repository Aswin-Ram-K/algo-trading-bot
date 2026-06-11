import logging

class RiskSentinel:
    """The 'Kill Switch' logic."""
    def check(self, current_metrics, config):
        if current_metrics.get("max_drawdown", 0) > config["metrics"]["max_drawdown"]["threshold"]:
            logging.warning("💀 SENTINEL: Max Drawdown exceeded. Kill switch engaged!")
            return False
        return True

def check_kill_switch():
    return True
