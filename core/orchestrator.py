import yaml
import time
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)

class Crest:
    """The Master Orchestrator Agent."""
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.loop_active = True
        self.regime_detector = None
        self._regime_config_path = config_path

    def start(self):
        logging.info("🦄 CREST: Orchestrator online. Initializing 2026 Trading Loop.")
        while self.loop_active:
            try:
                self.daily_maintenance()
                self.weekly_improvement()
                time.sleep(86400)  # 24-hour simulation loop
            except Exception as e:
                logging.error(f"CREST: Critical failure. {e}")
                time.sleep(3600)

    def daily_maintenance(self):
        logging.info("📅 Daily Maintenance: Data sync, health checks, trade logging.")
        from data.fetcher import update_all_markets
        from data.storage import log_daily_report
        update_all_markets()
        log_daily_report()

    def weekly_improvement(self):
        logging.info("🧬 Weekly Improvement: Running self-improvement loop.")
        from core.backtester import run_walk_forward
        from logic.metrics import evaluate_metrics
        from execution.broker import adjust_live_params
        from execution.risk import check_kill_switch

        if check_kill_switch():
            logging.warning("🚨 CREST: Kill switch triggered. Pausing execution.")
            return

        new_params = run_walk_forward()

        # ── HMM regime retraining ──────────────────────────────────────────
        self._retrain_regime_model()

        if new_params:
            live_metrics = evaluate_metrics()
            if live_metrics.get("live_decay_score", 1.0) < 0.1:
                adjust_live_params(new_params)
                logging.info(f"🚀 CREST: New parameters deployed. Decay is within tolerance.")

    def _retrain_regime_model(self):
        """
        Retrain the HMM regime detector with the latest data.
        Called every week as part of the self-improvement loop.
        """
        if not self.config.get("regime", {}).get("enabled", False):
            logging.info("ℹ️  CREST: HMM regime detection is disabled. Skipping retraining.")
            return

        from core.regime import RegimeDetector

        regime_cfg = self.config["regime"]
        retrain_freq = regime_cfg.get("retrain_frequency_days", 30)
        lookback = regime_cfg.get("lookback_days", 90)

        try:
            from data.fetcher import DataFetcher
            data_fetcher = DataFetcher(self.config)

            # Fetch latest data for regime detection (use SPY as benchmark)
            df = data_fetcher.get_historical_mass("SPY", period="2y")

            if df is not None and not df.empty and len(df) >= 60:
                if self.regime_detector is None:
                    self.regime_detector = RegimeDetector(
                        n_states=regime_cfg.get("n_states", 3),
                        lookback_days=lookback,
                        confidence_threshold=regime_cfg.get("confidence_threshold", 0.60),
                    )

                trained = self.regime_detector.retrain_with_window(df, warmup_bars=60)
                if trained:
                    logging.info(f"✅ CREST: HMM retrained on {len(df)} bars "
                                 f"(lookback={lookback} days).")
                else:
                    logging.warning("⚠️  CREST: HMM retrain had insufficient data.")
            else:
                logging.warning("⚠️  CREST: No data available for HMM retraining.")

        except Exception as e:
            logging.error(f"❌ CREST: HMM retraining failed: {e}")

    def update_regime_config(self, new_params: dict):
        """
        Adjust regime detection thresholds based on backtest results.

        Parameters
        ----------
        new_params : dict
            Keys can include:
              - confidence_threshold : float — adjust regime confidence threshold
              - n_states : int — adjust number of HMM states
              - lookback_days : int — adjust training window length
              - retrain_frequency_days : int — adjust retrain interval

        Usage::

            # After a backtest, if regime_breakdown shows a regime with
            # very low win rate, adjust the threshold:
            orchestrator.update_regime_config({
                "confidence_threshold": 0.65,
                "lookback_days": 120,
            })
        """
        regime_cfg = self.config.setdefault("regime", {})

        if "confidence_threshold" in new_params:
            regime_cfg["confidence_threshold"] = float(new_params["confidence_threshold"])
            logging.info(
                f"📊 CREST: Updated confidence_threshold to "
                f"{regime_cfg['confidence_threshold']}"
            )
        if "n_states" in new_params:
            regime_cfg["n_states"] = int(new_params["n_states"])
            logging.info(
                f"📊 CREST: Updated n_states to {regime_cfg['n_states']}"
            )
        if "lookback_days" in new_params:
            regime_cfg["lookback_days"] = int(new_params["lookback_days"])
            logging.info(
                f"📊 CREST: Updated lookback_days to {regime_cfg['lookback_days']}"
            )
        if "retrain_frequency_days" in new_params:
            regime_cfg["retrain_frequency_days"] = int(new_params["retrain_frequency_days"])
            logging.info(
                f"📊 CREST: Updated retrain_frequency_days to "
                f"{regime_cfg['retrain_frequency_days']}"
            )

        # If a regime_detector is already loaded, create a fresh one with
        # the updated config
        if self.regime_detector is not None:
            self.regime_detector = RegimeDetector(
                n_states=regime_cfg.get("n_states", 3),
                lookback_days=regime_cfg.get("lookback_days", 90),
                confidence_threshold=regime_cfg.get("confidence_threshold", 0.60),
            )
            logging.info("🔄 CREST: Reinitialized RegimeDetector with new config.")

        logging.info("✅ CREST: Regime config update applied.")
