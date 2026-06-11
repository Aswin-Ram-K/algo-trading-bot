"""
Regime Detection Module — HMM/Markov regime filter for the algo-trading-bot.

HMMs act as regime FILTERS, not alpha generators. They gate existing strategies,
allowing each to fire only when the current regime is compatible with its design.

Based on research:
  - Catello et al. (2023) arXiv:2310.03775v2 — DPA ~50% = coin flip, HMMs are filters
  - QuantInsti — regime-adaptive trading: HMMs adapt existing strategies
  - Multi-scale Markov-Switching GARCH (arXiv:2606.06190v1) — multi-scale detection
  - Hidden Regime package — production-ready HMM pipeline architecture
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Regime labels
# ──────────────────────────────────────────────────────────────────────────────

REGIME_LABELS = {
    0: "Bull/Trending",
    1: "Bear/Declining",
    2: "Sideways/Chop",
}

# Map a regime label → what strategy types are allowed to fire.
# This is the gating table: regime → set of compatible strategy names.
REGIME_COMPATIBLE_STRATEGIES = {
    "Bull/Trending":    {"momentum", "carry"},
    "Bear/Declining":   {"mean_reversion"},
    "Sideways/Chop":    {"mean_reversion"},
}

# Volatility labels (derived from realized vol features)
VOL_LABELS = {
    "high": "HighVol",
    "low": "LowVol",
}


def _compute_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build the observation matrix from OHLCV data.

    Features:
      0. Log returns (period-by-period)
      1. Realized volatility (20-period rolling std of log returns)
      2. Volume ratio (current volume / 20-period rolling mean volume)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'Close', 'Volume' (OHLCV data).

    Returns
    -------
    np.ndarray of shape (n_bars, n_features)
    """
    close = df["Close"].astype(float).values
    volume = df["Volume"].astype(float).values

    # Log returns
    log_returns = np.log(close[1:] / close[:-1])

    # 20-period rolling realized volatility
    returns_series = pd.Series(log_returns)
    realized_vol = returns_series.rolling(window=20, min_periods=5).std().values

    # Volume ratio: current / 20-period rolling mean
    volume_series = pd.Series(volume)
    vol_mean = volume_series.rolling(window=20, min_periods=5).mean()
    # Align lengths: vol_mean has NaN at start, volume ratio matches returns length
    vol_ratio = np.where(
        (vol_mean.iloc[1:].values > 0) & np.isfinite(vol_mean.iloc[1:].values),
        volume[1:] / vol_mean.iloc[1:].values,
        1.0,
    )

    # Stack features — pad first element so lengths match
    pad_len = len(close) - 1
    log_ret_pad = np.concatenate([[0.0], log_returns])
    vol_pad = np.concatenate([[0.0], realized_vol])
    vr_pad = np.concatenate([[1.0], vol_ratio])

    features = np.column_stack([log_ret_pad, vol_pad, vr_pad])

    # Replace NaN/inf with 0
    features = np.where(np.isfinite(features), features, 0.0)

    return features


class RegimeDetector:
    """
    Hidden Markov Model regime detector for market data.

    Uses a Gaussian HMM with 3 states (Bull, Bear, Sideways) trained via
    Baum-Welch (EM) on a rolling lookback window. Viterbi decoding produces
    regime probabilities for the latest observation.

    Parameters
    ----------
    n_states : int
        Number of hidden regime states (default 3).
    lookback_days : int
        Number of recent days to use for training (default 90).
    confidence_threshold : float
        Minimum dominant regime probability to consider a regime "active"
        (default 0.60).
    features : list of str
        Feature names to include: "log_returns", "realized_vol", "volume_ratio".
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback_days: int = 90,
        confidence_threshold: float = 0.60,
        features: Optional[List[str]] = None,
    ):
        self.n_states = max(n_states, 3)  # Minimum 3 for useful detection
        self.lookback_days = lookback_days
        self.confidence_threshold = confidence_threshold
        self.features = features or ["log_returns", "realized_vol", "volume_ratio"]

        # The HMM model (fitted later)
        self.model: Optional[hmm.GaussianHMM] = None
        self._trained = False

        # Feature standardization parameters (set during training)
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None

        # Historical regime assignments for backtracking
        self._regime_history: List[Tuple[int, float]] = []  # (state, max_prob)

    # ─── Training ──────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> bool:
        """
        Fit the HMM on the provided OHLCV dataframe.

        Uses a rolling window of the most recent `lookback_days` bars (or all
        available data if fewer bars exist).

        Parameters
        ----------
        df : pd.DataFrame — OHLCV data with at least 'Close' and 'Volume'.

        Returns
        -------
        bool — True if training succeeded.
        """
        # Trim to lookback window
        if len(df) > self.lookback_days:
            train_data = df.tail(self.lookback_days)
        else:
            train_data = df

        # Minimum data check: need at least 3×n_states data points for Baum-Welch
        min_samples = 3 * self.n_states
        if len(train_data) < min_samples:
            return False

        # Build features
        X = _compute_features(train_data)

        # Drop any rows with zero variance features (avoid numerical issues)
        nonzero_cols = np.any(np.abs(X) > 1e-10, axis=0)
        X = X[:, nonzero_cols]

        # Validate we still have data
        if X.shape[0] < 10:
            return False

        # Build Gaussian HMM
        best_model = None
        best_score = float('inf')

        # Multiple random restarts (hmmlearn can get stuck in local optima)
        for retry in range(3):
            try:
                model_candidate = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="diag",       # diagonal covariance (simpler, more stable)
                    n_iter=150,                    # EM iterations
                    min_covar=1e-4,                # floor on covariance to avoid singularity
                    means_prior=0,
                    means_weight=0,
                    covars_prior=1e-2,
                    covars_weight=1,
                    tol=1e-2,
                    verbose=False,
                    startprob_prior=1.0,           # uniform prior on start probs
                    transmat_prior=1.0,            # uniform prior on transitions
                )

                # Standardize features to help convergence
                mean = X.mean(axis=0)
                std = X.std(axis=0)
                std[std < 1e-8] = 1.0  # avoid division by zero
                X_scaled = (X - mean) / std

                model_candidate.fit(X_scaled)
                score = model_candidate.score(X_scaled)
                if score < best_score:
                    best_score = score
                    best_model = model_candidate

                # If convergence score is reasonable, stop early
                if abs(score) < 1e3:
                    break

            except Exception:
                continue

        if best_model is None:
            return False

        self.model = best_model

        # Standardize the same way for inference
        self._feature_mean = mean
        self._feature_std = std

        try:
            X_scaled = (X - mean) / std
            _, states = self.model.predict(X_scaled)
            _, probs = self.model.predict_proba(X_scaled)
            self._regime_history = [
                (int(s), float(p[s])) for s, p in zip(states, probs)
            ]
        except Exception:
            pass

        self._trained = True
        return True

    # ─── Inference ─────────────────────────────────────────────────────────

    def get_regime(self, df: pd.DataFrame) -> Tuple[str, Dict[str, float], float]:
        """
        Infer the current market regime from the latest data.

        Trains the HMM if needed, then returns the dominant regime and its
        probability distribution.

        Parameters
        ----------
        df : pd.DataFrame — OHLCV data (latest window).

        Returns
        -------
        dominant_regime : str — Human-readable regime label.
        probabilities : dict — regime_name → probability.
        confidence : float — Probability of the dominant regime.
        """
        if not self._trained or self.model is None:
            self.train(df)

        if not self._trained:
            # Fallback: no data to train on — return unknown
            return "Unknown", {"Unknown": 1.0}, 1.0

        # Use the latest bar for inference
        X_latest = _compute_features(df)
        if X_latest.shape[0] == 0:
            return "Unknown", {"Unknown": 1.0}, 1.0

        # Standardize using the same parameters from training
        if hasattr(self, '_feature_std') and self._feature_std is not None:
            X_infer = (X_latest[-1:] - self._feature_mean) / self._feature_std
        else:
            X_infer = X_latest[-1:].reshape(1, -1)

        # Viterbi decoding — most likely state sequence
        states = self.model.predict(X_infer)
        # Posterior state probabilities
        posteriors = self.model.predict_proba(X_infer)

        state_idx = int(states[0])
        posteriors_flat = posteriors[0]

        # Map state index → regime label
        # Since we don't have a guaranteed ordering, we infer from model parameters.
        # States with higher mean log-returns → Bull, lower → Bear, near-zero → Sideways.
        labeled_probs = self._label_regime_states(posteriors_flat, state_idx)

        dominant_label = max(labeled_probs, key=labeled_probs.get)
        confidence = labeled_probs[dominant_label]

        return dominant_label, labeled_probs, confidence

    def get_regime_for_backtest(
        self, df: pd.DataFrame, start_idx: int
    ) -> Tuple[str, Dict[str, float], float]:
        """
        During backtesting: decode regime at a specific historical index.

        The HMM is trained once on the full lookback window, then we decode
        regimes for the entire history at once (efficient for backtesting).

        Parameters
        ----------
        df : pd.DataFrame — full historical OHLCV data.
        start_idx : int — index to query regime for.

        Returns
        -------
        (dominant_regime, probabilities, confidence)
        """
        if not self._trained or self.model is None:
            self.train(df)

        if not self._trained:
            return "Unknown", {"Unknown": 1.0}, 1.0

        X = _compute_features(df)

        # Standardize using the same parameters from training
        if hasattr(self, '_feature_std') and self._feature_std is not None:
            X_scaled = (X - self._feature_mean) / self._feature_std
        else:
            X_scaled = X

        # Decode entire sequence once
        states_all = self.model.predict(X_scaled)
        probs_all = self.model.predict_proba(X_scaled)

        idx = min(start_idx, len(states_all) - 1)
        state_idx = int(states_all[idx])
        posteriors_flat = probs_all[idx]

        labeled_probs = self._label_regime_states(posteriors_flat, state_idx)

        dominant_label = max(labeled_probs, key=labeled_probs.get)
        confidence = labeled_probs[dominant_label]

        return dominant_label, labeled_probs, confidence

    # ─── Internal helpers ──────────────────────────────────────────────────

    def _label_regime_states(self, posteriors: np.ndarray, dominant_idx: int) -> Dict[str, float]:
        """
        Assign semantic labels to HMM states based on emission parameters.

        Uses the mean emission for the log-return feature (first dimension)
        to determine which state is Bull, Bear, or Sideways.
        """
        if self.model is None:
            return {label: 1.0 / self.n_states for label in REGIME_LABELS.values()}

        n = self.model.n_components

        # Get emission means (mean of each feature per state)
        try:
            means = self.model.means_  # shape: (n_states, n_features)
        except AttributeError:
            # Fallback: assign randomly
            return {label: 1.0 / self.n_states for label in REGIME_LABELS.values()}

        # Sort states by mean log-return (feature 0) — higher = more bullish
        # Only use log_returns if that feature was present
        if self.model.means_.shape[1] >= 1:
            sorted_order = np.argsort(means[:, 0])[::-1]  # descending
        else:
            sorted_order = np.arange(n)

        # Map: highest mean → Bull, lowest → Bear, middle → Sideways
        labeled_probs = {}
        for rank, state_idx in enumerate(sorted_order):
            if rank == 0 and n >= 1:
                label = "Bull/Trending"
            elif rank == n - 1 and n >= 2:
                label = "Bear/Declining"
            else:
                label = "Sideways/Chop"

            labeled_probs[label] = float(posteriors[state_idx])

        # Ensure all labels are present (pad with zeros if fewer states than labels)
        for label in ["Bull/Trending", "Bear/Declining", "Sideways/Chop"]:
            if label not in labeled_probs:
                labeled_probs[label] = 0.0

        return labeled_probs

    def get_vol_regime(self, df: pd.DataFrame) -> str:
        """
        Separate volatility regime classification (HighVol / LowVol).
        Uses the realized_vol feature from the latest bar.

        Returns 'HighVol' or 'LowVol'.
        """
        if len(df) < 21:
            return "LowVol"  # Not enough data, assume calm

        close = df["Close"].astype(float).values
        returns = np.log(close[1:] / close[:-1])
        recent_vol = pd.Series(returns).rolling(20, min_periods=5).std()

        if len(recent_vol) == 0:
            return "LowVol"

        current_vol = recent_vol.iloc[-1]
        avg_vol = recent_vol.mean()

        if np.isnan(current_vol) or np.isnan(avg_vol) or avg_vol == 0:
            return "LowVol"

        return "HighVol" if current_vol > avg_vol * 1.2 else "LowVol"

    def retrain_with_window(self, df: pd.DataFrame, warmup_bars: int = 60) -> bool:
        """
        Convenience: retrain on the last lookback_days bars, but only if
        the dataframe is long enough to provide the warmup period.

        Returns True if training succeeded.
        """
        if len(df) < warmup_bars:
            return False
        return self.train(df)
