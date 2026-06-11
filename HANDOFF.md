# Session Handoff — Algo-Trading-Bot
**Status:** PAUSED by user
**Restart Trigger:** User returns and restarts gateway

## Context
We are building a self-improving trading bot. The current objective is to integrate a **State-Based/Markov Chain** regime detection layer into the trading logic.

## 1. Completed Research (The "State of Research")
We have finished a comprehensive literature review and analysis of Hidden Markov Models (HMMs) and Markov Regime-Switching models for trading.

**Key Findings (Executive Summary):**
1.  **HMMs are NOT price predictors:** Direct price prediction using HMMs yields ~50% accuracy (coin flip). Do NOT use them to predict the next price.
2.  **HMMs ARE regime filters:** The value is detecting *what regime the market is in* (Bull, Bear, Sideways, HighVol) and activating specific strategies accordingly.
3.  **Multi-Scale is required:** Single-scale detection is noisy. The architecture should run HMMs at 3 timeframes (1H Micro, 4H Meso, 1D Macro) and use cross-scale voting.
4.  **TVTP is superior:** Time-Varying Transition Probabilities (where transition risks adapt to data) outperform fixed transition models.
5.  **Ensemble approaches work:** Combining tree models (Random Forest) with HMMs reduces false regime transitions.

**The Report:**
Full detailed research paper is saved at: `~/.hermes/research/hmm-markov-trading-strategies.md`
*Contains: 10 academic references, code architecture plans, parameter thresholds, and implementation timeline.*

## 2. The Plan (Execution Roadmap)
Based on the research, here is the plan we have agreed upon (Phase 1 is complete):

### Phase 1: Foundation (RESEARCH — COMPLETED)
- [x] Research state-of-the-art HMM/Markov literature (2023-2026).
- [x] Evaluate practical tools (e.g., `hidden-regime` Python package).
- [x] Design multi-scale regime detection architecture.

### Phase 2: Architecture Integration (NEXT STEP)
- [ ] Set up the project environment (`~/algo-trading-bot`).
- [ ] Implement the `hidden-regime` pipeline wrapper or custom HMM class.
- [ ] Configure the 3-scale HMM (1H, 4H, 1D) with 3-5 states.
- [ ] Implement "Cross-Scale Voting" to filter noisy signals.

### Phase 3: Strategy Integration (Future)
- [ ] Map existing strategies to compatible regimes:
    - Momentum → Only active in Trending regimes.
    - Mean Reversion → Only active in Sideways/HighVol regimes.
    - Carry/Stat Arb → Only active in LowVol regimes.
- [ ] Implement regime-probability-weighted signal generation.

### Phase 4: Backtest & Optimize (Future)
- [ ] Run walk-forward backtests comparing:
    - Baseline Strategy vs.
    - Strategy + Single-Scale HMM vs.
    - Strategy + Multi-Scale HMM.
- [ ] Stress test against 2020 Crash, 2022 Bear, 2024 Chop.

## 3. Critical Parameters (From Research)
- **Regime Probability Threshold (theta):** 0.60 - 0.75 (only trade when confident).
- **Deviation Threshold (delta):** 1.0 - 2.0 sigma (entry signals).
- **Retrain Frequency:** Every 30 days.
- **Lookback Window:** 60-120 days (HMM needs this to converge).
- **Number of States:** Start with 3 (Bull, Bear, Sideways), expand to 5 if needed.

## 4. Files & Locations
- **Research Report:** `~/.hermes/research/hmm-markov-trading-strategies.md`
- **Project Directory:** `~/algo-trading-bot/` (Create if needed)
- **Libraries Needed:** `hidden-regime`, `hmmlearn`, `statsmodels`, `yfinance`, `pandas`, `numpy`.

## 5. How to Resume
When you restart, simply say "resume". I will load the research report and begin **Phase 2: Architecture Integration**.
