#!/bin/bash
cd /home/zer0null/algo-trading-bot
python -m pip install -q ccxt httpx pandas numpy 2>&1 | tail -1
echo ""
echo "Running smoke tests..."
python tests/test_smoke.py
