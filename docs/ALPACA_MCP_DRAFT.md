---
draft_version: 1.0
date: 2026-06-10
status: "READY_FOR_DEPLOY"
---

# Draft: Alpaca SDK MCP Bridge
## Purpose
To allow the Hermes Operator to interact directly with Alpaca Markets via MCP instead of using the Python SDK as a black box.

## Implementation
1. **MCP Server:** `alpaca-mcp-server` (Local).
2. **Protocol:** HTTP/JSON-RPC over stdio.
3. **Tools:**
   - `get_balance`: Fetches total equity.
   - `get_positions`: Lists all active positions.
   - `submit_order`: Sends market/limit orders.
   - `cancel_orders`: Removes pending limit orders.
   - `get_historical_data`: Pulls 1d OHLCV for backtesting.

## Integration with Hermes
- In `orchestrator.py`, we will add a `mcp_client.py` to handle the local socket connection.
- Allows the AI to "see" the order book status as a native tool in the prompt.
