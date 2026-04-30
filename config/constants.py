"""
DCX-AgenticTrader — Constants & Static Configuration

All static values that don't change between environments live here.
Environment-specific values go in settings.py (loaded from .env).
"""

# =============================================================================
# CoinDCX API URLs
# =============================================================================

COINDCX_BASE_URL = "https://api.coindcx.com"
COINDCX_PUBLIC_URL = "https://public.coindcx.com"
COINDCX_SOCKET_URL = "https://stream.coindcx.com"

# =============================================================================
# CoinDCX API Endpoints — Public (no auth required)
# =============================================================================

ENDPOINTS_PUBLIC = {
    "ticker": "/exchange/ticker",
    "markets": "/exchange/v1/markets",
    "markets_details": "/exchange/v1/markets_details",
    "candles": "/market_data/candles",          # on public URL
    "orderbook": "/market_data/orderbook",      # on public URL
    "trades": "/market_data/trade_history",     # on public URL
}

# =============================================================================
# CoinDCX API Endpoints — Authenticated (HMAC signed)
# =============================================================================

ENDPOINTS_AUTH = {
    # Orders
    "create_order": "/exchange/v1/orders/create",
    "create_multiple_orders": "/exchange/v1/orders/create_multiple",
    "order_status": "/exchange/v1/orders/status",
    "active_orders": "/exchange/v1/orders/active_orders",
    "trade_history": "/exchange/v1/orders/trade_history",
    "cancel_order": "/exchange/v1/orders/cancel",
    "cancel_all": "/exchange/v1/orders/cancel_all",
    "cancel_multiple": "/exchange/v1/orders/cancel_by_ids",
    # User
    "balances": "/exchange/v1/users/balances",
    "user_info": "/exchange/v1/users/info",
}

# =============================================================================
# Supported Trading Pairs
# =============================================================================

# CoinDCX pair format: "B-BTC_USDT" (ecode-TARGET_BASE)
# Market format for orders: "BTCINR"
SUPPORTED_PAIRS = {
    "BTCINR": {
        "pair": "I-BTC_INR",
        "market": "BTCINR",
        "target_currency": "BTC",
        "base_currency": "INR",
        "min_quantity": 0.00001,
        "price_precision": 2,
        "quantity_precision": 5,
    },
    "USDTINR": {
        "pair": "I-USDT_INR",
        "market": "USDTINR",
        "target_currency": "USDT",
        "base_currency": "INR",
        "min_quantity": 0.01,
        "price_precision": 2,
        "quantity_precision": 2,
    },
    "ETHINR": {
        "pair": "I-ETH_INR",
        "market": "ETHINR",
        "target_currency": "ETH",
        "base_currency": "INR",
        "min_quantity": 0.0001,
        "price_precision": 2,
        "quantity_precision": 4,
    },
}

# =============================================================================
# Candle Intervals
# =============================================================================

CANDLE_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "1d", "3d", "1w", "1M"]
DEFAULT_CANDLE_INTERVAL = "15m"
DEFAULT_CANDLE_LIMIT = 100

# =============================================================================
# Indian Crypto Tax Constants
# =============================================================================

INDIA_TAX_FLAT_RATE = 0.30            # 30% flat tax on crypto profits
INDIA_TDS_RATE = 0.01                 # 1% TDS on transfers
INDIA_TDS_THRESHOLD_INR = 50_000      # TDS applies above ₹50,000
INDIA_GST_ON_FEES = 0.18              # 18% GST on exchange service fees

# =============================================================================
# CoinDCX Fee Structure
# =============================================================================

COINDCX_MAKER_FEE = 0.001             # 0.1% maker fee
COINDCX_TAKER_FEE = 0.001             # 0.1% taker fee

# =============================================================================
# Rate Limits
# =============================================================================

RATE_LIMIT_PUBLIC_PER_SEC = 10
RATE_LIMIT_PRIVATE_PER_SEC = 5

# =============================================================================
# Technical Analysis Defaults
# =============================================================================

TA_RSI_PERIOD = 14
TA_MACD_FAST = 12
TA_MACD_SLOW = 26
TA_MACD_SIGNAL = 9
TA_BOLLINGER_PERIOD = 20
TA_BOLLINGER_STD = 2.0
TA_EMA_PERIODS = [9, 21, 50, 200]
TA_ATR_PERIOD = 14

# =============================================================================
# Signal Thresholds
# =============================================================================

SIGNAL_STRONG_BUY_THRESHOLD = 0.7
SIGNAL_BUY_THRESHOLD = 0.3
SIGNAL_SELL_THRESHOLD = -0.3
SIGNAL_STRONG_SELL_THRESHOLD = -0.7

# =============================================================================
# Risk Management Defaults
# =============================================================================

DEFAULT_MAX_POSITION_SIZE_PCT = 10     # 10% of portfolio per trade
DEFAULT_MAX_DRAWDOWN_PCT = 8           # 8% max drawdown
DEFAULT_MAX_TRADES_PER_DAY = 10
DEFAULT_STOP_LOSS_PCT = 3              # 3% stop loss
DEFAULT_TAKE_PROFIT_PCT = 6            # 6% take profit
PAPER_TRADING_SLIPPAGE = 0.001         # 0.1% simulated slippage

# =============================================================================
# Memory & Storage
# =============================================================================

CHROMA_PERSIST_DIR = "data/chroma"
TRADE_DB_PATH = "data/trades.db"
LOG_DIR = "logs"

# =============================================================================
# Agent System Prompt Prefixes
# =============================================================================

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor of a multi-agent crypto trading desk operating on CoinDCX.
Your role is to orchestrate 6 specialized agents to make informed trading decisions.

Available agents:
1. market_data - Fetches live price, candle, and orderbook data from CoinDCX
2. technical - Runs technical analysis (RSI, MACD, Bollinger Bands, etc.)
3. sentiment - Analyzes India-focused crypto news and market sentiment
4. risk - Checks compliance with Indian regulations and enforces risk limits
5. orchestrator - Synthesizes all signals and makes the final trade decision
6. executor - Executes paper or live trades on CoinDCX

Follow this decision cycle:
1. First, call market_data to get current market state
2. Then call technical and sentiment in parallel for analysis
3. Call risk to validate compliance and position sizing
4. Call orchestrator to make the final decision
5. If orchestrator recommends a trade, route to executor (after human approval)

Always explain your routing decisions. Never skip the risk check."""

MARKET_DATA_PROMPT = """You are the Market Data Agent. Your job is to fetch and organize live market data from CoinDCX.
Use your tools to get:
- Current price and 24h stats for the target trading pairs
- Recent candlestick data (15m, 1h, 4h timeframes)
- Orderbook depth (top 20 bids and asks)
Present the data in a clean, structured format for other agents to analyze."""

TECHNICAL_ANALYST_PROMPT = """You are the Technical Analyst Agent. Your job is to analyze price action and generate trading signals.
Using the market data provided, compute and interpret:
- RSI (14): overbought >70, oversold <30
- MACD (12,26,9): crossover signals and histogram momentum
- Bollinger Bands (20,2): price relative to bands
- EMA crossovers (9/21 and 50/200)
- ATR (14): current volatility level
- Support and resistance levels

Provide a composite signal: STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL
Include your confidence level (0.0 - 1.0) and a brief reasoning."""

SENTIMENT_PROMPT = """You are the Sentiment Researcher Agent focusing on India's crypto market.
Analyze:
- Recent crypto news from Indian and global sources
- Crypto Fear & Greed Index
- Any regulatory news from India (RBI, SEBI, FIU-IND)
- Overall market momentum

Weight India-specific news 2x higher than global news.
Return a sentiment score from -1.0 (extreme bearish) to +1.0 (extreme bullish).
Include the top 3 most impactful news items with their individual sentiment scores."""

RISK_COMPLIANCE_PROMPT = """You are the Risk & Compliance Agent specializing in Indian crypto regulations.
Your responsibilities:
1. Calculate Value-at-Risk (VaR) for the proposed position
2. Determine maximum position size based on portfolio and risk limits
3. Check Indian tax implications:
   - 30% flat tax on any realized profits (Section 115BBH)
   - 1% TDS on transfers exceeding ₹50,000 (Section 194S)
   - No loss offsetting allowed between different VDAs
4. Verify FIU-IND/PMLA/AML compliance
5. Block any trade that violates risk limits or regulations

Return: max_position_size, compliance_status (PASS/FAIL), tax_estimate, risk_warnings"""

ORCHESTRATOR_PROMPT = """You are the Strategy Orchestrator. You synthesize all agent signals into a final trading decision.
You have access to:
- Technical signals (RSI, MACD, etc.)
- Sentiment score
- Risk assessment and compliance status
- Current portfolio state

Decision rules:
1. NEVER override a FAIL compliance status — block the trade
2. Require technical + sentiment agreement for high-confidence trades
3. Size positions according to the risk agent's max_position_size
4. Include detailed reasoning for every decision
5. Set confidence threshold: only trade if confidence >= 0.6

Output format:
{action: BUY/SELL/HOLD, pair, quantity, price_type, confidence, reasoning}"""

EXECUTOR_PROMPT = """You are the Executor Agent. You execute approved trades on CoinDCX.
Rules:
1. Check if PAPER_TRADING mode is enabled — if so, simulate the trade
2. For live trades, verify human approval was given
3. Use limit orders by default (safer than market orders)
4. Log every trade with full details: timestamp, pair, side, qty, price, fees, reasoning
5. Calculate and report realized PnL after execution
6. Update the portfolio state

Never execute a trade without checking the compliance flag first."""
