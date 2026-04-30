"""
DCX-AgenticTrader — Graph State Schema

Typed state that flows through the LangGraph pipeline.
Every agent reads from and writes to this shared state.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Annotated
from datetime import datetime

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


# =============================================================================
# Sub-Models — Each agent writes one of these
# =============================================================================

class MarketSnapshot(BaseModel):
    """Output of the Market Data Agent."""
    pair: str = ""
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    volume_24h: float = 0.0
    spread_pct: float = 0.0
    candles_15m: List[Dict[str, Any]] = Field(default_factory=list)
    candles_1h: List[Dict[str, Any]] = Field(default_factory=list)
    candles_4h: List[Dict[str, Any]] = Field(default_factory=list)
    candles_1d: List[Dict[str, Any]] = Field(default_factory=list)
    orderbook_bids: List[Dict[str, str]] = Field(default_factory=list)
    orderbook_asks: List[Dict[str, str]] = Field(default_factory=list)
    recent_trades: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = ""


class TechnicalSignals(BaseModel):
    """Output of the Technical Analyst Agent."""
    pair: str = ""
    rsi: float = 50.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    atr: float = 0.0
    vwap: float = 0.0
    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)
    composite_signal: str = "NEUTRAL"  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    signal_score: float = 0.0  # -1.0 to +1.0
    confidence: float = 0.5
    reasoning: str = ""


class SentimentResult(BaseModel):
    """Output of the Sentiment Researcher Agent."""
    overall_score: float = 0.0  # -1.0 to +1.0
    fear_greed_index: int = 50  # 0 = extreme fear, 100 = extreme greed
    fear_greed_label: str = "Neutral"
    india_sentiment: float = 0.0
    global_sentiment: float = 0.0
    top_news: List[Dict[str, Any]] = Field(default_factory=list)
    sources_analyzed: int = 0
    confidence: float = 0.5
    reasoning: str = ""


class RiskAssessment(BaseModel):
    """Output of the Risk & Compliance Agent."""
    compliance_status: str = "PENDING"  # PASS, FAIL, PENDING
    max_position_size: float = 0.0
    max_position_value_inr: float = 0.0
    value_at_risk: float = 0.0
    current_drawdown_pct: float = 0.0
    tax_estimate_inr: float = 0.0
    tds_applicable: bool = False
    risk_warnings: List[str] = Field(default_factory=list)
    compliance_notes: List[str] = Field(default_factory=list)
    reasoning: str = ""


class TradeDecision(BaseModel):
    """Output of the Strategy Orchestrator."""
    action: str = "HOLD"  # BUY, SELL, HOLD
    pair: str = ""
    market: str = ""
    quantity: float = 0.0
    price: float = 0.0
    order_type: str = "limit_order"  # market_order, limit_order
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    technical_weight: float = 0.0
    sentiment_weight: float = 0.0
    risk_weight: float = 0.0


class ExecutionResult(BaseModel):
    """Output of the Executor Agent."""
    order_id: str = ""
    status: str = ""  # filled, partial, rejected, simulated
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    fee: float = 0.0
    realized_pnl: float = 0.0
    is_paper: bool = True
    timestamp: str = ""
    error: str = ""


class PortfolioState(BaseModel):
    """Current portfolio state."""
    balances: Dict[str, float] = Field(default_factory=lambda: {"INR": 100000.0})
    positions: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    total_value_inr: float = 100000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    peak_value: float = 100000.0
    max_drawdown_pct: float = 0.0


# =============================================================================
# Main Trading State — flows through the entire graph
# =============================================================================

class TradingState(MessagesState):
    """
    Complete state schema for the LangGraph trading pipeline.

    Inherits `messages` from MessagesState (with add_messages reducer).
    Each agent updates its own field; the supervisor reads all fields
    to decide routing.
    """
    # Agent outputs
    market_data: Optional[Dict[str, Any]] = None
    technical_signals: Optional[Dict[str, Any]] = None
    sentiment_score: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    trade_decision: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None

    # Portfolio & meta
    portfolio: Dict[str, Any] = Field(
        default_factory=lambda: PortfolioState().model_dump()
    )
    current_pair: str = "BTCINR"
    current_step: str = "supervisor"
    human_approval_needed: bool = False
    human_approved: bool = False
    cycle_count: int = 0
    error: Optional[str] = None
