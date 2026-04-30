"""
DCX-AgenticTrader — LangGraph Supervisor

Central routing node that decides which agent to call next.
Uses conditional edges to orchestrate the full trading pipeline.
"""

from typing import Dict, Any, Literal

from graph.state import TradingState
from utils.logger import get_agent_logger

log = get_agent_logger("supervisor")


def supervisor_node(state: TradingState) -> Dict[str, Any]:
    """
    LangGraph node: Supervisor.

    Examines current state and determines which agent should run next.
    This is the entry point of every trading cycle.
    """
    cycle = state.get("cycle_count", 0)
    log.info(f"=== Supervisor — Cycle {cycle + 1} ===")

    return {
        "current_step": "supervisor",
        "cycle_count": cycle + 1,
        "error": None,
    }


def route_after_supervisor(state: TradingState) -> str:
    """Route from supervisor → always start with market data."""
    return "market_data"


def route_after_market_data(state: TradingState) -> str:
    """Route from market data → technical analysis."""
    if state.get("error"):
        log.warning(f"Market data error: {state['error']} — retrying via supervisor")
        return "end"
    return "technical"


def route_after_technical(state: TradingState) -> str:
    """Route from technical → sentiment."""
    return "sentiment"


def route_after_sentiment(state: TradingState) -> str:
    """Route from sentiment → risk check."""
    return "risk"


def route_after_risk(state: TradingState) -> str:
    """Route from risk → orchestrator."""
    return "orchestrator"


def route_after_orchestrator(state: TradingState) -> str:
    """Route from orchestrator → human review or end."""
    decision = state.get("trade_decision", {})
    action = decision.get("action", "HOLD")

    if action == "HOLD" or decision.get("blocked"):
        log.info("Decision is HOLD or blocked — ending cycle")
        return "end"

    if state.get("human_approval_needed"):
        log.info("Trade requires approval → human_review")
        return "human_review"

    return "executor"


def route_after_human_review(state: TradingState) -> str:
    """Route from human review → executor (if approved) or end."""
    if state.get("human_approved"):
        log.info("Human approved → executor")
        return "executor"
    else:
        log.info("Human rejected or timeout → ending cycle")
        return "end"


def human_review_node(state: TradingState) -> Dict[str, Any]:
    """
    Human-in-the-loop review node.

    In interactive mode, this pauses for user input.
    In auto-approve mode (for backtesting), it auto-approves.
    """
    decision = state.get("trade_decision", {})
    action = decision.get("action", "HOLD")
    pair = decision.get("pair", "")
    qty = decision.get("quantity", 0)
    price = decision.get("price", 0)
    confidence = decision.get("confidence", 0)
    reasoning = decision.get("reasoning", "")

    log.info(
        f"\n{'='*60}\n"
        f"🔔 TRADE APPROVAL REQUIRED\n"
        f"   Action:     {action}\n"
        f"   Pair:       {pair}\n"
        f"   Quantity:   {qty}\n"
        f"   Price:      ₹{price:,.2f}\n"
        f"   Confidence: {confidence:.1%}\n"
        f"   Reasoning:  {reasoning[:200]}\n"
        f"{'='*60}"
    )

    # Auto-approve in paper trading mode for now
    # In production, this would pause and wait for user input
    from config.settings import get_settings
    settings = get_settings()

    if settings.paper_trading:
        log.info("Auto-approving (paper trading mode)")
        return {
            "human_approved": True,
            "current_step": "human_review",
        }

    # For live trading, default to reject (safety first)
    log.warning("Live trading — defaulting to REJECT (manual approval needed via dashboard)")
    return {
        "human_approved": False,
        "current_step": "human_review",
    }
