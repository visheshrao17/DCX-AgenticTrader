"""
DCX-AgenticTrader — Executor Agent

Executes approved trade decisions via paper trading or live CoinDCX API.
Logs all trades to the trade store and updates portfolio state.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone

from config.settings import get_settings
from config.constants import SUPPORTED_PAIRS
from tools.coindcx_client import CoinDCXClient
from memory.paper_trading import PaperTradingEngine
from memory.trade_store import TradeStore
from memory.vector_store import VectorStore
from graph.state import TradingState
from utils.logger import get_agent_logger

log = get_agent_logger("executor")

# Module-level singletons (initialized on first use)
_paper_engine: Optional[PaperTradingEngine] = None
_trade_store: Optional[TradeStore] = None
_vector_store: Optional[VectorStore] = None


def _get_paper_engine() -> PaperTradingEngine:
    global _paper_engine
    if _paper_engine is None:
        settings = get_settings()
        _trade_store_inst = _get_trade_store()
        _paper_engine = PaperTradingEngine(
            initial_capital=settings.initial_capital_inr,
            trade_store=_trade_store_inst,
        )
    return _paper_engine


def _get_trade_store() -> TradeStore:
    global _trade_store
    if _trade_store is None:
        _trade_store = TradeStore()
    return _trade_store


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def executor_agent(state: TradingState) -> Dict[str, Any]:
    """
    LangGraph node: Executor Agent.

    Checks approval, executes trade (paper or live), logs results,
    and updates portfolio state.
    """
    log.info("=== Executor Agent running ===")

    settings = get_settings()
    decision = state.get("trade_decision", {})
    action = decision.get("action", "HOLD")

    # =========================================================================
    # 1. Gate checks
    # =========================================================================

    if action == "HOLD":
        log.info("Decision is HOLD — no execution needed")
        return {
            "execution_result": {
                "status": "skipped",
                "error": "",
                "is_paper": settings.paper_trading,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "current_step": "executor",
        }

    # Check human approval (in non-auto mode)
    if state.get("human_approval_needed") and not state.get("human_approved"):
        log.warning("Trade requires human approval but not yet approved — skipping")
        return {
            "execution_result": {
                "status": "pending_approval",
                "error": "Awaiting human approval",
                "is_paper": settings.paper_trading,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "current_step": "executor",
        }

    # Check compliance
    risk = state.get("risk_assessment", {})
    if risk.get("compliance_status") == "FAIL":
        log.error("Compliance FAIL — cannot execute trade")
        return {
            "execution_result": {
                "status": "blocked",
                "error": "Trade blocked by compliance",
                "is_paper": settings.paper_trading,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "current_step": "executor",
        }

    # =========================================================================
    # 2. Execute
    # =========================================================================

    market = decision.get("market", decision.get("pair", ""))
    side = "buy" if action == "BUY" else "sell"
    order_type = decision.get("order_type", "limit_order")
    quantity = decision.get("quantity", 0.0)
    price = decision.get("price", 0.0)
    reasoning = decision.get("reasoning", "")

    if quantity <= 0 or price <= 0:
        log.error(f"Invalid order params: qty={quantity}, price={price}")
        return {
            "execution_result": {
                "status": "rejected",
                "error": f"Invalid quantity ({quantity}) or price ({price})",
                "is_paper": settings.paper_trading,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "current_step": "executor",
        }

    log.info(f"Executing: {side.upper()} {quantity} {market} @ ₹{price:,.2f} (paper={settings.paper_trading})")

    if settings.paper_trading:
        result = _execute_paper(market, side, order_type, quantity, price, reasoning, state)
    else:
        result = _execute_live(market, side, order_type, quantity, price, reasoning)

    # =========================================================================
    # 3. Store trade memory for reflection
    # =========================================================================

    try:
        vs = _get_vector_store()
        market_data = state.get("market_data", {})
        technical = state.get("technical_signals", {})

        context = (
            f"Market: {market}, Price: ₹{price:,.2f}, "
            f"Signal: {technical.get('composite_signal', 'N/A')}, "
            f"RSI: {technical.get('rsi', 'N/A')}"
        )
        outcome = f"Status: {result.get('status')}, PnL: ₹{result.get('realized_pnl', 0):,.2f}"

        vs.store_trade_memory(
            trade_id=result.get("order_id", "unknown"),
            context=context,
            decision=f"{side.upper()} {quantity} {market}: {reasoning[:200]}",
            outcome=outcome,
        )
    except Exception as e:
        log.warning(f"Failed to store trade memory (non-critical): {e}")

    # =========================================================================
    # 4. Update portfolio state
    # =========================================================================

    updated_portfolio = state.get("portfolio", {})
    if settings.paper_trading:
        engine = _get_paper_engine()
        updated_portfolio = engine.get_portfolio_state()

    return {
        "execution_result": result,
        "portfolio": updated_portfolio,
        "current_step": "executor",
        "human_approval_needed": False,
        "human_approved": False,
    }


def _execute_paper(
    market: str, side: str, order_type: str,
    quantity: float, price: float, reasoning: str,
    state: dict,
) -> Dict[str, Any]:
    """Execute a paper trade."""
    engine = _get_paper_engine()

    # Collect agent signals for audit
    agent_signals = {
        "technical": state.get("technical_signals", {}),
        "sentiment": state.get("sentiment_score", {}),
        "risk": state.get("risk_assessment", {}),
    }

    result = engine.execute_order(
        market=market,
        side=side,
        order_type=order_type,
        quantity=quantity,
        current_price=price,
        reasoning=reasoning,
        agent_signals=agent_signals,
    )

    log.info(
        f"📝 Paper {side.upper()}: {quantity} {market} @ ₹{result.get('fill_price', 0):,.2f} "
        f"| PnL: ₹{result.get('realized_pnl', 0):,.2f} | Status: {result.get('status')}"
    )

    return result


def _execute_live(
    market: str, side: str, order_type: str,
    quantity: float, price: float, reasoning: str,
) -> Dict[str, Any]:
    """Execute a live trade on CoinDCX."""
    settings = get_settings()

    if not settings.has_coindcx_credentials:
        log.error("CoinDCX credentials not configured for live trading!")
        return {
            "status": "rejected",
            "error": "CoinDCX API credentials not configured",
            "is_paper": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    client = CoinDCXClient(
        api_key=settings.coindcx_api_key,
        api_secret=settings.coindcx_api_secret,
    )

    try:
        response = client.place_order(
            market=market,
            side=side,
            order_type=order_type,
            total_quantity=quantity,
            price_per_unit=price if order_type == "limit_order" else None,
        )

        result = {
            "order_id": response.get("id", ""),
            "status": response.get("status", "unknown"),
            "fill_price": float(response.get("avg_price", price)),
            "fill_quantity": quantity,
            "fee": float(response.get("fee_amount", 0)),
            "realized_pnl": 0.0,  # Calculated on fill
            "is_paper": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "",
        }

        # Record to trade store
        store = _get_trade_store()
        store.record_trade({
            "id": result["order_id"],
            "pair": SUPPORTED_PAIRS.get(market, {}).get("pair", market),
            "market": market,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "price": result["fill_price"],
            "fee": result["fee"],
            "is_paper": False,
            "status": result["status"],
            "reasoning": reasoning,
        })

        log.info(f"🔴 LIVE {side.upper()}: {quantity} {market} — Order ID: {result['order_id']}")
        return result

    except Exception as e:
        log.error(f"Live order failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "is_paper": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
