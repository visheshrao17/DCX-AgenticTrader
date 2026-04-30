"""
DCX-AgenticTrader — Strategy Orchestrator

Pure LLM reasoning node that synthesizes signals from all other agents
and produces the final trade decision with full reasoning chain.
"""

from typing import Dict, Any

from config.constants import SUPPORTED_PAIRS
from graph.state import TradingState
from utils.logger import get_agent_logger

log = get_agent_logger("orchestrator")


def orchestrator_agent(state: TradingState) -> Dict[str, Any]:
    """
    LangGraph node: Strategy Orchestrator.

    Synthesizes technical signals, sentiment, and risk assessment
    into a final BUY / SELL / HOLD decision. No external tools —
    pure decision logic.
    """
    log.info("=== Strategy Orchestrator running ===")

    technical = state.get("technical_signals", {})
    sentiment = state.get("sentiment_score", {})
    risk = state.get("risk_assessment", {})
    market = state.get("market_data", {})
    portfolio = state.get("portfolio", {})
    pair = state.get("current_pair", "BTCINR")

    # =========================================================================
    # 1. Check if trading is allowed (risk gate)
    # =========================================================================

    if risk.get("compliance_status") == "FAIL":
        warnings = risk.get("risk_warnings", [])
        reasoning = f"Trade BLOCKED by Risk Agent: {'; '.join(warnings)}"
        log.warning(reasoning)
        return {
            "trade_decision": {
                "action": "HOLD",
                "pair": pair,
                "confidence": 0.0,
                "reasoning": reasoning,
                "blocked": True,
            },
            "human_approval_needed": False,
            "current_step": "orchestrator",
        }

    # =========================================================================
    # 2. Extract signals
    # =========================================================================

    # Technical
    tech_signal = technical.get("composite_signal", "NEUTRAL")
    tech_score = technical.get("signal_score", 0.0)
    tech_confidence = technical.get("confidence", 0.0)

    # Sentiment
    sent_score = sentiment.get("overall_score", 0.0)
    fng_value = sentiment.get("fear_greed_index", 50)
    sent_confidence = sentiment.get("confidence", 0.0)

    # Risk constraints
    max_qty = risk.get("max_position_size", 0.0)
    max_value = risk.get("max_position_value_inr", 0.0)
    var_95 = risk.get("value_at_risk", 0.0)

    # Market
    current_price = market.get("last_price", 0.0) if market else 0.0

    # =========================================================================
    # 3. Decision Logic
    # =========================================================================

    # Weighted composite: 50% technical, 30% sentiment, 20% risk-adjusted
    tech_weight = 0.50
    sent_weight = 0.30
    risk_adj_weight = 0.20

    # Risk adjustment: penalize if high VaR or high drawdown
    risk_penalty = 0.0
    drawdown = risk.get("current_drawdown_pct", 0.0)
    if drawdown > 5:
        risk_penalty = -0.2
    elif drawdown > 3:
        risk_penalty = -0.1

    composite_score = (
        tech_score * tech_weight +
        sent_score * sent_weight +
        risk_penalty * risk_adj_weight
    )

    # Confidence from weighted average of agent confidences
    composite_confidence = (
        tech_confidence * tech_weight +
        sent_confidence * sent_weight +
        (1.0 - abs(risk_penalty)) * risk_adj_weight
    )

    # =========================================================================
    # 4. Determine Action
    # =========================================================================

    action = "HOLD"
    quantity = 0.0
    order_type = "limit_order"

    # Only trade if confidence >= 0.6
    MIN_CONFIDENCE = 0.6

    if composite_score >= 0.3 and composite_confidence >= MIN_CONFIDENCE:
        action = "BUY"
        # Size based on confidence: higher confidence = larger position (up to max)
        size_factor = min(composite_confidence, 0.9)
        quantity = max_qty * size_factor

    elif composite_score <= -0.3 and composite_confidence >= MIN_CONFIDENCE:
        # Check if we have a position to sell
        positions = portfolio.get("positions", {})
        pair_info = SUPPORTED_PAIRS.get(pair, {})
        target_currency = pair_info.get("target_currency", "")

        if target_currency in positions:
            pos = positions[target_currency]
            available = pos.get("quantity", 0.0) if isinstance(pos, dict) else 0.0
            if available > 0:
                action = "SELL"
                quantity = available  # Sell entire position
            else:
                action = "HOLD"
        else:
            action = "HOLD"  # No position to sell, can't short

    # Calculate price (use limit with small buffer for better fill)
    price = current_price
    if action == "BUY" and current_price > 0:
        price = current_price * 1.001  # Slightly above for limit buy
    elif action == "SELL" and current_price > 0:
        price = current_price * 0.999  # Slightly below for limit sell

    # Stop loss and take profit
    atr = technical.get("atr", 0)
    stop_loss = 0.0
    take_profit = 0.0
    if current_price > 0 and atr > 0:
        stop_loss = current_price - (atr * 2)  # 2 ATR stop loss
        take_profit = current_price + (atr * 3)  # 3 ATR take profit (1.5:1 R:R)

    # =========================================================================
    # 5. Build Decision
    # =========================================================================

    reasoning_parts = [
        f"Pair: {pair} @ ₹{current_price:,.2f}",
        f"Technical: {tech_signal} (score={tech_score:.2f}, conf={tech_confidence:.2f})",
        f"Sentiment: {sent_score:+.2f} (FNG={fng_value})",
        f"Risk: drawdown={drawdown:.1f}%, VaR=₹{var_95:,.2f}",
        f"Composite: score={composite_score:.3f}, conf={composite_confidence:.2f}",
        f"Decision: {action}",
    ]

    if action != "HOLD":
        reasoning_parts.append(f"Quantity: {quantity:.6f} (₹{quantity * current_price:,.2f})")
        reasoning_parts.append(f"SL: ₹{stop_loss:,.2f}, TP: ₹{take_profit:,.2f}")

    reasoning = " | ".join(reasoning_parts)

    decision = {
        "action": action,
        "pair": pair,
        "market": pair,
        "quantity": round(quantity, 8),
        "price": round(price, 2),
        "order_type": order_type,
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "confidence": round(composite_confidence, 3),
        "reasoning": reasoning,
        "technical_weight": tech_weight,
        "sentiment_weight": sent_weight,
        "risk_weight": risk_adj_weight,
    }

    needs_approval = action != "HOLD"

    log.info(f"Decision: {action} {quantity:.6f} {pair} (conf={composite_confidence:.2f})")

    return {
        "trade_decision": decision,
        "human_approval_needed": needs_approval,
        "current_step": "orchestrator",
    }
