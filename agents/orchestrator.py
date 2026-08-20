"""
DCX-AgenticTrader — Strategy Orchestrator

Pure LLM reasoning node that synthesizes signals from all other agents
and produces the final trade decision with full reasoning chain.
"""

from typing import Dict, Any
import traceback

from config.constants import SUPPORTED_PAIRS
from graph.state import TradingState
from utils.logger import get_agent_logger
from agents.orchestrator_prompt import call_llm_orchestrator
from memory.vector_store import VectorStore

log = get_agent_logger("orchestrator")


def orchestrator_agent(state: TradingState) -> Dict[str, Any]:
    """
    LangGraph node: Strategy Orchestrator.

    Synthesizes technical signals, sentiment, and risk assessment
    into a final BUY / SELL / HOLD decision. Uses LLM for primary
    decision logic with a deterministic fallback.
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
    # 3. Deterministic Fallback Logic (Calculated side-by-side)
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

    composite_confidence = (
        tech_confidence * tech_weight +
        sent_confidence * sent_weight +
        (1.0 - abs(risk_penalty)) * risk_adj_weight
    )

    fallback_action = "HOLD"
    MIN_CONFIDENCE = 0.6
    
    # Check if we have a position to sell
    positions = portfolio.get("positions", {})
    pair_info = SUPPORTED_PAIRS.get(pair, {})
    target_currency = pair_info.get("target_currency", "")
    available_to_sell = 0.0
    if target_currency in positions:
        pos = positions[target_currency]
        available_to_sell = pos.get("quantity", 0.0) if isinstance(pos, dict) else 0.0

    if composite_score >= 0.3 and composite_confidence >= MIN_CONFIDENCE:
        fallback_action = "BUY"
    elif composite_score <= -0.3 and composite_confidence >= MIN_CONFIDENCE:
        if available_to_sell > 0:
            fallback_action = "SELL"

    atr = technical.get("atr", 0)
    fallback_sl = 0.0
    fallback_tp = 0.0
    if current_price > 0 and atr > 0:
        fallback_sl = current_price - (atr * 2)
        fallback_tp = current_price + (atr * 3)
        
    fallback_reasoning_parts = [
        f"Pair: {pair} @ ₹{current_price:,.2f}",
        f"Technical: {tech_signal} (score={tech_score:.2f}, conf={tech_confidence:.2f})",
        f"Sentiment: {sent_score:+.2f} (FNG={fng_value})",
        f"Risk: drawdown={drawdown:.1f}%, VaR=₹{var_95:,.2f}",
        f"Composite: score={composite_score:.3f}, conf={composite_confidence:.2f}",
        f"Deterministic Decision: {fallback_action}",
    ]
    fallback_reasoning_str = " | ".join(fallback_reasoning_parts)

    # =========================================================================
    # 4. LLM Decision Logic
    # =========================================================================

    final_action = fallback_action
    final_confidence = composite_confidence
    final_sl = fallback_sl
    final_tp = fallback_tp
    final_reasoning = fallback_reasoning_str
    
    llm_success = False

    try:
        from config.settings import get_settings
        settings = get_settings()
        
        store = VectorStore()
        market_context_str = f"{pair} trading at {current_price}. Tech={tech_signal}, Sent={sent_score}."
        similar_trades = store.recall_similar_trades(market_context_str, k=3)
        
        if settings.use_llm_orchestrator:
            from agents.orchestrator_prompt import call_llm_orchestrator, LLMTradeDecision
            
            llm_decision = call_llm_orchestrator(
                pair=pair,
                market=market,
                technical=technical,
                sentiment=sentiment,
                risk=risk,
                portfolio=portfolio,
                similar_trades=similar_trades
            )
            
            final_action = llm_decision.action.upper()
            if final_action not in ["BUY", "SELL", "HOLD"]:
                final_action = "HOLD"
                
            # Prevent invalid SELLS
            if final_action == "SELL" and available_to_sell <= 0:
                final_action = "HOLD"
                
            final_confidence = llm_decision.confidence
            final_sl = llm_decision.stop_loss
            final_tp = llm_decision.take_profit
            
            # Side-by-side logging
            final_reasoning = (
                f"[LLM Decision] {llm_decision.reasoning}\n"
                f"[Fallback Comp] {fallback_reasoning_str}"
            )
            llm_success = True
            log.info(f"LLM Strategy Orchestrator decision: {final_action} with confidence {final_confidence:.2f}")
        else:
            log.info("LLM Orchestrator disabled in config. Using deterministic rules.")
            final_reasoning = f"[Deterministic] {fallback_reasoning_str}"

    except Exception as e:
        log.error(f"LLM decision failed, falling back to deterministic: {e}")
        log.debug(traceback.format_exc())
        final_reasoning = f"[Fallback Decision - LLM Failed] {fallback_reasoning_str}"

    # =========================================================================
    # 5. Determine Quantity & Price
    # =========================================================================

    quantity = 0.0
    if final_action == "BUY":
        size_factor = min(final_confidence, 0.9)
        quantity = max_qty * size_factor
    elif final_action == "SELL":
        quantity = available_to_sell

    price = current_price
    if final_action == "BUY" and current_price > 0:
        price = current_price * 1.001
    elif final_action == "SELL" and current_price > 0:
        price = current_price * 0.999

    # =========================================================================
    # 6. Build Decision Dictionary
    # =========================================================================

    decision = {
        "action": final_action,
        "pair": pair,
        "market": pair,
        "quantity": round(quantity, 8),
        "price": round(price, 2),
        "order_type": "limit_order",
        "stop_loss": round(final_sl, 2),
        "take_profit": round(final_tp, 2),
        "confidence": round(final_confidence, 3),
        "reasoning": final_reasoning,
        "technical_weight": tech_weight,
        "sentiment_weight": sent_weight,
        "risk_weight": risk_adj_weight,
        "llm_used": llm_success
    }

    needs_approval = final_action != "HOLD"

    log.info(f"Decision: {final_action} {quantity:.6f} {pair} (conf={final_confidence:.2f})")

    return {
        "trade_decision": decision,
        "human_approval_needed": needs_approval,
        "current_step": "orchestrator",
    }
