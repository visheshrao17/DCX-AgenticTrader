import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from config.settings import get_settings
from utils.logger import get_agent_logger
from tools.llm_client import call_llm

log = get_agent_logger("orchestrator_prompt")

class LLMTradeDecision(BaseModel):
    """Structured output expected from the LLM orchestrator."""
    action: str = Field(description="The trading action to take: BUY, SELL, or HOLD.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    stop_loss: float = Field(description="Suggested stop loss price (0.0 if HOLD).")
    take_profit: float = Field(description="Suggested take profit price (0.0 if HOLD).")
    reasoning: str = Field(description="A short natural language rationale explaining the decision.")

def build_orchestrator_prompt(
    pair: str,
    market: Dict[str, Any],
    technical: Dict[str, Any],
    sentiment: Dict[str, Any],
    risk: Dict[str, Any],
    portfolio: Dict[str, Any],
    similar_trades: List[Dict[str, Any]]
) -> str:
    """Builds the context prompt for the LLM based on all agent outputs and memories."""
    
    prompt = f"""
CURRENT PAIR: {pair}
CURRENT PRICE: {market.get('last_price', 'Unknown')}

=== TECHNICAL SIGNALS ===
{json.dumps(technical, indent=2)}

=== SENTIMENT RESEARCH ===
{json.dumps(sentiment, indent=2)}

=== RISK & COMPLIANCE ASSESSMENT ===
{json.dumps(risk, indent=2)}

=== CURRENT PORTFOLIO ===
{json.dumps(portfolio, indent=2)}

=== PAST SIMILAR TRADES ===
"""
    if similar_trades:
        for idx, trade in enumerate(similar_trades):
            prompt += f"\nTrade {idx + 1}:\n"
            prompt += f"{trade.get('document', '')}\n"
    else:
        prompt += "No past similar trades found in memory.\n"

    prompt += """
=== INSTRUCTIONS ===
1. Analyze the Technical Signals, Sentiment Research, and Risk Assessment.
2. Consider the outcomes of Past Similar Trades.
3. Formulate a short chain-of-thought rationale explaining what the technical picture looks like, what sentiment is saying, risk concerns, and insights from past trades.
4. Provide your decision and reasoning.
"""
    return prompt

def call_llm_orchestrator(
    pair: str,
    market: Dict[str, Any],
    technical: Dict[str, Any],
    sentiment: Dict[str, Any],
    risk: Dict[str, Any],
    portfolio: Dict[str, Any],
    similar_trades: List[Dict[str, Any]]
) -> LLMTradeDecision:
    """Calls the LLM to make a trading decision."""
    
    settings = get_settings()
    if not settings.has_llm_credentials:
        raise ValueError("LLM API Key not found in settings.")

    system_prompt = "You are the Strategy Orchestrator agent in a multi-agent trading system. Your job is to synthesize signals from various specialized agents to make a final trading decision."
    user_prompt = build_orchestrator_prompt(
        pair, market, technical, sentiment, risk, portfolio, similar_trades
    )
    
    log.debug("Calling LLM for Orchestrator decision...")
    result = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=LLMTradeDecision,
        agent_name="orchestrator",
        temperature=0.2,
    )
    
    return result
