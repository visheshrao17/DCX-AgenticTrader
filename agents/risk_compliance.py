"""
DCX-AgenticTrader — Risk & Compliance Agent

Validates trades against Indian regulations, calculates VaR,
enforces position sizing limits, and estimates tax implications.
This agent can BLOCK trades that violate rules.
"""

import json
import math
from typing import Dict, Any, List
from datetime import datetime, timezone

import numpy as np
from pydantic import BaseModel, Field

from config.constants import (
    INDIA_TAX_FLAT_RATE, INDIA_TDS_RATE, INDIA_TDS_THRESHOLD_INR,
    DEFAULT_MAX_POSITION_SIZE_PCT, DEFAULT_MAX_DRAWDOWN_PCT,
    DEFAULT_MAX_TRADES_PER_DAY, DEFAULT_STOP_LOSS_PCT,
)
from config.settings import get_settings
from tools.rag_compliance import query_compliance
from tools.tax_calculator import calculate_trade_tax
from memory.trade_store import TradeStore
from graph.state import TradingState
from utils.logger import get_agent_logger
from tools.llm_client import call_llm
from utils.error_handler import LLMError

log = get_agent_logger("risk")


class ComplianceRationale(BaseModel):
    """Structured output expected from the Risk & Compliance LLM."""
    rationale: str = Field(description="Natural-language compliance rationale citing specific regulations and risk limits.")
    regulatory_concerns: List[str] = Field(description="Any specific regulatory concerns identified from the compliance query.")


def risk_compliance_agent(state: TradingState) -> Dict[str, Any]:
    """
    LangGraph node: Risk & Compliance Agent.

    Validates the current trading context against:
    1. Indian crypto regulations (30% tax, 1% TDS, FIU-IND)
    2. Portfolio risk limits (position sizing, drawdown)
    3. Value-at-Risk calculations
    4. Daily trade count limits
    """
    log.info("=== Risk & Compliance Agent running ===")

    settings = get_settings()
    portfolio = state.get("portfolio", {})
    market_data = state.get("market_data", {})
    technical = state.get("technical_signals", {})
    pair = state.get("current_pair", "BTCINR")

    risk_warnings: List[str] = []
    compliance_notes: List[str] = []
    compliance_status = "PASS"

    # =========================================================================
    # 1. Portfolio Risk Assessment
    # =========================================================================

    total_value = portfolio.get("total_value_inr", settings.initial_capital_inr)
    current_drawdown = portfolio.get("max_drawdown_pct", 0.0)
    max_drawdown_limit = settings.max_drawdown_pct

    # Check drawdown
    if current_drawdown >= max_drawdown_limit:
        compliance_status = "FAIL"
        risk_warnings.append(
            f"🛑 BLOCKED: Current drawdown ({current_drawdown:.1f}%) exceeds "
            f"limit ({max_drawdown_limit:.1f}%). Trading halted."
        )

    # Check daily trade count
    trade_store = TradeStore()
    trades_today = trade_store.count_trades_today()
    max_daily = settings.max_trades_per_day

    if trades_today >= max_daily:
        compliance_status = "FAIL"
        risk_warnings.append(
            f"🛑 BLOCKED: Daily trade limit reached ({trades_today}/{max_daily})."
        )

    # =========================================================================
    # 2. Position Sizing
    # =========================================================================

    max_position_pct = settings.max_position_size_pct
    max_position_value = total_value * (max_position_pct / 100)

    current_price = market_data.get("last_price", 0) if market_data else 0

    if current_price > 0:
        max_quantity = max_position_value / current_price
    else:
        max_quantity = 0.0

    log.info(
        f"Position sizing: max {max_position_pct}% of ₹{total_value:,.2f} = "
        f"₹{max_position_value:,.2f} ({max_quantity:.6f} units @ ₹{current_price:,.2f})"
    )

    # =========================================================================
    # 3. Value-at-Risk (VaR) — Historical Volatility Method
    # =========================================================================

    var_95 = 0.0
    atr = technical.get("atr", 0) if technical else 0

    if current_price > 0 and atr > 0:
        # Daily volatility estimate from ATR
        daily_vol_pct = (atr / current_price) * 100
        # 95% VaR = 1.645 * daily volatility * position value
        var_95 = 1.645 * (daily_vol_pct / 100) * max_position_value
        log.info(f"VaR(95%): ₹{var_95:,.2f} (daily vol: {daily_vol_pct:.2f}%)")

        if var_95 > max_position_value * 0.05:  # VaR > 5% of position
            risk_warnings.append(
                f"⚠️ High volatility: VaR(95%) = ₹{var_95:,.2f} "
                f"({var_95/max_position_value*100:.1f}% of max position). Consider smaller size."
            )

    # =========================================================================
    # 4. Tax Impact Estimation
    # =========================================================================

    # Estimate tax on a hypothetical profitable trade
    tax_estimate = 0.0
    tds_applicable = False

    if current_price > 0:
        hypothetical_profit_pct = 5  # Assume 5% profit target
        hypothetical_sell = current_price * (1 + hypothetical_profit_pct / 100)
        hypothetical_qty = max_quantity

        if hypothetical_qty > 0:
            profit = (hypothetical_sell - current_price) * hypothetical_qty
            tax_estimate = profit * INDIA_TAX_FLAT_RATE
            trade_value = hypothetical_sell * hypothetical_qty

            if trade_value > INDIA_TDS_THRESHOLD_INR:
                tds_applicable = True
                compliance_notes.append(
                    f"TDS of 1% (≈₹{trade_value * INDIA_TDS_RATE:,.2f}) will apply "
                    f"on transaction value > ₹{INDIA_TDS_THRESHOLD_INR:,}"
                )

    compliance_notes.append(
        f"Estimated tax on 5% profit: ₹{tax_estimate:,.2f} "
        f"(30% flat + 4% cess = {INDIA_TAX_FLAT_RATE*100 + INDIA_TAX_FLAT_RATE*4:.1f}%)"
    )
    compliance_notes.append("Losses CANNOT be offset against other VDA gains (Section 115BBH)")

    # =========================================================================
    # 5. Regulatory Compliance Check
    # =========================================================================

    reg_info_text = ""
    # Query RAG for any specific concerns
    try:
        reg_info_text = query_compliance.invoke({"question": f"Is trading {pair} compliant in India?"})
        if reg_info_text and "not allowed" not in reg_info_text.lower():
            compliance_notes.append("Regulatory check: Trading is legal and compliant in India")
        else:
            risk_warnings.append("⚠️ Regulatory concern detected — review before trading")
    except Exception as e:
        log.warning(f"Compliance RAG query failed (non-blocking): {e}")
        compliance_notes.append("Regulatory check: Manual verification recommended")

    # =========================================================================
    # Build Result (Deterministic)
    # =========================================================================

    reasoning = (
        f"Risk assessment for {pair}: "
        f"Portfolio=₹{total_value:,.2f}, Drawdown={current_drawdown:.1f}%/{max_drawdown_limit:.1f}%, "
        f"Max position=₹{max_position_value:,.2f} ({max_quantity:.6f} units), "
        f"VaR(95%)=₹{var_95:,.2f}, Trades today={trades_today}/{max_daily}, "
        f"Status={compliance_status}."
    )

    if risk_warnings:
        reasoning += f" Warnings: {'; '.join(risk_warnings)}"

    result = {
        "compliance_status": compliance_status,
        "max_position_size": round(max_quantity, 8),
        "max_position_value_inr": round(max_position_value, 2),
        "value_at_risk": round(var_95, 2),
        "current_drawdown_pct": round(current_drawdown, 2),
        "tax_estimate_inr": round(tax_estimate, 2),
        "tds_applicable": tds_applicable,
        "trades_today": trades_today,
        "max_trades_per_day": max_daily,
        "risk_warnings": risk_warnings,
        "compliance_notes": compliance_notes,
        "reasoning": reasoning,
        "llm_used": False,
    }

    # =========================================================================
    # 6. LLM Explanation (Does NOT alter PASS/FAIL status)
    # =========================================================================
    
    if settings.use_llm_risk_explanation:
        try:
            system_prompt = "You are a Risk & Compliance Officer for an Indian crypto trading firm. Generate a brief compliance rationale based on the provided data."
            user_prompt = (
                f"Pair: {pair}\n"
                f"Status: {compliance_status}\n"
                f"Warnings: {json.dumps(risk_warnings)}\n"
                f"Notes: {json.dumps(compliance_notes)}\n"
                f"Regulatory Context:\n{reg_info_text}\n"
            )
            
            llm_decision = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=ComplianceRationale,
                agent_name="risk",
                temperature=0.1,
            )
            
            result["llm_compliance_rationale"] = llm_decision.rationale
            result["reasoning"] = f"[LLM] {llm_decision.rationale} (Deterministic: {reasoning})"
            if llm_decision.regulatory_concerns:
                result["compliance_notes"].extend(llm_decision.regulatory_concerns)
            result["llm_used"] = True
            log.info("LLM Risk Explanation generated successfully.")
            
        except LLMError as e:
            log.warning(f"Risk LLM failed, using deterministic reasoning: {e}")
        except Exception as e:
            log.warning(f"Risk LLM failed unexpectedly: {e}")

    log.info(f"Risk: {compliance_status} | Max position: ₹{max_position_value:,.2f} | Warnings: {len(risk_warnings)}")

    return {
        "risk_assessment": result,
        "current_step": "risk",
    }
