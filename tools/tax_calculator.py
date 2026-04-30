"""
DCX-AgenticTrader — Indian Crypto Tax Calculator

Calculates tax liability under Indian tax law:
- 30% flat tax on VDA profits (Section 115BBH)
- 1% TDS on transfers > ₹50,000 (Section 194S)
- No loss offsetting between different VDAs
"""

from typing import Dict, Any
from langchain_core.tools import tool

from config.constants import INDIA_TAX_FLAT_RATE, INDIA_TDS_RATE, INDIA_TDS_THRESHOLD_INR
from utils.logger import get_agent_logger

log = get_agent_logger("tax_calc")


@tool
def calculate_trade_tax(
    buy_price: float,
    sell_price: float,
    quantity: float,
) -> dict:
    """
    Calculate Indian crypto tax on a single trade.

    Args:
        buy_price: Purchase price per unit in INR.
        sell_price: Sale price per unit in INR.
        quantity: Number of units traded.

    Returns:
        Dict with profit, tax, TDS, and net proceeds.
    """
    cost = buy_price * quantity
    proceeds = sell_price * quantity
    profit = proceeds - cost

    # 30% flat tax on profits only (no tax on losses, but no offset either)
    tax = max(profit * INDIA_TAX_FLAT_RATE, 0.0)

    # 1% TDS on gross transaction value if > ₹50,000
    tds = 0.0
    tds_applicable = False
    if proceeds > INDIA_TDS_THRESHOLD_INR:
        tds = proceeds * INDIA_TDS_RATE
        tds_applicable = True

    # 4% health & education cess on tax
    cess = tax * 0.04
    total_tax = tax + cess

    net_proceeds = proceeds - total_tax - tds

    result = {
        "cost_basis": round(cost, 2),
        "gross_proceeds": round(proceeds, 2),
        "profit": round(profit, 2),
        "is_profitable": profit > 0,
        "tax_30_pct": round(tax, 2),
        "cess_4_pct": round(cess, 2),
        "total_tax": round(total_tax, 2),
        "tds_1_pct": round(tds, 2),
        "tds_applicable": tds_applicable,
        "net_proceeds": round(net_proceeds, 2),
        "effective_tax_rate": round((total_tax + tds) / proceeds * 100, 2) if proceeds else 0,
        "notes": [],
    }

    if profit <= 0:
        result["notes"].append("Loss cannot be offset against other VDA gains under Section 115BBH")
    if tds_applicable:
        result["notes"].append(f"1% TDS applies as transaction value (₹{proceeds:,.2f}) exceeds ₹{INDIA_TDS_THRESHOLD_INR:,}")

    log.info(
        f"Tax calc: profit=₹{profit:,.2f}, tax=₹{total_tax:,.2f}, "
        f"TDS=₹{tds:,.2f}, net=₹{net_proceeds:,.2f}"
    )
    return result


@tool
def estimate_portfolio_tax(
    trades_json: str,
) -> dict:
    """
    Estimate total tax liability for a list of completed trades.

    Args:
        trades_json: JSON string of trades with buy_price, sell_price, quantity fields.

    Returns:
        Dict with total tax, total TDS, and breakdown by trade.
    """
    import json
    try:
        trades = json.loads(trades_json)
    except Exception:
        return {"error": "Invalid JSON", "total_tax": 0}

    total_profit = 0.0
    total_loss = 0.0
    total_tax = 0.0
    total_tds = 0.0

    for trade in trades:
        buy = trade.get("buy_price", 0)
        sell = trade.get("sell_price", 0)
        qty = trade.get("quantity", 0)
        profit = (sell - buy) * qty

        if profit > 0:
            total_profit += profit
            total_tax += profit * INDIA_TAX_FLAT_RATE
        else:
            total_loss += abs(profit)

        proceeds = sell * qty
        if proceeds > INDIA_TDS_THRESHOLD_INR:
            total_tds += proceeds * INDIA_TDS_RATE

    cess = total_tax * 0.04
    total_tax_with_cess = total_tax + cess

    return {
        "total_profit": round(total_profit, 2),
        "total_loss": round(total_loss, 2),
        "net_pnl": round(total_profit - total_loss, 2),
        "taxable_amount": round(total_profit, 2),  # Losses can't offset!
        "tax_30_pct": round(total_tax, 2),
        "cess": round(cess, 2),
        "total_tax": round(total_tax_with_cess, 2),
        "total_tds": round(total_tds, 2),
        "total_liability": round(total_tax_with_cess + total_tds, 2),
        "important": "Losses CANNOT be offset against gains under Indian VDA tax rules",
    }


TAX_TOOLS = [calculate_trade_tax, estimate_portfolio_tax]
