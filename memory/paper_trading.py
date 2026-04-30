"""
DCX-AgenticTrader — Paper Trading Engine

Simulates order execution against live orderbook data without
hitting the real CoinDCX exchange. Maintains a virtual portfolio
with realistic slippage and fee simulation.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from config.constants import (
    COINDCX_MAKER_FEE, COINDCX_TAKER_FEE,
    PAPER_TRADING_SLIPPAGE, SUPPORTED_PAIRS,
)
from memory.trade_store import TradeStore
from utils.logger import get_agent_logger

log = get_agent_logger("paper_trading")


class PaperTradingEngine:
    """
    Simulates trading without real money.

    Maintains virtual balances, simulates fills against live prices,
    applies realistic slippage and fees, and records all trades to SQLite.

    Usage:
        engine = PaperTradingEngine(initial_capital=100000)
        result = engine.execute_order(
            market="BTCINR", side="buy", order_type="market_order",
            quantity=0.001, current_price=7500000
        )
    """

    def __init__(self, initial_capital: float = 100000.0, trade_store: Optional[TradeStore] = None):
        self.trade_store = trade_store or TradeStore()

        # Virtual portfolio
        self.balances: Dict[str, float] = {"INR": initial_capital}
        self.positions: Dict[str, Dict[str, float]] = {}
        self.initial_capital = initial_capital
        self.peak_value = initial_capital
        self.realized_pnl = 0.0
        self.total_fees = 0.0

        log.info(f"Paper trading engine initialized with ₹{initial_capital:,.2f}")

    # =========================================================================
    # Order Execution
    # =========================================================================

    def execute_order(
        self,
        market: str,
        side: str,
        order_type: str,
        quantity: float,
        current_price: float,
        limit_price: Optional[float] = None,
        reasoning: str = "",
        agent_signals: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Simulate an order execution.

        Args:
            market: Market name (e.g., "BTCINR").
            side: "buy" or "sell".
            order_type: "market_order" or "limit_order".
            quantity: Amount to trade.
            current_price: Current market price.
            limit_price: Price for limit orders.
            reasoning: Why this trade was made.
            agent_signals: Agent signal data for audit.

        Returns:
            Execution result dict.
        """
        pair_info = SUPPORTED_PAIRS.get(market, {})
        target_currency = pair_info.get("target_currency", market.replace("INR", ""))
        base_currency = pair_info.get("base_currency", "INR")

        # Determine fill price
        if order_type == "market_order":
            slippage = PAPER_TRADING_SLIPPAGE
            if side == "buy":
                fill_price = current_price * (1 + slippage)  # Slightly worse for buyer
            else:
                fill_price = current_price * (1 - slippage)  # Slightly worse for seller
        else:
            fill_price = limit_price or current_price

        # Calculate trade value and fees
        trade_value = quantity * fill_price
        fee_rate = COINDCX_TAKER_FEE if order_type == "market_order" else COINDCX_MAKER_FEE
        fee = trade_value * fee_rate

        # Validate balance
        if side == "buy":
            required = trade_value + fee
            available = self.balances.get(base_currency, 0.0)
            if available < required:
                log.warning(f"Insufficient {base_currency}: need ₹{required:,.2f}, have ₹{available:,.2f}")
                return {
                    "status": "rejected",
                    "error": f"Insufficient {base_currency} balance",
                    "required": required,
                    "available": available,
                    "is_paper": True,
                }
        else:  # sell
            available = self.balances.get(target_currency, 0.0)
            if available < quantity:
                log.warning(f"Insufficient {target_currency}: need {quantity}, have {available}")
                return {
                    "status": "rejected",
                    "error": f"Insufficient {target_currency} balance",
                    "required": quantity,
                    "available": available,
                    "is_paper": True,
                }

        # Execute the fill
        trade_id = str(uuid.uuid4())[:12]
        pnl = 0.0

        if side == "buy":
            # Deduct INR, add crypto
            self.balances[base_currency] = self.balances.get(base_currency, 0) - trade_value - fee
            self.balances[target_currency] = self.balances.get(target_currency, 0) + quantity

            # Track position entry
            if target_currency not in self.positions:
                self.positions[target_currency] = {
                    "quantity": 0.0,
                    "avg_entry_price": 0.0,
                    "total_cost": 0.0,
                }
            pos = self.positions[target_currency]
            total_qty = pos["quantity"] + quantity
            pos["total_cost"] = pos["total_cost"] + trade_value
            pos["avg_entry_price"] = pos["total_cost"] / total_qty if total_qty else 0
            pos["quantity"] = total_qty

        else:  # sell
            # Add INR, deduct crypto
            self.balances[base_currency] = self.balances.get(base_currency, 0) + trade_value - fee
            self.balances[target_currency] = self.balances.get(target_currency, 0) - quantity

            # Calculate realized PnL
            pos = self.positions.get(target_currency, {})
            avg_entry = pos.get("avg_entry_price", fill_price)
            pnl = (fill_price - avg_entry) * quantity - fee
            self.realized_pnl += pnl

            # Update position
            if target_currency in self.positions:
                self.positions[target_currency]["quantity"] -= quantity
                if self.positions[target_currency]["quantity"] <= 0.0001:
                    del self.positions[target_currency]

        self.total_fees += fee

        # Update peak value and drawdown
        total_value = self._calculate_total_value(current_price, market)
        if total_value > self.peak_value:
            self.peak_value = total_value

        # Record trade
        trade_record = {
            "id": trade_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair_info.get("pair", market),
            "market": market,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "price": fill_price,
            "fee": fee,
            "realized_pnl": pnl,
            "is_paper": True,
            "status": "filled",
            "reasoning": reasoning,
            "agent_signals": agent_signals or {},
        }
        self.trade_store.record_trade(trade_record)

        log.info(
            f"📝 Paper trade executed: {side.upper()} {quantity} {target_currency} "
            f"@ ₹{fill_price:,.2f} | Fee: ₹{fee:,.2f} | PnL: ₹{pnl:,.2f}"
        )

        return {
            "order_id": trade_id,
            "status": "filled",
            "fill_price": fill_price,
            "fill_quantity": quantity,
            "fee": fee,
            "realized_pnl": round(pnl, 2),
            "is_paper": True,
            "timestamp": trade_record["timestamp"],
            "error": "",
        }

    # =========================================================================
    # Portfolio State
    # =========================================================================

    def get_portfolio_state(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Get current portfolio state with unrealized PnL.

        Args:
            current_prices: Dict of market → current price for unrealized PnL calc.

        Returns:
            Portfolio state dict.
        """
        unrealized = 0.0
        if current_prices:
            for currency, pos in self.positions.items():
                # Find the market for this currency
                for market, info in SUPPORTED_PAIRS.items():
                    if info["target_currency"] == currency and market in current_prices:
                        current = current_prices[market]
                        entry = pos["avg_entry_price"]
                        unrealized += (current - entry) * pos["quantity"]
                        break

        total_inr = self.balances.get("INR", 0)
        if current_prices:
            for currency, pos in self.positions.items():
                for market, info in SUPPORTED_PAIRS.items():
                    if info["target_currency"] == currency and market in current_prices:
                        total_inr += pos["quantity"] * current_prices[market]
                        break

        drawdown = 0.0
        if self.peak_value > 0:
            drawdown = (self.peak_value - total_inr) / self.peak_value * 100

        return {
            "balances": self.balances.copy(),
            "positions": {k: v.copy() for k, v in self.positions.items()},
            "total_value_inr": round(total_inr, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_fees": round(self.total_fees, 2),
            "peak_value": round(self.peak_value, 2),
            "max_drawdown_pct": round(max(drawdown, 0), 2),
            "total_trades": len(self.trade_store.get_trade_history(limit=10000)),
        }

    def _calculate_total_value(self, current_price: float, market: str) -> float:
        """Calculate total portfolio value in INR."""
        total = self.balances.get("INR", 0)
        pair_info = SUPPORTED_PAIRS.get(market, {})
        target = pair_info.get("target_currency", "")
        if target in self.positions:
            total += self.positions[target]["quantity"] * current_price
        return total

    def save_snapshot(self, current_prices: Optional[Dict[str, float]] = None) -> None:
        """Save current portfolio state as a snapshot."""
        state = self.get_portfolio_state(current_prices)
        state["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.trade_store.save_portfolio_snapshot(state)
        log.debug(f"Portfolio snapshot saved: ₹{state['total_value_inr']:,.2f}")
