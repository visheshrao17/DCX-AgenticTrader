"""
DCX-AgenticTrader — Backtesting Engine

Replays historical candle data through the agent pipeline,
simulates fills, and tracks performance vs Buy-&-Hold.
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import pandas as pd

from config.constants import SUPPORTED_PAIRS
from tools.coindcx_client import CoinDCXClient
from tools.technical_indicators import generate_signal_summary
from agents.market_data import candles_to_dataframe
from memory.paper_trading import PaperTradingEngine
from memory.trade_store import TradeStore
from backtesting.metrics import calculate_metrics
from utils.logger import get_agent_logger

log = get_agent_logger("backtest")


class BacktestEngine:
    """
    Historical backtesting engine.

    Replays candle data through technical analysis → risk check → decision pipeline.
    Tracks equity curve and compares against Buy-&-Hold benchmark.

    Usage:
        engine = BacktestEngine("BTCINR", initial_capital=100000)
        results = engine.run(months=6)
    """

    def __init__(self, market: str = "BTCINR", initial_capital: float = 100000):
        self.market = market
        self.pair_info = SUPPORTED_PAIRS.get(market, {})
        self.pair = self.pair_info.get("pair", "B-BTC_INR")
        self.initial_capital = initial_capital

        self.client = CoinDCXClient()
        self.trade_store = TradeStore(db_path="data/backtest_trades.db")
        self.paper_engine = PaperTradingEngine(
            initial_capital=initial_capital,
            trade_store=self.trade_store,
        )

        # Results tracking
        self.equity_curve: List[Dict] = []
        self.trades: List[Dict] = []
        self.buy_hold_curve: List[Dict] = []

    def run(self, months: int = 6, interval: str = "1h") -> Dict[str, Any]:
        """
        Run backtest over historical data.

        Args:
            months: Number of months of historical data.
            interval: Candle interval for analysis.

        Returns:
            Dict with performance metrics, equity curve, and trade list.
        """
        log.info(f"Starting backtest: {self.market}, {months} months, {interval} interval")

        # Fetch historical candles
        log.info("Fetching historical data...")
        candles = self._fetch_historical(interval, months)
        if not candles or len(candles) < 50:
            log.error(f"Insufficient data: {len(candles) if candles else 0} candles")
            return {"error": "Insufficient historical data"}

        df = candles_to_dataframe(candles)
        log.info(f"Loaded {len(df)} candles from {interval} timeframe")

        # Buy & Hold benchmark
        start_price = float(df.iloc[0]["close"])
        bh_quantity = self.initial_capital / start_price

        # Sliding window analysis
        window_size = 50
        analysis_interval = 5  # Analyze every N candles

        for i in range(window_size, len(df), analysis_interval):
            window = df.iloc[i - window_size:i].copy().reset_index(drop=True)
            current = df.iloc[i]
            current_price = float(current["close"])
            timestamp = current.get("time", i)

            # Run technical analysis
            signals = generate_signal_summary(window)

            # Simple decision logic (mirrors orchestrator but simplified)
            action = self._decide(signals, current_price)

            if action != "HOLD":
                self._execute_backtest_trade(action, current_price, signals)

            # Track equity curve
            portfolio = self.paper_engine.get_portfolio_state(
                current_prices={self.market: current_price}
            )
            self.equity_curve.append({
                "step": i,
                "timestamp": timestamp,
                "price": current_price,
                "portfolio_value": portfolio["total_value_inr"],
                "realized_pnl": portfolio["realized_pnl"],
            })

            # Buy & Hold curve
            self.buy_hold_curve.append({
                "step": i,
                "timestamp": timestamp,
                "value": bh_quantity * current_price,
            })

        # Calculate metrics
        final_portfolio = self.paper_engine.get_portfolio_state(
            current_prices={self.market: float(df.iloc[-1]["close"])}
        )
        bh_final = bh_quantity * float(df.iloc[-1]["close"])

        metrics = calculate_metrics(
            equity_curve=self.equity_curve,
            initial_capital=self.initial_capital,
            buy_hold_final=bh_final,
        )

        trades = self.trade_store.get_trade_history(limit=10000)

        results = {
            "market": self.market,
            "interval": interval,
            "months": months,
            "total_candles": len(df),
            "start_price": start_price,
            "end_price": float(df.iloc[-1]["close"]),
            "metrics": metrics,
            "portfolio_final": final_portfolio,
            "buy_hold_final": round(bh_final, 2),
            "equity_curve": self.equity_curve,
            "buy_hold_curve": self.buy_hold_curve,
            "total_trades": len(trades),
            "trades": trades[:50],  # Last 50
        }

        log.info(
            f"Backtest complete: Return={metrics['total_return_pct']:.1f}% vs "
            f"B&H={metrics['buy_hold_return_pct']:.1f}%, "
            f"Drawdown={metrics['max_drawdown_pct']:.1f}%, "
            f"Trades={len(trades)}"
        )

        return results

    def _fetch_historical(self, interval: str, months: int) -> List[Dict]:
        """Fetch historical candles from CoinDCX."""
        try:
            # CoinDCX limit per request varies; fetch max
            limit = min(months * 30 * 24, 500)  # Rough estimate
            candles = self.client.get_candles(self.pair, interval, limit=limit)
            return candles
        except Exception as e:
            log.error(f"Failed to fetch historical data: {e}")
            return []

    def _decide(self, signals: Dict, current_price: float) -> str:
        """Simple signal-based decision logic for backtesting."""
        score = signals.get("signal_score", 0)
        confidence = signals.get("confidence", 0)

        if score >= 0.3 and confidence >= 0.5:
            # Check if we already have a position
            target = self.pair_info.get("target_currency", "BTC")
            if target not in self.paper_engine.positions:
                return "BUY"
        elif score <= -0.3 and confidence >= 0.5:
            target = self.pair_info.get("target_currency", "BTC")
            if target in self.paper_engine.positions:
                return "SELL"

        return "HOLD"

    def _execute_backtest_trade(self, action: str, price: float, signals: Dict):
        """Execute a backtest trade through the paper engine."""
        target = self.pair_info.get("target_currency", "BTC")
        side = "buy" if action == "BUY" else "sell"

        if side == "buy":
            # Use 10% of available INR
            available_inr = self.paper_engine.balances.get("INR", 0)
            trade_value = available_inr * 0.10
            quantity = trade_value / price
        else:
            quantity = self.paper_engine.positions.get(target, {}).get("quantity", 0)

        if quantity <= 0:
            return

        self.paper_engine.execute_order(
            market=self.market,
            side=side,
            order_type="market_order",
            quantity=quantity,
            current_price=price,
            reasoning=f"Backtest {action}: score={signals.get('signal_score', 0):.2f}",
        )
