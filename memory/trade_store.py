"""
DCX-AgenticTrader — Trade Store (SQLite)

Persistent storage for all trades (paper + live), portfolio snapshots,
and trading session history. Uses SQLite for zero-config local storage.
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path

from config.constants import TRADE_DB_PATH
from utils.logger import get_agent_logger

log = get_agent_logger("trade_store")


class TradeStore:
    """
    SQLite-backed persistent store for trade history and portfolio snapshots.

    Usage:
        store = TradeStore()
        store.record_trade({...})
        history = store.get_trade_history(market="BTCINR", limit=50)
    """

    def __init__(self, db_path: str = TRADE_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Initialize database tables."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    market TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    is_paper INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'filled',
                    reasoning TEXT DEFAULT '',
                    agent_signals TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    balances TEXT NOT NULL,
                    positions TEXT NOT NULL,
                    total_value_inr REAL NOT NULL,
                    realized_pnl REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    max_drawdown_pct REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS agent_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cycle_id TEXT,
                    agent_name TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reasoning TEXT DEFAULT '',
                    data TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(pair);
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp);
            """)
            conn.commit()
            log.info(f"Trade store initialized at {self.db_path}")
        finally:
            conn.close()

    # =========================================================================
    # Trade Recording
    # =========================================================================

    def record_trade(self, trade: Dict[str, Any]) -> str:
        """
        Record a completed trade.

        Args:
            trade: Dict with pair, market, side, order_type, quantity, price, etc.

        Returns:
            Trade ID.
        """
        trade_id = trade.get("id", str(uuid.uuid4()))
        timestamp = trade.get("timestamp", datetime.now(timezone.utc).isoformat())

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO trades
                   (id, timestamp, pair, market, side, order_type, quantity, price,
                    fee, realized_pnl, is_paper, status, reasoning, agent_signals)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade_id,
                    timestamp,
                    trade.get("pair", ""),
                    trade.get("market", ""),
                    trade.get("side", ""),
                    trade.get("order_type", "market_order"),
                    trade.get("quantity", 0.0),
                    trade.get("price", 0.0),
                    trade.get("fee", 0.0),
                    trade.get("realized_pnl", 0.0),
                    1 if trade.get("is_paper", True) else 0,
                    trade.get("status", "filled"),
                    trade.get("reasoning", ""),
                    json.dumps(trade.get("agent_signals", {})),
                ),
            )
            conn.commit()
            log.info(f"Recorded trade {trade_id}: {trade.get('side')} {trade.get('quantity')} {trade.get('pair')} @ {trade.get('price')}")
            return trade_id
        finally:
            conn.close()

    def get_trade_history(
        self,
        market: Optional[str] = None,
        limit: int = 100,
        is_paper: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Get trade history, optionally filtered by market."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params: list = []

            if market:
                query += " AND market = ?"
                params.append(market)
            if is_paper is not None:
                query += " AND is_paper = ?"
                params.append(1 if is_paper else 0)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_trades_today(self) -> List[Dict[str, Any]]:
        """Get all trades from today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM trades WHERE timestamp LIKE ? ORDER BY timestamp DESC",
                (f"{today}%",),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def count_trades_today(self) -> int:
        """Count trades executed today."""
        return len(self.get_trades_today())

    # =========================================================================
    # Portfolio Snapshots
    # =========================================================================

    def save_portfolio_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Save a portfolio state snapshot."""
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO portfolio_snapshots
                   (timestamp, balances, positions, total_value_inr,
                    realized_pnl, unrealized_pnl, max_drawdown_pct)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    json.dumps(snapshot.get("balances", {})),
                    json.dumps(snapshot.get("positions", {})),
                    snapshot.get("total_value_inr", 0.0),
                    snapshot.get("realized_pnl", 0.0),
                    snapshot.get("unrealized_pnl", 0.0),
                    snapshot.get("max_drawdown_pct", 0.0),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_portfolio_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get portfolio snapshot history."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["balances"] = json.loads(d["balances"])
                d["positions"] = json.loads(d["positions"])
                result.append(d)
            return result
        finally:
            conn.close()

    # =========================================================================
    # Agent Decisions Log
    # =========================================================================

    def log_agent_decision(
        self,
        agent_name: str,
        decision: str,
        reasoning: str = "",
        data: Optional[Dict] = None,
        cycle_id: Optional[str] = None,
    ) -> None:
        """Log an agent's decision for audit trail."""
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO agent_decisions
                   (timestamp, cycle_id, agent_name, decision, reasoning, data)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    cycle_id or "",
                    agent_name,
                    decision,
                    reasoning,
                    json.dumps(data or {}),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_performance_stats(self) -> Dict[str, Any]:
        """Calculate overall trading performance statistics."""
        conn = self._get_conn()
        try:
            trades = conn.execute("SELECT * FROM trades ORDER BY timestamp").fetchall()
            if not trades:
                return {"total_trades": 0, "win_rate": 0, "total_pnl": 0}

            trades_list = [dict(t) for t in trades]
            total = len(trades_list)
            winners = sum(1 for t in trades_list if t["realized_pnl"] > 0)
            total_pnl = sum(t["realized_pnl"] for t in trades_list)
            total_fees = sum(t["fee"] for t in trades_list)

            winning_pnl = [t["realized_pnl"] for t in trades_list if t["realized_pnl"] > 0]
            losing_pnl = [t["realized_pnl"] for t in trades_list if t["realized_pnl"] < 0]

            return {
                "total_trades": total,
                "winning_trades": winners,
                "losing_trades": total - winners,
                "win_rate": round(winners / total * 100, 1) if total else 0,
                "total_pnl": round(total_pnl, 2),
                "total_fees": round(total_fees, 2),
                "avg_win": round(sum(winning_pnl) / len(winning_pnl), 2) if winning_pnl else 0,
                "avg_loss": round(sum(losing_pnl) / len(losing_pnl), 2) if losing_pnl else 0,
                "largest_win": round(max(winning_pnl), 2) if winning_pnl else 0,
                "largest_loss": round(min(losing_pnl), 2) if losing_pnl else 0,
            }
        finally:
            conn.close()
