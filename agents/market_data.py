"""
DCX-AgenticTrader — Market Data Agent

Fetches live price, candle, and orderbook data from CoinDCX.
Cleans and structures data into the MarketSnapshot format
for consumption by downstream agents.
"""

from datetime import datetime, timezone
from typing import Dict, Any

import pandas as pd
from langchain_core.tools import tool

from config.constants import SUPPORTED_PAIRS, DEFAULT_CANDLE_INTERVAL
from tools.coindcx_client import CoinDCXClient
from graph.state import TradingState, MarketSnapshot
from utils.logger import get_agent_logger

log = get_agent_logger("market_data")


# =============================================================================
# LangGraph Tools (callable by the agent via tool-calling)
# =============================================================================

_client = CoinDCXClient()


@tool
def fetch_market_snapshot(market: str) -> dict:
    """
    Fetch a complete market snapshot for a trading pair from CoinDCX.

    Args:
        market: Market identifier like 'BTCINR' or 'USDTINR'.

    Returns:
        Dict with last_price, bid, ask, 24h stats, candles, orderbook, and recent trades.
    """
    pair_info = SUPPORTED_PAIRS.get(market)
    if not pair_info:
        return {"error": f"Unsupported market: {market}. Supported: {list(SUPPORTED_PAIRS.keys())}"}

    pair = pair_info["pair"]
    log.info(f"Fetching market snapshot for {market} (pair: {pair})")

    try:
        # Fetch ticker for current price + 24h stats
        ticker = _client.get_ticker_for_market(market)

        # Fetch candles at multiple timeframes
        candles_15m = _client.get_candles(pair, "15m", limit=100)
        candles_1h = _client.get_candles(pair, "1h", limit=100)
        candles_4h = _client.get_candles(pair, "4h", limit=50)
        candles_1d = _client.get_candles(pair, "1d", limit=30)

        # Fetch orderbook
        orderbook = _client.get_orderbook(pair)

        # Fetch recent trades
        trades = _client.get_trades(pair, limit=30)

        # Get best bid/ask
        bids = orderbook.get("bids", {})
        asks = orderbook.get("asks", {})

        # Normalize bids/asks — CoinDCX returns them as dicts or lists
        bid_list = []
        ask_list = []
        if isinstance(bids, dict):
            bid_list = [{"price": k, "quantity": v} for k, v in sorted(bids.items(), key=lambda x: float(x[0]), reverse=True)[:20]]
        elif isinstance(bids, list):
            bid_list = bids[:20]

        if isinstance(asks, dict):
            ask_list = [{"price": k, "quantity": v} for k, v in sorted(asks.items(), key=lambda x: float(x[0]))[:20]]
        elif isinstance(asks, list):
            ask_list = asks[:20]

        best_bid = float(bid_list[0]["price"]) if bid_list else 0.0
        best_ask = float(ask_list[0]["price"]) if ask_list else 0.0

        snapshot = {
            "pair": market,
            "last_price": float(ticker.get("last_price", 0)) if ticker else 0.0,
            "bid": best_bid,
            "ask": best_ask,
            "high_24h": float(ticker.get("high", 0)) if ticker else 0.0,
            "low_24h": float(ticker.get("low", 0)) if ticker else 0.0,
            "volume_24h": float(ticker.get("volume", 0)) if ticker else 0.0,
            "spread_pct": ((best_ask - best_bid) / best_bid * 100) if best_bid else 0.0,
            "candles_15m": candles_15m[-20:] if candles_15m else [],
            "candles_1h": candles_1h[-20:] if candles_1h else [],
            "candles_4h": candles_4h[-20:] if candles_4h else [],
            "candles_1d": candles_1d[-20:] if candles_1d else [],
            "orderbook_bids": bid_list[:10],
            "orderbook_asks": ask_list[:10],
            "recent_trades": trades[:10] if trades else [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            f"Snapshot for {market}: price=₹{snapshot['last_price']:,.2f}, "
            f"spread={snapshot['spread_pct']:.4f}%, vol={snapshot['volume_24h']:,.2f}"
        )
        return snapshot

    except Exception as e:
        log.error(f"Failed to fetch snapshot for {market}: {e}")
        return {"error": str(e), "pair": market}


@tool
def get_current_price(market: str) -> dict:
    """
    Get just the current price and basic stats for a market.

    Args:
        market: Market identifier like 'BTCINR'.

    Returns:
        Dict with last_price, bid, ask, high, low, volume.
    """
    ticker = _client.get_ticker_for_market(market)
    if not ticker:
        return {"error": f"Could not fetch ticker for {market}"}

    return {
        "market": market,
        "last_price": float(ticker.get("last_price", 0)),
        "bid": float(ticker.get("bid", 0)),
        "ask": float(ticker.get("ask", 0)),
        "high_24h": float(ticker.get("high", 0)),
        "low_24h": float(ticker.get("low", 0)),
        "volume_24h": float(ticker.get("volume", 0)),
    }


@tool
def get_candle_dataframe(market: str, interval: str = "15m", limit: int = 100) -> str:
    """
    Fetch candle data and return a summary string for LLM consumption.

    Args:
        market: Market identifier like 'BTCINR'.
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d).
        limit: Number of candles.

    Returns:
        String summary of candle data with OHLCV stats.
    """
    pair_info = SUPPORTED_PAIRS.get(market)
    if not pair_info:
        return f"Unsupported market: {market}"

    candles = _client.get_candles(pair_info["pair"], interval, limit)
    if not candles:
        return f"No candle data for {market} @ {interval}"

    df = candles_to_dataframe(candles)
    if df.empty:
        return f"Empty candle data for {market}"

    latest = df.iloc[-1]
    summary = (
        f"Candles for {market} @ {interval} (last {len(df)} candles):\n"
        f"  Latest: O={latest['open']:.2f} H={latest['high']:.2f} "
        f"L={latest['low']:.2f} C={latest['close']:.2f} V={latest['volume']:.2f}\n"
        f"  Range: High={df['high'].max():.2f}, Low={df['low'].min():.2f}\n"
        f"  Avg Volume: {df['volume'].mean():.2f}\n"
        f"  Price Change (period): {((latest['close'] - df.iloc[0]['open']) / df.iloc[0]['open'] * 100):.2f}%"
    )
    return summary


# =============================================================================
# Helper Functions
# =============================================================================

def candles_to_dataframe(candles: list) -> pd.DataFrame:
    """
    Convert CoinDCX candle response to a clean pandas DataFrame.

    Args:
        candles: Raw candle data from CoinDCX API.

    Returns:
        DataFrame with columns: time, open, high, low, close, volume
    """
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)

    # CoinDCX candles may have different column names
    rename_map = {}
    if "time" in df.columns:
        rename_map["time"] = "time"
    elif "T" in df.columns:
        rename_map["T"] = "time"

    for col_map in [("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")]:
        short, full = col_map
        if short in df.columns and full not in df.columns:
            rename_map[short] = full

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by time ascending
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)

    return df


# =============================================================================
# Agent Node Function (called by LangGraph)
# =============================================================================

def market_data_agent(state: TradingState) -> Dict[str, Any]:
    """
    LangGraph node: Market Data Agent.

    Fetches complete market snapshot for the current trading pair
    and updates the state.
    """
    current_pair = state.get("current_pair", "BTCINR")
    log.info(f"=== Market Data Agent running for {current_pair} ===")

    try:
        snapshot = fetch_market_snapshot.invoke({"market": current_pair})

        if "error" in snapshot:
            log.error(f"Snapshot error: {snapshot['error']}")
            return {
                "market_data": snapshot,
                "current_step": "market_data",
                "error": snapshot["error"],
            }

        log.info(f"Market data collected: ₹{snapshot.get('last_price', 0):,.2f}")

        return {
            "market_data": snapshot,
            "current_step": "market_data",
            "error": None,
        }

    except Exception as e:
        log.error(f"Market Data Agent failed: {e}")
        return {
            "market_data": None,
            "current_step": "market_data",
            "error": str(e),
        }


# List of tools this agent exposes
MARKET_DATA_TOOLS = [fetch_market_snapshot, get_current_price, get_candle_dataframe]
