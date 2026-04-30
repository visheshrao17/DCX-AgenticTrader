"""
DCX-AgenticTrader — CoinDCX WebSocket Client

Real-time data streaming via Socket.IO for live prices,
orderbook depth, new trades, and candlestick updates.
CoinDCX uses Socket.IO v2.x protocol.
"""

import threading
from typing import Callable, Optional, Dict, List

import socketio

from config.constants import COINDCX_SOCKET_URL
from utils.logger import get_agent_logger
from utils.error_handler import WebSocketError, CircuitBreaker

log = get_agent_logger("websocket")


class CoinDCXWebSocket:
    """
    Real-time WebSocket client for CoinDCX using Socket.IO.

    Usage:
        ws = CoinDCXWebSocket()
        ws.on_price_update(lambda data: print(data))
        ws.connect()
        ws.subscribe_to_pair("B-BTC_INR")
    """

    def __init__(self):
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=10,
            reconnection_delay=1,
            reconnection_delay_max=30,
            logger=False,
            engineio_logger=False,
        )
        self._connected = False
        self._subscribed_pairs: List[str] = []
        self._price_callbacks: List[Callable] = []
        self._trade_callbacks: List[Callable] = []
        self._depth_callbacks: List[Callable] = []
        self._candle_callbacks: List[Callable] = []
        self._latest_prices: Dict[str, Dict] = {}
        self._latest_depth: Dict[str, Dict] = {}
        self._breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60, name="coindcx_ws")
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all Socket.IO event handlers."""

        @self.sio.event
        def connect():
            self._connected = True
            self._breaker.record_success()
            log.info(f"WebSocket connected to {COINDCX_SOCKET_URL}")

        @self.sio.event
        def disconnect():
            self._connected = False
            log.warning("WebSocket disconnected")

        @self.sio.event
        def connect_error(data):
            self._breaker.record_failure()
            log.error(f"WebSocket connection error: {data}")

        @self.sio.on("price-change")
        def on_price_change(data):
            try:
                if isinstance(data, dict):
                    pair = data.get("s", "")
                    self._latest_prices[pair] = data
                    for cb in self._price_callbacks:
                        cb(data)
            except Exception as e:
                log.error(f"Error in price-change handler: {e}")

        @self.sio.on("new-trade")
        def on_new_trade(data):
            try:
                for cb in self._trade_callbacks:
                    cb(data)
            except Exception as e:
                log.error(f"Error in new-trade handler: {e}")

        @self.sio.on("depth-update")
        def on_depth_update(data):
            try:
                if isinstance(data, dict):
                    pair = data.get("s", "")
                    self._latest_depth[pair] = data
                    for cb in self._depth_callbacks:
                        cb(data)
            except Exception as e:
                log.error(f"Error in depth-update handler: {e}")

        @self.sio.on("candlestick")
        def on_candlestick(data):
            try:
                for cb in self._candle_callbacks:
                    cb(data)
            except Exception as e:
                log.error(f"Error in candlestick handler: {e}")

    def on_price_update(self, callback: Callable) -> None:
        self._price_callbacks.append(callback)

    def on_new_trade(self, callback: Callable) -> None:
        self._trade_callbacks.append(callback)

    def on_depth_update(self, callback: Callable) -> None:
        self._depth_callbacks.append(callback)

    def on_candle_update(self, callback: Callable) -> None:
        self._candle_callbacks.append(callback)

    def connect(self) -> None:
        """Connect to CoinDCX WebSocket server."""
        if self._connected:
            log.info("Already connected")
            return
        try:
            log.info(f"Connecting to {COINDCX_SOCKET_URL}...")
            self.sio.connect(COINDCX_SOCKET_URL, transports=["websocket"], wait=True, wait_timeout=10)
        except Exception as e:
            self._breaker.record_failure()
            raise WebSocketError(f"Failed to connect: {e}")

    def disconnect(self) -> None:
        """Disconnect from CoinDCX WebSocket server."""
        if self._connected:
            self.sio.disconnect()
            self._connected = False
            self._subscribed_pairs.clear()
            log.info("Disconnected gracefully")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def subscribe_to_pair(self, pair: str) -> None:
        """Subscribe to all data channels for a trading pair."""
        if not self._connected:
            raise WebSocketError("Not connected. Call connect() first.")
        if pair in self._subscribed_pairs:
            return
        for channel in [f"price-change@{pair}", f"new-trade@{pair}", f"depth-update@{pair}"]:
            self.sio.emit("join", {"channelName": channel})
        self._subscribed_pairs.append(pair)
        log.info(f"Subscribed to {pair}")

    def unsubscribe_from_pair(self, pair: str) -> None:
        if pair not in self._subscribed_pairs:
            return
        for channel in [f"price-change@{pair}", f"new-trade@{pair}", f"depth-update@{pair}"]:
            self.sio.emit("leave", {"channelName": channel})
        self._subscribed_pairs.remove(pair)
        log.info(f"Unsubscribed from {pair}")

    def subscribe_to_multiple(self, pairs: List[str]) -> None:
        for pair in pairs:
            self.subscribe_to_pair(pair)

    def get_latest_price(self, pair: str) -> Optional[Dict]:
        return self._latest_prices.get(pair)

    def get_latest_depth(self, pair: str) -> Optional[Dict]:
        return self._latest_depth.get(pair)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
