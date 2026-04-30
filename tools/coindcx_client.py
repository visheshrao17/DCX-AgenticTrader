"""
DCX-AgenticTrader — CoinDCX REST API Client

Full client with HMAC-SHA256 authentication for all CoinDCX endpoints.
Covers public market data (candles, orderbook, ticker) and authenticated
endpoints (orders, balances, user info).
"""

import hmac
import hashlib
import json
import time
from typing import Dict, Any, Optional, List

import requests
from pydantic import BaseModel

from config.constants import (
    COINDCX_BASE_URL,
    COINDCX_PUBLIC_URL,
    ENDPOINTS_PUBLIC,
    ENDPOINTS_AUTH,
    RATE_LIMIT_PUBLIC_PER_SEC,
    RATE_LIMIT_PRIVATE_PER_SEC,
)
from utils.logger import get_agent_logger
from utils.error_handler import (
    CoinDCXAPIError,
    AuthenticationError,
    RateLimitError,
    MarketDataError,
    retry_with_backoff,
)

log = get_agent_logger("coindcx_client")


# =============================================================================
# Response Models
# =============================================================================

class TickerData(BaseModel):
    """Ticker data for a single market."""
    market: str
    last_price: str
    bid: str
    ask: str
    high: str
    low: str
    volume: str
    timestamp: int = 0


class CandleData(BaseModel):
    """Single OHLCV candle."""
    time: int  # Unix timestamp in ms
    open: float
    high: float
    low: float
    close: float
    volume: float


class OrderBookEntry(BaseModel):
    """Single orderbook level."""
    price: str
    quantity: str


class OrderBookData(BaseModel):
    """Full orderbook snapshot."""
    bids: List[OrderBookEntry]
    asks: List[OrderBookEntry]


class OrderResponse(BaseModel):
    """Response from order creation/cancellation."""
    id: str = ""
    market: str = ""
    side: str = ""
    order_type: str = ""
    status: str = ""
    total_quantity: float = 0.0
    price_per_unit: float = 0.0
    fee: float = 0.0
    fee_amount: float = 0.0
    avg_price: float = 0.0
    remaining_quantity: float = 0.0
    created_at: str = ""


class BalanceEntry(BaseModel):
    """Single currency balance."""
    currency: str
    balance: str
    locked_balance: str


# =============================================================================
# CoinDCX Client
# =============================================================================

class CoinDCXClient:
    """
    Full CoinDCX REST API client with HMAC-SHA256 authentication.

    Usage:
        # Public endpoints (no auth needed)
        client = CoinDCXClient()
        candles = client.get_candles("B-BTC_INR", "15m")

        # Authenticated endpoints
        client = CoinDCXClient(api_key="...", api_secret="...")
        balances = client.get_balances()
        order = client.place_order("BTCINR", "buy", "limit_order", 0.001, 7500000)
    """

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Simple rate limiter tracking
        self._last_public_call = 0.0
        self._last_private_call = 0.0

    # =========================================================================
    # Authentication
    # =========================================================================

    def _sign(self, payload: Dict[str, Any]) -> str:
        """
        Generate HMAC-SHA256 signature for authenticated requests.

        Args:
            payload: Request body as a dict (timestamp will be added).

        Returns:
            Hex-encoded HMAC-SHA256 signature.
        """
        json_payload = json.dumps(payload, separators=(",", ":"))
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            json_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _auth_headers(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Build authenticated request headers."""
        signature = self._sign(payload)
        return {
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-SIGNATURE": signature,
            "Content-Type": "application/json",
        }

    def _ensure_auth(self) -> None:
        """Raise if API credentials are not configured."""
        if not self.api_key or not self.api_secret:
            raise AuthenticationError(
                "CoinDCX API key and secret are required for authenticated endpoints. "
                "Set COINDCX_API_KEY and COINDCX_API_SECRET in .env"
            )

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    def _rate_limit(self, is_private: bool = False) -> None:
        """Simple rate limiter — sleep if calling too fast."""
        now = time.time()
        if is_private:
            min_interval = 1.0 / RATE_LIMIT_PRIVATE_PER_SEC
            elapsed = now - self._last_private_call
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_private_call = time.time()
        else:
            min_interval = 1.0 / RATE_LIMIT_PUBLIC_PER_SEC
            elapsed = now - self._last_public_call
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_public_call = time.time()

    # =========================================================================
    # HTTP Methods
    # =========================================================================

    def _get(self, url: str, params: Optional[Dict] = None) -> Any:
        """Make a GET request to a public endpoint."""
        self._rate_limit(is_private=False)
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded", status_code=429)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise MarketDataError(f"GET request failed: {url} — {e}")

    def _post(self, url: str, payload: Dict[str, Any]) -> Any:
        """Make an authenticated POST request."""
        self._ensure_auth()
        self._rate_limit(is_private=True)

        # Add timestamp
        payload["timestamp"] = int(round(time.time() * 1000))
        headers = self._auth_headers(payload)
        json_body = json.dumps(payload, separators=(",", ":"))

        try:
            response = self.session.post(url, data=json_body, headers=headers, timeout=10)

            if response.status_code == 401:
                raise AuthenticationError(
                    "Authentication failed — check your API key and secret",
                    status_code=401,
                )
            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded", status_code=429)
            if response.status_code >= 400:
                error_body = {}
                try:
                    error_body = response.json()
                except Exception:
                    pass
                raise CoinDCXAPIError(
                    f"API error {response.status_code}: {error_body}",
                    status_code=response.status_code,
                    response=error_body,
                )

            return response.json()
        except requests.exceptions.RequestException as e:
            if isinstance(e, (CoinDCXAPIError, AuthenticationError, RateLimitError)):
                raise
            raise CoinDCXAPIError(f"POST request failed: {url} — {e}")

    # =========================================================================
    # Public Endpoints (No Auth)
    # =========================================================================

    @retry_with_backoff(max_retries=3)
    def get_ticker(self) -> List[Dict]:
        """
        Get ticker data for all markets.

        Returns:
            List of ticker dicts with market, last_price, bid, ask, high, low, volume.
        """
        url = f"{COINDCX_BASE_URL}{ENDPOINTS_PUBLIC['ticker']}"
        data = self._get(url)
        log.debug(f"Fetched ticker data for {len(data)} markets")
        return data

    @retry_with_backoff(max_retries=3)
    def get_ticker_for_market(self, market: str) -> Optional[Dict]:
        """
        Get ticker data for a specific market.

        Args:
            market: Market identifier (e.g., "BTCINR").

        Returns:
            Ticker dict or None if not found.
        """
        tickers = self.get_ticker()
        for t in tickers:
            if t.get("market") == market:
                return t
        log.warning(f"Market {market} not found in ticker data")
        return None

    @retry_with_backoff(max_retries=3)
    def get_markets(self) -> List[str]:
        """Get list of all active market names."""
        url = f"{COINDCX_BASE_URL}{ENDPOINTS_PUBLIC['markets']}"
        data = self._get(url)
        log.debug(f"Fetched {len(data)} active markets")
        return data

    @retry_with_backoff(max_retries=3)
    def get_markets_details(self) -> List[Dict]:
        """
        Get detailed info for all markets.

        Returns:
            List of dicts with min/max quantity, price precision, pair info, etc.
        """
        url = f"{COINDCX_BASE_URL}{ENDPOINTS_PUBLIC['markets_details']}"
        data = self._get(url)
        log.debug(f"Fetched details for {len(data)} markets")
        return data

    @retry_with_backoff(max_retries=3)
    def get_candles(
        self,
        pair: str,
        interval: str = "15m",
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get OHLCV candlestick data.

        Args:
            pair: CoinDCX pair format (e.g., "B-BTC_INR").
            interval: Candle interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w).
            limit: Number of candles to fetch (max varies by interval).

        Returns:
            List of candle dicts: [{"open", "high", "low", "close", "volume", "time"}, ...]
        """
        url = f"{COINDCX_PUBLIC_URL}{ENDPOINTS_PUBLIC['candles']}"
        params = {"pair": pair, "interval": interval, "limit": limit}
        data = self._get(url, params=params)
        log.debug(f"Fetched {len(data)} candles for {pair} @ {interval}")
        return data

    @retry_with_backoff(max_retries=3)
    def get_orderbook(self, pair: str) -> Dict:
        """
        Get current orderbook (bids and asks).

        Args:
            pair: CoinDCX pair format (e.g., "B-BTC_INR").

        Returns:
            Dict with "bids" and "asks" lists.
        """
        url = f"{COINDCX_PUBLIC_URL}{ENDPOINTS_PUBLIC['orderbook']}"
        params = {"pair": pair}
        data = self._get(url, params=params)
        log.debug(
            f"Fetched orderbook for {pair}: "
            f"{len(data.get('bids', []))} bids, {len(data.get('asks', []))} asks"
        )
        return data

    @retry_with_backoff(max_retries=3)
    def get_trades(self, pair: str, limit: int = 50) -> List[Dict]:
        """
        Get recent trade history for a pair.

        Args:
            pair: CoinDCX pair format (e.g., "B-BTC_INR").
            limit: Number of recent trades.

        Returns:
            List of trade dicts.
        """
        url = f"{COINDCX_PUBLIC_URL}{ENDPOINTS_PUBLIC['trades']}"
        params = {"pair": pair, "limit": limit}
        data = self._get(url, params=params)
        log.debug(f"Fetched {len(data)} recent trades for {pair}")
        return data

    # =========================================================================
    # Authenticated Endpoints
    # =========================================================================

    @retry_with_backoff(max_retries=2)
    def get_balances(self) -> List[Dict]:
        """
        Get account balances for all currencies.

        Returns:
            List of balance dicts: [{"currency", "balance", "locked_balance"}, ...]
        """
        url = f"{COINDCX_BASE_URL}{ENDPOINTS_AUTH['balances']}"
        data = self._post(url, {})
        log.info(f"Fetched balances for {len(data)} currencies")
        return data

    @retry_with_backoff(max_retries=2)
    def get_user_info(self) -> Dict:
        """Get user account information."""
        url = f"{COINDCX_BASE_URL}{ENDPOINTS_AUTH['user_info']}"
        data = self._post(url, {})
        log.info("Fetched user info")
        return data

    @retry_with_backoff(max_retries=2)
    def place_order(
        self,
        market: str,
        side: str,
        order_type: str,
        total_quantity: float,
        price_per_unit: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Dict:
        """
        Place a new order on CoinDCX.

        Args:
            market: Market name (e.g., "BTCINR").
            side: "buy" or "sell".
            order_type: "market_order", "limit_order", or "stop_limit".
            total_quantity: Amount to trade.
            price_per_unit: Price for limit orders (required for limit_order).
            stop_price: Stop/trigger price (required for stop_limit).

        Returns:
            Order response dict with order ID, status, etc.
        """
        url = f"{COINDCX_BASE_URL}{ENDPOINTS_AUTH['create_order']}"
        payload: Dict[str, Any] = {
            "market": market,
            "side": side,
            "order_type": order_type,
            "total_quantity": total_quantity,
        }

        if price_per_unit is not None:
            payload["price_per_unit"] = price_per_unit
        if stop_price is not None:
            payload["stop_price"] = stop_price

        log.info(
            f"Placing {order_type} {side} order: {total_quantity} on {market}"
            + (f" @ ₹{price_per_unit}" if price_per_unit else "")
        )

        data = self._post(url, payload)
        log.info(f"Order placed successfully: {data.get('id', 'unknown')}")
        return data

    @retry_with_backoff(max_retries=2)
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an active order.

        Args:
            order_id: The order ID to cancel.

        Returns:
            Cancellation response dict.
        """
        url = f"{COINDCX_BASE_URL}{ENDPOINTS_AUTH['cancel_order']}"
        payload = {"id": order_id}
        data = self._post(url, payload)
        log.info(f"Order {order_id} cancelled")
        return data

    @retry_with_backoff(max_retries=2)
    def cancel_all_orders(self, market: Optional[str] = None) -> Dict:
        """
        Cancel all active orders, optionally filtered by market.

        Args:
            market: Optional market filter (e.g., "BTCINR").

        Returns:
            Cancellation response dict.
        """
        url = f"{COINDCX_BASE_URL}{ENDPOINTS_AUTH['cancel_all']}"
        payload: Dict[str, Any] = {}
        if market:
            payload["market"] = market
        data = self._post(url, payload)
        log.info(f"All orders cancelled" + (f" for {market}" if market else ""))
        return data

    @retry_with_backoff(max_retries=2)
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get status of a specific order.

        Args:
            order_id: The order ID.

        Returns:
            Order status dict.
        """
        url = f"{COINDCX_BASE_URL}{ENDPOINTS_AUTH['order_status']}"
        payload = {"id": order_id}
        return self._post(url, payload)

    @retry_with_backoff(max_retries=2)
    def get_active_orders(self, market: Optional[str] = None) -> List[Dict]:
        """
        Get all active (open) orders.

        Args:
            market: Optional market filter.

        Returns:
            List of active order dicts.
        """
        url = f"{COINDCX_BASE_URL}{ENDPOINTS_AUTH['active_orders']}"
        payload: Dict[str, Any] = {}
        if market:
            payload["market"] = market
        data = self._post(url, payload)
        log.debug(f"Fetched {len(data)} active orders")
        return data

    @retry_with_backoff(max_retries=2)
    def get_trade_history(
        self,
        market: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Get account trade history.

        Args:
            market: Optional market filter.
            limit: Number of trades to fetch.

        Returns:
            List of trade history dicts.
        """
        url = f"{COINDCX_BASE_URL}{ENDPOINTS_AUTH['trade_history']}"
        payload: Dict[str, Any] = {"limit": limit}
        if market:
            payload["market"] = market
        data = self._post(url, payload)
        log.debug(f"Fetched {len(data)} trade history entries")
        return data

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_best_bid_ask(self, pair: str) -> Dict[str, float]:
        """
        Get the best bid and ask prices from the orderbook.

        Args:
            pair: CoinDCX pair format.

        Returns:
            {"best_bid": float, "best_ask": float, "spread": float}
        """
        orderbook = self.get_orderbook(pair)
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 0.0
        spread = best_ask - best_bid if best_bid and best_ask else 0.0

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": (spread / best_bid * 100) if best_bid else 0.0,
        }

    def get_balance_for_currency(self, currency: str) -> Dict[str, str]:
        """
        Get balance for a specific currency.

        Args:
            currency: Currency code (e.g., "INR", "BTC", "USDT").

        Returns:
            {"currency": str, "balance": str, "locked_balance": str}
        """
        balances = self.get_balances()
        for b in balances:
            if b.get("currency", "").upper() == currency.upper():
                return b
        return {"currency": currency, "balance": "0", "locked_balance": "0"}
