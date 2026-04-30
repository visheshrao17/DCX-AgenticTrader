"""
DCX-AgenticTrader — Error Handling

Custom exceptions, retry decorator with exponential backoff,
and circuit breaker for API resilience.
"""

import time
import functools
from typing import Type, Tuple, Optional, Callable, Any

from utils.logger import get_agent_logger

log = get_agent_logger("error_handler")


# =============================================================================
# Custom Exceptions
# =============================================================================

class DCXBaseError(Exception):
    """Base exception for all DCX-AgenticTrader errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class CoinDCXAPIError(DCXBaseError):
    """Raised when CoinDCX API returns an error response."""

    def __init__(self, message: str, status_code: int = 0, response: Optional[dict] = None):
        self.status_code = status_code
        self.response = response or {}
        super().__init__(message, {"status_code": status_code, "response": self.response})


class AuthenticationError(CoinDCXAPIError):
    """Raised when API authentication fails (invalid key/secret)."""
    pass


class RateLimitError(CoinDCXAPIError):
    """Raised when API rate limit is exceeded."""
    pass


class InsufficientBalanceError(DCXBaseError):
    """Raised when account balance is insufficient for a trade."""
    pass


class ComplianceViolationError(DCXBaseError):
    """Raised when a trade violates Indian regulatory compliance rules."""
    pass


class RiskLimitExceededError(DCXBaseError):
    """Raised when a trade exceeds configured risk limits."""
    pass


class MarketDataError(DCXBaseError):
    """Raised when market data fetch fails or returns invalid data."""
    pass


class WebSocketError(DCXBaseError):
    """Raised when WebSocket connection fails or drops."""
    pass


class PaperTradingError(DCXBaseError):
    """Raised when paper trading simulation encounters an error."""
    pass


# =============================================================================
# Retry Decorator with Exponential Backoff
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        CoinDCXAPIError,
        RateLimitError,
        MarketDataError,
        ConnectionError,
        TimeoutError,
    ),
    on_retry: Optional[Callable] = None,
):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before first retry.
        backoff_factor: Multiplier for delay after each retry.
        max_delay: Maximum delay between retries.
        retryable_exceptions: Tuple of exception types that trigger retry.
        on_retry: Optional callback called on each retry with (attempt, exception, delay).

    Usage:
        @retry_with_backoff(max_retries=3)
        def fetch_data():
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        log.error(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise

                    # Special handling for rate limits — wait longer
                    if isinstance(e, RateLimitError):
                        delay = max(delay, 5.0)

                    log.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} for {func.__name__} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(attempt + 1, e, delay)

                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            raise last_exception  # Should never reach here, but just in case

        return wrapper

    return decorator


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreaker:
    """
    Simple circuit breaker to prevent hammering a failing service.

    States:
        CLOSED  — normal operation, requests go through
        OPEN    — too many failures, requests are blocked
        HALF_OPEN — testing if service recovered

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        @breaker
        def call_api():
            ...
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        if self._state == self.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = self.HALF_OPEN
                log.info(f"Circuit breaker [{self.name}] → HALF_OPEN (testing recovery)")
        return self._state

    def record_success(self) -> None:
        """Record a successful call — reset failure count."""
        self._failure_count = 0
        if self._state != self.CLOSED:
            log.info(f"Circuit breaker [{self.name}] → CLOSED (recovered)")
        self._state = self.CLOSED

    def record_failure(self) -> None:
        """Record a failed call — potentially trip the breaker."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            log.warning(
                f"Circuit breaker [{self.name}] → OPEN "
                f"({self._failure_count} failures, blocking for {self.recovery_timeout}s)"
            )

    def __call__(self, func):
        """Use as a decorator."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == self.OPEN:
                raise WebSocketError(
                    f"Circuit breaker [{self.name}] is OPEN — service unavailable. "
                    f"Will retry in {self.recovery_timeout - (time.time() - self._last_failure_time):.0f}s"
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        return wrapper
