"""Circuit breaker for external API calls.

Tracks consecutive failures and opens the circuit after a threshold is reached.
While open, calls fail immediately without hitting the remote service.
After a cooldown period, the circuit enters half-open state and allows one probe request.
"""

from __future__ import annotations

import time
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and rejecting calls."""

    def __init__(self, service: str, retry_after: float):
        self.service = service
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker open for {service!r}. Retry after {retry_after:.1f}s.")


class CircuitBreaker:
    """Simple circuit breaker with configurable thresholds.

    Args:
        failure_threshold: Number of consecutive failures before opening.
        cooldown_seconds: Seconds to wait before transitioning to half-open.
        service_name: Identifier for logging/error messages.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        service_name: str = "external-api",
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.service_name = service_name

        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        """Current circuit state, accounting for cooldown expiry."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.cooldown_seconds:
                return CircuitState.HALF_OPEN
        return self._state

    def check(self) -> None:
        """Check if the circuit allows a request. Raises CircuitOpenError if not."""
        current = self.state
        if current == CircuitState.OPEN:
            retry_after = self.cooldown_seconds - (time.monotonic() - self._opened_at)
            raise CircuitOpenError(self.service_name, max(0.0, retry_after))
        # CLOSED and HALF_OPEN both allow requests

    def record_success(self) -> None:
        """Record a successful call. Resets the circuit to CLOSED."""
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call. Opens the circuit if threshold is reached."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at = 0.0
