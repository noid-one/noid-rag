"""Tests for the circuit breaker module."""

from __future__ import annotations

import time

import pytest

from noid_rag.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


class TestCircuitBreakerInit:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_custom_thresholds(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=10.0, service_name="test")
        assert cb.failure_threshold == 3
        assert cb.cooldown_seconds == 10.0
        assert cb.service_name == "test"


class TestCircuitBreakerTransitions:
    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_success_in_half_open_closes(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerCheck:
    def test_check_passes_when_closed(self):
        cb = CircuitBreaker()
        cb.check()  # Should not raise

    def test_check_raises_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0, service_name="embedding")
        cb.record_failure()
        with pytest.raises(CircuitOpenError, match="embedding") as exc_info:
            cb.check()
        assert exc_info.value.retry_after > 0

    def test_check_passes_when_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.check()  # Should not raise in half-open


class TestCircuitBreakerReset:
    def test_reset_closes_open_circuit(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED


class TestCircuitOpenError:
    def test_error_attributes(self):
        err = CircuitOpenError("test-service", 15.5)
        assert err.service == "test-service"
        assert err.retry_after == 15.5
        assert "test-service" in str(err)
        assert "15.5" in str(err)
