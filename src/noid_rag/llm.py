"""LLM client for answer synthesis via OpenAI-compatible API."""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from noid_rag.circuit_breaker import CircuitBreaker
from noid_rag.config import LLMConfig


class LLMClient:
    """OpenAI-compatible chat completions client."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        if not self.config.api_key.get_secret_value():
            raise ValueError(
                "No LLM API key configured. "
                "Set NOID_RAG_LLM__API_KEY in your .env file or environment."
            )
        self._client: httpx.AsyncClient | None = None
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5, cooldown_seconds=30.0, service_name="llm-api"
        )

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared httpx client, creating it lazily."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
        ),
    )
    async def generate(self, prompt: str, context: str) -> str:
        """Generate an answer given a prompt and context."""
        self.circuit_breaker.check()
        user_message = f"Context:\n{context}\n\nQuestion: {prompt}"
        client = self._get_client()
        try:
            resp = await client.post(
                self.config.api_url,
                json={
                    "model": self.config.model,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "messages": [
                        {"role": "system", "content": self.config.system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                },
                headers={
                    "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException):
            self.circuit_breaker.record_failure()
            raise
        self.circuit_breaker.record_success()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            # Truncate the response body to avoid echoing large or sensitive
            # content (e.g. user prompt reflected by some providers) into
            # tracebacks and log aggregators.
            preview = repr(data)[:200]
            raise ValueError(f"Unexpected LLM response shape: {preview}...") from exc

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
