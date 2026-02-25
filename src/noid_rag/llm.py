"""LLM client for answer synthesis via OpenAI-compatible API."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from noid_rag.config import LLMConfig

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based ONLY on the "
    "provided context. If the context doesn't contain enough information to answer, "
    "say so clearly. Be concise and direct."
)


class LLMClient:
    """OpenAI-compatible chat completions client."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        if not self.config.api_key.get_secret_value():
            raise ValueError(
                "No LLM API key configured. "
                "Set NOID_RAG_LLM__API_KEY in your .env file or environment."
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
        ),
    )
    async def generate(self, prompt: str, context: str) -> str:
        """Generate an answer given a prompt and context."""
        user_message = f"Context:\n{context}\n\nQuestion: {prompt}"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                self.config.api_url,
                json={
                    "model": self.config.model,
                    "max_tokens": self.config.max_tokens,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                },
                headers={
                    "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                # Truncate the response body to avoid echoing large or sensitive
                # content (e.g. user prompt reflected by some providers) into
                # tracebacks and log aggregators.
                preview = repr(data)[:200]
                raise ValueError(
                    f"Unexpected LLM response shape: {preview}..."
                ) from exc
