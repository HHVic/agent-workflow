"""OpenAI-compatible chat completions client."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from app.core.config import Settings


class LLMConfigurationError(ValueError):
    """Raised when required LLM settings are missing."""


class LLMClient:
    """Small async wrapper that preserves raw chat completion responses."""

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise LLMConfigurationError("LLM_API_KEY is required")

        kwargs: dict[str, Any] = {
            "api_key": settings.llm_api_key,
            "timeout": 120.0,
            "max_retries": 2,
        }
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = settings.llm_model

    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatCompletion:
        """Send one tool-enabled chat completion request."""

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return await self._client.chat.completions.create(
            **kwargs,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""

        await self._client.close()
