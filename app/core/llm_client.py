"""OpenAI-compatible chat completions client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from app.core.config import Settings


class LLMConfigurationError(ValueError):
    """Raised when required LLM settings are missing."""


class MockLLMResponseError(ValueError):
    """Raised when replay responses are invalid or exhausted."""


class ChatCompletionClient(Protocol):
    """Client behavior required by CodeExplorerAgent."""

    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatCompletion: ...

    async def close(self) -> None: ...


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


class ReplayLLMClient:
    """Return deterministic local Chat Completions responses from JSONL."""

    def __init__(self, responses_file: Path) -> None:
        self._responses_file = responses_file
        self._responses = _load_replay_responses(responses_file)
        self._next_response = 0

    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatCompletion:
        """Return the next local response without making an HTTP request."""

        del messages, tools
        if self._next_response >= len(self._responses):
            raise MockLLMResponseError(
                "mock LLM responses exhausted after "
                f"{len(self._responses)} calls: {self._responses_file}"
            )
        response = self._responses[self._next_response]
        self._next_response += 1
        return response

    async def close(self) -> None:
        """Match the real client lifecycle without allocating resources."""


def create_llm_client(settings: Settings) -> ChatCompletionClient:
    """Create the configured real or local replay client."""

    if settings.mock_llm_responses_file is not None:
        return ReplayLLMClient(settings.mock_llm_responses_file)
    return LLMClient(settings)


def _load_replay_responses(responses_file: Path) -> list[ChatCompletion]:
    try:
        lines = responses_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MockLLMResponseError(
            f"failed to read mock LLM responses file {responses_file}: {exc}"
        ) from exc

    responses: list[ChatCompletion] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MockLLMResponseError(
                f"invalid mock LLM JSONL at {responses_file}, line {line_number}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise MockLLMResponseError(
                f"mock LLM response at {responses_file}, line {line_number} "
                "must be a JSON object"
            )
        try:
            responses.append(_build_replay_completion(payload, len(responses) + 1))
        except (TypeError, ValueError) as exc:
            raise MockLLMResponseError(
                f"invalid mock LLM response at {responses_file}, "
                f"line {line_number}: {exc}"
            ) from exc
    if not responses:
        raise MockLLMResponseError(
            f"mock LLM responses file is empty: {responses_file}"
        )
    return responses


def _build_replay_completion(payload: dict[str, Any], index: int) -> ChatCompletion:
    if "choices" in payload:
        return ChatCompletion.model_validate(payload)

    content = payload.get("content")
    if content is not None and not isinstance(content, str):
        raise TypeError("content must be a string or null")
    tool_calls = payload.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        raise TypeError("tool_calls must be an array")

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        message["tool_calls"] = [
            _build_replay_tool_call(tool_call, call_index)
            for call_index, tool_call in enumerate(tool_calls, start=1)
        ]
    return ChatCompletion.model_validate(
        {
            "id": payload.get("id", f"mock-chatcmpl-{index}"),
            "object": "chat.completion",
            "created": 0,
            "model": payload.get("model", "mock-replay"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": message,
                }
            ],
        }
    )


def _build_replay_tool_call(tool_call: Any, index: int) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        raise TypeError("each tool_call must be an object")
    name = tool_call.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("each tool_call requires a non-empty name")
    arguments = tool_call.get("arguments", {})
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": tool_call.get("id", f"mock-call-{index}"),
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }
