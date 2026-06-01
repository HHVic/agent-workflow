"""Lightweight context management for long exploration runs."""

from __future__ import annotations

from typing import Any


class ConversationMemory:
    """Truncates older tool results without mutating the full local log."""

    def __init__(
        self,
        *,
        recent_tool_results: int = 5,
        old_tool_result_chars: int = 600,
        recent_tool_result_chars: int = 12_000,
    ) -> None:
        self._recent_tool_results = recent_tool_results
        self._old_tool_result_chars = old_tool_result_chars
        self._recent_tool_result_chars = recent_tool_result_chars

    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return an LLM-facing copy with bounded tool result content."""

        prepared = [dict(message) for message in messages]
        tool_indices = [
            index
            for index, message in enumerate(prepared)
            if message.get("role") == "tool"
        ]
        recent_indices = set(tool_indices[-self._recent_tool_results :])

        for index in tool_indices:
            content = str(prepared[index].get("content", ""))
            if index in recent_indices:
                prepared[index]["content"] = _truncate_recent(
                    content,
                    self._recent_tool_result_chars,
                )
            else:
                prepared[index]["content"] = _truncate_old(
                    content,
                    self._old_tool_result_chars,
                )
        return prepared


def _truncate_recent(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    return f"{content[:limit]}\n[当前工具结果已截断，原始长度 {len(content)} 字符]"


def _truncate_old(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    return f"[历史工具结果已截断，原始长度 {len(content)} 字符]\n{content[:limit]}"
