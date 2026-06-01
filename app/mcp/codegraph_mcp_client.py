"""CodeGraph-specific MCP adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.types import Tool

from app.mcp.stdio_client import StdioMCPClient


class CodeGraphMCPClient:
    """Starts CodeGraph MCP and exposes JSON-serializable tool results."""

    def __init__(
        self,
        command: str,
        args: tuple[str, ...],
        *,
        project_path: Path,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._stdio = StdioMCPClient(
            command,
            args,
            cwd=project_path,
            timeout_seconds=timeout_seconds,
        )
        self._tools: list[Tool] | None = None

    async def __aenter__(self) -> "CodeGraphMCPClient":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        await self.close()

    async def connect(self) -> None:
        """Start CodeGraph MCP and cache its available tools."""

        await self._stdio.connect()
        self._tools = await self._stdio.list_tools()

    async def list_tools(self) -> list[Tool]:
        """Return CodeGraph's MCP tool definitions."""

        if self._tools is None:
            raise RuntimeError("CodeGraph MCP client is not connected")
        return list(self._tools)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call one CodeGraph tool and serialize the SDK result for logs and LLMs."""

        result = await self._stdio.call_tool(name, arguments)
        return result.model_dump(mode="json", by_alias=True, exclude_none=True)

    async def close(self) -> None:
        """Close the underlying stdio subprocess."""

        self._tools = None
        await self._stdio.close()
