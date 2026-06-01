"""Timeout-aware stdio MCP client built on the official Python SDK."""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Awaitable, TypeVar

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool

T = TypeVar("T")


class MCPClientError(RuntimeError):
    """Raised when an MCP subprocess or protocol request fails."""


class StdioMCPClient:
    """Starts an MCP subprocess and exposes its tool endpoints."""

    def __init__(
        self,
        command: str,
        args: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._command = command
        self._args = args
        self._cwd = cwd
        self._timeout_seconds = timeout_seconds
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "StdioMCPClient":
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
        """Start the subprocess and complete the MCP initialize handshake."""

        if self._session is not None:
            return

        stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self._command,
            args=list(self._args),
            cwd=self._cwd,
        )
        try:
            read_stream, write_stream = await self._await(
                stack.enter_async_context(stdio_client(params, errlog=sys.stderr)),
                "start MCP server",
            )
            session = await self._await(
                stack.enter_async_context(
                    ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(seconds=self._timeout_seconds),
                    )
                ),
                "open MCP session",
            )
            await self._await(session.initialize(), "initialize MCP session")
        except Exception:
            await stack.aclose()
            raise

        self._exit_stack = stack
        self._session = session

    async def list_tools(self) -> list[Tool]:
        """Return every MCP tool, following pagination if present."""

        session = self._require_session()
        tools: list[Tool] = []
        cursor: str | None = None
        while True:
            result = await self._await(
                session.list_tools(cursor=cursor),
                "list MCP tools",
            )
            tools.extend(result.tools)
            cursor = result.nextCursor
            if cursor is None:
                return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> CallToolResult:
        """Invoke an MCP tool and return its raw SDK result."""

        session = self._require_session()
        return await self._await(
            session.call_tool(
                name,
                arguments,
                read_timeout_seconds=timedelta(seconds=self._timeout_seconds),
            ),
            f"call MCP tool {name}",
        )

    async def close(self) -> None:
        """Close the MCP session and terminate the subprocess."""

        stack = self._exit_stack
        self._session = None
        self._exit_stack = None
        if stack is None:
            return
        try:
            await asyncio.wait_for(stack.aclose(), timeout=self._timeout_seconds)
        except TimeoutError as exc:
            raise MCPClientError("timed out while closing MCP server") from exc

    async def _await(self, awaitable: Awaitable[T], operation: str) -> T:
        try:
            return await asyncio.wait_for(awaitable, timeout=self._timeout_seconds)
        except TimeoutError as exc:
            raise MCPClientError(f"timed out while trying to {operation}") from exc
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPClientError(f"failed to {operation}: {exc}") from exc

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise MCPClientError("MCP client is not connected")
        return self._session
