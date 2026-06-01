import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from app.agents.code_explorer.tools import (
    EXPLORER_TOOL_SPECS,
    ExplorerTools,
    MissingCodeGraphToolsError,
    ToolArgumentValidationError,
)


@dataclass
class FakeMCPTool:
    name: str
    inputSchema: dict[str, Any]


class FakeCodeGraphClient:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self._result = result

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._result is not None:
            return self._result
        return {"name": name, "arguments": arguments}


def make_mcp_tools() -> list[FakeMCPTool]:
    return [
        FakeMCPTool(
            spec.mcp_name,
            {
                "type": "object",
                "properties": {"projectPath": {"type": "string"}},
            },
        )
        for spec in EXPLORER_TOOL_SPECS
    ]


def test_explorer_tools_wrap_codegraph_tools_for_llm() -> None:
    tools = ExplorerTools(FakeCodeGraphClient(), make_mcp_tools())

    llm_tools = tools.as_llm_tools()

    assert [tool["function"]["name"] for tool in llm_tools] == [
        "check_code_index",
        "get_code_context",
        "search_code",
        "explore_symbol",
        "get_symbol_detail",
        "find_callers",
        "find_callees",
        "trace_path",
        "analyze_impact",
        "list_code_files",
    ]
    assert all(tool["type"] == "function" for tool in llm_tools)
    assert all(tool["function"]["description"] for tool in llm_tools)
    assert all(
        tool["function"]["parameters"]["additionalProperties"] is False
        for tool in llm_tools
    )
    assert llm_tools[0]["function"]["parameters"]["properties"] == {}
    descriptions = {
        tool["function"]["name"]: tool["function"]["description"]
        for tool in llm_tools
    }
    assert "PRIMARY TOOL" in descriptions["get_code_context"] or "primary tool" in descriptions["get_code_context"].lower()
    assert "complex code understanding" in descriptions["get_code_context"]
    assert "architecture" in descriptions["get_code_context"]
    assert "feature exploration" in descriptions["get_code_context"]
    assert "bug" in descriptions["get_code_context"].lower()
    assert "quick symbol search" in descriptions["search_code"].lower()
    assert "returns locations only" in descriptions["search_code"].lower()
    assert "not suitable for understanding complex feature or architecture flow" in descriptions["search_code"].lower()
    assert "bag of symbol/file names" in descriptions["explore_symbol"].lower()
    assert "prefer over many repeated get_symbol_detail calls" in descriptions["explore_symbol"].lower()
    assert "call path between two known symbols" in descriptions["trace_path"].lower()
    assert "flow questions" in descriptions["trace_path"].lower()
    assert "skip unless debugging" in descriptions["check_code_index"].lower()
    assert "symbol, not a file path" in descriptions["get_symbol_detail"].lower()
    assert "not suitable for repo_overview" in descriptions["analyze_impact"].lower()
    assert "vague symbol" in descriptions["analyze_impact"].lower()
    assert "requires a specific symbol" in descriptions["find_callers"].lower()
    assert "requires a specific symbol" in descriptions["find_callees"].lower()
    assert "glob" in descriptions["list_code_files"]
    assert all(
        "alternate repository root containing .codegraph" in description
        and "not a module filter" in description
        and "not a symbol-disambiguation parameter" in description
        for name, description in descriptions.items()
        if name != "check_code_index"
    )


def test_explorer_tools_report_missing_mcp_tool() -> None:
    tools = make_mcp_tools()
    tools.pop()

    with pytest.raises(MissingCodeGraphToolsError, match="codegraph_files"):
        ExplorerTools(FakeCodeGraphClient(), tools)


def test_explorer_tools_translate_tool_name() -> None:
    tools = ExplorerTools(FakeCodeGraphClient(), make_mcp_tools())

    result = asyncio.run(tools.call("check_code_index", {}, "feature_exploration"))

    assert result == {"name": "codegraph_status", "arguments": {}}


@pytest.mark.parametrize(
    "ambiguous_text",
    (
        "Found 3 symbols named WidgetService",
        "Showing results for WidgetService.calculate",
        "Others: more matches",
        "Aggregated results across modules",
    ),
)
def test_explorer_tools_warn_about_ambiguous_symbols(ambiguous_text: str) -> None:
    client = FakeCodeGraphClient(
        {"content": [{"type": "text", "text": ambiguous_text}]}
    )
    tools = ExplorerTools(client, make_mcp_tools())

    result = asyncio.run(
        tools.call("get_symbol_detail", {"symbol": "WidgetService"}, "feature_exploration")
    )

    assert result["warning"].startswith("[AMBIGUOUS_SYMBOL_WARNING]")
    assert "不要默认选择第一个结果" in result["warning"]
    assert "explore_symbol" in result["warning"]
    assert "projectPath" in result["warning"]


def test_explorer_tools_do_not_call_mcp_when_validation_fails() -> None:
    client = CountingCodeGraphClient()
    tools = ExplorerTools(client, make_mcp_tools())

    with pytest.raises(ToolArgumentValidationError, match="search_code 不支持参数"):
        asyncio.run(
            tools.call(
                "search_code",
                {"query": "index", "pattern": "**/*.tsx"},
                "feature_exploration",
            )
        )

    assert client.calls == 0


def test_explorer_tools_mark_no_result() -> None:
    client = FakeCodeGraphClient(
        {"content": [{"type": "text", "text": 'No results found for "Widget"'}]}
    )
    tools = ExplorerTools(client, make_mcp_tools())

    result = asyncio.run(
        tools.call("search_code", {"query": "Widget"}, "feature_exploration")
    )

    assert result["warning"].startswith("[NO_RESULT]")


def test_explorer_tools_skip_repeated_successful_call() -> None:
    client = CountingCodeGraphClient()
    tools = ExplorerTools(client, make_mcp_tools())

    first = asyncio.run(
        tools.call("get_symbol_detail", {"symbol": "WidgetService"}, "feature_exploration")
    )
    second = asyncio.run(
        tools.call("get_symbol_detail", {"symbol": "WidgetService"}, "feature_exploration")
    )

    assert first["name"] == "codegraph_node"
    assert second["warning"].startswith("[REPEATED_TOOL_CALL_WARNING]")
    assert "explore_symbol" in second["warning"]
    assert client.calls == 1


class CountingCodeGraphClient:
    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {"name": name, "arguments": arguments}
