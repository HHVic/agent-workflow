"""LLM-visible wrappers for CodeGraph MCP tools."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from app.agents.code_explorer.tool_validation import validate_tool_args

AMBIGUOUS_SYMBOL_WARNING = """\
[AMBIGUOUS_SYMBOL_WARNING]
当前工具结果包含多个同名符号。不要默认选择第一个结果；请先按路径、仓库和模块归类，必要时使用更精确的路径或符号继续探索。"""

AMBIGUOUS_SYMBOL_WARNING += """
不要重复当前 get_symbol_detail 调用，也不要使用 projectPath 进行符号消歧。请改用 explore_symbol，将符号名和模块、文件或路径线索放入 query。"""

NO_RESULT_WARNING = """\
[NO_RESULT]
当前查询没有返回结果。可能原因：
- 查询词不适合 CodeGraph 符号搜索
- 传入了路径但工具期望符号名
- 需要先使用 list_code_files 查看目录结构
- 需要改用 explore_symbol 进行宽泛探索"""

REPEATED_TOOL_CALL_WARNING = """\
[REPEATED_TOOL_CALL_WARNING]
完全相同的 Explorer Tool 调用此前已经成功执行，不会再次请求 MCP。重复调用不会增加新信息。
请根据已有结果选择新的探索方向。若此前结果包含多个同名符号，请改用 explore_symbol，将符号名和模块、文件或路径线索放入 query；不要使用 projectPath 进行符号消歧。"""

_AMBIGUOUS_SYMBOL_MARKERS = (
    "symbols named",
    "Showing results for",
    "Others:",
    "Aggregated results across",
)
_NO_RESULT_MARKERS = (
    "No results found",
    "No files found",
    "not found in the codebase",
)
_PROJECT_PATH_GUIDANCE = (
    " projectPath is only for an alternate repository root containing .codegraph. "
    "Omit it for the current repository. It is not a module filter and not a "
    "symbol-disambiguation parameter."
)


class MCPToolDefinition(Protocol):
    """Subset of an MCP tool definition needed by the wrapper."""

    name: str
    inputSchema: dict[str, Any]


class CodeGraphToolCaller(Protocol):
    """Client behavior required by ExplorerTools."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ExplorerToolSpec:
    explorer_name: str
    mcp_name: str
    description: str


EXPLORER_TOOL_SPECS = (
    ExplorerToolSpec(
        "check_code_index",
        "codegraph_status",
        "Check CodeGraph index health and scale, including indexed file counts, node counts, edge counts, and language coverage. Suitable for repo_overview when you need to confirm index status, for MCP/index debugging, or when the user explicitly asks to inspect the index. Skip unless debugging or doing repository-level index inspection. Not suitable as a routine first step for ordinary feature exploration, and avoid repeating it once index health is already confirmed. Takes no arguments; call check_code_index({}).",
    ),
    ExplorerToolSpec(
        "get_code_context",
        "codegraph_context",
        "PRIMARY TOOL for complex code understanding. Use this first for how-does-X-work questions, architecture questions, bug or root-cause analysis, feature exploration, and cross-module business or data flow where you do not yet know the right entry points. Accepts a natural-language code exploration goal and returns entry points, related symbols, key code context, and an initial code map. This provides code context, not product-requirements analysis. For repo_overview, use repository-structure terms such as modules, directories, entry points, layers, and business domains; avoid vague ordinary English verbs that may accidentally match code symbols. For complex business flow exploration, prefer this to build the candidate map first, then use search_code, explore_symbol, or trace_path for targeted follow-up. Not suitable when you already know one exact class or method and only need a quick symbol lookup, a single symbol detail, or simple file-structure listing. Supports task, maxNodes, includeCode, projectPath.",
    ),
    ExplorerToolSpec(
        "search_code",
        "codegraph_search",
        "Quick symbol search by exact or partial name. Returns locations only or lightweight candidate matches, usually not full source code. Suitable when you already know a keyword, class name, method name, interface name, route name, error code, or log clue and need to find candidate symbols or verify whether something exists. Not suitable for understanding complex feature or architecture flow, and does not replace get_code_context for comprehensive code understanding. For complex business flow work, usually build context with get_code_context first, then use search_code for targeted lookup. Supports query, kind, limit, projectPath. Does not accept pattern, path, format, maxDepth, includeCode, or maxFiles. If you need glob file filtering, use list_code_files instead.",
    ),
    ExplorerToolSpec(
        "explore_symbol",
        "codegraph_explore",
        "Explore source for several related symbols or files grouped by file in one capped call. Query should be a bag of symbol/file names or short code terms, not a long natural-language question. Suitable after get_code_context or search_code has identified candidate classes, methods, files, or modules and you want to inspect several related code areas at once. Use it when you have file-name, class-name, or module clues but not one exact symbol id. Prefer over many repeated get_symbol_detail calls. Not suitable as the main tool for global architecture or business-flow mapping, not ideal for single exact symbol detail, and not intended for long natural-language prompts. Returned source is read-equivalent for the files shown, so do not re-open those files separately. Supports query, maxFiles, projectPath.",
    ),
    ExplorerToolSpec(
        "get_symbol_detail",
        "codegraph_node",
        "Get details, location, signature, and nearby graph relationships for one explicit symbol. Suitable when search_code, get_code_context, or explore_symbol has already given you a clear class, method, interface, enum, or other symbol name and you want focused detail on that one target. This is for a symbol, not a file path, directory path, or file name such as WidgetService.java. If the result is ambiguous, do not repeat the same call and do not pass projectPath to disambiguate it; use explore_symbol with symbol plus module, file, or path clues in query. If you need several related symbols or files, use explore_symbol instead. If you need to verify a full call chain between two known symbols, use trace_path instead. Supports symbol, includeCode, projectPath.",
    ),
    ExplorerToolSpec(
        "find_callers",
        "codegraph_callers",
        "Find callers of one specific symbol so you can understand who triggers that logic. Suitable after you have identified a key method or class and want upstream entry points such as controllers, jobs, events, RPC handlers, or service callers. Useful as targeted follow-up during feature exploration, bug analysis, or impact analysis. Requires a specific symbol; classify ambiguous matches first. Not suitable for repo_overview on generic symbols, vague symbol guesses, or file paths. Supports symbol, limit, projectPath.",
    ),
    ExplorerToolSpec(
        "find_callees",
        "codegraph_callees",
        "Find downstream callees of one specific symbol so you can follow what logic it invokes next. Suitable after you have identified an entry method or class and want to trace service, repository, client, or persistence calls further downstream. Useful as targeted follow-up during feature exploration and flow completion. Requires a specific symbol; classify ambiguous matches first. Not suitable for repo_overview on generic symbols, vague symbol guesses, or file paths. Supports symbol, limit, projectPath.",
    ),
    ExplorerToolSpec(
        "trace_path",
        "codegraph_trace",
        "Call path between two known symbols. Use this for flow questions such as how does A reach B, when you already know explicit from and to symbols and want to verify whether a concrete call chain exists. Returns the call chain with each hop's body inlined plus destination-side callees in one call. If no static path exists, the result can still help reveal dynamic-dispatch boundaries. Not suitable for repo_overview, vague symbols, ambiguous same-name symbols that have not been disambiguated, or module names and file paths instead of symbols. Supports from, to, projectPath.",
    ),
    ExplorerToolSpec(
        "analyze_impact",
        "codegraph_impact",
        "Analyze the likely impact of changing one explicit symbol or logic point, including affected callers, downstream modules, and related paths. Suitable for change-impact analysis, refactor risk assessment, and requirement-change evaluation after you have already identified the concrete symbol that may change. Not suitable for repo_overview, vague symbol guesses, ambiguous matches, or file paths. Supports symbol, depth, projectPath.",
    ),
    ExplorerToolSpec(
        "list_code_files",
        "codegraph_files",
        "List indexed repository files and directories, or filter files by glob. This is usually better than a generic Glob pass when the project is already indexed by CodeGraph. Suitable for repo_overview, checking top-level repository or module structure, finding files with glob patterns such as **/Application.java, **/*.proto, or **/*Controller.java, and confirming a directory exists before deeper exploration. Not suitable as the main tool for understanding complex business flow or architecture, and does not replace get_code_context or symbol exploration. Use path for a directory and pattern for a glob; pattern must be glob syntax, not regex. Supports path, pattern, format, includeMetadata, maxDepth, projectPath.",
    ),
)


class MissingCodeGraphToolsError(RuntimeError):
    """Raised when the MCP server does not expose every required tool."""


class UnknownExplorerToolError(ValueError):
    """Raised when the LLM requests an unknown Explorer Tool."""


class ToolArgumentValidationError(ValueError):
    """Raised when an LLM tool call must not reach CodeGraph MCP."""

    def __init__(self, message: str, mapped_mcp_tool_name: str) -> None:
        super().__init__(message)
        self.mapped_mcp_tool_name = mapped_mcp_tool_name


class ExplorerTools:
    """Validates and translates the LLM-visible Explorer Tool surface."""

    def __init__(
        self,
        client: CodeGraphToolCaller,
        mcp_tools: Sequence[MCPToolDefinition],
    ) -> None:
        self._client = client
        self._mcp_tools = {tool.name: tool for tool in mcp_tools}
        self._specs = {spec.explorer_name: spec for spec in EXPLORER_TOOL_SPECS}
        self._successful_call_signatures: set[str] = set()

        missing = [
            spec.mcp_name
            for spec in EXPLORER_TOOL_SPECS
            if spec.mcp_name not in self._mcp_tools
        ]
        if missing:
            joined = ", ".join(sorted(missing))
            raise MissingCodeGraphToolsError(
                f"CodeGraph MCP server is missing required tools: {joined}"
            )

    def as_llm_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function tool definitions."""

        return [
            {
                "type": "function",
                "function": {
                    "name": spec.explorer_name,
                    "description": _llm_description(spec),
                    "parameters": _strict_schema(
                        spec.explorer_name,
                        self._mcp_tools[spec.mcp_name].inputSchema
                    ),
                },
            }
            for spec in EXPLORER_TOOL_SPECS
        ]

    async def call(
        self,
        explorer_name: str,
        arguments: dict[str, Any],
        task_mode: str,
    ) -> dict[str, Any]:
        """Translate an Explorer Tool call into a CodeGraph MCP call."""

        spec = self._specs.get(explorer_name)
        if spec is None:
            raise UnknownExplorerToolError(f"unknown Explorer Tool: {explorer_name}")
        validation = validate_tool_args(explorer_name, arguments, task_mode)
        if not validation.ok:
            raise ToolArgumentValidationError(
                validation.message or "[TOOL_ARGUMENT_ERROR]",
                spec.mcp_name,
            )
        normalized_args = validation.normalized_args or {}
        signature = _call_signature(explorer_name, normalized_args)
        if signature in self._successful_call_signatures:
            return {"warning": REPEATED_TOOL_CALL_WARNING}
        result = await self._client.call_tool(
            spec.mcp_name,
            normalized_args,
        )
        self._successful_call_signatures.add(signature)
        if _contains_ambiguous_symbol_marker(result):
            return {"warning": AMBIGUOUS_SYMBOL_WARNING, **result}
        if _contains_no_result_marker(result):
            return {"warning": NO_RESULT_WARNING, **result}
        return result


def _contains_ambiguous_symbol_marker(result: dict[str, Any]) -> bool:
    serialized = str(result)
    return any(marker in serialized for marker in _AMBIGUOUS_SYMBOL_MARKERS)


def _contains_no_result_marker(result: dict[str, Any]) -> bool:
    serialized = str(result)
    return any(marker in serialized for marker in _NO_RESULT_MARKERS)


def _strict_schema(
    explorer_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    strict_schema = deepcopy(schema)
    if explorer_name == "check_code_index":
        strict_schema["properties"] = {}
    strict_schema["additionalProperties"] = False
    return strict_schema


def _llm_description(spec: ExplorerToolSpec) -> str:
    if spec.explorer_name == "check_code_index":
        return spec.description
    return f"{spec.description}{_PROJECT_PATH_GUIDANCE}"


def _call_signature(explorer_name: str, arguments: dict[str, Any]) -> str:
    serialized = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return f"{explorer_name}:{serialized}"
