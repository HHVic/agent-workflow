"""Local validation for LLM-generated Explorer Tool arguments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

_FILE_EXTENSIONS = (
    ".java",
    ".kt",
    ".py",
    ".xml",
    ".yaml",
    ".yml",
    ".properties",
    ".proto",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".md",
    ".json",
)
_GENERIC_SYMBOLS = {
    "application",
    "main",
    "controller",
    "service",
    "repository",
    "mapper",
    "prehandle",
    "handle",
    "execute",
    "run",
    "process",
}
_SYMBOL_TOOLS = {"find_callers", "find_callees", "analyze_impact"}
_ALLOWED_ARGS = {
    "check_code_index": set(),
    "list_code_files": {"path", "pattern", "format", "includeMetadata", "maxDepth", "projectPath"},
    "get_code_context": {"task", "maxNodes", "includeCode", "projectPath"},
    "search_code": {"query", "kind", "limit", "projectPath"},
    "explore_symbol": {"query", "maxFiles", "projectPath"},
    "get_symbol_detail": {"symbol", "includeCode", "projectPath"},
    "find_callers": {"symbol", "limit", "projectPath"},
    "find_callees": {"symbol", "limit", "projectPath"},
    "trace_path": {"from", "to", "projectPath"},
    "analyze_impact": {"symbol", "depth", "projectPath"},
}


@dataclass(frozen=True, slots=True)
class ToolValidationResult:
    """Whether an Explorer Tool call may reach MCP."""

    ok: bool
    message: str | None = None
    normalized_args: dict[str, Any] | None = None


def validate_tool_args(
    tool_name: str,
    args: dict[str, Any],
    task_mode: str,
) -> ToolValidationResult:
    """Validate arguments before invoking CodeGraph MCP."""

    allowed_args = _ALLOWED_ARGS.get(tool_name)
    if allowed_args is None:
        return _error(
            f"未知工具：{tool_name}。",
            "请只使用当前提供的 Explorer Tools。",
        )
    extra_args = sorted(set(args) - allowed_args)
    if extra_args:
        return _unsupported_args_error(tool_name, extra_args)
    if not _is_optional_string(args, "projectPath"):
        return _error(
            f"{tool_name} 的 projectPath 必须是字符串。",
            '正确示例：projectPath="/path/to/repository"',
        )
    project_path_error = _validate_project_path(args, tool_name)
    if project_path_error:
        return project_path_error

    validators = {
        "check_code_index": _validate_check_code_index,
        "list_code_files": validate_list_code_files,
        "get_code_context": _validate_get_code_context,
        "search_code": validate_search_code,
        "explore_symbol": validate_explore_symbol,
        "get_symbol_detail": validate_get_symbol_detail,
        "find_callers": lambda value: validate_symbol_tool(
            "find_callers", value, task_mode
        ),
        "find_callees": lambda value: validate_symbol_tool(
            "find_callees", value, task_mode
        ),
        "trace_path": lambda value: validate_trace_path(value, task_mode),
        "analyze_impact": lambda value: validate_symbol_tool(
            "analyze_impact", value, task_mode
        ),
    }
    return validators[tool_name](args)


def looks_like_file_path(value: str) -> bool:
    """Return whether a value resembles a file or directory path."""

    lowered = value.strip().lower()
    return "/" in value or "\\" in value or lowered.endswith(_FILE_EXTENSIONS)


def validate_search_code(args: dict[str, Any]) -> ToolValidationResult:
    """Validate symbol-search arguments."""

    query_error = _require_non_empty_string(args, "query", "search_code")
    if query_error:
        return query_error
    if not _is_optional_positive_int(args, "limit"):
        return _error(
            "search_code 的 limit 必须是正整数。",
            '正确示例：search_code(query="IndexSearcher", kind="class", limit=10)',
        )
    if not _is_optional_string(args, "kind"):
        return _error(
            "search_code 的 kind 必须是字符串。",
            '正确示例：search_code(query="IndexSearcher", kind="class")',
        )
    return _ok(args)


def validate_list_code_files(args: dict[str, Any]) -> ToolValidationResult:
    """Validate indexed-file listing arguments."""

    if not _is_optional_string(args, "path"):
        return _error(
            "list_code_files 的 path 必须是字符串。",
            '正确示例：list_code_files(path="src/main/java", format="tree")',
        )
    pattern = args.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str) or not pattern.strip():
            return _error(
                "list_code_files 的 pattern 必须是非空 glob 字符串。",
                '正确示例：list_code_files(pattern="**/*.java", format="flat")',
            )
        if pattern.startswith(".*") or r"\." in pattern:
            return _error(
                "list_code_files 的 pattern 应使用 glob，不要使用正则。",
                f"你传入的是：\n{pattern}\n\n"
                '正确示例：list_code_files(pattern="**/*.tsx", format="flat")\n'
                '或：list_code_files(pattern="**/*.java", format="flat")',
            )
    if args.get("format") not in (None, "tree", "flat", "grouped"):
        return _error(
            "list_code_files 的 format 只支持 tree、flat、grouped。",
            '正确示例：list_code_files(pattern="**/*.java", format="flat")',
        )
    if not _is_optional_bool(args, "includeMetadata"):
        return _error(
            "list_code_files 的 includeMetadata 必须是布尔值。",
            '正确示例：list_code_files(format="tree", includeMetadata=true)',
        )
    if not _is_optional_positive_int(args, "maxDepth"):
        return _error(
            "list_code_files 的 maxDepth 必须是正整数。",
            '正确示例：list_code_files(format="tree", maxDepth=3)',
        )
    return _ok(args)


def validate_explore_symbol(args: dict[str, Any]) -> ToolValidationResult:
    """Validate broad local exploration arguments."""

    query_error = _require_non_empty_string(args, "query", "explore_symbol")
    if query_error:
        return query_error
    if not _is_optional_positive_int(args, "maxFiles"):
        return _error(
            "explore_symbol 的 maxFiles 必须是正整数。",
            '正确示例：explore_symbol(query="WidgetService.java widget impl", maxFiles=4)',
        )
    return _ok(args)


def validate_get_symbol_detail(args: dict[str, Any]) -> ToolValidationResult:
    """Validate precise symbol-detail arguments."""

    symbol_error = _validate_symbol_name(args, "get_symbol_detail")
    if symbol_error:
        return symbol_error
    if not _is_optional_bool(args, "includeCode"):
        return _error(
            "get_symbol_detail 的 includeCode 必须是布尔值。",
            '正确示例：get_symbol_detail(symbol="WidgetService", includeCode=false)',
        )
    return _ok(args)


def validate_symbol_tool(
    tool_name: str,
    args: dict[str, Any],
    task_mode: str,
) -> ToolValidationResult:
    """Validate callers, callees, and impact arguments."""

    symbol_error = _validate_symbol_name(args, tool_name)
    if symbol_error:
        return symbol_error
    symbol = args["symbol"]
    if task_mode == "repo_overview" and symbol.casefold() in _GENERIC_SYMBOLS:
        return _error(
            f"{tool_name} 不适合在 repo_overview 任务中分析泛符号：{symbol}。",
            "仓库级概览应先按路径、模块和启动入口归类，不应对单个泛符号做调用链深挖。\n\n"
            "如果要继续做仓库概览，请使用：\n"
            '* list_code_files(format="tree", maxDepth=3)\n'
            '* search_code(query="Application", kind="class")\n'
            '* search_code(query="Controller", kind="class")',
        )
    numeric_arg = "depth" if tool_name == "analyze_impact" else "limit"
    if not _is_optional_positive_int(args, numeric_arg):
        return _error(
            f"{tool_name} 的 {numeric_arg} 必须是正整数。",
            f'正确示例：{tool_name}(symbol="WidgetService", {numeric_arg}=2)',
        )
    return _ok(args)


def validate_trace_path(
    args: dict[str, Any],
    task_mode: str,
) -> ToolValidationResult:
    """Validate explicit call-path tracing arguments."""

    if task_mode == "repo_overview":
        return _error(
            "trace_path 不适合仓库级概览任务。",
            "trace_path 需要明确的调用起点和终点，适用于具体调用链分析。"
            "当前任务是 repo_overview，应优先使用 list_code_files、search_code 和 "
            "explore_symbol 对模块结构进行归纳。",
        )
    for key in ("from", "to"):
        value_error = _require_non_empty_string(args, key, "trace_path")
        if value_error:
            return value_error
        if looks_like_file_path(args[key]):
            return _error(
                f"trace_path 的 {key} 参数需要明确符号名，不接受文件路径或文件名。",
                '正确示例：trace_path(from="calculateWidget", to="saveWidgetDetail")',
            )
    return _ok(args)


def _validate_check_code_index(args: dict[str, Any]) -> ToolValidationResult:
    if args:
        return _error(
            "check_code_index 不接受参数。",
            "正确示例：check_code_index({})",
        )
    return _ok(args)


def _validate_get_code_context(args: dict[str, Any]) -> ToolValidationResult:
    task_error = _require_non_empty_string(args, "task", "get_code_context")
    if task_error:
        return task_error
    if not _is_optional_positive_int(args, "maxNodes"):
        return _error(
            "get_code_context 的 maxNodes 必须是正整数。",
            '正确示例：get_code_context(task="widget processing flow", maxNodes=20)',
        )
    if not _is_optional_bool(args, "includeCode"):
        return _error(
            "get_code_context 的 includeCode 必须是布尔值。",
            '正确示例：get_code_context(task="widget processing flow", includeCode=false)',
        )
    return _ok(args)


def _validate_symbol_name(
    args: dict[str, Any],
    tool_name: str,
) -> ToolValidationResult | None:
    symbol_error = _require_non_empty_string(args, "symbol", tool_name)
    if symbol_error:
        return symbol_error
    symbol = args["symbol"]
    if looks_like_file_path(symbol):
        filename = PurePath(symbol.replace("\\", "/")).name
        stem = PurePath(filename).stem or "WidgetService"
        return _error(
            f"{tool_name} 的 symbol 参数需要代码符号名，不接受文件路径或文件名。\n\n"
            f"你传入的是：\n{symbol}",
            f'正确示例：get_symbol_detail(symbol="{stem}", includeCode=false)\n\n'
            "如果你只有文件名、路径或模块线索，请改用：\n"
            f'explore_symbol(query="{filename} related symbols", maxFiles=4)',
        )
    return None


def _validate_project_path(
    args: dict[str, Any],
    tool_name: str,
) -> ToolValidationResult | None:
    project_path = args.get("projectPath")
    if project_path is None:
        return None
    if not project_path.strip():
        return _error(
            f"{tool_name} 的 projectPath 不能为空。",
            "当前仓库请省略 projectPath。",
        )
    expanded_path = Path(project_path).expanduser()
    if not expanded_path.is_dir() or not (expanded_path / ".codegraph").is_dir():
        return _error(
            f"{tool_name} 的 projectPath 必须指向包含 .codegraph 的已初始化项目根目录。\n\n"
            f"你传入的是：\n{project_path}",
            "当前仓库请省略 projectPath。projectPath 不是模块过滤参数，也不是符号消歧参数。\n\n"
            "如果你只有模块、文件路径或同名符号线索，请改用 explore_symbol 的 query，"
            "或先用 search_code / list_code_files 按路径归类。",
        )
    return None


def _unsupported_args_error(
    tool_name: str,
    extra_args: list[str],
) -> ToolValidationResult:
    joined = "、".join(extra_args)
    allowed = "、".join(sorted(_ALLOWED_ARGS[tool_name])) or "无"
    if tool_name == "search_code":
        guidance = (
            "你当前似乎想按文件类型过滤。请改用 list_code_files，并使用 glob pattern，不要使用正则。\n\n"
            '正确示例：list_code_files(pattern="**/*.tsx", format="flat")\n\n'
            '如果只是搜索代码符号，请使用：search_code(query="IndexSearcher", kind="class")'
        )
    else:
        guidance = f"请仅使用允许参数。{tool_name} 支持：{allowed}"
    return _error(
        f"{tool_name} 不支持参数：{joined}。\n\n{tool_name} 只支持：{allowed}",
        guidance,
    )


def _require_non_empty_string(
    args: dict[str, Any],
    key: str,
    tool_name: str,
) -> ToolValidationResult | None:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        return _error(
            f"{tool_name} 的 {key} 必须是非空字符串。",
            f'正确示例：{tool_name}({key}="WidgetService")',
        )
    return None


def _is_optional_string(args: dict[str, Any], key: str) -> bool:
    return key not in args or isinstance(args[key], str)


def _is_optional_bool(args: dict[str, Any], key: str) -> bool:
    return key not in args or isinstance(args[key], bool)


def _is_optional_positive_int(args: dict[str, Any], key: str) -> bool:
    value = args.get(key)
    return key not in args or (
        isinstance(value, int) and not isinstance(value, bool) and value > 0
    )


def _ok(args: dict[str, Any]) -> ToolValidationResult:
    return ToolValidationResult(ok=True, normalized_args=dict(args))


def _error(reason: str, guidance: str) -> ToolValidationResult:
    return ToolValidationResult(
        ok=False,
        message=f"[TOOL_ARGUMENT_ERROR]\n\n{reason}\n\n{guidance}",
    )
