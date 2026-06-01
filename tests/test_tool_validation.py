from pathlib import Path

import pytest

from app.agents.code_explorer.tool_validation import (
    looks_like_file_path,
    validate_tool_args,
)


def test_search_code_rejects_pattern_parameter() -> None:
    result = validate_tool_args(
        "search_code",
        {"query": "index", "pattern": r".*\.(ts|tsx)$"},
        "feature_exploration",
    )

    assert not result.ok
    assert "pattern" in (result.message or "")
    assert "list_code_files" in (result.message or "")
    assert '**/*.tsx' in (result.message or "")


def test_search_code_accepts_supported_parameters(tmp_path: Path) -> None:
    project_path = tmp_path / "repo"
    (project_path / ".codegraph").mkdir(parents=True)
    args = {
        "query": "IndexSearcher",
        "kind": "class",
        "limit": 10,
        "projectPath": str(project_path),
    }

    result = validate_tool_args("search_code", args, "feature_exploration")

    assert result.ok
    assert result.normalized_args == args


def test_project_path_rejects_module_filter_without_codegraph_root() -> None:
    result = validate_tool_args(
        "get_symbol_detail",
        {
            "symbol": "WidgetService",
            "projectPath": "miniapp-open-platform-bilispec",
        },
        "feature_exploration",
    )

    assert not result.ok
    assert ".codegraph" in (result.message or "")
    assert "模块过滤" in (result.message or "")
    assert "符号消歧" in (result.message or "")


def test_list_code_files_rejects_regex_pattern() -> None:
    result = validate_tool_args(
        "list_code_files",
        {"pattern": r".*\.(ts|tsx|js|jsx)$", "format": "flat"},
        "repo_overview",
    )

    assert not result.ok
    assert "glob" in (result.message or "")
    assert '**/*.tsx' in (result.message or "")


def test_list_code_files_accepts_glob_pattern() -> None:
    result = validate_tool_args(
        "list_code_files",
        {"pattern": "**/*Controller.java", "format": "flat"},
        "repo_overview",
    )

    assert result.ok


def test_get_symbol_detail_rejects_java_file_path() -> None:
    result = validate_tool_args(
        "get_symbol_detail",
        {"symbol": "src/main/java/example/AccrualService.java"},
        "feature_exploration",
    )

    assert not result.ok
    assert "explore_symbol" in (result.message or "")
    assert "AccrualService.java related symbols" in (result.message or "")


def test_get_symbol_detail_accepts_symbol_name() -> None:
    result = validate_tool_args(
        "get_symbol_detail",
        {"symbol": "AccrualService", "includeCode": False},
        "feature_exploration",
    )

    assert result.ok


def test_explore_symbol_accepts_file_and_module_clues() -> None:
    result = validate_tool_args(
        "explore_symbol",
        {"query": "AccrualService.java accrual impl", "maxFiles": 4},
        "feature_exploration",
    )

    assert result.ok


@pytest.mark.parametrize(
    ("tool_name", "symbol"),
    (
        ("find_callers", "Application"),
        ("analyze_impact", "Service"),
    ),
)
def test_repo_overview_symbol_tools_reject_generic_symbols(
    tool_name: str,
    symbol: str,
) -> None:
    result = validate_tool_args(tool_name, {"symbol": symbol}, "repo_overview")

    assert not result.ok
    assert "repo_overview" in (result.message or "")
    assert "list_code_files" in (result.message or "")


def test_trace_path_rejects_repo_overview() -> None:
    result = validate_tool_args(
        "trace_path",
        {"from": "calculate", "to": "save"},
        "repo_overview",
    )

    assert not result.ok
    assert "trace_path 不适合仓库级概览任务" in (result.message or "")


def test_trace_path_accepts_feature_exploration() -> None:
    args = {"from": "calculate", "to": "save"}

    result = validate_tool_args("trace_path", args, "feature_exploration")

    assert result.ok
    assert result.normalized_args == args


@pytest.mark.parametrize(
    "value",
    (
        "/xxx/yyy.java",
        r"src\\main\\Example.kt",
        "AccrualService.java",
    ),
)
def test_looks_like_file_path_recognizes_paths_and_filenames(value: str) -> None:
    assert looks_like_file_path(value)


def test_looks_like_file_path_does_not_reject_symbol_name() -> None:
    assert not looks_like_file_path("AccrualService")
