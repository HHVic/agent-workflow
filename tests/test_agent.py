import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import app.agents.code_explorer.agent as agent_module
from app.agents.code_explorer.agent import (
    CodeExplorerAgent,
    ExplorationLimitError,
    _needs_repo_overview_rewrite,
)
from app.agents.code_explorer.tools import EXPLORER_TOOL_SPECS
from app.core.config import Settings


@dataclass
class FakeMCPTool:
    name: str
    inputSchema: dict[str, Any]


class FakeCodeGraphMCPClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> "FakeCodeGraphMCPClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def list_tools(self) -> list[FakeMCPTool]:
        return [
            FakeMCPTool(spec.mcp_name, {"type": "object", "properties": {}})
            for spec in EXPLORER_TOOL_SPECS
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": f"{name}: indexed"}]}


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        self.calls += 1
        if self.calls == 1:
            return _response(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        id="call-1",
                        type="function",
                        function=SimpleNamespace(
                            name="check_code_index",
                            arguments="{}",
                        ),
                    )
                ],
            )

        assert messages[-1]["role"] == "tool"
        assert len(tools) == 10
        return _response(
            content="# 代码探索结果\n\n索引可用。\n\n```java\nclass Hidden {}\n```",
            tool_calls=None,
        )


class FakeOverviewLLMClient:
    def __init__(self) -> None:
        self.tool_arguments: list[list[dict[str, Any]] | None] = []

    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        self.tool_arguments.append(tools)
        if len(self.tool_arguments) == 1:
            return _response(
                content=(
                    "# 代码探索结果\n\n"
                    "## 调用关系\n"
                    "某个局部流程的调用关系。"
                ),
                tool_calls=None,
            )

        assert "仓库级概览" in messages[-1]["content"]
        return _response(
            content=(
                "# 代码探索结果\n\n"
                "## 索引状态\n可用。\n\n"
                "## 顶层仓库结构\n包含多个仓库。\n\n"
                "## 主要模块职责\n模块职责概览。\n\n"
                "## 关键启动入口\n多个 Application。\n\n"
                "## 主要业务域\n开放平台。\n\n"
                "## 建议进一步探索方向\n局部链路可作为方向之一。\n\n"
                "## 不确定项\n需进一步确认。"
            ),
            tool_calls=None,
        )


class FakeSelfCorrectingLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        self.calls += 1
        if self.calls == 1:
            return _tool_call(
                "call-invalid",
                "get_symbol_detail",
                '{"symbol":"src/main/java/example/Widget.java"}',
            )
        if self.calls == 2:
            assert "[TOOL_ARGUMENT_ERROR]" in messages[-1]["content"]
            return _tool_call(
                "call-corrected",
                "explore_symbol",
                '{"query":"Widget.java example","maxFiles":4}',
            )
        assert messages[-1]["role"] == "tool"
        return _response("# 代码探索结果\n\n探索完成。", None)


class FakeContinuingLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        self.calls += 1
        if self.calls == 1:
            return _response(
                "# 代码探索结果\n\n"
                "## 调用关系\n"
                "WidgetService -> WidgetRepository\n",
                None,
            )
        if self.calls == 2:
            assert "不要输出最终报告，请继续使用工具探索" in messages[-1]["content"]
            return _tool_call(
                "call-follow-up",
                "search_code",
                '{"query":"WidgetRepository"}',
            )

        assert messages[-1]["role"] == "tool"
        return _response(
            "# 代码探索结果\n\n"
            "## 调用关系\n"
            "WidgetService -> WidgetRepository\n\n"
            "## 不确定项\n"
            "仍需结合业务配置确认外部边界。\n",
            None,
        )


class FakeConvergenceCheckpointLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        self.calls += 1
        if self.calls <= 3:
            return _tool_call(
                f"call-{self.calls}",
                "search_code",
                f'{{"query":"Widget{self.calls}"}}',
            )
        if self.calls == 4:
            assert "探索收敛检查点" in messages[-1]["content"]
            return _response("# 代码探索结果\n\nWidgetService -> WidgetRepository", None)

        assert "不要输出最终报告，请继续使用工具探索" in messages[-1]["content"]
        return _response("# 代码探索结果\n\nWidgetService -> WidgetRepository", None)


class FakeStagnatingLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        self.calls += 1
        if self.calls <= 6:
            return _tool_call(
                f"call-{self.calls}",
                "search_code",
                '{"query":"WidgetService"}',
            )
        if self.calls == 7:
            assert tools is not None
            assert "反停滞继续探索" in messages[-1]["content"]
            return _response("# 代码探索结果\n\nWidgetService -> WidgetRepository", None)

        assert tools is not None
        assert "不要输出最终报告，请继续使用工具探索" in messages[-1]["content"]
        return _response("# 代码探索结果\n\nWidgetService -> WidgetRepository", None)


class FakeFailingLLMClient:
    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        raise RuntimeError("upstream summary failed")


class FakeCancelledLLMClient:
    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        raise asyncio.CancelledError


def test_agent_runs_tool_loop_and_saves_complete_logs(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_module, "CodeGraphMCPClient", FakeCodeGraphMCPClient)
    settings = Settings(
        llm_api_key="test-key",
        llm_base_url=None,
        llm_model="test-model",
        codegraph_mcp_command="codegraph",
        codegraph_mcp_args=("serve", "--mcp"),
        codegraph_project_path=tmp_path,
        max_tool_calls=120,
        max_seconds=900,
        runs_dir=tmp_path / "runs",
    )
    llm_client = FakeLLMClient()

    result = asyncio.run(
        CodeExplorerAgent(settings, llm_client).run("请调查 codegraph_status 符号")
    )

    tool_log = result.artifacts.path("tool_calls.jsonl").read_text(encoding="utf-8")
    messages = result.artifacts.path("messages.md").read_text(encoding="utf-8")
    report = result.artifacts.path("report.md").read_text(encoding="utf-8")
    assert json.loads(tool_log)["tool"] == "check_code_index"
    assert "codegraph_status: indexed" in messages
    assert "````json" in messages
    assert "&quot;" not in messages
    assert "[源码正文已省略]" in report
    assert "class Hidden" not in report


def test_agent_writes_recovery_report_when_llm_request_fails(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_module, "CodeGraphMCPClient", FakeCodeGraphMCPClient)
    agent = CodeExplorerAgent(_settings(tmp_path), FakeFailingLLMClient())

    with pytest.raises(RuntimeError, match="upstream summary failed"):
        asyncio.run(agent.run("探索 Widget 功能的完整业务流程"))

    assert agent.last_artifacts is not None
    report = agent.last_artifacts.path("report.md").read_text(encoding="utf-8")
    assert "# 代码探索结果（未完成）" in report
    assert "RuntimeError: upstream summary failed" in report
    assert "不是完整最终结论" in report


def test_agent_writes_recovery_report_when_run_is_cancelled(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_module, "CodeGraphMCPClient", FakeCodeGraphMCPClient)
    agent = CodeExplorerAgent(_settings(tmp_path), FakeCancelledLLMClient())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agent.run("探索 Widget 功能的完整业务流程"))

    assert agent.last_artifacts is not None
    report = agent.last_artifacts.path("report.md").read_text(encoding="utf-8")
    assert "# 代码探索结果（未完成）" in report
    assert "CancelledError" in report


def test_repo_overview_rewrites_incomplete_report_without_tools(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_module, "CodeGraphMCPClient", FakeCodeGraphMCPClient)
    settings = _settings(tmp_path)
    llm_client = FakeOverviewLLMClient()

    result = asyncio.run(
        CodeExplorerAgent(settings, llm_client).run(
            "请检查代码索引，并概述这个仓库的主要模块结构、关键入口和建议进一步探索的方向。"
        )
    )

    assert len(llm_client.tool_arguments) == 2
    assert len(llm_client.tool_arguments[0] or []) == 10
    assert llm_client.tool_arguments[1] is None
    assert "## 顶层仓库结构" in result.report
    assert "## 主要模块职责" in result.report


def test_repo_overview_requires_every_section() -> None:
    report = (
        "# 代码探索结果\n\n"
        "## 索引状态\n可用。\n\n"
        "## 顶层仓库结构\n多个仓库。\n"
    )

    assert _needs_repo_overview_rewrite(report)


def test_repo_overview_accepts_complete_report_without_symbol_blacklist() -> None:
    report = (
        "# 代码探索结果\n\n"
        "## 索引状态\n可用。\n\n"
        "## 顶层仓库结构\n多个仓库。\n\n"
        "## 主要模块职责\n模块职责。\n\n"
        "## 关键启动入口\n入口。\n\n"
        "## 主要业务域\n业务域。\n\n"
        "## 建议进一步探索方向\n某个局部链路。\n\n"
        "## 不确定项\n待确认。\n"
    )

    assert not _needs_repo_overview_rewrite(report)


def test_agent_records_validation_error_and_allows_llm_to_self_correct(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_module, "CodeGraphMCPClient", FakeCodeGraphMCPClient)
    llm_client = FakeSelfCorrectingLLMClient()

    result = asyncio.run(
        CodeExplorerAgent(_settings(tmp_path), llm_client).run("调查 Widget 符号")
    )

    tool_logs = [
        json.loads(line)
        for line in result.artifacts.path("tool_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert tool_logs[0]["status"] == "validation_error"
    assert tool_logs[0]["mapped_mcp_tool_name"] == "codegraph_node"
    assert tool_logs[0]["started_at"]
    assert "[TOOL_ARGUMENT_ERROR]" in tool_logs[0]["result_preview"]
    assert tool_logs[1]["status"] == "success"
    assert not result.artifacts.path("errors.log").read_text(encoding="utf-8")


def test_feature_exploration_does_not_save_unclosed_candidate_report(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_module, "CodeGraphMCPClient", FakeCodeGraphMCPClient)
    llm_client = FakeContinuingLLMClient()

    agent = CodeExplorerAgent(
        _settings(tmp_path, max_continuations=1, fail_on_incomplete=True),
        llm_client,
    )
    with pytest.raises(ExplorationLimitError, match="remained incomplete"):
        asyncio.run(agent.run("探索 Widget 功能的完整业务流程"))

    assert llm_client.calls == 3
    assert agent.last_artifacts is not None
    artifacts = agent.last_artifacts
    assert artifacts.path("continuation_1.md").exists()
    assert artifacts.path("stop_decision_1.md").exists()
    assert artifacts.path("stop_decision_2.md").exists()
    assert artifacts.path("notebook.md").read_text(encoding="utf-8")
    assert "# 代码探索结果（未完成）" in artifacts.path("report.md").read_text(
        encoding="utf-8"
    )
    state = json.loads(
        artifacts.path("exploration_state.json").read_text(encoding="utf-8")
    )
    assert state["continuation_count"] == 1
    assert state["stop_decision"]["can_stop"] is False
    tool_log = json.loads(
        artifacts.path("tool_calls.jsonl").read_text(encoding="utf-8")
    )
    assert tool_log["tool_name"] == "search_code"


def test_feature_exploration_adds_one_convergence_checkpoint_near_tool_limit(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_module, "CodeGraphMCPClient", FakeCodeGraphMCPClient)
    llm_client = FakeConvergenceCheckpointLLMClient()

    agent = CodeExplorerAgent(
        _settings(
            tmp_path,
            max_continuations=1,
            max_tool_calls=4,
            fail_on_incomplete=True,
        ),
        llm_client,
    )
    with pytest.raises(ExplorationLimitError, match="remained incomplete"):
        asyncio.run(agent.run("探索 Widget 功能的完整业务流程"))

    assert agent.last_artifacts is not None
    checkpoint = agent.last_artifacts.path("convergence_checkpoint.md")
    assert checkpoint.exists()
    assert "3 / 4" in checkpoint.read_text(encoding="utf-8")
    messages = agent.last_artifacts.path("messages.md").read_text(encoding="utf-8")
    assert messages.count("探索收敛检查点") == 1


def test_feature_exploration_keeps_tools_after_repeated_stagnation_when_unclosed(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_module, "CodeGraphMCPClient", FakeCodeGraphMCPClient)
    llm_client = FakeStagnatingLLMClient()

    agent = CodeExplorerAgent(
        _settings(tmp_path, max_continuations=1, fail_on_incomplete=True),
        llm_client,
    )
    with pytest.raises(ExplorationLimitError, match="remained incomplete"):
        asyncio.run(agent.run("探索 Widget 功能的完整业务流程"))

    assert llm_client.calls == 8
    assert agent.last_artifacts is not None
    checkpoint = agent.last_artifacts.path("stagnation_checkpoint_1.md")
    assert checkpoint.exists()
    assert "反停滞继续探索" in checkpoint.read_text(encoding="utf-8")
    assert "禁止重复相同工具和参数" in checkpoint.read_text(encoding="utf-8")
    tool_calls = [
        json.loads(line)
        for line in agent.last_artifacts.path("tool_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(
        "REPEATED_TOOL_CALL_WARNING" in log["result_preview"] for log in tool_calls
    ) == 5


def _settings(
    tmp_path: Path,
    *,
    max_continuations: int = 3,
    max_tool_calls: int = 120,
    fail_on_incomplete: bool = False,
) -> Settings:
    return Settings(
        llm_api_key="test-key",
        llm_base_url=None,
        llm_model="test-model",
        codegraph_mcp_command="codegraph",
        codegraph_mcp_args=("serve", "--mcp"),
        codegraph_project_path=tmp_path,
        max_tool_calls=max_tool_calls,
        max_seconds=900,
        runs_dir=tmp_path / "runs",
        max_continuations=max_continuations,
        fail_on_incomplete=fail_on_incomplete,
    )


def _response(content: str, tool_calls: list[Any] | None) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
    )


def _tool_call(call_id: str, name: str, arguments: str) -> Any:
    return _response(
        "",
        [
            SimpleNamespace(
                id=call_id,
                type="function",
                function=SimpleNamespace(name=name, arguments=arguments),
            )
        ],
    )
