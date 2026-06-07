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
    _build_incomplete_report,
)
from app.agents.code_explorer.exploration_state import (
    BusinessStage,
    Evidence,
    ExplorationState,
    StageEdge,
    StopDecision,
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


class FakeIncompleteLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        self.calls += 1
        if self.calls == 2:
            assert "不要输出最终报告，请继续使用工具探索" in messages[-1]["content"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "# 代码探索结果\n\n"
                            "## 调用关系\n"
                            "WidgetService -> WidgetRepository\n\n"
                            "所有阶段均已确认，主链完整闭合，无断链。"
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )


def test_incomplete_feature_returns_phase_report_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_module, "CodeGraphMCPClient", FakeCodeGraphMCPClient)
    agent = CodeExplorerAgent(_settings(tmp_path), FakeIncompleteLLMClient())

    result = asyncio.run(agent.run("探索 Widget 功能的完整业务流程"))

    assert "# 代码探索结果（阶段性）" in result.report
    assert "Evidence Gate 未完全通过" in result.report
    assert "## 运行状态" in result.report
    assert "Evidence Gate: not passed" in result.report
    assert "continuation count: 1" in result.report
    assert "tool call count: 0" in result.report
    assert "## 已确认主线事实" in result.report
    assert "## 已识别业务阶段" in result.report
    assert "## 已确认阶段连接" in result.report
    assert "## 未闭合的关键断点" in result.report
    assert "Missing Stage Fields" in result.report
    assert "## 不确定项" in result.report
    assert "## 下一步最有价值探索方向" in result.report
    assert "所有阶段均已确认" not in result.report
    assert "完整闭合" not in result.report
    assert "无断链" not in result.report
    assert result.artifacts.path("report.md").read_text(encoding="utf-8") == result.report

    state = json.loads(
        result.artifacts.path("exploration_state.json").read_text(encoding="utf-8")
    )
    assert state["stop_decision"]["can_stop"] is False


def test_incomplete_feature_still_raises_in_fail_fast_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_module, "CodeGraphMCPClient", FakeCodeGraphMCPClient)
    agent = CodeExplorerAgent(
        _settings(tmp_path, fail_on_incomplete=True),
        FakeIncompleteLLMClient(),
    )

    with pytest.raises(ExplorationLimitError, match="remained incomplete"):
        asyncio.run(agent.run("探索 Widget 功能的完整业务流程"))

    assert agent.last_artifacts is not None
    state = json.loads(
        agent.last_artifacts.path("exploration_state.json").read_text(encoding="utf-8")
    )
    assert state["stop_decision"]["can_stop"] is False


def test_phase_report_uses_strong_state_and_relevance_gated_next_actions() -> None:
    strong_fact = Evidence(
        id="strong-fact",
        claim="WidgetJob invokes WidgetService",
        source_type="call_edge",
        summary="WidgetJob -> WidgetService",
        confidence="confirmed",
        symbol="WidgetJob -> WidgetService",
        strength="strong",
        can_satisfy_stage_edge=True,
        reason="Matched direct call edge.",
    )
    weak_edge = Evidence(
        id="weak-edge",
        claim="WidgetService may call Response",
        source_type="call_edge",
        summary="WidgetService -> Response",
        confidence="confirmed",
        symbol="WidgetService -> Response",
        strength="weak",
        reason="Wrapper edge.",
    )
    decision = StopDecision(
        can_stop=False,
        reason="Evidence Gate is incomplete.",
        blocking_gaps=["Need persistence evidence."],
        missing_stage_fields={"入口触发": ["persistence"]},
        evidence_problems=["Stage 入口触发.persistence lacks strong evidence."],
    )
    state = ExplorationState(
        goal="widget flow",
        task_mode="feature_exploration",
        stages=[
            BusinessStage(
                name="入口触发",
                confidence="confirmed",
                evidence=[strong_fact],
            ),
            BusinessStage(name="后续回调", confidence="inferred"),
            BusinessStage(name="候选支线", optional=True),
        ],
        edges=[
            StageEdge(
                from_stage="入口触发",
                to_stage="业务计算",
                connection_type="direct_call",
                evidence=[strong_fact],
                confidence="confirmed",
            ),
            StageEdge(
                from_stage="业务计算",
                to_stage="响应包装",
                connection_type="direct_call",
                evidence=[weak_edge],
                confidence="confirmed",
            ),
        ],
        confirmed_facts=[strong_fact],
        candidate_next_actions=["WidgetRepository", "OpenOrderEvent"],
        stop_decision=decision,
        continuation_count=2,
    )

    report = _build_incomplete_report(state, "continuation fuse reached", 7)

    assert "WidgetJob invokes WidgetService" in report
    assert "入口触发 (confirmed)" in report
    assert "后续回调 (inferred)" in report
    assert "候选支线 (optional)" in report
    assert "入口触发 -> 业务计算: direct_call" in report
    assert "业务计算 -> 响应包装" not in report
    assert "WidgetRepository" in report
    assert "OpenOrderEvent" not in report
    assert "Need persistence evidence." in report


def _settings(tmp_path: Path, *, fail_on_incomplete: bool = False) -> Settings:
    return Settings(
        llm_api_key="test-key",
        llm_base_url=None,
        llm_model="test-model",
        codegraph_mcp_command="codegraph",
        codegraph_mcp_args=("serve", "--mcp"),
        codegraph_project_path=tmp_path,
        max_tool_calls=120,
        max_seconds=900,
        runs_dir=tmp_path / "runs",
        max_continuations=1,
        fail_on_incomplete=fail_on_incomplete,
    )
