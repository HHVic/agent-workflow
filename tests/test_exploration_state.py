import json

from app.agents.code_explorer.exploration_state import (
    BusinessStage,
    Evidence,
    ExplorationState,
    StageEdge,
    StopDecision,
    is_confirmed_tool_evidence,
    update_state_from_candidate_report,
    update_state_from_tool_logs,
)


def test_exploration_state_serializes_stages_edges_and_decision() -> None:
    evidence = Evidence(
        id="e-1",
        claim="Job triggers calculation",
        source_type="call_edge",
        file_path="src/Job.java",
        symbol="run",
        summary="run calls calculate",
        confidence="confirmed",
    )
    first = BusinessStage(
        name="raw detail",
        trigger=evidence,
        input=evidence,
        transform=evidence,
        output=evidence,
        persistence=evidence,
        state_change=evidence,
        evidence=[evidence],
        confidence="confirmed",
    )
    second = BusinessStage(
        name="settlement",
        trigger=evidence,
        input=evidence,
        transform=evidence,
        output=evidence,
        persistence=evidence,
        state_change=evidence,
        evidence=[evidence],
        confidence="confirmed",
    )
    edge = StageEdge(
        from_stage="raw detail",
        to_stage="settlement",
        connection_type="direct_call",
        evidence=[evidence],
        confidence="confirmed",
    )
    decision = StopDecision(can_stop=True, reason="main chain is closed")
    state = ExplorationState(
        goal="explore workflow",
        task_mode="feature_exploration",
        stages=[first, second],
        edges=[edge],
        confirmed_facts=[evidence],
        open_questions=["external callback semantics"],
        stop_decision=decision,
    )

    payload = json.loads(state.to_json())

    assert payload["stages"][0]["name"] == "raw detail"
    assert payload["edges"][0]["connection_type"] == "direct_call"
    assert payload["open_questions"] == ["external callback semantics"]
    assert payload["stop_decision"]["can_stop"] is True


def test_exploration_state_can_add_unique_candidate_actions() -> None:
    state = ExplorationState(goal="explore", task_mode="feature_exploration")

    state.add_candidate_action("Inspect WidgetService")
    state.add_candidate_action("Inspect WidgetService")

    assert state.candidate_next_actions == ["Inspect WidgetService"]


def test_candidate_report_extracts_only_explicit_business_stages_and_main_flow() -> None:
    state = ExplorationState(goal="explore", task_mode="feature_exploration")
    report = """\
# 代码探索结果

## 调用关系与业务流程

核心流程分为两个阶段：收入明细计算 → 结算单创建。

### 阶段一：收入明细计算
- 状态变化：PENDING → DISCLOSED。
- 调用：calculate() -> save()。

### 阶段二：结算单创建
- 状态变化：RELEASED → PAID。

## 建议的后续验证
- 需确认外部回调语义。
"""

    update_state_from_candidate_report(state, report)

    assert [stage.name for stage in state.stages] == ["收入明细计算", "结算单创建"]
    assert [(edge.from_stage, edge.to_stage) for edge in state.edges] == [
        ("收入明细计算", "结算单创建")
    ]
    assert all(stage.confidence != "confirmed" for stage in state.stages)
    assert all(not stage.optional for stage in state.stages)
    assert all(
        getattr(stage, field) is None
        for stage in state.stages
        for field in (
            "trigger",
            "input",
            "transform",
            "output",
            "persistence",
            "state_change",
        )
    )
    assert all(edge.confidence != "confirmed" for edge in state.edges)


def test_inferred_report_evidence_is_not_confirmed_tool_evidence() -> None:
    evidence = Evidence(
        id="report-1",
        claim="Candidate report mentions stage",
        source_type="inferred",
        summary="Candidate report mentions stage",
        confidence="inferred",
    )

    assert not is_confirmed_tool_evidence(evidence)


def test_located_tool_evidence_is_confirmed_tool_evidence() -> None:
    evidence = Evidence(
        id="tool-1",
        claim="Found WidgetRepository.save",
        source_type="repository",
        summary="WidgetRepository.save",
        confidence="confirmed",
        file_path="src/WidgetRepository.java:10",
        symbol="WidgetRepository.save",
    )

    assert is_confirmed_tool_evidence(evidence)


def test_tool_log_response_preserves_multiline_mcp_text_for_extraction() -> None:
    state = ExplorationState(goal="explore", task_mode="feature_exploration")

    update_state_from_tool_logs(
        state,
        [
            {
                "status": "success",
                "tool_name": "find_callers",
                "arguments": {"symbol": "WidgetService.calculate"},
                "response": {
                    "ok": True,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "## Callers of WidgetService.calculate (1 found)\n\n"
                                    "- WidgetJob.doExecute (method) - "
                                    "src/main/java/example/WidgetJob.java:22"
                                ),
                            }
                        ]
                    },
                },
            }
        ],
    )

    assert any(
        item.source_type == "call_edge"
        and item.symbol == "WidgetJob.doExecute -> WidgetService.calculate"
        for item in state.evidence_items
    )


def test_business_stage_default_stage_type() -> None:
    stage = BusinessStage(name="test stage")
    assert stage.stage_type == "unknown"


def test_business_stage_default_is_mainline() -> None:
    stage = BusinessStage(name="test stage")
    assert stage.is_mainline is False


def test_business_stage_default_is_optional() -> None:
    stage = BusinessStage(name="test stage")
    assert stage.is_optional is False


def test_business_stage_default_capabilities() -> None:
    stage = BusinessStage(name="test stage")
    assert stage.capabilities == []


def test_business_stage_default_required_fields() -> None:
    stage = BusinessStage(name="test stage")
    assert stage.required_fields == []


def test_business_stage_from_dict_backward_compat() -> None:
    old_format = {
        "name": "收入明细计算",
        "description": "计算收入明细",
        "confidence": "inferred",
        "evidence": [],
        "open_questions": [],
        "optional": False,
    }
    stage = BusinessStage.from_dict(old_format)
    assert stage.name == "收入明细计算"
    assert stage.stage_type == "unknown"
    assert stage.is_mainline is False
    assert stage.is_optional is False
    assert stage.capabilities == []
    assert stage.required_fields == []


def test_business_stage_from_dict_with_new_fields() -> None:
    full_format = {
        "name": "结算单创建",
        "stage_type": "settlement",
        "is_mainline": True,
        "is_optional": False,
        "capabilities": ["create", "validate"],
        "required_fields": ["trigger", "output"],
        "confidence": "confirmed",
    }
    stage = BusinessStage.from_dict(full_format)
    assert stage.stage_type == "settlement"
    assert stage.is_mainline is True
    assert stage.is_optional is False
    assert stage.capabilities == ["create", "validate"]
    assert stage.required_fields == ["trigger", "output"]
