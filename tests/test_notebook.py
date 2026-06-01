from app.agents.code_explorer.exploration_state import (
    BusinessStage,
    Evidence,
    ExplorationState,
    StageEdge,
)
from app.agents.code_explorer.notebook import (
    build_notebook_update_prompt,
    update_notebook,
)


def test_notebook_prompt_requests_internal_chain_tracking_not_final_report() -> None:
    state = ExplorationState(
        goal="explore workflow",
        task_mode="feature_exploration",
        confirmed_facts=[
            Evidence(
                id="e-1",
                claim="Job invokes service",
                source_type="call_edge",
                summary="caller result",
                confidence="confirmed",
                symbol="WidgetJob.doExecute -> WidgetService.calculate",
                strength="strong",
                can_satisfy_stage_edge=True,
                reason="Matched direct business call edge.",
            )
        ],
        branches=["Job -> Service"],
        unknowns=["persistence is unknown"],
    )

    prompt = build_notebook_update_prompt(state, [], [])

    assert "已确认业务事实" in prompt
    assert "Job invokes service" in prompt
    assert "当前业务链路假设" in prompt
    assert "Job -> Service" in prompt
    assert "当前断链点" in prompt
    assert "不要输出最终报告" in prompt


def test_update_notebook_renders_required_sections() -> None:
    evidence = Evidence(
        id="tool-7-edge-1",
        claim="WidgetJob.doExecute calls WidgetService.calculate",
        source_type="call_edge",
        summary="WidgetJob.doExecute -> WidgetService.calculate",
        confidence="confirmed",
        symbol="WidgetJob.doExecute -> WidgetService.calculate",
        file_path="src/main/java/example/WidgetJob.java:22",
        strength="strong",
        can_satisfy_stage_field=True,
        can_satisfy_stage_edge=True,
        reason="Matched direct business call edge.",
    )
    state = ExplorationState(
        goal="explore workflow",
        task_mode="feature_exploration",
        confirmed_facts=[evidence],
        stages=[BusinessStage(name="入口触发", trigger=evidence, evidence=[evidence])],
        edges=[
            StageEdge(
                from_stage="入口触发",
                to_stage="业务计算与转换",
                connection_type="direct_call",
                evidence=[evidence],
                confidence="confirmed",
            )
        ],
        unknowns=["mapping table is unknown"],
        candidate_next_actions=["Inspect Repository candidates"],
    )

    notebook = update_notebook(state, [], [])

    assert "# Exploration Notebook" in notebook
    assert "## 已确认业务事实" in notebook
    assert "## 已确认调用/连接" in notebook
    assert "## 已识别业务阶段" in notebook
    assert "## 阶段字段缺口" in notebook
    assert "## 阶段间断链" in notebook
    assert "## 外部/动态边界" in notebook
    assert "## 下一步最高价值探索方向" in notebook
    assert "[strong][call_edge][tool-7-edge-1]" in notebook
    assert "WidgetJob.doExecute -> WidgetService.calculate" in notebook
    assert "src/main/java/example/WidgetJob.java:22" in notebook
    assert "mapping table is unknown" in notebook
    assert "Inspect Repository candidates" in notebook


def test_notebook_omits_generic_tool_success_messages() -> None:
    state = ExplorationState(
        goal="explore workflow",
        task_mode="feature_exploration",
        confirmed_facts=[
            Evidence(
                id="tool-1",
                claim="get_code_context returned exploration evidence",
                source_type="tool_result",
                summary="get_code_context returned exploration evidence",
                confidence="confirmed",
            )
        ],
    )

    notebook = update_notebook(state, [], [])

    assert "returned exploration evidence" not in notebook


def test_notebook_omits_test_source_facts_but_local_evidence_can_keep_them() -> None:
    test_evidence = Evidence(
        id="test-path",
        claim="Found test helper",
        source_type="tool_result",
        summary="Found test helper",
        confidence="confirmed",
        symbol="should_calculate_amount",
        file_path="src/test/java/example/WidgetServiceTest.java:22",
    )
    state = ExplorationState(
        goal="explore workflow",
        task_mode="feature_exploration",
        confirmed_facts=[test_evidence],
        evidence_items=[test_evidence],
    )

    notebook = update_notebook(state, [], [])

    assert "should_calculate_amount" not in notebook
    assert state.evidence_items == [test_evidence]


def test_notebook_keeps_weak_leads_separate_and_omits_noise() -> None:
    weak = Evidence(
        id="weak-1",
        claim="Found WidgetRepository",
        source_type="tool_result",
        summary="WidgetRepository",
        confidence="confirmed",
        strength="weak",
        reason="A located symbol is only a lead.",
    )
    noise = Evidence(
        id="noise-1",
        claim="Response.SUCCESS wrapper",
        source_type="semantic_noise",
        summary="Response.SUCCESS(value)",
        confidence="confirmed",
        strength="noise",
        reason="Wrapper noise.",
    )
    state = ExplorationState(
        goal="explore workflow",
        task_mode="feature_exploration",
        evidence_items=[weak, noise],
    )

    notebook = update_notebook(state, [], [])

    assert "## 弱线索" in notebook
    assert "[weak][tool_result][weak-1]" in notebook
    assert "Response.SUCCESS wrapper" not in notebook
