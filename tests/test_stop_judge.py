from app.agents.code_explorer.continuation import (
    build_anti_stagnation_prompt,
    build_continuation_prompt,
    build_convergence_checkpoint_prompt,
    build_stagnation_report_prompt,
)
from app.agents.code_explorer.exploration_state import (
    BusinessStage,
    Evidence,
    ExplorationState,
    StageEdge,
)
from app.agents.code_explorer.stop_judge import judge_stop, render_evidence_quality_report


def test_feature_exploration_shallow_three_part_report_cannot_stop() -> None:
    state = ExplorationState(goal="explore flow", task_mode="feature_exploration")

    decision = judge_stop(
        state,
        "# 代码探索结果\n\nService A -> Service B -> Service C",
    )

    assert not decision.can_stop
    assert decision.blocking_gaps


def test_feature_exploration_service_list_without_fields_cannot_stop() -> None:
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[BusinessStage(name="stage A"), BusinessStage(name="stage B")],
    )

    decision = judge_stop(state, "ServiceA\nServiceB\nServiceC")

    assert not decision.can_stop
    assert decision.missing_stage_fields["stage A"]
    assert "trigger" in decision.missing_stage_fields["stage A"]
    assert "persistence" in decision.missing_stage_fields["stage A"]
    assert "state_change" in decision.missing_stage_fields["stage A"]


def test_feature_exploration_missing_edge_cannot_stop() -> None:
    state = _closed_state()
    state.edges.clear()

    decision = judge_stop(state, _complete_report())

    assert not decision.can_stop
    assert decision.missing_edges == [
        "stage A -> stage B",
        "stage B -> stage C",
        "stage C -> stage D",
    ]


def test_two_confirmed_edges_do_not_create_spurious_adjacency_gaps() -> None:
    state = _closed_state()
    state.stages.append(BusinessStage(name="additional stage", evidence=[_evidence("extra")]))

    decision = judge_stop(state, _complete_report())

    assert not decision.missing_edges


def test_trigger_must_reach_persistence_over_confirmed_edge_graph() -> None:
    evidence = _evidence("confirmed")
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[
            BusinessStage(name="入口触发", trigger=evidence, evidence=[evidence]),
            BusinessStage(name="业务计算与转换", transform=evidence, evidence=[evidence]),
            BusinessStage(name="持久化写入", persistence=evidence, evidence=[evidence]),
            BusinessStage(name="状态流转", state_change=evidence, evidence=[evidence]),
        ],
        edges=[
            StageEdge(
                from_stage="入口触发",
                to_stage="业务计算与转换",
                connection_type="direct_call",
                evidence=[evidence],
                confidence="confirmed",
            ),
            StageEdge(
                from_stage="状态流转",
                to_stage="业务计算与转换",
                connection_type="direct_call",
                evidence=[evidence],
                confidence="confirmed",
            ),
        ],
    )

    decision = judge_stop(state, _complete_report())

    assert not decision.can_stop
    assert any("持久化" in gap and "连通" in gap for gap in decision.blocking_gaps)


def test_feature_exploration_actionable_follow_up_cannot_stop() -> None:
    state = _closed_state()
    report = _complete_report() + "\n建议后续验证：需检查 Job / Event / MQ / Repository。"

    decision = judge_stop(state, report)

    assert not decision.can_stop
    assert any("继续探索" in gap for gap in decision.blocking_gaps)


def test_external_boundary_unknown_can_stop_when_main_chain_is_closed() -> None:
    state = _closed_state()
    state.external_boundaries.append("payment provider callback")

    decision = judge_stop(state, _complete_report())

    assert decision.can_stop
    assert not decision.blocking_gaps


def test_all_inferred_stages_cannot_stop() -> None:
    inferred = Evidence(
        id="report-1",
        claim="Candidate report mentions stage",
        source_type="inferred",
        summary="Candidate report mentions stage",
        confidence="inferred",
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[
            BusinessStage(name="stage A", evidence=[inferred], confidence="inferred"),
            BusinessStage(name="stage B", evidence=[inferred], confidence="inferred"),
            BusinessStage(name="stage C", evidence=[inferred], confidence="inferred"),
        ],
    )

    decision = judge_stop(state, "主链已闭合。")

    assert not decision.can_stop
    assert any("strong evidence" in problem for problem in decision.evidence_problems)


def test_inferred_edges_cannot_satisfy_stop_judge() -> None:
    state = _closed_state()
    inferred = Evidence(
        id="report-edge",
        claim="stage A -> stage B",
        source_type="inferred",
        summary="candidate report arrow",
        confidence="inferred",
    )
    state.edges = [
        StageEdge(
            from_stage="stage A",
            to_stage="stage B",
            connection_type="direct_call",
            evidence=[inferred],
            confidence="inferred",
        ),
        StageEdge(
            from_stage="stage B",
            to_stage="stage C",
            connection_type="direct_call",
            evidence=[inferred],
            confidence="inferred",
        ),
    ]

    decision = judge_stop(state, "主链已闭合。")

    assert not decision.can_stop
    assert decision.missing_edges


def test_critical_unknown_event_consumer_blocks_stop() -> None:
    state = _closed_state()
    state.unknowns.append("事件下游 Listener / consumer 尚未确认")

    decision = judge_stop(state, _complete_report())

    assert not decision.can_stop
    assert any("事件下游" in gap for gap in decision.blocking_gaps)


def test_critical_unknown_status_transition_blocks_stop() -> None:
    state = _closed_state()
    state.unknowns.append("状态流转未确认")

    decision = judge_stop(state, _complete_report())

    assert not decision.can_stop
    assert any("状态流转" in gap for gap in decision.blocking_gaps)


def test_status_related_report_requires_confirmed_state_change_evidence() -> None:
    state = _closed_state()
    for stage in state.stages:
        stage.state_change = None

    decision = judge_stop(state, _complete_report())

    assert not decision.can_stop
    assert any("状态" in gap for gap in decision.blocking_gaps)


def test_event_publish_requires_consumer_evidence_or_external_boundary() -> None:
    state = _closed_state()

    decision = judge_stop(state, _complete_report() + "\n发布 WidgetEvent。")

    assert not decision.can_stop
    assert any("事件" in gap for gap in decision.blocking_gaps)


def test_external_boundary_does_not_hide_missing_main_stage_fields() -> None:
    state = _closed_state()
    for stage in state.stages:
        stage.persistence = None
    state.external_boundaries.append("external persistence API")

    decision = judge_stop(state, _complete_report())

    assert not decision.can_stop
    assert decision.missing_stage_fields


def test_missing_main_stage_field_is_a_hard_stop_even_with_strong_graph() -> None:
    state = _closed_state()
    state.stages[0].input = None

    decision = judge_stop(state, _complete_report())

    assert not decision.can_stop
    assert decision.missing_stage_fields == {"stage A": ["input"]}
    assert "CAN_STOP: no" in decision.to_markdown()


def test_optional_stage_can_keep_missing_fields_without_blocking_stop() -> None:
    state = _closed_state()
    state.stages.append(BusinessStage(name="optional follow-up", optional=True))

    decision = judge_stop(state, _complete_report())

    assert decision.can_stop
    assert "optional follow-up" not in decision.missing_stage_fields


def test_evidence_quality_report_lists_counts_and_unqualified_graph_items() -> None:
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[BusinessStage(name="stage without evidence")],
        edges=[
            StageEdge(
                from_stage="stage without evidence",
                to_stage="unknown destination",
                connection_type="direct_call",
            )
        ],
    )

    report = render_evidence_quality_report(state)

    assert "Total evidence: 0" in report
    assert "Stages without strong evidence" in report
    assert "stage without evidence" in report
    assert "Edges without strong evidence" in report
    assert "stage without evidence -> unknown destination" in report
    assert "Why StopJudge blocked" in report


def test_evidence_quality_report_lists_downgrade_reason() -> None:
    weak = Evidence(
        id="weak-path",
        claim="Found repository directory",
        source_type="file_path",
        summary="src/repository",
        confidence="confirmed",
        strength="weak",
        reason="A directory name is only a navigation lead.",
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        evidence_items=[weak],
    )

    report = render_evidence_quality_report(state)

    assert "## Downgraded Evidence" in report
    assert "A directory name is only a navigation lead." in report


def test_repo_overview_does_not_use_feature_stop_rules() -> None:
    state = ExplorationState(goal="overview", task_mode="repo_overview")

    decision = judge_stop(state, "short overview")

    assert decision.can_stop


def test_continuation_prompt_contains_missing_fields_and_edges() -> None:
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        confirmed_facts=[_evidence("confirmed call")],
        candidate_next_actions=["Inspect Repository and Event candidates"],
    )
    decision = judge_stop(state, "ServiceA -> ServiceB")

    prompt = build_continuation_prompt(decision, state)

    assert "不要输出最终报告，请继续使用工具探索" in prompt
    assert "缺少必要证据" in prompt
    assert "相邻阶段之间仍然断链" in prompt
    assert "Inspect Repository and Event candidates" in prompt
    assert "不合格证据" in prompt
    assert "不要重复已经成功执行过的完全相同工具调用" in prompt


def test_convergence_checkpoint_prompt_requests_candidate_or_targeted_follow_up() -> None:
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        candidate_next_actions=["Inspect candidate symbol: WidgetRepository"],
    )

    prompt = build_convergence_checkpoint_prompt(state, 90, 120)

    assert "探索收敛检查点" in prompt
    assert "90 / 120" in prompt
    assert "候选 Markdown《代码探索结果》" in prompt
    assert "仍可继续自主选择工具" in prompt
    assert "WidgetRepository" in prompt


def test_stagnation_report_prompt_requests_one_no_tools_candidate_report() -> None:
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        candidate_next_actions=["Inspect candidate symbol: WidgetRepository"],
    )

    prompt = build_stagnation_report_prompt(state, 5)

    assert "探索停滞检查点" in prompt
    assert "连续 5 次" in prompt
    assert "本轮不要调用工具" in prompt
    assert "候选 Markdown《代码探索结果》" in prompt
    assert "已确认事实、合理推断和未知项" in prompt


def test_anti_stagnation_prompt_keeps_tools_available_for_unclosed_state() -> None:
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        candidate_next_actions=["Inspect candidate symbol: WidgetRepository"],
    )
    decision = judge_stop(state, "")

    prompt = build_anti_stagnation_prompt(
        decision,
        state,
        5,
        ['get_symbol_detail({"symbol":"WidgetService"})'],
    )

    assert "反停滞继续探索" in prompt
    assert "不要输出最终报告" in prompt
    assert "禁止重复相同工具和参数" in prompt
    assert "WidgetService" in prompt
    assert decision.blocking_gaps[0] in prompt


def _closed_state() -> ExplorationState:
    evidence = _evidence("confirmed")
    first = BusinessStage(
        name="stage A",
        trigger=evidence,
        input=evidence,
        transform=evidence,
        output=evidence,
        persistence=evidence,
        state_change=evidence,
    )
    second = BusinessStage(
        name="stage B",
        trigger=evidence,
        input=evidence,
        transform=evidence,
        output=evidence,
        persistence=evidence,
        state_change=evidence,
    )
    third = BusinessStage(
        name="stage C",
        trigger=evidence,
        input=evidence,
        transform=evidence,
        output=evidence,
        persistence=evidence,
        state_change=evidence,
    )
    fourth = BusinessStage(
        name="stage D",
        trigger=evidence,
        input=evidence,
        transform=evidence,
        output=evidence,
        persistence=evidence,
        state_change=evidence,
    )
    return ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[first, second, third, fourth],
        edges=[
            StageEdge(
                from_stage="stage A",
                to_stage="stage B",
                connection_type="direct_call",
                evidence=[evidence],
                confidence="confirmed",
            ),
            StageEdge(
                from_stage="stage B",
                to_stage="stage C",
                connection_type="repository_read_write",
                evidence=[evidence],
                confidence="confirmed",
            ),
            StageEdge(
                from_stage="stage C",
                to_stage="stage D",
                connection_type="status_transition",
                evidence=[evidence],
                confidence="confirmed",
            ),
        ],
    )


def _evidence(claim: str) -> Evidence:
    return Evidence(
        id="e-1",
        claim=claim,
        source_type="call_edge",
        summary=claim,
        confidence="confirmed",
        strength="strong",
        can_satisfy_stage_field=True,
        can_satisfy_stage_edge=True,
        reason="Explicit strong fixture evidence.",
    )


def _complete_report() -> str:
    return """\
# 代码探索结果

## 业务阶段
- stage A: 触发者 Job，输入 DTO，转换为 PO，输出 detail，持久化到 Repository，状态变更为 READY。
- stage B: 触发者 callback，输入 detail，转换为 order，输出 response，持久化到 Repository，状态变更为 DONE。

## 阶段连接
- stage A -> stage B: direct_call，已确认。

## 外部边界
- 外部回调语义需确认。
"""
