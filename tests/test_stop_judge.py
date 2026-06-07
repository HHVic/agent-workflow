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
from app.agents.code_explorer.stage_normalizer import normalize_stage_candidates


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


# -- Stage-type-aware stop judge tests --


def _strong_evidence() -> Evidence:
    return Evidence(
        id="e-strong",
        claim="strong evidence",
        source_type="call_edge",
        summary="strong evidence",
        confidence="confirmed",
        strength="strong",
        can_satisfy_stage_field=True,
        can_satisfy_stage_edge=True,
    )


def test_supporting_stage_missing_fields_does_not_block() -> None:
    """Supporting stage with all fields missing should not block CAN_STOP."""
    strong = _strong_evidence()
    supporting = BusinessStage(
        name="持久化写入",
        stage_type="supporting",
        is_mainline=False,
        capabilities=["has_persistence_write"],
    )
    main_stage = BusinessStage(
        name="订单创建",
        trigger=strong, input=strong, transform=strong,
        output=strong, persistence=strong, state_change=strong,
        evidence=[strong],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main_stage, supporting],
    )
    decision = judge_stop(state, "")
    assert decision.can_stop
    assert "持久化写入" not in decision.missing_stage_fields


def test_optional_stage_missing_fields_does_not_block() -> None:
    """Optional stage with missing fields should not block."""
    strong = _strong_evidence()
    main = BusinessStage(
        name="订单创建",
        trigger=strong, input=strong, transform=strong,
        output=strong, persistence=strong, state_change=strong,
        evidence=[strong],
    )
    optional = BusinessStage(name="可选通知", optional=True)
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main, optional],
    )
    decision = judge_stop(state, "")
    assert decision.can_stop
    assert "可选通知" not in decision.missing_stage_fields


def test_scheduled_job_missing_persistence_state_change_does_not_block() -> None:
    """Scheduled job stage missing persistence/state_change should not block."""
    strong = _strong_evidence()
    main = BusinessStage(
        name="订单创建",
        trigger=strong, input=strong, transform=strong,
        output=strong, persistence=strong, state_change=strong,
        evidence=[strong],
    )
    scheduled = BusinessStage(
        name="定时任务触发",
        stage_type="scheduled_job",
        is_mainline=True,
        capabilities=["has_trigger"],
        trigger=strong,
        input=strong,
        output=strong,
        evidence=[strong],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main, scheduled],
    )
    decision = judge_stop(state, "")
    assert decision.can_stop


def test_calculation_missing_required_fields_blocks() -> None:
    """Calculation stage missing input/transform/output should block."""
    strong = _strong_evidence()
    main = BusinessStage(
        name="订单创建",
        trigger=strong, input=strong, transform=strong,
        output=strong, persistence=strong, state_change=strong,
        evidence=[strong],
    )
    calc = BusinessStage(
        name="业务计算与转换",
        stage_type="calculation",
        is_mainline=True,
        capabilities=["has_transform"],
        evidence=[strong],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main, calc],
    )
    decision = judge_stop(state, "")
    assert not decision.can_stop
    assert "业务计算与转换" in decision.missing_stage_fields


def test_persistence_read_missing_state_change_does_not_block() -> None:
    """Persistence read stage missing state_change should not block."""
    strong = _strong_evidence()
    main = BusinessStage(
        name="订单创建",
        trigger=strong, input=strong, transform=strong,
        output=strong, persistence=strong, state_change=strong,
        evidence=[strong],
    )
    pread = BusinessStage(
        name="可持久化数据读取",
        stage_type="persistence_read",
        is_mainline=True,
        capabilities=["has_persistence_read"],
        input=strong,
        persistence=strong,
        output=strong,
        evidence=[strong],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main, pread],
    )
    decision = judge_stop(state, "")
    assert decision.can_stop


def test_event_publish_missing_persistence_does_not_block() -> None:
    """Event publish stage missing persistence should not block."""
    strong = _strong_evidence()
    main = BusinessStage(
        name="订单创建",
        trigger=strong, input=strong, transform=strong,
        output=strong, persistence=strong, state_change=strong,
        evidence=[strong],
    )
    epub = BusinessStage(
        name="事件发布与消费",
        stage_type="event_publish",
        is_mainline=True,
        capabilities=["has_event_publish"],
        trigger=strong,
        output=strong,
        evidence=[strong],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main, epub],
    )
    decision = judge_stop(state, "")
    assert decision.can_stop


def test_settlement_creation_missing_persistence_blocks() -> None:
    """Settlement creation missing persistence should block."""
    strong = _strong_evidence()
    main = BusinessStage(
        name="订单创建",
        trigger=strong, input=strong, transform=strong,
        output=strong, persistence=strong, state_change=strong,
        evidence=[strong],
    )
    settlement = BusinessStage(
        name="结算创建",
        stage_type="settlement_creation",
        is_mainline=True,
        trigger=strong,
        input=strong,
        transform=strong,
        output=strong,
        evidence=[strong],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main, settlement],
    )
    decision = judge_stop(state, "")
    assert not decision.can_stop
    assert "结算创建" in decision.missing_stage_fields


def test_blocking_mainline_missing_required_fields_blocks() -> None:
    """Blocking mainline stage missing required fields should block CAN_STOP."""
    strong = _strong_evidence()
    main = BusinessStage(
        name="订单创建",
        trigger=strong, input=strong, transform=strong,
        output=strong, persistence=strong, state_change=strong,
        evidence=[strong],
    )
    blocking = BusinessStage(
        name="结算创建",
        stage_type="settlement_creation",
        is_mainline=True,
        trigger=strong,
        input=strong,
        transform=strong,
        # missing output and persistence
        evidence=[strong],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main, blocking],
    )
    decision = judge_stop(state, "")
    assert not decision.can_stop
    assert "结算创建" in decision.missing_stage_fields


def test_supporting_evidence_problems_do_not_block() -> None:
    """Evidence problems in supporting stages should not block."""
    strong = _strong_evidence()
    main = BusinessStage(
        name="订单创建",
        trigger=strong, input=strong, transform=strong,
        output=strong, persistence=strong, state_change=strong,
        evidence=[strong],
    )
    supporting = BusinessStage(
        name="持久化写入",
        stage_type="supporting",
        is_mainline=False,
        capabilities=["has_persistence_write"],
        # No evidence at all
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main, supporting],
    )
    decision = judge_stop(state, "")
    # Supporting stage problems should not block
    assert decision.can_stop


def test_blocking_evidence_problems_block() -> None:
    """Evidence problems in blocking stages should block."""
    strong = _strong_evidence()
    main = BusinessStage(
        name="订单创建",
        trigger=strong, input=strong, transform=strong,
        output=strong, persistence=strong, state_change=strong,
        evidence=[strong],
    )
    calc = BusinessStage(
        name="业务计算与转换",
        stage_type="calculation",
        is_mainline=True,
        capabilities=["has_transform"],
        evidence=[],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main, calc],
    )
    decision = judge_stop(state, "")
    assert not decision.can_stop
    assert any("业务计算与转换" in p for p in decision.evidence_problems)


def test_only_supporting_gaps_allows_stop() -> None:
    """When only supporting stages have gaps, CAN_STOP should be yes."""
    strong = _strong_evidence()
    main = BusinessStage(
        name="订单创建",
        trigger=strong, input=strong, transform=strong,
        output=strong, persistence=strong, state_change=strong,
        evidence=[strong],
    )
    supporting = BusinessStage(
        name="持久化写入",
        stage_type="supporting",
        is_mainline=False,
        capabilities=["has_persistence_write"],
        evidence=[],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main, supporting],
    )
    decision = judge_stop(state, "")
    assert decision.can_stop


def test_qualified_stages_do_not_count_supporting() -> None:
    """Qualified business stages should not include supporting technical stages."""
    from app.agents.code_explorer.stop_judge import _confirmed_stages

    supporting = BusinessStage(
        name="持久化写入",
        stage_type="supporting",
        is_mainline=False,
        capabilities=["has_persistence_write"],
        trigger=_strong_evidence(),
        input=_strong_evidence(),
        transform=_strong_evidence(),
        output=_strong_evidence(),
        persistence=_strong_evidence(),
        state_change=_strong_evidence(),
        evidence=[_strong_evidence()],
    )
    main = BusinessStage(
        name="订单创建",
        trigger=_strong_evidence(),
        input=_strong_evidence(),
        transform=_strong_evidence(),
        output=_strong_evidence(),
        persistence=_strong_evidence(),
        state_change=_strong_evidence(),
        evidence=[_strong_evidence()],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main, supporting],
    )
    confirmed = _confirmed_stages(state)
    assert len(confirmed) == 1
    assert confirmed[0].name == "订单创建"


def test_qualified_edges_do_not_count_supporting_edges() -> None:
    """Qualified stage edges should not count supporting edges as mainline."""
    from app.agents.code_explorer.stop_judge import _blocking_edge_count

    strong = _strong_evidence()
    main = BusinessStage(
        name="订单创建",
        trigger=strong, input=strong, transform=strong,
        output=strong, persistence=strong, state_change=strong,
        evidence=[strong],
    )
    supporting = BusinessStage(
        name="持久化写入",
        stage_type="supporting",
        is_mainline=False,
        capabilities=["has_persistence_write"],
        evidence=[strong],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[main, supporting],
        edges=[
            StageEdge(
                from_stage="订单创建",
                to_stage="持久化写入",
                connection_type="direct_call",
                evidence=[strong],
                confidence="confirmed",
            ),
        ],
    )
    # The edge from main to supporting should not count as mainline qualified
    count = _blocking_edge_count(state)
    assert count == 0


def test_stage_normalizer_supporting_stages_not_counted_in_qualified() -> None:
    """Stages normalized by StageNormalizer should not count as qualified."""
    from app.agents.code_explorer.stop_judge import _confirmed_stages

    stages = normalize_stage_candidates([_strong_evidence()])
    main = BusinessStage(
        name="订单创建",
        trigger=_strong_evidence(),
        input=_strong_evidence(),
        transform=_strong_evidence(),
        output=_strong_evidence(),
        persistence=_strong_evidence(),
        state_change=_strong_evidence(),
        evidence=[_strong_evidence()],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=stages + [main],
    )
    confirmed = _confirmed_stages(state)
     # Only the manually created main stage should be counted
    assert len(confirmed) == 1


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


# -- _main_chain_reaches_persistence tightened single-stage early return tests --


def test_controller_or_rpc_single_stage_with_repository_read_should_not_early_return() -> None:
    """controller_or_rpc stage with only trigger + repository read persistence
    should NOT be considered as main chain reaching persistence via single-stage
    early return."""
    trigger = _strong_evidence()
    # repository read evidence: has can_satisfy_stage_field but indicates read only
    read_evidence = Evidence(
        id="e-repo-read",
        claim="Found repository read in WidgetRepository.findById()",
        source_type="repository",
        summary="repository read from WidgetRepository",
        confidence="confirmed",
        strength="strong",
        can_satisfy_stage_field=True,
        can_satisfy_stage_edge=False,
        reason="Repository read operation detected.",
    )
    controller = BusinessStage(
        name="RPC入口",
        stage_type="controller_or_rpc",
        is_mainline=True,
        capabilities=["has_controller"],
        trigger=trigger,
        persistence=read_evidence,
        evidence=[trigger, read_evidence],
    )
    state = ExplorationState(
        goal="explore RPC flow",
        task_mode="feature_exploration",
        stages=[controller],
    )

    from app.agents.code_explorer.stop_judge import _main_chain_reaches_persistence

    assert not _main_chain_reaches_persistence([controller], state)


def test_settlement_creation_single_stage_with_full_fields_should_early_return() -> None:
    """settlement_creation stage with trigger + input + transform + output + persistence_write
    should be considered as main chain reaching persistence via single-stage early return."""
    strong = _strong_evidence()
    settlement = BusinessStage(
        name="结算创建",
        stage_type="settlement_creation",
        is_mainline=True,
        capabilities=["has_persistence_write"],
        trigger=strong,
        input=strong,
        transform=strong,
        output=strong,
        persistence=strong,
        evidence=[strong],
    )
    state = ExplorationState(
        goal="explore settlement flow",
        task_mode="feature_exploration",
        stages=[settlement],
    )

    from app.agents.code_explorer.stop_judge import _main_chain_reaches_persistence

    assert _main_chain_reaches_persistence([settlement], state)


def test_main_chain_reaches_persistence_bfs_through_edges() -> None:
    """When no single stage has both trigger and persistence, BFS over edges
    should still detect reachability from trigger stage to persistence stage."""
    trigger = _strong_evidence()
    transform_ev = _strong_evidence()
    persistence_ev = _strong_evidence()

    trigger_stage = BusinessStage(
        name="入口触发",
        stage_type="entry",
        is_mainline=True,
        capabilities=["has_trigger"],
        trigger=trigger,
        evidence=[trigger],
    )
    transform_stage = BusinessStage(
        name="业务计算",
        stage_type="calculation",
        is_mainline=True,
        capabilities=["has_transform"],
        transform=transform_ev,
        evidence=[transform_ev],
    )
    persist_stage = BusinessStage(
        name="持久化写入",
        stage_type="persistence_write",
        is_mainline=True,
        capabilities=["has_persistence_write"],
        persistence=persistence_ev,
        evidence=[persistence_ev],
    )
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[trigger_stage, transform_stage, persist_stage],
        edges=[
            StageEdge(
                from_stage="入口触发",
                to_stage="业务计算",
                connection_type="direct_call",
                evidence=[trigger],
                confidence="confirmed",
            ),
            StageEdge(
                from_stage="业务计算",
                to_stage="持久化写入",
                connection_type="direct_call",
                evidence=[transform_ev],
                confidence="confirmed",
            ),
        ],
    )

    from app.agents.code_explorer.stop_judge import _main_chain_reaches_persistence

    assert _main_chain_reaches_persistence(
        [trigger_stage, transform_stage, persist_stage], state
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
