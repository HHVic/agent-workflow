"""Tests for evidence quality report edge classification."""

from app.agents.code_explorer.exploration_state import (
    BusinessStage,
    Evidence,
    ExplorationState,
    StageEdge,
)
from app.agents.code_explorer.stop_judge import (
    judge_stop,
    render_evidence_quality_report,
)


def _strong_edge(from_name: str, to_name: str) -> StageEdge:
    return StageEdge(
        from_stage=from_name,
        to_stage=to_name,
        connection_type="direct_call",
        evidence=[
            Evidence(
                id="e-strong",
                claim="strong edge evidence",
                source_type="call_edge",
                summary="call edge",
                confidence="confirmed",
                strength="strong",
                can_satisfy_stage_edge=True,
                can_satisfy_stage_field=True,
            )
        ],
        confidence="confirmed",
    )


def _supporting_edge(from_name: str, to_name: str) -> StageEdge:
    return StageEdge(
        from_stage=from_name,
        to_stage=to_name,
        connection_type="direct_call",
        evidence=[
            Evidence(
                id="e-weak",
                claim="weak edge evidence",
                source_type="call_edge",
                summary="call edge",
                confidence="confirmed",
                strength="weak",
                can_satisfy_stage_edge=False,
            )
        ],
        confidence="confirmed",
    )


def _weak_edge(from_name: str, to_name: str) -> StageEdge:
    return StageEdge(
        from_stage=from_name,
        to_stage=to_name,
        connection_type="unknown",
        evidence=[],
        confidence="unknown",
    )


def _strong_stage(name: str) -> BusinessStage:
    e = Evidence(
        id=f"e-{name}",
        claim=f"strong {name}",
        source_type="call_edge",
        summary="strong evidence",
        confidence="confirmed",
        strength="strong",
        can_satisfy_stage_field=True,
    )
    return BusinessStage(
        name=name,
        trigger=e,
        input=e,
        transform=e,
        output=e,
        persistence=e,
        state_change=e,
        evidence=[e],
    )


def _report(state: ExplorationState) -> str:
    decision = judge_stop(state, "")
    return render_evidence_quality_report(state, decision)


# -- Test scenarios --


def test_strong_edge_not_in_edges_without_strong_evidence() -> None:
    """Strong edge should NOT appear in Edges without strong evidence."""
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[_strong_stage("stage A"), _strong_stage("stage B")],
        edges=[_strong_edge("stage A", "stage B")],
    )
    report = _report(state)

    section = report.split("## Edges without strong evidence")[1].split("##")[0]
    assert "stage A -> stage B" not in section


def test_strong_edge_in_qualified_stage_edges() -> None:
    """Strong edge should appear in Qualified Stage Edges."""
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[_strong_stage("stage A"), _strong_stage("stage B")],
        edges=[_strong_edge("stage A", "stage B")],
    )
    report = _report(state)

    section = report.split("## Qualified Stage Edges")[1].split("##")[0]
    assert "stage A -> stage B" in section


def test_supporting_edge_in_supporting_edges_section() -> None:
    """Supporting (weak but has evidence) edge should appear in Supporting Edges."""
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[_strong_stage("stage A"), _strong_stage("stage B")],
        edges=[_supporting_edge("stage A", "stage B")],
    )
    report = _report(state)

    supporting_section = report.split("## Supporting Edges")[1].split("##")[0]
    assert "stage A -> stage B" in supporting_section


def test_weak_edge_in_edges_without_strong_evidence() -> None:
    """Missing/weak edge should appear in Edges without strong evidence."""
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[_strong_stage("stage A"), _strong_stage("stage B")],
        edges=[_weak_edge("stage A", "stage B")],
    )
    report = _report(state)

    section = report.split("## Edges without strong evidence")[1].split("##")[0]
    assert "stage A -> stage B" in section


def test_can_stop_consistent_with_blocking_gaps() -> None:
    """Can stop decision should be consistent with blocking gaps."""
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[_strong_stage("stage A"), _strong_stage("stage B")],
        edges=[_strong_edge("stage A", "stage B")],
    )
    decision = judge_stop(state, "")
    report = render_evidence_quality_report(state, decision)

    if decision.can_stop:
        assert "Can stop: yes" in report
        assert not decision.blocking_gaps
    else:
        assert "Can stop: no" in report
        assert decision.blocking_gaps


def test_mixed_edges_classified_correctly() -> None:
    """When edges are mixed, each should go to the correct section."""
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[
            _strong_stage("A"),
            _strong_stage("B"),
            _strong_stage("C"),
        ],
        edges=[
            _strong_edge("A", "B"),
            _supporting_edge("B", "C"),
        ],
    )
    report = _report(state)

    qualified = report.split("## Qualified Stage Edges")[1].split("##")[0]
    supporting = report.split("## Supporting Edges")[1].split("##")[0]

    assert "A -> B" in qualified
    assert "B -> C" in supporting
    assert "A -> B" not in supporting


def test_no_contradiction_strong_in_both_qualified_and_unqualified() -> None:
    """A strong edge must never appear in both Qualified and Without sections."""
    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[
            _strong_stage("A"),
            _strong_stage("B"),
        ],
        edges=[_strong_edge("A", "B")],
    )
    report = _report(state)

    qualified = report.split("## Qualified Stage Edges")[1].split("##")[0]
    without = report.split("## Edges without strong evidence")[1].split("##")[0]

    label = "A -> B"
    in_qualified = label in qualified
    in_without = label in without

    assert not (in_qualified and in_without), (
        f"Edge '{label}' should not appear in both Qualified and Without sections"
    )


def test_downgraded_evidence_reflects_test_code_weakness() -> None:
    """Test code evidence that was downgraded should appear in Downgraded Evidence."""
    from app.agents.code_explorer.evidence_extractor import extract_evidence_from_tool_result

    evidence = extract_evidence_from_tool_result(
        "find_callers",
        {"symbol": "WidgetService.calculate"},
        "- WidgetJob.doExecute (method) - src/main/java/example/test/WidgetJob.java:22",
    )

    state = ExplorationState(
        goal="explore flow",
        task_mode="feature_exploration",
        stages=[_strong_stage("stage A")],
        evidence_items=evidence,
    )
    report = _report(state)

    assert "## Downgraded Evidence" in report
    # The downgraded item should have strength=weak
    assert "[weak]" in report
