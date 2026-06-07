"""Rule-based stop gate for feature exploration reports."""

from __future__ import annotations

import re

from app.agents.code_explorer.exploration_state import (
    BusinessStage,
    Evidence,
    ExplorationState,
    StopDecision,
    can_satisfy_stage_edge,
    can_satisfy_stage_field,
    has_strong_evidence,
    is_strong_evidence,
)
from app.agents.code_explorer.stage_requirements import (
    get_required_fields_for_stage,
    is_blocking_stage,
)

_STAGE_FIELDS = (
    "trigger",
    "input",
    "transform",
    "output",
    "persistence",
    "state_change",
)
_MAIN_CHAIN_FIELDS = ("trigger", "transform", "persistence")
_ACTIONABLE_FOLLOW_UP = re.compile(
    r"建议后续验证[\s\S]{0,500}(?:需检查|待检查|Job|Event|MQ|Repository|Mapper|"
    r"Listener|Controller|Service|调用链|落库|状态)",
    re.IGNORECASE,
)
_CRITICAL_UNKNOWN = re.compile(
    r"(?:事件下游|listener|consumer|状态流转|status|state|原始.*来源|数据源|"
    r"upstream|补偿|重试|retry|compensation|分支|branch|持久化|映射)",
    re.IGNORECASE,
)
_STATUS_RELATED = re.compile(
    r"(?:状态|结算|支付|status|state|settlement|payment)",
    re.IGNORECASE,
)
_EVENT_RELATED = re.compile(r"(?:事件|event|publish)", re.IGNORECASE)
_EVENT_CONSUMER = re.compile(
    r"(?:listener|consumer|consume|消费|订阅|onapplicationevent)",
    re.IGNORECASE,
)
_OVERCLAIM = re.compile(
    r"(?:所有阶段均已确认|所有阶段均有明确证据|完整闭合|无断链|全部确认)",
    re.IGNORECASE,
)
_MIN_QUALIFIED_STAGES = 1
_MIN_QUALIFIED_EDGES = 3


def judge_stop(
    state: ExplorationState,
    candidate_report: str,
) -> StopDecision:
    """Decide whether confirmed evidence closes the feature's main chain."""

    if state.task_mode != "feature_exploration":
        return StopDecision(
            can_stop=True,
            reason="Feature exploration stop rules do not apply to this task mode.",
        )

    blocking_gaps: list[str] = []
    evidence_problems = _evidence_problems(state)
    confirmed_stages = _confirmed_stages(state)
    missing_stage_fields = _missing_stage_fields(state)
    missing_edges = _missing_edges(confirmed_stages, state)

    # Minimum qualified stages: total confirmed (backward compat)
    if len(confirmed_stages) < _MIN_QUALIFIED_STAGES:
        blocking_gaps.append(
            f"尚未识别至少 {_MIN_QUALIFIED_STAGES} 个有 strong evidence 支持的业务阶段。"
        )
    if missing_stage_fields:
        blocking_gaps.append("主链业务阶段仍有 Missing Stage Fields，不能结束探索。")
    if evidence_problems:
        blocking_gaps.append("主链仍有 Evidence Problems，不能结束探索。")
    missing_main_fields = [
        field_name
        for field_name in _MAIN_CHAIN_FIELDS
        if not (
            field_name == "persistence" and state.external_boundaries
        )
        and not any(
            _qualified_field(getattr(stage, field_name)) for stage in confirmed_stages
        )
    ]
    if missing_main_fields:
        blocking_gaps.append(
            "主链仍缺少 strong evidence 支撑的关键字段："
            + ", ".join(missing_main_fields)
            + "。"
        )
    # For typed blocking stages, only count blocking edges and edges between blocking stages
    has_typed_stages = any(s.stage_type not in ("unknown", "supporting") for s in state.stages)
    edge_count = _blocking_edge_count(state) if has_typed_stages else _confirmed_edge_count(state)
    # Only enforce edge count when there are 2+ blocking/mainline stages
    blocking_stage_count = sum(1 for s in state.stages if _stage_is_blocking_or_untyped(s))
    # For typed stages, only enforce when there are enough stages to plausibly
    # have that many edges (need at least N+1 stages for N edges).
    # For untyped (backward compat), always enforce if count is insufficient.
    if has_typed_stages:
        enforce_edge_check = (
            blocking_stage_count >= 2
            and blocking_stage_count >= _MIN_QUALIFIED_EDGES + 1
            and edge_count < _MIN_QUALIFIED_EDGES
        )
    else:
        enforce_edge_check = blocking_stage_count >= 2 and edge_count < _MIN_QUALIFIED_EDGES
    if enforce_edge_check:
        blocking_gaps.append(
            f"相邻业务阶段之间仍缺少至少 {_MIN_QUALIFIED_EDGES} 条 strong 连接证据。"
        )
    if (
        not state.external_boundaries
        and not _main_chain_reaches_persistence(confirmed_stages, state)
    ):
        blocking_gaps.append("入口到持久化阶段尚未通过 strong edge 图连通。")
    if _STATUS_RELATED.search(f"{state.goal}\n{candidate_report}") and not (
        any(_qualified_field(stage.state_change) for stage in confirmed_stages)
        or _has_evidence_type(state, "status")
    ):
        blocking_gaps.append("任务涉及状态或结算，但仍缺少 strong 状态流转证据。")
    if (
        _EVENT_RELATED.search(candidate_report)
        and not state.external_boundaries
        and not _has_event_consumer_evidence(state)
    ):
        blocking_gaps.append("报告涉及事件发布，但仍缺少 listener / consumer 证据或明确外部边界。")
    for unknown in state.unknowns + state.open_questions:
        if _CRITICAL_UNKNOWN.search(unknown):
            blocking_gaps.append(f"关键不确定项仍会改变主结论：{unknown}")
    if _ACTIONABLE_FOLLOW_UP.search(candidate_report):
        blocking_gaps.append("建议后续验证中仍包含可以继续探索确认的代码问题。")
    if _OVERCLAIM.search(candidate_report) and (
        missing_stage_fields or missing_edges or evidence_problems
    ):
        blocking_gaps.append("候选报告存在过度确认表述，但 evidence quality 仍有缺口。")

    return StopDecision(
        can_stop=not blocking_gaps,
        reason=(
            "关键业务链路已经由 strong evidence 闭合，剩余未知不会改变主结论。"
            if not blocking_gaps
            else "关键业务链路尚未由 strong evidence 闭合，需要继续探索。"
        ),
        blocking_gaps=_deduplicate(blocking_gaps),
        missing_stage_fields=missing_stage_fields,
        missing_edges=missing_edges,
        evidence_problems=evidence_problems,
    )


def render_evidence_quality_report(
    state: ExplorationState,
    decision: StopDecision | None = None,
) -> str:
    """Render an audit artifact explaining whether evidence can close the run."""

    decision = decision or judge_stop(state, "")
    strong = [item for item in state.evidence_items if is_strong_evidence(item)]
    weak = [item for item in state.evidence_items if item.strength == "weak"]
    noise = [item for item in state.evidence_items if item.strength == "noise"]
    inferred = [
        item for item in state.inferred_facts if item.confidence == "inferred"
    ]
    stages_without_evidence = [
        stage.name for stage in state.stages if stage not in _confirmed_stages(state)
    ]

    # Classify edges into three buckets
    qualified_edges: list[str] = []
    supporting_edges: list[str] = []
    weak_edges: list[str] = []
    for edge in state.edges:
        has_strong = any(can_satisfy_stage_edge(item) for item in edge.evidence)
        has_any = any(
            item.strength in ("strong", "weak") and item.source_type != "report"
            for item in edge.evidence
        )
        label = f"{edge.from_stage} -> {edge.to_stage}"
        if has_strong:
            qualified_edges.append(label)
        elif has_any:
            supporting_edges.append(label)
        else:
            weak_edges.append(label)

    downgraded = [
        f"[{item.strength}][{item.id}] {item.reason} Claim: {item.claim}"
        for item in state.evidence_items
        if item.strength != "strong"
    ]
    lines = [
        "# Evidence Quality Report",
        "",
        f"- Total evidence: {len(state.evidence_items) + len(inferred)}",
        f"- Strong evidence: {len(strong)}",
        f"- Weak signals: {len(weak)}",
        f"- Noise filtered: {len(noise)}",
        f"- Inferred/report-only evidence: {len(inferred)}",
        f"- Qualified business stages: {len(_confirmed_stages(state))}",
        f"- Qualified stage edges: {_confirmed_edge_count(state)}",
        f"- Can stop: {'yes' if decision.can_stop else 'no'}",
        "",
        f"## Why StopJudge {'allowed' if decision.can_stop else 'blocked'}",
        decision.reason,
        "",
        "## Stages without strong evidence",
        *_markdown_items(stages_without_evidence),
        "",
        "## Qualified Stage Edges",
        *_markdown_items(qualified_edges),
        "",
        "## Supporting Edges",
        *_markdown_items(supporting_edges),
        "",
        "## Edges without strong evidence",
        *_markdown_items(weak_edges),
        "",
        "## Blocking Gaps",
        *_markdown_items(decision.blocking_gaps),
        "",
        "## Evidence Problems",
        *_markdown_items(decision.evidence_problems),
        "",
        "## Missing Stage Fields",
    ]
    if decision.missing_stage_fields:
        for stage, fields in decision.missing_stage_fields.items():
            lines.append(f"- {stage}: {', '.join(fields)}")
    else:
        lines.append("- None")
    lines.extend(("", "## Missing Edges", *_markdown_items(decision.missing_edges)))
    lines.extend(("", "## Downgraded Evidence", *_markdown_items(downgraded)))
    return "\n".join(lines) + "\n"


def _confirmed_stages(state: ExplorationState) -> list[BusinessStage]:
    return [
        stage
        for stage in state.stages
        if stage.stage_type != "supporting"
        and has_strong_evidence(
            stage.evidence
            + [
                evidence
                for field_name in _STAGE_FIELDS
                if (evidence := getattr(stage, field_name)) is not None
            ]
        )
    ]


def _missing_stage_fields(state: ExplorationState) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for stage in state.stages:
        if stage.optional:
            continue
        # Skip supporting stages entirely
        if stage.stage_type == "supporting":
            continue
        # Stage-type-aware: use per-type required fields for blocking stages
        if is_blocking_stage(stage):
            required = get_required_fields_for_stage(stage)
        else:
            # Untyped: check all fields for backward compatibility
            required = list(_STAGE_FIELDS)
        fields = [
            field_name
            for field_name in required
            if not _qualified_field(getattr(stage, field_name))
        ]
        if fields:
            missing[stage.name] = fields
    return missing


def _missing_edges(
    stages: list[BusinessStage],
    state: ExplorationState,
) -> list[str]:
    # Only check edges between blocking/untyped stages (not supporting)
    has_typed_stages = any(s.stage_type not in ("unknown", "supporting") for s in state.stages)
    edge_count = _blocking_edge_count(state) if has_typed_stages else _confirmed_edge_count(state)
    if edge_count >= _MIN_QUALIFIED_EDGES:
        return []
    # Filter to blocking/untyped stages only; skip if fewer than 2
    blocking_stages = [s for s in stages if _stage_is_blocking_or_untyped(s)] or stages
    if len(blocking_stages) < 2:
        return []
    missing: list[str] = []
    for first, second in zip(blocking_stages, blocking_stages[1:]):
        matching = [
            edge
            for edge in state.edges
            if edge.from_stage == first.name
            and edge.to_stage == second.name
            and edge.connection_type != "unknown"
            and edge.confidence == "confirmed"
            and any(can_satisfy_stage_edge(item) for item in edge.evidence)
        ]
        if not matching:
            missing.append(f"{first.name} -> {second.name}")
    return missing


def _confirmed_edge_count(state: ExplorationState) -> int:
    return sum(
        1
        for edge in state.edges
        if edge.connection_type != "unknown"
        and edge.confidence == "confirmed"
        and any(can_satisfy_stage_edge(item) for item in edge.evidence)
    )


def _blocking_edge_count(state: ExplorationState) -> int:
    """Count strong edges between blocking/mainline stages only."""
    blocking_names = {stage.name for stage in state.stages if is_blocking_stage(stage)}
    return sum(
        1
        for edge in state.edges
        if edge.connection_type != "unknown"
        and edge.confidence == "confirmed"
        and any(can_satisfy_stage_edge(item) for item in edge.evidence)
        and edge.from_stage in blocking_names
        and edge.to_stage in blocking_names
    )


def _stage_is_blocking_or_untyped(stage: BusinessStage) -> bool:
    """A stage that should be considered for blocking checks.
    Typed blocking stages AND untyped stages (backward compat) are blocking.
    Supporting and optional stages are excluded."""
    return stage.stage_type != "supporting" and not stage.optional


def _main_chain_reaches_persistence(
    stages: list[BusinessStage],
    state: ExplorationState,
) -> bool:
    # Tightened single-stage early return:
    # A single stage can close the main chain to persistence only if ALL
    # of the following are true:
    #   1. stage is blocking (or untyped for backward compat)
    #   2. stage.stage_type is NOT a "no-persistence-required" type
    #      (entry / controller_or_rpc / scheduled_job)
    #   3. stage has qualified trigger OR upstream edge
    #   4. stage has qualified persistence with write capability
    #   5. stage has qualified transform OR output
    for stage in stages:
        if not _stage_is_blocking_or_untyped(stage):
            continue
        if _qualified_field(stage.trigger) or _has_upstream_edge(stage, stages, state):
            pass
        else:
            continue
        if stage.stage_type in ("entry", "controller_or_rpc", "scheduled_job"):
            continue
        if not _qualified_field(stage.persistence):
            continue
        if not _has_write_capability(stage):
            continue
        if not (_qualified_field(stage.transform) or _qualified_field(stage.output)):
            continue
        return True
    # Only consider blocking/untyped stages (exclude supporting)
    relevant = {s.name for s in stages if _stage_is_blocking_or_untyped(s)}
    trigger_stages = {
        stage.name for stage in stages if _stage_is_blocking_or_untyped(stage) and _qualified_field(stage.trigger)
    }
    persistence_stages = {
        stage.name for stage in stages if _stage_is_blocking_or_untyped(stage) and _qualified_field(stage.persistence)
    }
    if not trigger_stages or not persistence_stages:
        return False
    graph: dict[str, set[str]] = {}
    for edge in state.edges:
        if (
            edge.connection_type != "unknown"
            and edge.confidence == "confirmed"
            and any(can_satisfy_stage_edge(item) for item in edge.evidence)
        ):
            graph.setdefault(edge.from_stage, set()).add(edge.to_stage)
    for start in trigger_stages:
        pending = list(graph.get(start, ()))
        visited = {start}
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            if current in persistence_stages:
                return True
            visited.add(current)
            pending.extend(graph.get(current, ()))
    return False


def _has_write_capability(stage: BusinessStage) -> bool:
    """Return True if the stage or its persistence evidence indicates write."""
    if stage.capabilities:
        if any("has_persistence_write" in cap for cap in stage.capabilities):
            return True
        if any("has_persistence_read" in cap for cap in stage.capabilities):
            return False
    return True


def _has_read_only_capability(stage: BusinessStage) -> bool:
    """Return True if the stage is explicitly read-only.

    Kept for backward compatibility — callers should prefer
    ``_has_write_capability`` which returns the negative default.
    """
    return any("has_persistence_read" in cap for cap in stage.capabilities)


def _has_upstream_edge(
    stage: BusinessStage,
    stages: list[BusinessStage],
    state: ExplorationState,
) -> bool:
    """Return True if a qualified edge points INTO this stage."""
    names = {s.name for s in stages}
    for edge in state.edges:
        if (
            edge.to_stage == stage.name
            and edge.from_stage in names
            and edge.connection_type != "unknown"
            and edge.confidence == "confirmed"
            and any(can_satisfy_stage_edge(item) for item in edge.evidence)
        ):
            return True
    return False


def _evidence_problems(state: ExplorationState) -> list[str]:
    problems: list[str] = []
    for stage in state.stages:
        if stage.optional:
            continue
        # Skip supporting stage problems (they don't block)
        if stage.stage_type == "supporting":
            continue
        required = get_required_fields_for_stage(stage) if is_blocking_stage(stage) else list(_STAGE_FIELDS)
        if not has_strong_evidence(
            stage.evidence
            + [
                evidence
                for field_name in required
                if (evidence := getattr(stage, field_name)) is not None
            ]
        ):
            problems.append(f"Stage {stage.name} lacks strong evidence.")
        for field_name in required:
            evidence = getattr(stage, field_name)
            if evidence is not None and not can_satisfy_stage_field(
                evidence,
                field_name,
            ):
                problems.append(
                    f"Stage {stage.name}.{field_name} is not backed by strong stage-field evidence."
                )
    for edge in state.edges:
        if (
            edge.confidence == "confirmed"
            and not any(can_satisfy_stage_edge(item) for item in edge.evidence)
        ):
            problems.append(
                f"Edge {edge.from_stage} -> {edge.to_stage} lacks strong stage-edge evidence."
            )
    return _deduplicate(problems)


def _has_evidence_type(state: ExplorationState, source_type: str) -> bool:
    return any(
        item.source_type == source_type and is_strong_evidence(item)
        for item in state.evidence_items
    )


def _has_event_consumer_evidence(state: ExplorationState) -> bool:
    return any(
        item.source_type == "event"
        and is_strong_evidence(item)
        and _EVENT_CONSUMER.search(f"{item.claim}\n{item.summary}")
        for item in state.evidence_items
    )


def _qualified_field(evidence: Evidence | None) -> bool:
    return can_satisfy_stage_field(evidence)


def _deduplicate(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _markdown_items(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- None"]
