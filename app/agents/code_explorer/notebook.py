"""Markdown notebook generation for internal exploration state."""

from __future__ import annotations

from typing import Any

from app.agents.code_explorer.exploration_state import (
    Evidence,
    ExplorationState,
    can_satisfy_stage_edge,
    can_satisfy_stage_field,
    has_strong_evidence,
    is_strong_evidence,
)

_STAGE_FIELDS = (
    "trigger",
    "input",
    "transform",
    "output",
    "persistence",
    "state_change",
)


def build_notebook_update_prompt(
    state: ExplorationState,
    recent_messages: list[dict[str, Any]],
    recent_tool_logs: list[dict[str, Any]],
) -> str:
    """Build an optional LLM prompt for future notebook refinement."""

    del recent_messages, recent_tool_logs
    return f"""\
# Exploration Notebook Update

不要输出最终报告。只更新内部探索状态，并区分已确认、推断和未知。

## 已确认业务事实
{_evidence_items(_notebook_business_evidence(state.confirmed_facts))}

## 弱线索
{_weak_evidence_items(_notebook_weak_evidence(state))}

## 当前业务链路假设
{_items(state.branches)}

## 当前断链点
{_items(_current_gaps(state))}

请标记业务阶段、阶段间断链、不确定项和下一步候选探索方向。
"""


def update_notebook(
    state: ExplorationState,
    recent_messages: list[dict[str, Any]],
    recent_tool_logs: list[dict[str, Any]],
) -> str:
    """Render a compact evidence-driven notebook without an extra LLM request."""

    del recent_messages, recent_tool_logs
    notebook = f"""\
# Exploration Notebook

## 已确认业务事实
{_evidence_items(_notebook_business_evidence(state.confirmed_facts))}

## 弱线索
{_weak_evidence_items(_notebook_weak_evidence(state))}

## 已确认调用/连接
{_evidence_items(_confirmed_connections(state))}

## 当前业务链路假设
{_items(state.branches)}

## 已识别业务阶段
{_items(_stage_items(state))}

## 阶段字段缺口
{_items(_stage_field_gap_items(state))}

## 阶段间断链
{_items(_edge_gap_items(state))}

## 外部/动态边界
{_items(state.external_boundaries + state.dynamic_boundaries)}

## 不确定项
{_items(state.unknowns + state.open_questions)}

## 下一步最高价值探索方向
{_items(state.candidate_next_actions)}
"""
    state.notebook_markdown = notebook
    return notebook


def _confirmed_connections(state: ExplorationState) -> list[Evidence]:
    evidence: list[Evidence] = []
    for edge in state.edges:
        evidence.extend(
            item
            for item in edge.evidence
            if item.source_type == "call_edge" and can_satisfy_stage_edge(item)
        )
    return _deduplicate_evidence(evidence)


def _notebook_business_evidence(items: list[Evidence]) -> list[Evidence]:
    filtered = [
        item
        for item in items
        if is_strong_evidence(item) and not _is_test_evidence(item)
    ]
    return filtered[-80:]


def _notebook_weak_evidence(state: ExplorationState) -> list[Evidence]:
    return [
        item
        for item in state.evidence_items
        if item.strength == "weak" and not _is_test_evidence(item)
    ][-40:]


def _stage_items(state: ExplorationState) -> list[str]:
    items: list[str] = []
    for stage in state.stages:
        confirmed = has_strong_evidence(
            stage.evidence + [
                item
                for field_name in _STAGE_FIELDS
                if (item := getattr(stage, field_name)) is not None
            ]
        )
        confidence = "confirmed" if confirmed else stage.confidence
        items.append(f"{stage.name} ({confidence})")
    return items


def _stage_field_gap_items(state: ExplorationState) -> list[str]:
    items: list[str] = []
    for stage in state.stages:
        if stage.optional:
            continue
        missing = [
            field_name
            for field_name in _STAGE_FIELDS
            if not _qualified_field(getattr(stage, field_name))
        ]
        if missing:
            items.append(f"{stage.name}: {', '.join(missing)}")
    return items


def _edge_gap_items(state: ExplorationState) -> list[str]:
    if state.stop_decision and state.stop_decision.missing_edges:
        return state.stop_decision.missing_edges
    items: list[str] = []
    for first, second in zip(state.stages, state.stages[1:]):
        matching = [
            edge
            for edge in state.edges
            if edge.from_stage == first.name
            and edge.to_stage == second.name
            and edge.confidence == "confirmed"
            and any(can_satisfy_stage_edge(item) for item in edge.evidence)
        ]
        if not matching:
            items.append(f"{first.name} -> {second.name}")
    return items


def _current_gaps(state: ExplorationState) -> list[str]:
    if state.stop_decision and state.stop_decision.blocking_gaps:
        return state.stop_decision.blocking_gaps
    return state.unknowns or state.open_questions


def _evidence_items(items: list[Evidence]) -> str:
    strong = _deduplicate_evidence(
        [item for item in items if is_strong_evidence(item)]
    )
    return _items([_format_evidence(item) for item in strong])


def _weak_evidence_items(items: list[Evidence]) -> str:
    weak = _deduplicate_evidence(
        [item for item in items if item.strength == "weak"]
    )
    return _items([_format_evidence(item) for item in weak])


def _format_evidence(evidence: Evidence) -> str:
    locators: list[str] = []
    if evidence.symbol:
        locators.append(f"symbol: `{evidence.symbol}`")
    if evidence.file_path:
        locators.append(f"path: `{evidence.file_path}`")
    locator = f" | {' | '.join(locators)}" if locators else ""
    return (
        f"[{evidence.strength}][{evidence.source_type}][{evidence.id}] "
        f"{evidence.claim}{locator} | reason: {evidence.reason}"
    )


def _qualified_field(evidence: Evidence | None) -> bool:
    return can_satisfy_stage_field(evidence)


def _is_test_evidence(evidence: Evidence) -> bool:
    path = (evidence.file_path or "").casefold()
    return "/src/test/" in path or "/test/" in path or "test." in path


def _deduplicate_evidence(items: list[Evidence]) -> list[Evidence]:
    unique: dict[str, Evidence] = {}
    for item in items:
        unique.setdefault(item.id, item)
    return list(unique.values())


def _items(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None"
