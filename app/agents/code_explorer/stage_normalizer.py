"""Normalize located tool evidence into generic business stages."""

from __future__ import annotations

from dataclasses import replace

from app.agents.code_explorer.exploration_state import (
    BusinessStage,
    Evidence,
    StageEdge,
    can_satisfy_stage_edge,
    can_satisfy_stage_field,
    is_confirmed_tool_evidence,
    is_strong_evidence,
)

_CATEGORY_MARKERS = (
    ("事件发布与消费", ("event", "listener", "publish", "consumer", "onapplicationevent")),
    ("状态流转", ("status", "state", "transition", "changestatus", "updatestatus")),
    ("持久化写入", ("repository", "mapper", "dao", "save", "insert", "update", "persist")),
    ("可持久化数据读取", ("query", "find", "load", "select", "fetch", "getby")),
    ("业务计算与转换", ("calculate", "compute", "match", "assemble", "convert", "transform", "process", "handle")),
    ("入口触发", ("controller", "delegate", "grpc", "rpc", "job", "listener", "execute", "trigger")),
)
_SOURCE_STAGE_NAMES = {
    "repository": "持久化写入",
    "table": "持久化写入",
    "event": "事件发布与消费",
    "status": "状态流转",
}


def normalize_stage_candidates(
    evidence_items: list[Evidence],
    candidate_report: str | None = None,
) -> list[BusinessStage]:
    """Group confirmed evidence into generic stages instead of method-name stages."""

    del candidate_report
    stages: dict[str, BusinessStage] = {}
    fallback: tuple[str, Evidence] | None = None
    for evidence in evidence_items:
        if (
            not is_confirmed_tool_evidence(evidence)
            or not evidence.symbol
            or evidence.source_type in {"call_edge", "file_path"}
            or _is_test_evidence(evidence)
        ):
            continue
        stage_name = stage_name_for_evidence(evidence)
        if not is_strong_evidence(evidence):
            fallback = fallback or (stage_name, evidence)
            continue
        if stage_name.startswith("待确认阶段："):
            fallback = fallback or (stage_name, evidence)
            continue
        stage = stages.setdefault(
            stage_name,
            BusinessStage(
                name=stage_name,
                description="Normalized from located CodeGraph tool evidence.",
                confidence=("unknown" if stage_name.startswith("待确认阶段：") else "confirmed"),
            ),
        )
        _attach_evidence(stage, evidence)
    if not stages and fallback:
        stage_name, evidence = fallback
        stage = BusinessStage(
            name=stage_name,
            description="Unclassified located symbol; keep as a review candidate.",
            confidence="unknown",
            optional=not is_strong_evidence(evidence),
        )
        _attach_evidence(stage, evidence)
        return [stage]
    return list(stages.values())


def normalize_stage_edges(
    evidence_items: list[Evidence],
) -> list[StageEdge]:
    """Convert confirmed call edges into connections between normalized stages."""

    edges: dict[tuple[str, str], StageEdge] = {}
    for evidence in evidence_items:
        if (
            evidence.source_type != "call_edge"
            or not can_satisfy_stage_edge(evidence)
            or _is_test_evidence(evidence)
        ):
            continue
        if not evidence.symbol or " -> " not in evidence.symbol:
            continue
        source, destination = evidence.symbol.split(" -> ", maxsplit=1)
        source_stage = stage_name_for_symbol(source)
        destination_stage = stage_name_for_symbol(destination)
        if (
            source_stage == destination_stage
            or source_stage.startswith("待确认阶段：")
            or destination_stage.startswith("待确认阶段：")
        ):
            continue
        key = (source_stage, destination_stage)
        edge = edges.setdefault(
            key,
            StageEdge(
                from_stage=source_stage,
                to_stage=destination_stage,
                connection_type="direct_call",
                confidence="confirmed",
            ),
        )
        if evidence not in edge.evidence:
            edge.evidence.append(evidence)
    return list(edges.values())


def stage_name_for_evidence(evidence: Evidence) -> str:
    """Return the generic business-stage label for one evidence item."""

    if stage_name := _SOURCE_STAGE_NAMES.get(evidence.source_type):
        return stage_name
    return stage_name_for_symbol(evidence.symbol or evidence.claim, evidence.source_type)


def stage_name_for_symbol(symbol: str, source_type: str = "tool_result") -> str:
    """Classify a symbol without hard-coding product-specific names."""

    lowered = f"{source_type} {symbol}".casefold()
    for stage_name, markers in _CATEGORY_MARKERS:
        if any(marker in lowered for marker in markers):
            return stage_name
    return f"待确认阶段：{symbol}"


def merge_normalized_stages(
    existing: list[BusinessStage],
    normalized: list[BusinessStage],
) -> list[BusinessStage]:
    """Merge fresh tool-backed stages while preserving inferred report candidates."""

    if any(not stage.name.startswith("待确认阶段：") for stage in normalized):
        existing = [
            stage for stage in existing if not stage.name.startswith("待确认阶段：")
        ]
    merged = {stage.name: stage for stage in existing}
    for candidate in normalized:
        current = merged.get(candidate.name)
        if current is None:
            merged[candidate.name] = candidate
            continue
        if current.confidence != "confirmed" and candidate.confidence == "confirmed":
            current.confidence = "confirmed"
            current.optional = False
        for evidence in candidate.evidence:
            _attach_evidence(current, evidence)
    return list(merged.values())


def merge_normalized_edges(
    existing: list[StageEdge],
    normalized: list[StageEdge],
) -> list[StageEdge]:
    """Merge fresh confirmed edges without promoting report-only edges."""

    merged = {(edge.from_stage, edge.to_stage): edge for edge in existing}
    for candidate in normalized:
        key = (candidate.from_stage, candidate.to_stage)
        current = merged.get(key)
        if current is None:
            merged[key] = candidate
            continue
        if candidate.confidence == "confirmed":
            current.confidence = "confirmed"
            current.connection_type = candidate.connection_type
        for evidence in candidate.evidence:
            if evidence not in current.evidence:
                current.evidence.append(evidence)
    return list(merged.values())


def _attach_evidence(stage: BusinessStage, evidence: Evidence) -> None:
    if evidence not in stage.evidence:
        stage.evidence.append(evidence)
    fields = _fields_for_evidence(evidence)
    for field_name in fields:
        if (
            getattr(stage, field_name) is None
            and can_satisfy_stage_field(evidence, field_name)
        ):
            setattr(stage, field_name, replace(evidence))


def _fields_for_evidence(evidence: Evidence) -> tuple[str, ...]:
    lowered = (
        f"{evidence.source_type} {evidence.symbol} {evidence.claim} {evidence.summary}"
    ).casefold()
    fields: list[str] = []
    if evidence.source_type == "event":
        if any(marker in lowered for marker in ("listener", "consumer", "onapplicationevent")):
            fields.append("trigger")
        if "publish" in lowered:
            fields.append("output")
    if evidence.source_type == "business_method" and any(
        marker in lowered
        for marker in ("controller", "delegate", "grpc", "rpc", "job", "listener", "execute", "trigger")
    ):
        fields.append("trigger")
    if evidence.source_type == "business_method" and any(
        marker in lowered for marker in ("request", "query", "find", "load", "select", "fetch")
    ):
        fields.append("input")
    if evidence.source_type == "business_method" and any(
        marker in lowered
        for marker in ("calculate", "compute", "match", "assemble", "convert", "transform", "process", "handle")
    ):
        fields.append("transform")
    if evidence.source_type == "business_method" and any(
        marker in lowered for marker in ("result", "create", "build", "publish")
    ):
        fields.append("output")
    if evidence.source_type in {"repository", "table"}:
        fields.append("persistence")
    if evidence.source_type == "status":
        fields.append("state_change")
    return tuple(dict.fromkeys(fields))


def _is_test_evidence(evidence: Evidence) -> bool:
    path = (evidence.file_path or "").casefold()
    return "/src/test/" in path or "/test/" in path or "test." in path
