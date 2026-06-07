"""Lightweight internal state for feature exploration."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields as dataclasses_fields
from typing import Any, Literal

_CANDIDATE_SYMBOL = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:Service|Job|Controller|Repository|Converter|"
    r"Coordinator|Policy|Status|Event|Listener|Po|Bo|Mapper|Dao|Client|Delegate)\b"
)
_ARROW = re.compile(r"\s*(?:->|→)\s*")
_STAGE_HEADING = re.compile(r"^#{2,4}\s+阶段\s*(.+?)\s*$", re.MULTILINE)
_STAGE_ENUMERATION = re.compile(r"^[一二三四五六七八九十0-9A-Za-z]+\s*[:：]\s*")
_FLOW_MARKERS = ("核心流程", "主链", "业务链路")
_LOCATED_PATH = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_$.-]+\.\w+(?::\d+)?")


@dataclass(slots=True)
class Evidence:
    """One confirmed, inferred, or unknown claim."""

    id: str
    claim: str
    source_type: str
    summary: str
    confidence: str
    file_path: str | None = None
    symbol: str | None = None
    strength: Literal["strong", "weak", "noise"] = "weak"
    can_satisfy_stage_field: bool = False
    can_satisfy_stage_edge: bool = False
    reason: str = "No explicit strong-evidence rule matched."


@dataclass(slots=True)
class BusinessStage:
    """A business stage rather than an individual code symbol."""

    name: str
    description: str | None = None
    trigger: Evidence | None = None
    input: Evidence | None = None
    transform: Evidence | None = None
    output: Evidence | None = None
    persistence: Evidence | None = None
    state_change: Evidence | None = None
    evidence: list[Evidence] = field(default_factory=list)
    confidence: str = "unknown"
    open_questions: list[str] = field(default_factory=list)
    optional: bool = False
    stage_type: str = "unknown"
    is_mainline: bool = False
    is_optional: bool = False
    capabilities: list[str] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BusinessStage:
        """Construct from a dict, ignoring unknown keys for backward compatibility."""

        field_names = {f.name for f in dataclasses_fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)


@dataclass(slots=True)
class StageEdge:
    """Evidence-backed connection between adjacent business stages."""

    from_stage: str
    to_stage: str
    connection_type: str
    evidence: list[Evidence] = field(default_factory=list)
    confidence: str = "unknown"
    open_questions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StopDecision:
    """Decision produced before accepting a feature exploration report."""

    can_stop: bool
    reason: str
    blocking_gaps: list[str] = field(default_factory=list)
    missing_stage_fields: dict[str, list[str]] = field(default_factory=dict)
    missing_edges: list[str] = field(default_factory=list)
    evidence_problems: list[str] = field(default_factory=list)
    continuation_prompt: str | None = None

    def to_markdown(self) -> str:
        """Render a local audit artifact."""

        lines = [
            f"CAN_STOP: {'yes' if self.can_stop else 'no'}",
            "",
            "## Reason",
            self.reason,
            "",
            "## Blocking Gaps",
            *_markdown_items(self.blocking_gaps),
            "",
            "## Missing Stage Fields",
        ]
        if self.missing_stage_fields:
            for stage, fields in self.missing_stage_fields.items():
                lines.append(f"- {stage}: {', '.join(fields)}")
        else:
            lines.append("- None")
        lines.extend(("", "## Missing Edges", *_markdown_items(self.missing_edges)))
        lines.extend(("", "## Evidence Problems", *_markdown_items(self.evidence_problems)))
        return "\n".join(lines) + "\n"


@dataclass(slots=True)
class ExplorationState:
    """Persistent notebook state for one exploration run."""

    goal: str
    task_mode: str
    branches: list[str] = field(default_factory=list)
    stages: list[BusinessStage] = field(default_factory=list)
    edges: list[StageEdge] = field(default_factory=list)
    evidence_items: list[Evidence] = field(default_factory=list)
    confirmed_facts: list[Evidence] = field(default_factory=list)
    inferred_facts: list[Evidence] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    dynamic_boundaries: list[str] = field(default_factory=list)
    external_boundaries: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    candidate_next_actions: list[str] = field(default_factory=list)
    notebook_markdown: str | None = None
    continuation_count: int = 0
    stop_decision: StopDecision | None = None

    def add_candidate_action(self, action: str) -> None:
        """Append a non-empty candidate action once."""

        if action and action not in self.candidate_next_actions:
            self.candidate_next_actions.append(action)

    def to_dict(self) -> dict[str, Any]:
        """Serialize nested dataclasses into plain Python structures."""

        return asdict(self)

    def to_json(self) -> str:
        """Serialize state as stable local JSON."""

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def update_state_from_tool_logs(
    state: ExplorationState,
    tool_logs: list[dict[str, Any]],
) -> ExplorationState:
    """Add lightweight evidence and candidate actions from recent tool logs."""

    from app.agents.code_explorer.evidence_extractor import (
        extract_evidence_from_tool_result,
    )
    from app.agents.code_explorer.stage_normalizer import (
        merge_normalized_edges,
        merge_normalized_stages,
        normalize_stage_candidates,
        normalize_stage_edges,
    )

    known_evidence_ids = {evidence.id for evidence in state.evidence_items}
    for log in tool_logs:
        if log.get("status") != "success":
            continue
        result_text = _tool_result_text(
            log.get("response", log.get("result_preview", ""))
        )
        extracted = extract_evidence_from_tool_result(
            str(log.get("tool_name", "tool")),
            dict(log.get("arguments", {})),
            result_text,
        )
        for evidence in extracted:
            if evidence.id in known_evidence_ids:
                continue
            state.evidence_items.append(evidence)
            known_evidence_ids.add(evidence.id)
            if is_strong_evidence(evidence):
                state.confirmed_facts.append(evidence)
            else:
                state.inferred_facts.append(evidence)
        source = f"{log.get('arguments', {})}\n{log.get('result_preview', '')}"
        for symbol in _CANDIDATE_SYMBOL.findall(source):
            state.add_candidate_action(f"Inspect candidate symbol: {symbol}")
    state.stages = merge_normalized_stages(
        state.stages,
        normalize_stage_candidates(state.evidence_items),
    )
    state.edges = merge_normalized_edges(
        state.edges,
        normalize_stage_edges(state.evidence_items),
    )
    return state


def _tool_result_text(value: Any) -> str:
    """Flatten MCP response containers while preserving text-node newlines."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(
            text for item in value.values() if (text := _tool_result_text(item))
        )
    if isinstance(value, list):
        return "\n".join(
            text for item in value if (text := _tool_result_text(item))
        )
    return str(value) if value is not None else ""


def update_state_from_candidate_report(
    state: ExplorationState,
    candidate_report: str,
) -> ExplorationState:
    """Extract cautious stage hypotheses and edges from a candidate report."""

    branches = _extract_branches(candidate_report)
    for branch in branches:
        if branch not in state.branches:
            state.branches.append(branch)
    stage_names = _extract_stage_names(candidate_report, branches)
    for stage_name in stage_names:
        if not any(stage.name == stage_name for stage in state.stages):
            stage = _build_stage_hypothesis(stage_name)
            state.stages.append(stage)
            state.inferred_facts.extend(stage.evidence)
    for edge in _extract_edges(branches, stage_names):
        if not any(
            existing.from_stage == edge.from_stage
            and existing.to_stage == edge.to_stage
            for existing in state.edges
        ):
            state.edges.append(edge)
    for line in candidate_report.splitlines():
        stripped = line.strip(" -*")
        if any(marker in stripped for marker in ("需确认", "待确认", "未知", "不确定")):
            if stripped and stripped not in state.unknowns:
                state.unknowns.append(stripped)
    return state


def _extract_branches(report: str) -> list[str]:
    branches: list[str] = []
    in_stage_edges = False
    for line in report.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            in_stage_edges = "阶段连接" in stripped
            continue
        if "->" not in stripped and "→" not in stripped:
            continue
        if in_stage_edges or any(marker in stripped for marker in _FLOW_MARKERS):
            branches.append(_branch_expression(stripped))
    return branches


def _extract_stage_names(report: str, branches: list[str]) -> list[str]:
    explicit_names: list[str] = []
    for heading in _STAGE_HEADING.findall(report):
        normalized = _STAGE_ENUMERATION.sub("", heading.strip(" *`"))
        if normalized:
            explicit_names.append(normalized)
    if explicit_names:
        return list(dict.fromkeys(explicit_names))

    names: list[str] = []
    for branch in branches:
        for part in _ARROW.split(branch):
            normalized = _clean_stage_token(part)
            if normalized and len(normalized) <= 80:
                names.append(normalized)
    return list(dict.fromkeys(names))


def is_confirmed_tool_evidence(evidence: Evidence) -> bool:
    """Return whether evidence is tool-backed and specific enough for stop checks."""

    if evidence.confidence != "confirmed":
        return False
    if evidence.source_type in {"inferred", "report"}:
        return False
    if "returned exploration evidence" in evidence.claim.casefold():
        return False
    if "candidate report mentions stage" in evidence.summary.casefold():
        return False
    return bool(
        evidence.file_path
        or evidence.symbol
        or evidence.source_type
        in {"call_edge", "repository", "table", "event", "status"}
        or _LOCATED_PATH.search(evidence.summary)
        or " -> " in evidence.summary
        or " → " in evidence.summary
    )


def has_confirmed_tool_evidence(items: list[Evidence]) -> bool:
    """Return whether a collection contains at least one qualified evidence item."""

    return any(is_confirmed_tool_evidence(evidence) for evidence in items)


def is_strong_evidence(evidence: Evidence) -> bool:
    """Return whether evidence matched an explicit strong-evidence rule."""

    return is_confirmed_tool_evidence(evidence) and evidence.strength == "strong"


def has_strong_evidence(items: list[Evidence]) -> bool:
    """Return whether a collection contains a strong evidence item."""

    return any(is_strong_evidence(evidence) for evidence in items)


def can_satisfy_stage_field(
    evidence: Evidence | None,
    field_name: str | None = None,
) -> bool:
    """Return whether evidence may satisfy a normalized stage field."""

    del field_name
    return bool(
        evidence
        and is_strong_evidence(evidence)
        and evidence.can_satisfy_stage_field
    )


def can_satisfy_stage_edge(evidence: Evidence | None) -> bool:
    """Return whether evidence may satisfy a normalized stage edge."""

    return bool(
        evidence
        and is_strong_evidence(evidence)
        and evidence.can_satisfy_stage_edge
    )


def _build_stage_hypothesis(name: str) -> BusinessStage:
    evidence = Evidence(
        id=f"report-stage-{len(name)}-{abs(hash(name)) % 10000}",
        claim=f"Candidate business stage: {name}",
        source_type="report",
        summary=f"Candidate report mentions stage {name}",
        confidence="inferred",
    )
    return BusinessStage(
        name=name,
        description="Inferred from candidate report; verify with code evidence.",
        evidence=[evidence],
        confidence="inferred",
        open_questions=["Need tool evidence"],
    )


def _extract_edges(
    branches: list[str],
    stage_names: list[str],
) -> list[StageEdge]:
    edges: list[StageEdge] = []
    for branch in branches:
        parts = [_clean_stage_token(part) for part in _ARROW.split(branch)]
        parts = [part for part in parts if part and len(part) <= 80]
        if stage_names and len(parts) == len(stage_names):
            parts = stage_names
        for source, target in zip(parts, parts[1:]):
            connection_type = _connection_type(branch)
            if connection_type == "unknown":
                connection_type = "inferred"
            confidence = "inferred"
            evidence = Evidence(
                id=f"report-edge-{abs(hash((source, target))) % 10000}",
                claim=f"{source} -> {target}",
                source_type="report",
                summary=branch,
                confidence=confidence,
            )
            edges.append(
                StageEdge(
                    from_stage=source,
                    to_stage=target,
                    connection_type=connection_type,
                    evidence=[evidence],
                    confidence=confidence,
                )
            )
    return edges


def _branch_expression(line: str) -> str:
    for separator in ("：", ":"):
        prefix, found, suffix = line.partition(separator)
        if found and any(marker in prefix for marker in _FLOW_MARKERS):
            return suffix.strip(" -*`。")
    return line.strip(" -*`。")


def _clean_stage_token(value: str) -> str:
    normalized = value.strip(" -*`。.;；")
    for separator in ("：", ":"):
        normalized = normalized.partition(separator)[0]
    return normalized.strip(" -*`。.;；")


def _connection_type(text: str) -> str:
    lowered = text.casefold()
    mapping = (
        ("direct_call", ("direct_call", "调用")),
        ("repository_read_write", ("repository", "落库", "持久化")),
        ("shared_table", ("shared_table", "共享表")),
        ("mapping_table", ("mapping_table", "映射表")),
        ("event", ("event", "事件")),
        ("rpc", ("rpc", "grpc")),
        ("job", ("job", "任务")),
        ("status_transition", ("status_transition", "状态")),
        ("external_boundary", ("external_boundary", "外部")),
    )
    for connection_type, markers in mapping:
        if any(marker in lowered for marker in markers):
            return connection_type
    return "unknown"


def _markdown_items(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] or ["- None"]
