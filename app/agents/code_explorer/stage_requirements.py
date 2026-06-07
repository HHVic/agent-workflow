"""Required fields and blocking logic per BusinessStage stage_type."""

from __future__ import annotations

from app.agents.code_explorer.exploration_state import BusinessStage, Evidence


def _evidence_has_state_change(evidence: Evidence | None) -> bool:
    """Check if an Evidence object indicates state change."""
    if evidence is None:
        return False
    if evidence.can_satisfy_stage_field:
        return True
    text = (evidence.claim + " " + evidence.summary).casefold()
    state_markers = ("state", "status", "transition", "update", "change")
    return any(m in text for m in state_markers)


def _stage_has_state_change_evidence(stage: BusinessStage) -> bool:
    """Check if stage evidence collectively indicates state change."""
    for e in stage.evidence:
        if _evidence_has_state_change(e):
            return True
    return False


def _has_state_change_capability(stage: BusinessStage) -> bool:
    """Check if stage capabilities include has_state_change."""
    return "has_state_change" in stage.capabilities


def _evidence_has_edge(evidence: Evidence | None) -> bool:
    """Check if Evidence indicates an edge/connection."""
    if evidence is None:
        return False
    text = (evidence.claim + " " + evidence.summary + " " + evidence.source_type).casefold()
    edge_markers = ("edge", "upstream", "connection", "call", "invoke")
    return any(m in text for m in edge_markers)


def _evidence_is_external_boundary(evidence: Evidence | None) -> bool:
    """Check if Evidence indicates external boundary."""
    if evidence is None:
        return False
    text = (evidence.claim + " " + evidence.summary + " " + evidence.source_type).casefold()
    return "external_boundary" in text or "external" in text


def _evidence_is_event(evidence: Evidence | None) -> bool:
    """Check if Evidence indicates an event."""
    if evidence is None:
        return False
    text = (evidence.claim + " " + evidence.summary + " " + evidence.source_type).casefold()
    return "event" in text


def _is_status_write_stage(stage: BusinessStage) -> bool:
    """Heuristic: is this stage primarily writing status fields?"""
    if stage.capabilities:
        return any("status" in cap for cap in stage.capabilities)
    text = (stage.name + " " + str(stage.description or "")).casefold()
    status_markers = ("update_status", "set_status", "status_change", "mark", "finalize")
    return any(m in text for m in status_markers)


def get_required_fields_for_stage(stage: BusinessStage) -> list[str]:
    """Return the list of required field names for the given stage type."""

    # Supporting and unknown stages have no blocking requirements.
    if stage.is_optional:
        return []
    if stage.stage_type in ("supporting", "unknown"):
        return []

    stage_type = stage.stage_type

    if stage_type in ("entry", "controller_or_rpc"):
        return ["trigger", "input", "output"]

    if stage_type == "scheduled_job":
        return ["trigger", "input", "output"]

    if stage_type == "calculation":
        return ["input", "transform", "output"]

    if stage_type in ("aggregation", "disclosure"):
        fields = ["input", "transform", "output", "persistence"]
        if _has_state_change_capability(stage) or _stage_has_state_change_evidence(stage):
            fields.append("state_change")
        return fields

    if stage_type == "settlement_creation":
        return ["trigger", "input", "transform", "output", "persistence"]

    if stage_type == "payment_or_invoice":
        return ["trigger", "input", "output"]

    if stage_type == "persistence_read":
        return ["input", "persistence", "output"]

    if stage_type == "persistence_write":
        fields = ["input", "persistence", "output"]
        if _is_status_write_stage(stage):
            fields.append("state_change")
        return fields

    if stage_type == "state_transition":
        return ["input", "state_change", "output"]

    if stage_type == "event_publish":
        return ["trigger", "output"]

    if stage_type == "event_consume":
        return ["trigger", "input"]

    # Fallback: all fields required if stage_type is unrecognized but not unknown/supporting.
    return ["trigger", "input", "transform", "output", "persistence", "state_change"]


def is_blocking_stage(stage: BusinessStage) -> bool:
    """Return True if this stage blocks completion when its required fields are missing."""
    if stage.stage_type in ("supporting", "unknown"):
        return False
    if stage.is_optional:
        return False
    return stage.is_mainline
