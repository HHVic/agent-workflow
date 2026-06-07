"""Tests for stage_requirements pure functions."""

from app.agents.code_explorer.exploration_state import BusinessStage, Evidence
from app.agents.code_explorer.stage_requirements import (
    get_required_fields_for_stage,
    is_blocking_stage,
)


def _stage(stage_type: str, **kwargs) -> BusinessStage:
    return BusinessStage(name=f"{stage_type} stage", stage_type=stage_type, **kwargs)


# -- get_required_fields_for_stage --

def test_unknown_stage_returns_empty() -> None:
    assert get_required_fields_for_stage(_stage("unknown")) == []


def test_supporting_stage_returns_empty() -> None:
    assert get_required_fields_for_stage(_stage("supporting")) == []


def test_optional_stage_returns_empty() -> None:
    stage = _stage("entry")
    stage.is_optional = True
    assert get_required_fields_for_stage(stage) == []


def test_entry_stage_required_fields() -> None:
    assert get_required_fields_for_stage(_stage("entry")) == ["trigger", "input", "output"]


def test_controller_or_rpc_stage_required_fields() -> None:
    assert get_required_fields_for_stage(_stage("controller_or_rpc")) == ["trigger", "input", "output"]


def test_scheduled_job_stage_required_fields() -> None:
    assert get_required_fields_for_stage(_stage("scheduled_job")) == ["trigger", "input", "output"]


def test_calculation_stage_required_fields() -> None:
    assert get_required_fields_for_stage(_stage("calculation")) == ["input", "transform", "output"]


def test_aggregation_stage_no_state_change_when_no_indicator() -> None:
    stage = _stage("aggregation")
    assert get_required_fields_for_stage(stage) == ["input", "transform", "output", "persistence"]


def test_aggregation_stage_requires_state_change_when_capability_set() -> None:
    stage = _stage("aggregation", capabilities=["has_state_change"])
    assert "state_change" in get_required_fields_for_stage(stage)


def test_aggregation_stage_requires_state_change_when_evidence_indicates() -> None:
    stage = _stage(
        "aggregation",
        evidence=[
            Evidence(
                id="e1", claim="status updated", summary="status transition",
                source_type="repository", confidence="confirmed",
                can_satisfy_stage_field=True,
            )
        ],
    )
    assert "state_change" in get_required_fields_for_stage(stage)


def test_disclosure_stage_required_fields() -> None:
    stage = _stage("disclosure")
    assert get_required_fields_for_stage(stage) == ["input", "transform", "output", "persistence"]


def test_settlement_creation_required_fields() -> None:
    assert get_required_fields_for_stage(_stage("settlement_creation")) == [
        "trigger", "input", "transform", "output", "persistence",
    ]


def test_payment_or_invoice_required_fields() -> None:
    assert get_required_fields_for_stage(_stage("payment_or_invoice")) == [
        "trigger", "input", "output",
    ]


def test_persistence_read_required_fields() -> None:
    assert get_required_fields_for_stage(_stage("persistence_read")) == [
        "input", "persistence", "output",
    ]


def test_persistence_write_no_state_change_by_default() -> None:
    assert get_required_fields_for_stage(_stage("persistence_write")) == [
        "input", "persistence", "output",
    ]


def test_persistence_write_requires_state_change_when_status_write() -> None:
    stage = _stage("persistence_write", capabilities=["status_update"])
    assert "state_change" in get_required_fields_for_stage(stage)


def test_state_transition_required_fields() -> None:
    assert get_required_fields_for_stage(_stage("state_transition")) == [
        "input", "state_change", "output",
    ]


def test_event_publish_required_fields() -> None:
    assert get_required_fields_for_stage(_stage("event_publish")) == ["trigger", "output"]


def test_event_consume_required_fields() -> None:
    assert get_required_fields_for_stage(_stage("event_consume")) == ["trigger", "input"]


# -- is_blocking_stage --

def test_blocking_unknown_stage() -> None:
    assert not is_blocking_stage(_stage("unknown"))


def test_blocking_supporting_stage() -> None:
    assert not is_blocking_stage(_stage("supporting"))


def test_blocking_optional_stage() -> None:
    stage = _stage("entry")
    stage.is_optional = True
    assert not is_blocking_stage(stage)


def test_blocking_non_mainline_stage() -> None:
    stage = _stage("entry", is_mainline=False)
    assert not is_blocking_stage(stage)


def test_blocking_mainline_stage() -> None:
    stage = _stage("entry", is_mainline=True)
    assert is_blocking_stage(stage)


def test_blocking_mainline_calculation() -> None:
    stage = _stage("calculation", is_mainline=True)
    assert is_blocking_stage(stage)
