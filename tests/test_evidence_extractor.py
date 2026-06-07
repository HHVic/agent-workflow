"""Tests for evidence_extractor, including test-code downgrade rules."""

from app.agents.code_explorer.exploration_state import Evidence
from app.agents.code_explorer.evidence_extractor import extract_evidence_from_tool_result


def test_extracts_symbol_and_file_path_from_search_result() -> None:
    evidence = extract_evidence_from_tool_result(
        "search_code",
        {"query": "WidgetService"},
        "- WidgetService (class) - src/main/java/example/WidgetService.java:18",
    )

    assert any(
        item.symbol == "WidgetService"
        and item.file_path == "src/main/java/example/WidgetService.java:18"
        and item.confidence == "confirmed"
        for item in evidence
    )


def test_extracts_standalone_file_path_evidence() -> None:
    evidence = extract_evidence_from_tool_result(
        "list_code_files",
        {"pattern": "**/*Application.java"},
        "miniapp/open-service/src/main/java/example/WidgetApplication.java:12",
    )

    assert any(
        item.source_type == "file_path"
        and item.file_path
        == "miniapp/open-service/src/main/java/example/WidgetApplication.java:12"
        for item in evidence
    )


def test_extracts_call_edge_from_callers_result() -> None:
    evidence = extract_evidence_from_tool_result(
        "find_callers",
        {"symbol": "WidgetService.calculate"},
        """\
## Callers of WidgetService.calculate (1 found)

- WidgetJob.doExecute (method) - src/main/java/example/WidgetJob.java:22
""",
    )

    assert any(
        item.source_type == "call_edge"
        and item.symbol == "WidgetJob.doExecute -> WidgetService.calculate"
        for item in evidence
    )


def test_extracts_call_edge_from_arrow_relationship() -> None:
    evidence = extract_evidence_from_tool_result(
        "trace_path",
        {"from": "WidgetJob.doExecute", "to": "WidgetRepository.save"},
        "WidgetJob.doExecute -> WidgetService.calculate -> WidgetRepository.save",
    )

    assert any(
        item.source_type == "call_edge"
        and item.symbol == "WidgetService.calculate -> WidgetRepository.save"
        for item in evidence
    )


def test_extracts_persistence_event_and_status_evidence() -> None:
    evidence = extract_evidence_from_tool_result(
        "explore_symbol",
        {"query": "widget flow"},
        """\
## WidgetRepository.save (method)
**Location:** src/main/java/example/WidgetRepository.java:31
save MiniAppWidgetPo into mini_app_widget_record

WidgetEvent published by publishEvent and consumed by WidgetEventListener.onApplicationEvent.
status changes from PENDING_REVIEW -> SUCCESS.
""",
    )

    assert any(item.source_type == "repository" for item in evidence)
    assert any(item.source_type == "table" for item in evidence)
    assert any(item.source_type == "event" for item in evidence)
    assert any(item.source_type == "status" for item in evidence)


def test_extracts_standalone_uppercase_status_enum() -> None:
    evidence = extract_evidence_from_tool_result(
        "explore_symbol",
        {"query": "widget status"},
        "READY_TO_WITHDRAW",
    )

    assert any(item.source_type == "status" for item in evidence)


def test_does_not_treat_unrelated_uppercase_constant_as_status() -> None:
    evidence = extract_evidence_from_tool_result(
        "explore_symbol",
        {"query": "widget date"},
        "private static final DateTimeFormatter BASIC_ISO_DATE = formatter;",
    )

    assert not any(item.source_type == "status" for item in evidence)


def test_does_not_guess_edge_path_from_multi_file_result() -> None:
    evidence = extract_evidence_from_tool_result(
        "explore_symbol",
        {"query": "widget flow"},
        "src/main/java/example/First.java:10\n"
        "WidgetJob.doExecute -> WidgetService.calculate\n"
        "src/main/java/example/Second.java:20",
    )

    edge = next(item for item in evidence if item.source_type == "call_edge")
    assert edge.file_path is None


def test_path_containing_repository_is_not_by_itself_persistence_evidence() -> None:
    evidence = extract_evidence_from_tool_result(
        "list_code_files",
        {},
        "src/main/java/example/repository/WidgetRepository.java:10",
    )

    assert not any(item.source_type == "repository" for item in evidence)


def test_does_not_create_business_evidence_from_generic_tool_success_text() -> None:
    evidence = extract_evidence_from_tool_result(
        "get_code_context",
        {"task": "widget flow"},
        "get_code_context returned exploration evidence",
    )

    assert not evidence


# -- Test-code downgrade rules --


def test_src_test_file_path_evidence_is_weak() -> None:
    """Test evidence from /src/test/ paths is downgraded to weak."""
    evidence = extract_evidence_from_tool_result(
        "find_callers",
        {"symbol": "WidgetService.calculate"},
        """\
## Callers of WidgetService.calculate (1 found)

- WidgetJob.doExecute (method) - src/main/java/example/test/WidgetJob.java:22
""",
    )

    edge = next((e for e in evidence if e.source_type == "call_edge"), None)
    assert edge is not None
    assert edge.strength == "weak"
    assert edge.can_satisfy_stage_edge is False


def test_test_java_file_is_weak() -> None:
    """Test evidence from Test.java files is downgraded to weak."""
    evidence = extract_evidence_from_tool_result(
        "find_callers",
        {"symbol": "WidgetService.calculate"},
        """\
## Callers of WidgetService.calculate (1 found)

- WidgetJob.doExecute (method) - src/main/java/example/WidgetJobTest.java:22
""",
    )

    edge = next((e for e in evidence if e.source_type == "call_edge"), None)
    assert edge is not None
    assert edge.strength == "weak"
    assert edge.can_satisfy_stage_edge is False


def test_should_call_edge_is_weak() -> None:
    """Call edges from test method names (should_*) are downgraded to weak."""
    evidence = extract_evidence_from_tool_result(
        "trace_path",
        {"from": "should_createUser", "to": "UserService.createUser"},
        "should_createUser -> UserService.createUser",
    )

    edge = next((e for e in evidence if e.source_type == "call_edge"), None)
    assert edge is not None
    assert edge.strength == "weak"
    assert edge.can_satisfy_stage_edge is False


def test_when_then_return_mockito_stub_is_weak() -> None:
    """Mockito stub when(...).thenReturn is detected as test code and downgraded."""
    evidence = extract_evidence_from_tool_result(
        "explore_symbol",
        {"query": "widget flow"},
        "when(xxxDao.selectByExample(any())).thenReturn(mockResult)",
    )

    repo = next((e for e in evidence if e.source_type == "repository"), None)
    assert repo is not None
    assert repo.strength == "weak"
    assert repo.can_satisfy_stage_field is False


def test_verify_keyword_in_evidence_is_weak() -> None:
    """Evidence containing verify() is detected as test code and downgraded."""
    evidence = extract_evidence_from_tool_result(
        "explore_symbol",
        {"query": "widget flow"},
        "verify(mockService).doSomething()",
    )

    semantic_items = [
        e for e in evidence if e.source_type in ("repository", "event", "status", "call_edge")
    ]
    for item in semantic_items:
        assert item.strength == "weak"
        assert item.can_satisfy_stage_field is False
        assert item.can_satisfy_stage_edge is False


def test_prod_repository_update_is_strong() -> None:
    """Production repository.update remains strong."""
    evidence = extract_evidence_from_tool_result(
        "explore_symbol",
        {"query": "widget flow"},
        """\
## WidgetRepository.update (method)
**Location:** src/main/java/example/WidgetRepository.java:31
update Widget record in widget_record table
""",
    )

    repo = next((e for e in evidence if e.source_type == "repository"), None)
    assert repo is not None
    assert repo.strength == "strong"
    assert repo.can_satisfy_stage_field is True


def test_prod_only_update_status_is_strong() -> None:
    """Production onlyUpdateSettlementStatus remains strong."""
    evidence = extract_evidence_from_tool_result(
        "explore_symbol",
        {"query": "widget flow"},
        """\
## SettlementService.onlyUpdateSettlementStatus (method)
**Location:** src/main/java/example/SettlementService.java:45
onlyUpdateSettlementStatus(id, NEW_STATUS)
""",
    )

    status = next((e for e in evidence if e.source_type == "status"), None)
    assert status is not None
    assert status.strength == "strong"
    assert status.can_satisfy_stage_field is True


def test_prod_publish_event_is_strong() -> None:
    """Production publishEvent remains strong."""
    evidence = extract_evidence_from_tool_result(
        "explore_symbol",
        {"query": "widget flow"},
        """\
## WidgetService.publishEvent (method)
**Location:** src/main/java/example/WidgetService.java:50
WidgetEvent published by publishEvent
""",
    )

    event = next((e for e in evidence if e.source_type == "event"), None)
    assert event is not None
    assert event.strength == "strong"
    assert event.can_satisfy_stage_field is True


def test_prod_call_edge_is_strong() -> None:
    """Production call edges remain strong."""
    evidence = extract_evidence_from_tool_result(
        "find_callers",
        {"symbol": "WidgetService.calculate"},
        """\
## Callers of WidgetService.calculate (1 found)

- WidgetJob.doExecute (method) - src/main/java/example/WidgetJob.java:22
""",
    )

    edge = next((e for e in evidence if e.source_type == "call_edge"), None)
    assert edge is not None
    assert edge.strength == "strong"
    assert edge.can_satisfy_stage_edge is True
