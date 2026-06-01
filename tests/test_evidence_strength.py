from app.agents.code_explorer.evidence_extractor import extract_evidence_from_tool_result
from app.agents.code_explorer.exploration_state import (
    Evidence,
    can_satisfy_stage_edge,
    can_satisfy_stage_field,
    is_strong_evidence,
)


def _extract(tool_name: str, result: str) -> list[Evidence]:
    return extract_evidence_from_tool_result(tool_name, {}, result)


def test_evidence_defaults_to_weak_and_cannot_satisfy_stage() -> None:
    evidence = Evidence(
        id="ev-1",
        claim="A candidate symbol exists",
        source_type="symbol",
        summary="WidgetService at src/WidgetService.java:10",
        confidence="confirmed",
    )

    assert evidence.strength == "weak"
    assert evidence.can_satisfy_stage_field is False
    assert evidence.can_satisfy_stage_edge is False
    assert is_strong_evidence(evidence) is False
    assert can_satisfy_stage_field(evidence, "transform") is False
    assert can_satisfy_stage_edge(evidence) is False


def test_search_result_is_a_weak_lead() -> None:
    evidence = _extract(
        "search_code",
        "WidgetService src/service/WidgetService.java:12",
    )

    assert evidence
    assert all(item.strength == "weak" for item in evidence)
    assert all(item.can_satisfy_stage_field is False for item in evidence)


def test_import_mapper_conversion_and_response_wrapper_do_not_satisfy_stage() -> None:
    evidence = _extract(
        "explore_symbol",
        "\n".join(
            [
                "import app.repo.WidgetRepository;",
                "WidgetVO vo = WidgetMapper.MAPPER.boToVo(widgetBO);",
                "return Response.SUCCESS(PageResult.of(items));",
                "class WidgetController extends AbstractController {",
                "  void list(Context context) {}",
                "}",
            ]
        ),
    )

    assert evidence
    assert all(item.strength != "strong" for item in evidence)
    assert all(item.can_satisfy_stage_field is False for item in evidence)
    assert all(item.can_satisfy_stage_edge is False for item in evidence)


def test_actual_repository_write_and_status_write_are_strong() -> None:
    evidence = _extract(
        "explore_symbol",
        "\n".join(
            [
                "widgetRepository.save(widget);",
                "widgetRepository.onlyUpdateWidgetStatus(widgetId, DONE);",
            ]
        ),
    )

    assert any(item.source_type == "repository" and is_strong_evidence(item) for item in evidence)
    assert any(item.source_type == "status" and is_strong_evidence(item) for item in evidence)
    assert any(can_satisfy_stage_field(item, "persistence") for item in evidence)
    assert any(can_satisfy_stage_field(item, "state_change") for item in evidence)


def test_business_call_edge_is_strong_but_generic_edge_is_noise() -> None:
    evidence = _extract(
        "find_callees",
        "\n".join(
            [
                "WidgetService.process -> WidgetRepository.save",
                "WidgetService.process -> Context.getUser",
            ]
        ),
    )

    strong_edges = [item for item in evidence if can_satisfy_stage_edge(item)]
    noise_edges = [item for item in evidence if item.strength == "noise"]

    assert any("WidgetRepository.save" in item.summary for item in strong_edges)
    assert any("Context.getUser" in item.summary for item in noise_edges)


def test_event_publish_and_listener_are_strong() -> None:
    evidence = _extract(
        "explore_symbol",
        "\n".join(
            [
                "applicationEventPublisher.publishEvent(new WidgetEvent(widgetId));",
                "void onApplicationEvent(WidgetEvent event) {",
            ]
        ),
    )

    event_evidence = [item for item in evidence if item.source_type == "event"]
    assert len(event_evidence) >= 2
    assert all(is_strong_evidence(item) for item in event_evidence)
    assert all(item.reason for item in event_evidence)
