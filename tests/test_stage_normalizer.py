from app.agents.code_explorer.exploration_state import Evidence
from app.agents.code_explorer.stage_normalizer import (
    merge_normalized_stages,
    normalize_stage_candidates,
    normalize_stage_edges,
)


def _evidence(symbol: str, source_type: str = "tool_result") -> Evidence:
    return Evidence(
        id=f"e-{symbol}",
        claim=f"Found {symbol}",
        source_type=source_type,
        symbol=symbol,
        file_path="src/main/java/example/Widget.java:10",
        summary=symbol,
        confidence="confirmed",
        strength="strong",
        can_satisfy_stage_field=True,
        can_satisfy_stage_edge=True,
        reason="Explicit strong fixture evidence.",
    )


def test_normalizes_methods_into_generic_business_stages() -> None:
    stages = normalize_stage_candidates(
        [
            _evidence("WidgetGrpcDelegate.doCreateWidget"),
            _evidence("WidgetService.calculateAmount"),
            _evidence("WidgetRepository.saveWidget", "repository"),
        ]
    )

    names = [stage.name for stage in stages]
    assert "入口触发" in names
    assert "业务计算与转换" in names
    assert "持久化写入" in names
    assert "WidgetService.calculateAmount" not in names
    assert all(stage.confidence == "confirmed" for stage in stages)


def test_unknown_symbol_becomes_unconfirmed_stage_candidate() -> None:
    stages = normalize_stage_candidates([_evidence("WidgetOrchestrator.begin")])

    assert stages[0].name.startswith("待确认阶段：")
    assert stages[0].confidence == "unknown"


def test_call_edge_does_not_become_whole_arrow_stage() -> None:
    evidence = _evidence(
        "WidgetGrpcDelegate.doCreateWidget -> WidgetService.calculateAmount",
        "call_edge",
    )

    stages = normalize_stage_candidates([evidence])
    edges = normalize_stage_edges([evidence])

    assert not stages
    assert [(edge.from_stage, edge.to_stage) for edge in edges] == [
        ("入口触发", "业务计算与转换")
    ]


def test_test_source_evidence_does_not_become_business_stage() -> None:
    evidence = _evidence("should_calculate_amount_when_valid")
    evidence.file_path = "src/test/java/example/WidgetServiceTest.java:22"

    assert not normalize_stage_candidates([evidence])


def test_weak_evidence_cannot_fill_stage_field() -> None:
    evidence = _evidence("WidgetService.calculateAmount")
    evidence.strength = "weak"
    evidence.can_satisfy_stage_field = False

    stages = normalize_stage_candidates([evidence])

    assert stages
    assert stages[0].confidence == "unknown"
    assert stages[0].optional
    assert stages[0].transform is None


def test_strong_evidence_promotes_matching_weak_candidate_to_main_stage() -> None:
    weak = _evidence("WidgetStatus", "status")
    weak.strength = "weak"
    weak.can_satisfy_stage_field = False
    strong = _evidence("WidgetStatus", "status")

    stages = merge_normalized_stages(
        normalize_stage_candidates([weak]),
        normalize_stage_candidates([strong]),
    )

    assert stages[0].name == "状态流转"
    assert stages[0].confidence == "confirmed"
    assert not stages[0].optional
