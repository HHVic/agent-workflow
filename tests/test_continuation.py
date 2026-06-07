"""Tests for relevance-gate integration in continuation prompts."""

from __future__ import annotations

from app.agents.code_explorer.exploration_state import (
    Evidence,
    ExplorationState,
    StopDecision,
)
from app.agents.code_explorer.continuation import (
    build_continuation_prompt,
    build_convergence_checkpoint_prompt,
    build_stagnation_report_prompt,
    build_anti_stagnation_prompt,
    _extract_confirmed_symbols,
    _extract_confirmed_paths,
    _extract_connected_symbols,
    _filter_candidates,
)

# ---- Fixtures ----


def _make_evidence(
    claim: str = "",
    source_type: str = "call_edge",
    summary: str = "",
    strength: str = "weak",
    file_path: str | None = None,
    symbol: str | None = None,
    can_satisfy_stage_field: bool = False,
    can_satisfy_stage_edge: bool = False,
    reason: str = "No explicit rule",
) -> Evidence:
    return Evidence(
        id=f"e-{hash((claim, source_type)) % 10000}",
        claim=claim,
        source_type=source_type,
        summary=summary,
        confidence="confirmed" if strength == "strong" else "inferred",
        strength=strength,
        file_path=file_path,
        symbol=symbol,
        can_satisfy_stage_field=can_satisfy_stage_field,
        can_satisfy_stage_edge=can_satisfy_stage_edge,
        reason=reason,
    )


def _make_state(
    goal: str = "income settlement flow",
    candidate_actions: list[str] | None = None,
    evidence_items: list[Evidence] | None = None,
    confirmed_facts: list[Evidence] | None = None,
    open_questions: list[str] | None = None,
    unknowns: list[str] | None = None,
    notebook: str | None = None,
    stop_decision: StopDecision | None = None,
) -> ExplorationState:
    return ExplorationState(
        goal=goal,
        task_mode="feature_exploration",
        candidate_next_actions=candidate_actions or [],
        evidence_items=evidence_items or [],
        confirmed_facts=confirmed_facts or [],
        open_questions=open_questions or [],
        unknowns=unknowns or [],
        notebook_markdown=notebook,
        stop_decision=stop_decision,
    )


def _assert_in_prioritized(prompt: str, text: str) -> bool:
    """Check if text appears in the prioritized section of a prompt."""
    lines = prompt.split("\n")
    in_prioritized = False
    for line in lines:
        stripped = line.strip()
        # Detect prioritized section heading
        if stripped.startswith("#") and ("优先补齐方向" in stripped or "优先探索方向" in stripped):
            in_prioritized = True
            continue
        # Detect downgraded section heading
        if stripped.startswith("#") and "已降级" in stripped:
            in_prioritized = False
            continue
        if in_prioritized and text in stripped:
            return True
    return False


def _assert_not_in_prioritized(prompt: str, text: str) -> None:
    """Assert that text does NOT appear in the prioritized section."""
    assert not _assert_in_prioritized(prompt, text), (
        f"'{text}' should NOT be in prioritized section"
    )


def _assert_in_downgraded(prompt: str, text: str) -> bool:
    """Check if text appears in the downgraded section of a prompt."""
    lines = prompt.split("\n")
    in_downgraded = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and "已降级" in stripped:
            in_downgraded = True
            continue
        if stripped.startswith("#"):
            in_downgraded = False
            continue
        if in_downgraded and text in stripped:
            return True
    return False


# ---- Test 1: continuation prompt preserves relevant candidates ----

def test_continuation_prompt_preserves_relevant_candidates() -> None:
    """Candidates matching goal/mainline terms should appear in prioritized section."""
    evidence = [
        _make_evidence(
            claim="IncomeDailyCalculationJob is a strong trigger",
            source_type="job",
            summary="IncomeDailyCalculationJob",
            strength="strong",
            symbol="IncomeDailyCalculationJob",
            file_path="src/main/java/com/example/biz/income/DailyCalculationJob.java",
        ),
    ]
    state = _make_state(
        goal="income settlement flow",
        candidate_actions=[
            "IncomeHourlyCalculationJob",
            "SettlementCreateJob",
            "OpenOrderEvent",
        ],
        evidence_items=evidence,
        confirmed_facts=evidence,
    )
    decision = StopDecision(
        can_stop=False,
        reason="needs more evidence",
        blocking_gaps=["Missing persistence"],
    )
    prompt = build_continuation_prompt(decision, state)

    # Relevant candidates should be in prioritized
    _assert_in_prioritized(prompt, "IncomeHourlyCalculationJob")
    _assert_in_prioritized(prompt, "SettlementCreateJob")


# ---- Test 2: continuation prompt filters obviously irrelevant candidates ----

def test_continuation_prompt_filters_irrelevant_candidates() -> None:
    """Candidates with no overlap with goal/confirmed should be filtered out."""
    state = _make_state(
        goal="income settlement flow",
        candidate_actions=[
            "OpenOrderEvent",
            "AppletSeasonService",
            "GenericStatusManager",
        ],
        evidence_items=[],
    )
    decision = StopDecision(
        can_stop=False,
        reason="needs more evidence",
        blocking_gaps=["Missing evidence"],
    )
    prompt = build_continuation_prompt(decision, state)

    # These should NOT appear in prioritized section
    _assert_not_in_prioritized(prompt, "OpenOrderEvent")
    _assert_not_in_prioritized(prompt, "AppletSeasonService")
    _assert_not_in_prioritized(prompt, "GenericStatusManager")

    # They should appear in downgraded section
    assert "已降级候选" in prompt, "Should show downgraded section"
    _assert_in_downgraded(prompt, "OpenOrderEvent")


# ---- Test 3: suffix-only candidates (Event/Service/Job) without overlap are filtered ----

def test_suffix_only_candidates_filtered() -> None:
    """Symbols with only generic suffixes and no domain overlap should not be prioritized."""
    state = _make_state(
        goal="income settlement flow",
        candidate_actions=[
            "OpenOrderEvent",  # open, order, event all generic
            "MiniAppLongVideoSearchIntervenePo",  # no overlap
            "AppletService",  # app is generic
            "IncomeDailyCalculationJob",  # has income overlap
        ],
        evidence_items=[],
    )
    decision = StopDecision(
        can_stop=False,
        reason="needs more evidence",
        blocking_gaps=["Missing evidence"],
    )
    prompt = build_continuation_prompt(decision, state)

    # IncomeDailyCalculationJob has "income" which overlaps with goal -> prioritized
    _assert_in_prioritized(prompt, "IncomeDailyCalculationJob")

    # Generic suffix-only candidates should NOT be in prioritized
    _assert_not_in_prioritized(prompt, "OpenOrderEvent")
    _assert_not_in_prioritized(prompt, "MiniAppLongVideoSearchIntervenePo")
    _assert_not_in_prioritized(prompt, "AppletService")


# ---- Test 4: connected_symbols candidates are preserved ----

def test_connected_symbols_preserved() -> None:
    """Candidates in connected_symbols should pass even with low goal overlap."""
    strong_evidence = [
        _make_evidence(
            claim="source: IncomeDailyCalculationJob -> target: SettlementCreateJob",
            source_type="call_edge",
            summary="IncomeDailyCalculationJob -> SettlementCreateJob",
            strength="strong",
            symbol="IncomeDailyCalculationJob",
        ),
    ]
    # ConnectedButUnrelated has no domain overlap but is in connected_symbols
    state = _make_state(
        goal="income settlement flow",
        candidate_actions=[
            "ConnectedButUnrelatedSymbol",
            "IncomeDailyCalculationJob",
        ],
        evidence_items=strong_evidence,
        confirmed_facts=strong_evidence,
    )
    decision = StopDecision(
        can_stop=False,
        reason="needs more evidence",
        blocking_gaps=["Missing evidence"],
    )
    prompt = build_continuation_prompt(decision, state)

    # ConnectedButUnrelatedSymbol should appear (connected_symbols pass)
    _assert_in_prioritized(prompt, "ConnectedButUnrelatedSymbol")


# ---- Test 5: test symbols should_* do not enter prioritized section ----

def test_test_symbols_filtered_from_prioritized() -> None:
    """Symbols prefixed with should_ should be filtered out from prioritized section."""
    state2 = _make_state(
        goal="income settlement flow",
        candidate_actions=[
            "should_TestOnlySymbol",
            "when_MockReturnsNull",
            "IncomeDailyJob",
        ],
        evidence_items=[],
    )
    decision = StopDecision(
        can_stop=False,
        reason="needs more evidence",
        blocking_gaps=["Missing evidence"],
    )
    prompt2 = build_continuation_prompt(decision, state2)

    # Test symbols should NOT be in the prioritized section
    _assert_not_in_prioritized(prompt2, "should_TestOnlySymbol")
    _assert_not_in_prioritized(prompt2, "when_MockReturnsNull")

    # IncomeDailyJob has "income" -> should pass via goal overlap
    _assert_in_prioritized(prompt2, "IncomeDailyJob")


# ---- Test 6: when no high-relevance candidates, prompt doesn't force irrelevant ones ----

def test_no_forced_irrelevant_candidates_when_none_relevant() -> None:
    """When all candidates are irrelevant, prioritized section should say so."""
    state = _make_state(
        goal="income settlement flow",
        candidate_actions=[
            "OpenOrderEvent",
            "AppletSeasonService",
            "GenericStatusManager",
        ],
        evidence_items=[],
    )
    decision = StopDecision(
        can_stop=False,
        reason="needs more evidence",
        blocking_gaps=["Missing evidence"],
    )
    prompt = build_continuation_prompt(decision, state)

    # The prioritized section should say "无可" when no candidates are relevant
    # Find the boundary: from "优先补齐方向" / "优先探索方向" heading to "已降级" heading
    lines = prompt.split("\n")
    in_target_prioritized = False
    found_no_candidates_msg = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and ("优先补齐方向" in stripped or "优先探索方向" in stripped):
            in_target_prioritized = True
            continue
        if in_target_prioritized and stripped.startswith("#"):
            in_target_prioritized = False
            continue
        if in_target_prioritized and stripped.startswith("- "):
            assert "无可" in stripped or "没有" in stripped or "暂无" in stripped, (
                f"Prioritized section should not list irrelevant candidates, got: {stripped}"
            )
            found_no_candidates_msg = True

    assert found_no_candidates_msg, "Should have 'no candidates' message in prioritized section"


# ---- Test 7: downgraded candidates are shown with clear disclaimer ----

def test_downgraded_candidates_have_disclaimer() -> None:
    """Downgraded candidates should appear with a clear disclaimer."""
    state = _make_state(
        goal="income settlement flow",
        candidate_actions=[
            "IncomeDailyCalculationJob",
            "OpenOrderEvent",
            "AppletSeasonService",
        ],
        evidence_items=[],
    )
    decision = StopDecision(
        can_stop=False,
        reason="needs more evidence",
        blocking_gaps=["Missing evidence"],
    )
    prompt = build_continuation_prompt(decision, state)

    assert "已降级候选" in prompt
    assert "相关性不足" in prompt or "暂不优先" in prompt


# ---- Test 8: convergence checkpoint uses relevance gate ----

def test_convergence_checkpoint_uses_relevance_gate() -> None:
    """Convergence checkpoint should filter candidates via relevance gate."""
    state = _make_state(
        goal="income settlement flow",
        candidate_actions=[
            "IncomeHourlyJob",
            "OpenOrderEvent",
        ],
        evidence_items=[],
    )
    prompt = build_convergence_checkpoint_prompt(state, used_tool_calls=50, max_tool_calls=100)

    # Should contain prioritized section with filtered candidates
    assert "优先探索方向" in prompt
    # OpenOrderEvent should be filtered (generic suffix only)
    _assert_not_in_prioritized(prompt, "OpenOrderEvent")

    # Should have the "no high-relevance" guidance
    assert "高相关候选" in prompt or "阶段性报告" in prompt


# ---- Test 9: no high-relevance candidates → prompt says don't continue blind search ----

def test_convergence_no_candidates_says_dont_search_blindly() -> None:
    """When no high-relevance candidates remain, prompt should advise against blind search."""
    state = _make_state(
        goal="income settlement flow",
        candidate_actions=[
            "OpenOrderEvent",
            "MiniAppLongVideoSearchIntervenePo",
        ],
        evidence_items=[],
    )
    prompt = build_convergence_checkpoint_prompt(state, used_tool_calls=90, max_tool_calls=100)

    # Should advise against blind search when no high-relevance candidates
    assert "盲搜" in prompt or "不要继续" in prompt or "阶段性报告" in prompt

    # And should mention strong evidence for report generation
    assert "strong evidence" in prompt or "已确认" in prompt


# ---- Test 10: _filter_candidates uses confirmed_facts symbols and paths ----

def test_filter_candidates_uses_confirmed_facts_symbols_and_paths() -> None:
    """Reproduce the user's scenario: confirmed_facts has IncomeDisclosureJob and SettlementService."""

    # Strong evidence ONLY in confirmed_facts (not in evidence_items)
    # This is the exact scenario the user described
    strong_evidence = [
        _make_evidence(
            claim="IncomeDisclosureJob is the income disclosure entry point",
            source_type="job",
            summary="IncomeDisclosureJob",
            strength="strong",
            symbol="IncomeDisclosureJob",
            file_path="src/main/java/com/example/biz/income/disclosure/IncomeDisclosureJob.java",
        ),
        _make_evidence(
            claim="SettlementService handles the settlement flow",
            source_type="service",
            summary="SettlementService",
            strength="strong",
            symbol="SettlementService",
            file_path="src/main/java/com/example/biz/settlement/impl/SettlementService.java",
        ),
    ]

    # Only set confirmed_facts, NOT evidence_items (to test the fix)
    state = ExplorationState(
        goal="帮我探索一下收入分成结算的完整链路，包括收入明细，分成计算，结算",
        task_mode="feature_exploration",
        candidate_next_actions=[
            "IncomeDisclosureFreezeServiceImpl",
            "IapIncomeDailyService",
            "SettlementEvent",
            "AppletSeasonService",
            "MiniAppLongVideoSearchIntervenePo",
            "BeiAnDownloadProcessorJob",
            "OpenOrderEvent",
            "should_CallIaaService_When_CreateAdjustmentAccrualWithValidId",
        ],
        evidence_items=[],
        confirmed_facts=strong_evidence,
    )

    prioritized, downgraded = _filter_candidates(state)

    # High-relevance candidates should be in prioritized
    assert "IncomeDisclosureFreezeServiceImpl" in prioritized, (
        f"IncomeDisclosureFreezeServiceImpl should be prioritized (shares 'income'/'disclosure'). "
        f"prioritized={prioritized}"
    )
    assert "IapIncomeDailyService" in prioritized, (
        f"IapIncomeDailyService should be prioritized (shares 'income'/'daily' with confirmed_symbols). "
        f"prioritized={prioritized}"
    )
    assert "SettlementEvent" in prioritized, (
        f"SettlementEvent should be prioritized (shares 'settlement'). "
        f"prioritized={prioritized}"
    )

    # Low-relevance candidates that FAIL the gate should be in downgraded
    # (they pass is_relevant_candidate=False so they're NOT in ranked, but ARE in raw)
    assert "OpenOrderEvent" in downgraded, (
        f"OpenOrderEvent should be downgraded (generic suffix only, no overlap). "
        f"downgraded={downgraded}"
    )
    assert "should_CallIaaService_When_CreateAdjustmentAccrualWithValidId" in downgraded, (
        f"should_ prefixed candidate should be downgraded. "
        f"downgraded={downgraded}"
    )
    assert "MiniAppLongVideoSearchIntervenePo" in downgraded, (
        f"MiniAppLongVideoSearchIntervenePo should be downgraded. "
        f"downgraded={downgraded}"
    )
    assert "BeiAnDownloadProcessorJob" in downgraded, (
        f"BeiAnDownloadProcessorJob should be downgraded (job is generic, no domain overlap). "
        f"downgraded={downgraded}"
    )


# ---- Test 11: _filter_candidates returns downgraded when no prioritized ----

def test_filter_candidates_returns_downgraded_when_no_prioritized() -> None:
    """When all candidates are irrelevant (no confirmed evidence), downgraded should still list them."""

    state = ExplorationState(
        goal="income settlement flow",
        task_mode="feature_exploration",
        candidate_next_actions=["UnrelatedA", "UnrelatedB"],
        evidence_items=[],
        confirmed_facts=[],
    )

    prioritized, downgraded = _filter_candidates(state)

    # No candidates pass -> prioritized is empty
    assert prioritized == [], f"prioritized should be empty, got {prioritized}"
    # But all candidates should still be in downgraded
    assert downgraded == ["UnrelatedA", "UnrelatedB"], (
        f"downgraded should contain all irrelevant candidates, got {downgraded}"
    )
