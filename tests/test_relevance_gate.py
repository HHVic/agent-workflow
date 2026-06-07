"""Tests for the relevance gate pure functions."""

from app.agents.code_explorer.relevance_gate import (
    ALWAYS_GENERIC,
    CONTEXTUAL_GENERIC,
    extract_relevance_terms,
    is_relevant_candidate,
    rank_relevant_candidates,
)


# ---- Test 1: Chinese goal extracts Chinese keywords ----

def test_extract_chinese_terms_from_cn_goal() -> None:
    terms = extract_relevance_terms("收入分成结算完整链路", [], [])

    # Should contain Chinese chunks
    assert terms
    assert any("\u4e00" <= ch <= "\u9fff" for ch in terms)
    # "收入", "分成", "结算" should appear in some chunk
    found_cn = "".join(t for t in terms if "\u4e00" <= t[0] <= "\u9fff")
    assert "收入" in found_cn or "分成" in found_cn or "结算" in found_cn


# ---- Test 2: camelCase symbol tokens ----

def test_extract_tokens_from_camel_case_symbol() -> None:
    terms = extract_relevance_terms("", ["IncomeDailyCalculationAndAssembleJob"], [])

    tokens = _tokenize_set("IncomeDailyCalculationAndAssembleJob")
    assert "income" in tokens
    assert "daily" in tokens
    assert "calculation" in tokens
    assert "assemble" in tokens


def test_extract_tokens_from_snake_case_symbol() -> None:
    terms = extract_relevance_terms("", ["some_long_snake_case_symbol"], [])

    tokens = _tokenize_set("some_long_snake_case_symbol")
    for part in ["some", "long", "snake", "case", "symbol"]:
        assert part in tokens


def test_extract_tokens_from_kebab_case_symbol() -> None:
    terms = extract_relevance_terms("", ["some-kebab-case-name"], [])

    tokens = _tokenize_set("some-kebab-case-name")
    for part in ["some", "kebab", "case", "name"]:
        assert part in tokens


# ---- Test 3: Path extracts domain terms ----

def test_extract_domain_terms_from_path() -> None:
    terms = extract_relevance_terms("", [], [
        "open-service/src/main/java/com/example/biz/income/disclosure/IncomeJob.java",
    ])

    assert "income" in terms
    assert "disclosure" in terms
    # Generic path parts should be filtered
    assert "src" not in terms
    assert "main" not in terms
    assert "java" not in terms
    assert "com" not in terms


# ---- Test 4: Related candidate returns True ----

def test_related_candidate_returns_true() -> None:
    goal = "收入分成结算完整链路"
    confirmed_symbols = ["IncomeDailyCalculationJob"]
    confirmed_paths = ["src/main/java/com/example/biz/income/disclosure/"]

    # This symbol shares "income" and "daily" with confirmed_symbols
    result = is_relevant_candidate(
        "IncomeHourlyCalculationAndAssembleJob",
        goal, confirmed_symbols, confirmed_paths,
    )
    assert result is True

    # This symbol shares "disclosure" with confirmed_paths
    result = is_relevant_candidate(
        "IncomeDisclosureDetailService",
        goal, confirmed_symbols, confirmed_paths,
    )
    assert result is True


# ---- Test 5: OpenOrderEvent with only generic words returns False ----

def test_generic_only_candidate_returns_false() -> None:
    goal = "收入分成结算完整链路"
    confirmed_symbols = ["IncomeDailyCalculationJob"]
    confirmed_paths = ["src/main/java/com/example/biz/income/"]

    # "OpenOrderEvent": open, order, event are all generic
    result = is_relevant_candidate(
        "OpenOrderEvent", goal, confirmed_symbols, confirmed_paths,
    )
    assert result is False


# ---- Test 6: AppletSeasonService without mainline overlap returns False ----

def test_unrelated_candidate_returns_false() -> None:
    goal = "收入分成结算完整链路"
    confirmed_symbols = ["IncomeDailyCalculationJob"]
    confirmed_paths = ["src/main/java/com/example/biz/income/"]

    # "AppletSeasonService": applet is generic (app), season has no overlap
    result = is_relevant_candidate(
        "AppletSeasonService", goal, confirmed_symbols, confirmed_paths,
    )
    assert result is False


# ---- Test 7: IncomeDisclosureJob returns True when goal/terms contain income/disclosure ----

def test_income_disclosure_job_returns_true() -> None:
    goal = "收入分成结算完整链路"
    confirmed_symbols = ["IncomeDailyCalculationJob"]
    confirmed_paths = ["src/main/java/com/example/biz/income/disclosure/"]

    result = is_relevant_candidate(
        "IncomeDisclosureJob", goal, confirmed_symbols, confirmed_paths,
    )
    assert result is True

    # Also via goal terms (收入 maps to tokens that may not match directly,
    # but confirmed_symbols has "income")
    result2 = is_relevant_candidate(
        "IncomeDisclosureJob", "explore revenue", ["IncomeDailyJob"], [],
    )
    # "income" from candidate matches "income" from confirmed_symbols
    assert result2 is True


# ---- Test 8: SettlementEvent returns True when confirmed terms include settlement ----

def test_settlement_event_returns_true() -> None:
    result = is_relevant_candidate(
        "SettlementEvent",
        "收入分成结算完整链路",
        ["SettlementCreateJob"],
        ["src/main/java/com/example/biz/settlement/"],
    )
    assert result is True


# ---- Test 9: Test symbol defaults to False ----

def test_test_symbol_defaults_false() -> None:
    result = is_relevant_candidate(
        "should_CallIaaService_When_Xxx",
        "income flow",
        ["IncomeJob"],
        ["src/main/java/com/example/income/"],
    )
    assert result is False

    result2 = is_relevant_candidate(
        "when_ServiceReturnsNull_Then_Handle",
        "income flow",
        ["IncomeJob"],
        ["src/main/java/com/example/income/"],
    )
    assert result2 is False


# ---- Test 10: connected_symbols candidate with low overlap returns True ----

def test_connected_symbols_passes_with_low_overlap() -> None:
    result = is_relevant_candidate(
        "SomeRandomUnrelatedSymbol",
        "income settlement flow",
        ["IncomeDailyJob"],
        ["src/main/java/com/example/income/"],
        connected_symbols=["SomeRandomUnrelatedSymbol"],
    )
    assert result is True


# ---- Test 11: rank_relevant_candidates filters irrelevant candidates ----

def test_rank_filters_irrelevant_candidates() -> None:
    candidates = [
        "IncomeDailyCalculationJob",
        "OpenOrderEvent",
        "AppletSeasonService",
        "IncomeDisclosureDetailService",
        "GenericStatusManager",
        "SettlementCreateJob",
    ]
    goal = "收入分成结算完整链路"
    confirmed_symbols = ["IncomeDailyCalculationJob"]
    confirmed_paths = ["src/main/java/com/example/biz/income/disclosure/"]

    ranked = rank_relevant_candidates(
        candidates, goal, confirmed_symbols, confirmed_paths, limit=10,
    )

    # IncomeDailyCalculationJob should be in result (highest overlap)
    assert "IncomeDailyCalculationJob" in ranked
    # OpenOrderEvent should NOT be in result
    assert "OpenOrderEvent" not in ranked
    # AppletSeasonService should NOT be in result
    assert "AppletSeasonService" not in ranked
    # GenericStatusManager should NOT be in result
    assert "GenericStatusManager" not in ranked


# ---- Test 12: rank_relevant_candidates puts connected candidate first ----

def test_rank_places_connected_first() -> None:
    candidates = [
        "IncomeDailyCalculationJob",
        "ConnectedButWeakSymbol",
        "SettlementCreateJob",
        "IrrelevantAppletService",
    ]
    goal = "收入分成结算完整链路"
    confirmed_symbols = ["IncomeDailyCalculationJob", "SettlementCreateJob"]
    confirmed_paths = ["src/main/java/com/example/biz/income/"]

    ranked = rank_relevant_candidates(
        candidates, goal, confirmed_symbols, confirmed_paths,
        connected_symbols=["ConnectedButWeakSymbol"],
        limit=10,
    )

    # ConnectedButWeakSymbol should be first (score +100)
    assert ranked[0] == "ConnectedButWeakSymbol"
    # IncomeDailyCalculationJob should be second (high overlap with confirmed_symbols, no connected)
    assert ranked[1] == "IncomeDailyCalculationJob"
    # SettlementCreateJob should be third (overlap with confirmed_symbols)
    assert ranked[2] == "SettlementCreateJob"
    # IrrelevantAppletService should NOT be in results (all ALWAYS_GENERIC)
    assert "IrrelevantAppletService" not in ranked


# ---- Additional: extract_relevance_terms filters generics ----

def test_generic_words_filtered() -> None:
    terms = extract_relevance_terms("", ["OrderEventServiceJobController"], [])
    # All parts are generic
    assert "event" not in terms
    assert "order" not in terms
    assert "service" not in terms
    assert "job" not in terms


# ---- Additional: is_relevant_candidate suffix-only rejection ----

def test_suffix_only_match_rejected() -> None:
    goal = "income settlement flow"
    confirmed_symbols: list[str] = []
    confirmed_paths: list[str] = []

    # "StatusEvent" — status and event are both generic
    result = is_relevant_candidate(
        "StatusEvent", goal, confirmed_symbols, confirmed_paths,
    )
    assert result is False


# ---- Task 8.5: ALWAYS_GENERIC vs CONTEXTUAL_GENERIC ----

def test_always_generic_excludes_project_specific() -> None:
    """Project-specific words should NOT be in ALWAYS_GENERIC."""
    assert "bilibili" not in ALWAYS_GENERIC
    assert "platform" not in ALWAYS_GENERIC
    assert "open" not in ALWAYS_GENERIC
    assert "mini" not in ALWAYS_GENERIC
    assert "java" not in ALWAYS_GENERIC
    # These should still be in ALWAYS_GENERIC
    assert "service" in ALWAYS_GENERIC
    assert "impl" in ALWAYS_GENERIC
    assert "controller" in ALWAYS_GENERIC


def test_contextual_generic_includes_domain_words() -> None:
    """order/event/status/job/document/video should be CONTEXTUAL_GENERIC."""
    assert "order" in CONTEXTUAL_GENERIC
    assert "event" in CONTEXTUAL_GENERIC
    assert "status" in CONTEXTUAL_GENERIC
    assert "job" in CONTEXTUAL_GENERIC
    assert "document" in CONTEXTUAL_GENERIC
    assert "video" in CONTEXTUAL_GENERIC
    assert "applet" in CONTEXTUAL_GENERIC
    assert "season" in CONTEXTUAL_GENERIC
    assert "query" in CONTEXTUAL_GENERIC
    assert "condition" in CONTEXTUAL_GENERIC
    assert "process" in CONTEXTUAL_GENERIC
    assert "handle" in CONTEXTUAL_GENERIC
    assert "create" in CONTEXTUAL_GENERIC
    assert "update" in CONTEXTUAL_GENERIC
    assert "processor" in CONTEXTUAL_GENERIC
    assert "handler" in CONTEXTUAL_GENERIC
    assert "task" in CONTEXTUAL_GENERIC
    assert "batch" in CONTEXTUAL_GENERIC
    # These should NOT be in CONTEXTUAL_GENERIC
    assert "service" not in CONTEXTUAL_GENERIC
    assert "bo" not in CONTEXTUAL_GENERIC


def test_contextual_generic_in_goal_contributes() -> None:
    """When contextual generic appears in goal (Chinese), its chars are in terms."""
    terms = extract_relevance_terms("订单退款链路", ["IncomeJob"], [])
    # Chinese chars from the goal are included (e.g. "订", "单")
    assert "订" in terms
    assert "单" in terms
    # English "order" won't be in terms from Chinese goal,
    # but _goal_supports_goalword handles the cross-language mapping


def test_contextual_generic_in_goal_not_filtered_from_candidate() -> None:
    """OpenOrderEvent with goal="订单退款链路" should pass (order is in goal)."""
    result = is_relevant_candidate(
        "OpenOrderEvent",
        "订单退款链路",
        ["IncomeJob"],
        [],
    )
    assert result is True


def test_open_order_event_order_goal() -> None:
    """goal=订单退款链路, OpenOrderEvent → True (order in goal)."""
    result = is_relevant_candidate(
        "OpenOrderEvent",
        "订单退款链路",
        ["RefundService"],
        [],
    )
    assert result is True


def test_payment_event_consumer_event_goal() -> None:
    """goal=事件消费链路, PaymentEventConsumer → True (event in goal)."""
    result = is_relevant_candidate(
        "PaymentEventConsumer",
        "事件消费链路",
        [],
        [],
    )
    assert result is True


def test_document_review_service_document_goal() -> None:
    """goal=文档审核链路, DocumentReviewService → True (document in goal)."""
    result = is_relevant_candidate(
        "DocumentReviewService",
        "文档审核链路",
        [],
        [],
    )
    assert result is True


def test_long_video_search_intervene_video_goal() -> None:
    """goal=视频搜索干预链路, LongVideoSearchIntervenePo → True (video in goal)."""
    result = is_relevant_candidate(
        "LongVideoSearchIntervenePo",
        "视频搜索干预链路",
        [],
        [],
    )
    assert result is True


def test_settlement_job_task_goal() -> None:
    """goal=定时任务调度链路, SettlementJob → True (job/task in goal)."""
    result = is_relevant_candidate(
        "SettlementJob",
        "定时任务调度链路",
        [],
        [],
    )
    assert result is True


def test_open_order_event_without_order_goal_is_false() -> None:
    """goal不含order/event时, OpenOrderEvent应为False, 除非connected_symbols."""
    result = is_relevant_candidate(
        "OpenOrderEvent",
        "用户登录链路",
        ["LoginService"],
        [],
    )
    assert result is False

    # connected_symbols still passes
    result2 = is_relevant_candidate(
        "OpenOrderEvent",
        "用户登录链路",
        ["LoginService"],
        [],
        connected_symbols=["OpenOrderEvent"],
    )
    assert result2 is True


def test_income_settlement_quick_check() -> None:
    """Goal=收入分成结算, confirm prioritized/downgraded behavior."""
    goal = "收入分成结算"
    confirmed = ["IncomeDisclosureJob", "SettlementService"]
    candidates = [
        "IncomeDisclosureFreezeServiceImpl",
        "IapIncomeDailyService",
        "SettlementEvent",
        "AppletSeasonService",
        "MiniAppLongVideoSearchIntervenePo",
        "BeiAnDownloadProcessorJob",
        "OpenOrderEvent",
    ]
    ranked = rank_relevant_candidates(candidates, goal, confirmed, [])

    assert "IncomeDisclosureFreezeServiceImpl" in ranked
    assert "IapIncomeDailyService" in ranked
    assert "SettlementEvent" in ranked
    assert "AppletSeasonService" not in ranked
    assert "MiniAppLongVideoSearchIntervenePo" not in ranked
    assert "BeiAnDownloadProcessorJob" not in ranked
    assert "OpenOrderEvent" not in ranked


def test_contextual_generic_only_in_candidate_no_goal_support_is_false() -> None:
    """Candidate with contextual generic but goal doesn't contain it → False."""
    result = is_relevant_candidate(
        "OrderManagement",
        "用户登录链路",
        ["LoginService"],
        [],
    )
    assert result is False  # order + management are contextual, not in goal


def test_contextual_generic_in_candidate_with_goal_support_passes() -> None:
    """Candidate with contextual generic AND goal contains it → passes."""
    result = is_relevant_candidate(
        "OrderProcessing",
        "订单处理链路",
        [],
        [],
    )
    assert result is True  # "order" and "processing" are in goal


def test_always_generic_goal_word_cannot_release_candidate() -> None:
    """ALWAYS_GENERIC words stay irrelevant even when the goal contains them."""
    result = is_relevant_candidate(
        "WidgetService",
        "service",
        [],
        [],
    )

    assert result is False


def test_project_context_words_are_not_globally_filtered() -> None:
    """Project and language context words may contribute when the goal uses them."""
    terms = extract_relevance_terms("open platform java gateway", [], [])

    assert {"open", "platform", "java", "gateway"} <= terms
    assert is_relevant_candidate("OpenGateway", "open gateway", [], []) is True


def test_goal_terms_keep_contextual_but_drop_always_generic_words() -> None:
    terms = extract_relevance_terms("order service event controller", [], [])

    assert "order" in terms
    assert "event" in terms
    assert "service" not in terms
    assert "controller" not in terms


def test_contextual_suffix_alone_cannot_release_candidate() -> None:
    """A contextual suffix still needs a substantive domain token."""
    assert is_relevant_candidate("Event", "event consumption flow", [], []) is False
    assert is_relevant_candidate("SettlementJob", "定时任务调度链路", [], []) is True


def test_chinese_contextual_word_requires_full_concept_match() -> None:
    """A shared Chinese character is not enough to activate a contextual word."""
    result = is_relevant_candidate(
        "DocumentReviewService",
        "文章审核链路",
        [],
        [],
    )

    assert result is False


def test_income_settlement_full_paths_do_not_release_open_order_event() -> None:
    """Repository-level path words such as open must not become one-token evidence."""
    result = is_relevant_candidate(
        "OpenOrderEvent",
        "帮我探索一下收入分成结算的完整链路，包括收入明细，分成计算，结算",
        ["IncomeDisclosureJob", "SettlementService"],
        [
            "miniapp-open-platform-bilispec/open-service/src/main/java/"
            "com/bilibili/miniapp/open/service/biz/income/disclosure/"
            "IncomeDisclosureJob.java",
            "miniapp-open-platform-bilispec/open-service/src/main/java/"
            "com/bilibili/miniapp/open/service/biz/settlement/impl/"
            "SettlementService.java",
        ],
    )

    assert result is False


def test_income_settlement_exact_quick_check() -> None:
    goal = "帮我探索一下收入分成结算的完整链路，包括收入明细，分成计算，结算"
    confirmed_symbols = ["IncomeDisclosureJob", "SettlementService"]
    confirmed_paths = [
        "miniapp-open-platform-bilispec/open-service/src/main/java/"
        "com/bilibili/miniapp/open/service/biz/income/disclosure/"
        "IncomeDisclosureJob.java",
        "miniapp-open-platform-bilispec/open-service/src/main/java/"
        "com/bilibili/miniapp/open/service/biz/settlement/impl/"
        "SettlementService.java",
    ]
    candidates = [
        "IncomeDisclosureFreezeServiceImpl",
        "IapIncomeDailyService",
        "SettlementEvent",
        "AppletSeasonService",
        "MiniAppLongVideoSearchIntervenePo",
        "BeiAnDownloadProcessorJob",
        "OpenOrderEvent",
        "should_CallIaaService_When_CreateAdjustmentAccrualWithValidId",
    ]

    prioritized = rank_relevant_candidates(
        candidates,
        goal,
        confirmed_symbols,
        confirmed_paths,
    )
    downgraded = [candidate for candidate in candidates if candidate not in prioritized]

    assert prioritized == [
        "IncomeDisclosureFreezeServiceImpl",
        "IapIncomeDailyService",
        "SettlementEvent",
    ]
    assert downgraded == [
        "AppletSeasonService",
        "MiniAppLongVideoSearchIntervenePo",
        "BeiAnDownloadProcessorJob",
        "OpenOrderEvent",
        "should_CallIaaService_When_CreateAdjustmentAccrualWithValidId",
    ]


# ---- Helper ----

def _tokenize_set(name: str) -> set[str]:
    from app.agents.code_explorer.relevance_gate import _tokenize
    return set(_tokenize(name))


# ---- Task 8 specific tests ----

def test_applet_season_service_filtered_with_income_disclosure_confirmed() -> None:
    """AppletSeasonService with goal=收入分成结算, confirmed_symbols=[IncomeDisclosureJob, SettlementService] -> False."""
    result = is_relevant_candidate(
        "AppletSeasonService",
        "收入分成结算",
        ["IncomeDisclosureJob", "SettlementService"],
        [],
    )
    assert result is False


def test_beian_download_processor_job_filtered_with_income_disclosure_only() -> None:
    """BeiAnDownloadProcessorJob with confirmed_symbols=[IncomeDisclosureJob] -> False.
    Cannot pass just because 'job' suffix matches."""
    result = is_relevant_candidate(
        "BeiAnDownloadProcessorJob",
        "收入分成结算",
        ["IncomeDisclosureJob"],
        [],
    )
    assert result is False


def test_income_disclosure_freeze_service_returns_true() -> None:
    """IncomeDisclosureFreezeServiceImpl shares 'income'/'disclosure' with confirmed_symbols."""
    result = is_relevant_candidate(
        "IncomeDisclosureFreezeServiceImpl",
        "收入分成结算",
        ["IncomeDisclosureJob"],
        [],
    )
    assert result is True


def test_iap_income_daily_service_returns_true() -> None:
    """IapIncomeDailyService shares 'income'/'daily' with confirmed_symbols."""
    result = is_relevant_candidate(
        "IapIncomeDailyService",
        "收入分成结算",
        ["IncomeDisclosureJob"],
        [],
    )
    assert result is True


def test_settlement_event_returns_true() -> None:
    """SettlementEvent passes because 'settlement' is a non-generic word."""
    result = is_relevant_candidate(
        "SettlementEvent",
        "收入分成结算",
        ["IncomeDisclosureJob", "SettlementService"],
        [],
    )
    assert result is True


def test_open_order_event_returns_false() -> None:
    """OpenOrderEvent returns False because 'order' and 'event' are both generic."""
    result = is_relevant_candidate(
        "OpenOrderEvent",
        "收入分成结算",
        ["IncomeDisclosureJob"],
        [],
    )
    assert result is False


def test_connected_open_order_event_returns_true() -> None:
    """OpenOrderEvent in connected_symbols returns True even though it's generic-only."""
    result = is_relevant_candidate(
        "OpenOrderEvent",
        "收入分成结算",
        ["IncomeDisclosureJob"],
        [],
        connected_symbols=["OpenOrderEvent"],
    )
    assert result is True


def test_rank_relevant_candidates_excludes_generic_only() -> None:
    """rank_relevant_candidates should not include AppletSeasonService or BeiAnDownloadProcessorJob."""
    candidates = [
        "IncomeDisclosureFreezeServiceImpl",
        "IapIncomeDailyService",
        "SettlementEvent",
        "AppletSeasonService",
        "MiniAppLongVideoSearchIntervenePo",
        "BeiAnDownloadProcessorJob",
        "OpenOrderEvent",
        "should_CallIaaService_When_CreateAdjustmentAccrualWithValidId",
    ]
    ranked = rank_relevant_candidates(
        candidates,
        "收入分成结算",
        ["IncomeDisclosureJob", "SettlementService"],
        [],
    )

    # Should be in prioritized
    assert "IncomeDisclosureFreezeServiceImpl" in ranked
    assert "IapIncomeDailyService" in ranked
    assert "SettlementEvent" in ranked

    # Should NOT be in prioritized
    assert "AppletSeasonService" not in ranked
    assert "BeiAnDownloadProcessorJob" not in ranked
    assert "OpenOrderEvent" not in ranked
    assert "MiniAppLongVideoSearchIntervenePo" not in ranked
    assert "should_CallIaaService_When_CreateAdjustmentAccrualWithValidId" not in ranked
