from app.agents.code_explorer.agent import guard_final_report


def test_final_report_guard_removes_overclaim_when_quality_report_has_gaps() -> None:
    report = "# 探索结果\n\n所有阶段均已确认，主链完整闭合，无断链。"
    quality_report = """# Evidence Quality Report

## Missing Stage Fields
- WidgetStage: persistence

## Evidence Problems
- WidgetStage persistence only has a weak import lead
"""

    guarded = guard_final_report(report, quality_report)

    assert "阶段性结论" in guarded
    assert "所有阶段均已确认" not in guarded
    assert "完整闭合" not in guarded
    assert "无断链" not in guarded


def test_final_report_guard_keeps_report_when_quality_report_has_no_gaps() -> None:
    report = "# 探索结果\n\n当前主链已确认。"
    quality_report = """# Evidence Quality Report

## Missing Stage Fields
- None

## Evidence Problems
- None

## Missing Edges
- None

## Blocking Gaps
- None
"""

    assert guard_final_report(report, quality_report) == report
