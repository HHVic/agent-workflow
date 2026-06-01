from app.agents.code_explorer.prompts import (
    SYSTEM_PROMPT,
    TaskMode,
    build_repo_overview_rewrite_prompt,
    build_task_prompt,
    detect_task_mode,
)


def test_system_prompt_sets_report_boundary() -> None:
    assert "Markdown" in SYSTEM_PROMPT
    assert "源码正文" in SYSTEM_PROMPT
    assert "只读" in SYSTEM_PROMPT
    assert "仓库级概览" in SYSTEM_PROMPT
    assert "多个同名符号" in SYSTEM_PROMPT
    assert "callers / callees / trace / impact" in SYSTEM_PROMPT


def test_build_task_prompt_includes_user_task() -> None:
    prompt = build_task_prompt("追踪订单结算完成后的对账调用链")

    assert "追踪订单结算完成后的对账调用链" in prompt
    assert "代码探索任务" in prompt
    assert "源码正文" in prompt


def test_detect_task_mode_for_repo_overview_keywords() -> None:
    task = "请检查代码索引，并概述这个仓库的主要模块结构、关键入口和建议进一步探索的方向。"

    assert detect_task_mode(task) is TaskMode.REPO_OVERVIEW


def test_detect_task_mode_for_symbol_investigation() -> None:
    assert (
        detect_task_mode("请分析 WidgetService.calculate 的调用链和改动影响")
        is TaskMode.SYMBOL_INVESTIGATION
    )


def test_detect_task_mode_defaults_to_feature_exploration() -> None:
    assert (
        detect_task_mode("我想增加一个收入明细计算完成之后对账的逻辑")
        is TaskMode.FEATURE_EXPLORATION
    )


def test_feature_exploration_prompt_requests_stage_closure() -> None:
    prompt = build_task_prompt(
        "探索订单结算流程",
        task_mode=TaskMode.FEATURE_EXPLORATION,
    )

    assert "关键业务阶段及相邻阶段连接" in prompt
    assert "触发入口、输入、核心转换、输出、持久化和状态变化" in prompt
    assert "不要把待验证线索写成已确认事实" in prompt
    assert "区分已确认事实、合理推断和未知项" in prompt


def test_repo_overview_prompt_prioritizes_repository_structure() -> None:
    prompt = build_task_prompt(
        "请概述仓库结构",
        task_mode=TaskMode.REPO_OVERVIEW,
    )

    assert "这是仓库级概览任务" in prompt
    assert "顶层仓库结构" in prompt
    assert "主要模块职责" in prompt
    assert "关键启动入口" in prompt
    assert "主要业务域" in prompt
    assert "不要过早深入单个业务链路" in prompt
    assert "不要把偶然命中的局部符号当成全仓主线" in prompt
    assert "多个同名符号" in prompt
    assert "区分已确认事实、合理推断和不确定项" in prompt
    assert "可能" in prompt
    assert "体现领域驱动" in prompt
    assert "工具参数或符号消歧问题" in prompt


def test_repo_overview_rewrite_prompt_is_generic() -> None:
    prompt = build_repo_overview_rewrite_prompt(["顶层仓库结构", "主要模块职责"])

    assert "当前报告没有完成仓库级概览任务" in prompt
    assert "请基于已有探索记录重写" in prompt
    assert "不要继续调用工具" in prompt
    assert "顶层仓库结构、主要模块职责" in prompt
    assert "不要把单个局部链路作为全仓主线" in prompt
    assert "身份识别" not in prompt
