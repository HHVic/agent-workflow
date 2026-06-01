"""Prompts for the code exploration agent."""

from __future__ import annotations

from enum import Enum


class TaskMode(str, Enum):
    """Supported exploration strategies."""

    REPO_OVERVIEW = "repo_overview"
    FEATURE_EXPLORATION = "feature_exploration"
    SYMBOL_INVESTIGATION = "symbol_investigation"


_REPO_OVERVIEW_KEYWORDS = (
    "检查代码索引",
    "仓库",
    "模块结构",
    "关键入口",
    "概述",
    "进一步探索方向",
)
_SYMBOL_INVESTIGATION_KEYWORDS = (
    "符号",
    "调用链",
    "调用关系",
    "callers",
    "callees",
    "trace",
    "impact",
    "改动影响",
)

SYSTEM_PROMPT = """\
你是 CodeExplorerAgent，一个只读代码探索 Agent。

边界：
- 仅通过提供的 CodeGraph 工具分析代码，不修改仓库，不编造未验证的事实。
- 根据任务自主选择工具并持续探索，直到足以回答问题。
- 中间工具结果可以包含源码片段，但最终报告不得输出源码正文。
- 如果任务是仓库级概览，优先概述全局模块结构，不要过早深入单个业务链路。
- 当工具返回多个同名符号时，不要默认选择第一个，应先按路径和模块归类。
- 只有当任务明确要求分析某个符号、调用链或改动影响时，才对单个符号执行 callers / callees / trace / impact。

最终输出：
- 使用 Markdown，标题为《代码探索结果》。
- 说明结论、关键文件与符号、调用关系、影响范围、不确定项和建议的后续验证。
- 引用文件路径和符号名称；不要粘贴源码正文。
"""


def detect_task_mode(task: str) -> TaskMode:
    """Choose a deterministic exploration strategy from the user task."""

    normalized_task = task.strip()
    if not normalized_task:
        raise ValueError("task must not be empty")
    if any(keyword in normalized_task for keyword in _REPO_OVERVIEW_KEYWORDS):
        return TaskMode.REPO_OVERVIEW
    if any(keyword in normalized_task for keyword in _SYMBOL_INVESTIGATION_KEYWORDS):
        return TaskMode.SYMBOL_INVESTIGATION
    return TaskMode.FEATURE_EXPLORATION


def build_task_prompt(task: str, *, task_mode: TaskMode | None = None) -> str:
    """Build the user message sent to the exploration agent."""

    normalized_task = task.strip()
    if not normalized_task:
        raise ValueError("task must not be empty")
    actual_mode = task_mode or detect_task_mode(normalized_task)
    if actual_mode is TaskMode.REPO_OVERVIEW:
        return _build_repo_overview_prompt(normalized_task)
    if actual_mode is TaskMode.SYMBOL_INVESTIGATION:
        return _build_symbol_investigation_prompt(normalized_task)
    return _build_feature_exploration_prompt(normalized_task)


def build_repo_overview_rewrite_prompt(missing_sections: list[str]) -> str:
    """Build a no-tools correction request for an incomplete overview report."""

    missing = "、".join(missing_sections)
    return f"""\
## 仓库级概览报告重写

当前报告没有完成仓库级概览任务。请基于已有探索记录重写最终 Markdown《代码探索结果》，不要继续调用工具。

必须补齐缺失章节：{missing}。
必须包含这些章节：索引状态、顶层仓库结构、主要模块职责、关键启动入口、主要业务域、建议进一步探索方向、不确定项。
不要把单个局部链路作为全仓主线。
请区分已确认事实、合理推断和不确定项。未直接由工具结果确认的架构判断，应使用“可能”“推测”“需确认”等表述。
不要写缺少证据的结论，例如“体现领域驱动”或“符合高可用和弹性伸缩设计”。
如果 search_code 找到符号，但 get_symbol_detail 返回 not found，不要解释成索引未覆盖；更可能是工具参数或符号消歧问题。
最终报告不要输出源码正文。
"""


def _build_repo_overview_prompt(task: str) -> str:
    return f"""\
## 代码探索任务

任务模式：repo_overview
这是仓库级概览任务。

{task}

目标：
- 系统概述顶层仓库结构、主要模块职责、关键启动入口、分层结构、主要业务域和建议进一步探索方向。
- 先检查索引并列出顶层文件结构，再按路径归纳模块；搜索启动入口时，将多个同名 Application 等符号按路径和模块归类。
- 不要过早深入单个业务链路。
- 不要把偶然命中的局部符号当成全仓主线。
- 多个同名符号要先按路径和模块归类。
- 区分已确认事实、合理推断和不确定项。未直接由工具结果确认的架构判断，应使用“可能”“推测”“需确认”等表述。
- 不要写缺少证据的结论，例如“体现领域驱动”或“符合高可用和弹性伸缩设计”。
- 如果 search_code 找到符号，但 get_symbol_detail 返回 not found，不要解释成索引未覆盖；更可能是工具参数或符号消歧问题。

最终使用 Markdown 输出《代码探索结果》，并包含这些章节：索引状态、顶层仓库结构、主要模块职责、关键启动入口、主要业务域、建议进一步探索方向、不确定项。最终报告不要输出源码正文。
"""


def _build_feature_exploration_prompt(task: str) -> str:
    return f"""\
## 代码探索任务

任务模式：feature_exploration

{task}

请围绕目标功能自主使用工具探索相关模块、入口、业务流程和潜在影响，并在最后输出 Markdown《代码探索结果》。最终报告要区分已确认事实、合理推断和未知项，不要输出源码正文。
在输出最终报告前，确认关键业务阶段及相邻阶段连接。对每个阶段尽量说明触发入口、输入、核心转换、输出、持久化和状态变化；无法确认的内容明确列入不确定项，不要把待验证线索写成已确认事实。
"""


def _build_symbol_investigation_prompt(task: str) -> str:
    return f"""\
## 代码探索任务

任务模式：symbol_investigation

{task}

请围绕明确符号、调用链或改动影响自主使用工具探索。多个同名符号要先按路径和模块归类，再选择需要深入的对象。最后输出 Markdown《代码探索结果》，不要输出源码正文。
"""
