"""Continuation prompt generation for incomplete feature exploration."""

from __future__ import annotations

from app.agents.code_explorer.exploration_state import ExplorationState, StopDecision


def build_continuation_prompt(
    decision: StopDecision,
    state: ExplorationState,
) -> str:
    """Tell the model which evidence gaps still block the final report."""

    missing_fields = [
        f"- {stage}: {', '.join(fields)}"
        for stage, fields in decision.missing_stage_fields.items()
    ]
    return f"""\
## 继续代码链路探索

当前探索还不足以输出最终报告。

### 原因
{_items(decision.blocking_gaps)}

### 已经确认
{_items([fact.claim for fact in state.confirmed_facts[-12:]])}

### 以下阶段缺少必要证据
{_items(missing_fields, already_markdown=True)}

### 以下相邻阶段之间仍然断链
{_items(decision.missing_edges)}

### 不合格证据
{_items(decision.evidence_problems)}

### 被降级为 weak / noise 的证据
{_items(_downgraded_evidence(state))}

### 当前最高优先级断链点
{_items(decision.blocking_gaps[:3])}

### 候选下一步探索方向
{_items(state.candidate_next_actions[-12:])}

不要输出最终报告，请继续使用工具探索。
不要重复已经成功执行过的完全相同工具调用。
请优先补齐 strong trigger / input / transform / output / persistence / state_change，以及相邻阶段间的 strong direct_call / repository_read_write / shared_table / mapping_table / event / rpc / job / status_transition / external_boundary 证据。
这些只是候选方向，不是固定步骤。请根据工具结果自行判断下一步。
"""


def build_convergence_checkpoint_prompt(
    state: ExplorationState,
    used_tool_calls: int,
    max_tool_calls: int,
) -> str:
    """Ask the model to synthesize or focus before the safety fuse is reached."""

    return f"""\
## 探索收敛检查点

当前 feature_exploration 已使用工具调用：{used_tool_calls} / {max_tool_calls}。

请基于已有探索记录和内部 notebook 判断关键业务链路是否已经闭合，不要机械继续钻取局部实现细节。

- 如果关键业务阶段、相邻阶段连接、持久化和状态变化已经足够支撑主结论，请现在输出候选 Markdown《代码探索结果》。
- 如果仍有会改变主结论的关键断链，你仍可继续自主选择工具，但只优先补齐最高优先级断链。
- 避免重复已经成功执行过的工具调用。候选符号只是线索，不是固定步骤。

### 当前断链点
{_items((state.stop_decision.blocking_gaps if state.stop_decision else []) or state.unknowns or state.open_questions)}

### 候选下一步探索方向
{_items(state.candidate_next_actions[-12:])}
"""


def build_stagnation_report_prompt(
    state: ExplorationState,
    repeated_tool_calls: int,
) -> str:
    """Request a candidate report when repeated calls no longer add evidence."""

    return f"""\
## 探索停滞检查点

最近连续 {repeated_tool_calls} 次工具请求都重复了已经成功执行过的调用，没有增加新证据。

本轮不要调用工具。请基于已有探索记录和内部 notebook 输出一份候选 Markdown《代码探索结果》，按业务阶段组织，并区分已确认事实、合理推断和未知项。不要输出源码正文。

这只是候选报告：程序仍会检查关键业务阶段、相邻连接、持久化和状态变化是否闭合。如果仍有会改变主结论的关键断链，后续会恢复工具供你继续探索。

### 当前断链点
{_items((state.stop_decision.blocking_gaps if state.stop_decision else []) or state.unknowns or state.open_questions)}

### 候选下一步探索方向
{_items(state.candidate_next_actions[-12:])}
"""


def build_anti_stagnation_prompt(
    decision: StopDecision,
    state: ExplorationState,
    repeated_tool_calls: int,
    repeated_calls: list[str],
) -> str:
    """Redirect exploration when duplicate calls recur before evidence closes."""

    return f"""\
## 反停滞继续探索

最近连续 {repeated_tool_calls} 次工具请求重复了已经成功执行过的调用，但关键业务链路仍未闭合。

不要输出最终报告。工具仍然可用，请更换探索路径或提高查询精度。禁止重复相同工具和参数。

### 禁止继续重复的调用
{_items(repeated_calls)}

### 当前阻塞结束的断链点
{_items(decision.blocking_gaps)}

### 不合格证据
{_items(decision.evidence_problems)}

### 被降级为 weak / noise 的证据
{_items(_downgraded_evidence(state))}

### 以下阶段缺少必要证据
{_items([f"{stage}: {', '.join(fields)}" for stage, fields in decision.missing_stage_fields.items()])}

### 以下相邻阶段之间仍然断链
{_items(decision.missing_edges)}

### 候选下一步探索方向
{_items(state.candidate_next_actions[-12:])}

请选择不同的工具、不同的符号或更精确的参数补齐最高价值缺口。
"""


def _downgraded_evidence(state: ExplorationState) -> list[str]:
    return [
        f"[{item.strength}] {item.reason} Claim: {item.claim}"
        for item in state.evidence_items
        if item.strength != "strong"
    ][-20:]


def _items(items: list[str], *, already_markdown: bool = False) -> str:
    if not items:
        return "- 尚未识别，请继续探索。"
    if already_markdown:
        return "\n".join(items)
    return "\n".join(f"- {item}" for item in items)
