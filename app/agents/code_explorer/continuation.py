"""Continuation prompt generation for incomplete feature exploration."""

from __future__ import annotations

from app.agents.code_explorer.exploration_state import (
    Evidence,
    ExplorationState,
    StopDecision,
    is_strong_evidence,
)
from app.agents.code_explorer.relevance_gate import rank_relevant_candidates

# --- Helper: extract relevance-gate inputs from state ---


def _extract_confirmed_symbols(state: ExplorationState) -> list[str]:
    """Extract symbol names from strong evidence items.

    Reads from: state.evidence_items, state.confirmed_facts,
    stage.evidence, edge.evidence.
    """
    symbols: list[str] = []
    seen: set[str] = set()

    def _process(items: list[Evidence]) -> None:
        for item in items:
            if not is_strong_evidence(item):
                continue
            sym = item.symbol or item.claim
            if not sym:
                continue
            if sym.startswith("test_") or sym.startswith("should_"):
                continue
            if sym not in seen:
                seen.add(sym)
                symbols.append(sym)

    _process(state.evidence_items)
    _process(state.confirmed_facts)
    for stage in state.stages:
        _process(stage.evidence)
    for edge in state.edges:
        _process(edge.evidence)

    return symbols


def _extract_confirmed_paths(state: ExplorationState) -> list[str]:
    """Extract file paths from strong evidence items.

    Reads from: state.evidence_items, state.confirmed_facts,
    stage.evidence, edge.evidence.
    """
    paths: list[str] = []
    seen: set[str] = set()

    def _process(items: list[Evidence]) -> None:
        for item in items:
            if not is_strong_evidence(item):
                continue
            fp = item.file_path
            if not fp or fp in seen:
                continue
            seen.add(fp)
            paths.append(fp)

    _process(state.evidence_items)
    _process(state.confirmed_facts)
    for stage in state.stages:
        _process(stage.evidence)
    for edge in state.edges:
        _process(edge.evidence)

    return paths


def _extract_connected_symbols(state: ExplorationState) -> list[str]:
    """Extract source/target symbols from strong call_edge evidence.

    Reads from: state.evidence_items, state.confirmed_facts,
    stage.evidence, edge.evidence.

    Also handles "A -> B" or "A \u2192 B" format in evidence.symbol.
    """
    symbols: list[str] = []
    seen: set[str] = set()

    def _add_sym(s: str) -> None:
        if s and s not in seen:
            seen.add(s)
            symbols.append(s)

    def _process(item: Evidence) -> None:
        if not is_strong_evidence(item):
            return
        if item.source_type != "call_edge":
            return
        # If evidence.symbol is "A -> B" format, extract both
        if item.symbol and " -> " in item.symbol:
            for part in item.symbol.split(" -> ", 1):
                _add_sym(part.strip().split()[0] if part.strip() else "")
            return
        if item.symbol and " \u2192 " in item.symbol:
            for part in item.symbol.split(" \u2192 ", 1):
                _add_sym(part.strip().split()[0] if part.strip() else "")
            return
        # Try to extract source and target from the claim
        for src in ("source:", "Source:", "caller:", "from:"):
            if src in item.claim:
                parts = item.claim.split(src, 1)
                if len(parts) > 1:
                    candidate = parts[1].strip().split()[0]
                    _add_sym(candidate)
        # Also check summary for " -> " or " → " pattern
        for sep in (" -> ", " \u2192 "):
            if sep in item.summary:
                parts = item.summary.split(sep, 1)
                for part in parts:
                    tok = part.strip().split()[0] if part.strip() else ""
                    _add_sym(tok)

    for item in state.evidence_items:
        _process(item)
    for item in state.confirmed_facts:
        _process(item)
    for stage in state.stages:
        for item in stage.evidence:
            _process(item)
    for edge in state.edges:
        for item in edge.evidence:
            _process(item)

    return symbols


def _filter_candidates(
    state: ExplorationState,
    limit: int = 10,
    downgraded_limit: int = 5,
) -> tuple[list[str], list[str]]:
    """Filter candidates via relevance gate.

    Returns (prioritized, downgraded) lists.
    """
    raw = list(state.candidate_next_actions)
    if not raw:
        return [], []

    goal = state.goal
    confirmed_symbols = _extract_confirmed_symbols(state)
    confirmed_paths = _extract_confirmed_paths(state)
    connected_symbols = _extract_connected_symbols(state)

    ranked = rank_relevant_candidates(
        raw,
        goal=goal,
        confirmed_symbols=confirmed_symbols,
        confirmed_paths=confirmed_paths,
        connected_symbols=connected_symbols or None,
        limit=limit,
    )

    # Downgraded = raw candidates not in ranked
    ranked_set = set(ranked)
    downgraded: list[str] = []
    for c in raw:
        if c not in ranked_set:
            downgraded.append(c)
            if len(downgraded) >= downgraded_limit:
                break

    return ranked, downgraded


def _build_candidate_section(
    candidates: list[str],
    heading: str,
) -> str:
    """Build a candidate section with optional downgraded note.

    The heading may already include a markdown level (e.g. "### ...").
    If so we use it as-is; otherwise we prefix with "### ".
    """
    if heading.startswith("#"):
        lines = [heading]
    else:
        lines = [f"### {heading}"]
    if not candidates:
        lines.append("- 当前无可优先探索的候选方向。")
        return "\n".join(lines)
    for c in candidates:
        lines.append(f"- {c}")
    return "\n".join(lines)


def _build_full_candidate_section(
    state: ExplorationState,
    prioritized_heading: str,
    downgraded_heading: str | None = None,
) -> str:
    """Build a full candidate section with prioritized and optional downgraded."""
    prioritized, downgraded = _filter_candidates(state)
    result = _build_candidate_section(prioritized, prioritized_heading)
    if downgraded_heading and downgraded:
        # Ensure consistent heading format
        if not downgraded_heading.startswith("#"):
            downgraded_heading = f"### {downgraded_heading}"
        result += f"\n\n{downgraded_heading}\n以下候选与当前主线相关性不足，暂不优先探索。\n"
        for d in downgraded:
            result += f"- {d}\n"
    return result


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

{_build_full_candidate_section(
        state,
        prioritized_heading="### 优先补齐方向",
        downgraded_heading="### 已降级候选",
    )}

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

{_build_full_candidate_section(
        state,
        prioritized_heading="### 优先探索方向",
        downgraded_heading="### 已降级候选",
    )}

- 如果没有高相关候选，应基于已确认 strong evidence 输出阶段性报告或说明阻塞，而不是继续盲搜。
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

{_build_full_candidate_section(
        state,
        prioritized_heading="### 候选下一步探索方向",
    )}
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

{_build_full_candidate_section(
        state,
        prioritized_heading="### 优先探索方向",
        downgraded_heading="### 已降级候选",
    )}

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
