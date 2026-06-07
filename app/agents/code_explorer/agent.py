"""Hand-written CodeExplorerAgent tool-calling loop."""

from __future__ import annotations

import asyncio
import json
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import monotonic, perf_counter
from typing import Any

from openai.types.chat import ChatCompletionMessage

from app.agents.code_explorer.continuation import (
    _filter_candidates,
    build_anti_stagnation_prompt,
    build_continuation_prompt,
    build_convergence_checkpoint_prompt,
    build_stagnation_report_prompt,
)
from app.agents.code_explorer.exploration_state import (
    BusinessStage,
    ExplorationState,
    StopDecision,
    can_satisfy_stage_edge,
    is_strong_evidence,
    update_state_from_candidate_report,
    update_state_from_tool_logs,
)
from app.agents.code_explorer.memory import ConversationMemory
from app.agents.code_explorer.notebook import update_notebook
from app.agents.code_explorer.prompts import (
    SYSTEM_PROMPT,
    TaskMode,
    build_repo_overview_rewrite_prompt,
    build_task_prompt,
    detect_task_mode,
)
from app.agents.code_explorer.stop_judge import (
    judge_stop,
    render_evidence_quality_report,
)
from app.agents.code_explorer.tools import (
    EXPLORER_TOOL_SPECS,
    ExplorerTools,
    ToolArgumentValidationError,
)
from app.core.config import Settings
from app.core.llm_client import ChatCompletionClient
from app.mcp.codegraph_mcp_client import CodeGraphMCPClient
from app.storage.run_store import RunArtifacts, RunStore

_FENCED_BLOCK = re.compile(r"```.*?```", flags=re.DOTALL)
_REPO_OVERVIEW_REQUIRED_SECTIONS = (
    "索引状态",
    "顶层仓库结构",
    "主要模块职责",
    "关键启动入口",
    "主要业务域",
    "建议进一步探索方向",
    "不确定项",
)
_MAX_CONSECUTIVE_REPEATED_TOOL_CALLS = 5
_OVERCLAIM_PHRASES = re.compile(
    r"(?:所有阶段均已确认|所有阶段均有明确证据|完整闭合|无断链|全部确认)",
    re.IGNORECASE,
)
_QUALITY_GAP_SECTIONS = (
    "Stages without strong evidence",
    "Edges without strong evidence",
    "Blocking Gaps",
    "Evidence Problems",
    "Missing Stage Fields",
    "Missing Edges",
)


class ExplorationLimitError(RuntimeError):
    """Raised when a configured safety fuse stops exploration."""


class RepoOverviewReportError(RuntimeError):
    """Raised when a rewritten repository overview is still incomplete."""


@dataclass(frozen=True, slots=True)
class ExplorationResult:
    """Successful run output paths and report."""

    run_id: str
    report: str
    artifacts: RunArtifacts


class CodeExplorerAgent:
    """Runs a ReAct-style loop using LLM-selected CodeGraph tools."""

    def __init__(
        self,
        settings: Settings,
        llm_client: ChatCompletionClient,
        *,
        memory: ConversationMemory | None = None,
        run_store: RunStore | None = None,
    ) -> None:
        self._settings = settings
        self._llm_client = llm_client
        self._memory = memory or ConversationMemory()
        self._run_store = run_store or RunStore(settings.runs_dir)
        self.last_artifacts: RunArtifacts | None = None

    async def run(self, task: str) -> ExplorationResult:
        """Create one run, explore with tools, and persist the Markdown report."""

        artifacts = self._run_store.create_run()
        self.last_artifacts = artifacts
        task_mode = detect_task_mode(task)
        task_prompt = build_task_prompt(task, task_mode=task_mode)
        state = ExplorationState(goal=task.strip(), task_mode=task_mode.value)
        artifacts.write_text("input.md", task.strip())
        artifacts.write_text("system_prompt.md", SYSTEM_PROMPT)
        artifacts.write_text("task_prompt.md", task_prompt)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]
        self._save_messages(artifacts, messages)
        update_notebook(state, messages, [])
        self._save_exploration_artifacts(artifacts, state)
        started_at = monotonic()

        try:
            async with CodeGraphMCPClient(
                self._settings.codegraph_mcp_command,
                self._settings.codegraph_mcp_args,
                project_path=self._settings.codegraph_project_path,
            ) as codegraph:
                tools = ExplorerTools(codegraph, await codegraph.list_tools())
                result = await self._run_loop(
                    artifacts,
                    messages,
                    tools,
                    started_at,
                    task_mode,
                    state,
                )
        except BaseException as exc:
            self._save_failure_artifacts(artifacts, messages, state, exc)
            raise

        artifacts.write_text("report.md", result)
        self._save_messages(artifacts, messages)
        return ExplorationResult(
            run_id=artifacts.run_id,
            report=result,
            artifacts=artifacts,
        )

    async def _run_loop(
        self,
        artifacts: RunArtifacts,
        messages: list[dict[str, Any]],
        tools: ExplorerTools,
        started_at: float,
        task_mode: TaskMode,
        state: ExplorationState,
    ) -> str:
        tool_count = 0
        stop_decision_count = 0
        convergence_checkpoint_sent = False
        consecutive_repeated_tool_calls = 0
        stagnation_checkpoint_count = 0
        force_no_tools_report = False
        tool_logs: list[dict[str, Any]] = []
        llm_tools = tools.as_llm_tools()

        while True:
            remaining = self._remaining_seconds(started_at)
            response = await asyncio.wait_for(
                self._llm_client.create_chat_completion(
                    self._memory.prepare(messages),
                    None if force_no_tools_report else llm_tools,
                ),
                timeout=remaining,
            )
            message = response.choices[0].message
            messages.append(_assistant_message_to_dict(message))
            self._save_messages(artifacts, messages)
            if force_no_tools_report:
                force_no_tools_report = False
                if message.tool_calls:
                    raise ExplorationLimitError(
                        "stagnation report request returned unexpected tool calls"
                    )

            if not message.tool_calls:
                report = _sanitize_report(message.content or "")
                if (
                    task_mode is TaskMode.REPO_OVERVIEW
                    and _needs_repo_overview_rewrite(report)
                ):
                    return await self._rewrite_repo_overview_report(
                        artifacts,
                        messages,
                        started_at,
                        report,
                    )
                if task_mode is TaskMode.FEATURE_EXPLORATION:
                    update_state_from_candidate_report(state, report)
                    decision = judge_stop(state, report)
                    state.stop_decision = decision
                    update_notebook(state, messages, tool_logs)
                    stop_decision_count += 1
                    artifacts.write_text(
                        f"stop_decision_{stop_decision_count}.md",
                        decision.to_markdown(),
                    )
                    self._save_exploration_artifacts(artifacts, state)
                    if not decision.can_stop and (
                        state.continuation_count < self._settings.max_continuations
                    ):
                        state.continuation_count += 1
                        continuation_prompt = build_continuation_prompt(
                            decision,
                            state,
                        )
                        decision.continuation_prompt = continuation_prompt
                        artifacts.write_text(
                            f"continuation_{state.continuation_count}.md",
                            continuation_prompt,
                        )
                        messages.append(
                            {"role": "user", "content": continuation_prompt}
                        )
                        self._save_messages(artifacts, messages)
                        self._save_exploration_artifacts(artifacts, state)
                        continue
                    if not decision.can_stop:
                        incomplete_reason = (
                            "CODE_EXPLORER_MAX_CONTINUATIONS safety fuse reached "
                            "before Evidence Gate passed"
                        )
                        if self._settings.fail_on_incomplete:
                            raise ExplorationLimitError(
                                "feature exploration remained incomplete after "
                                "CODE_EXPLORER_MAX_CONTINUATIONS safety fuse reached"
                            )
                        return _build_incomplete_report(
                            state,
                            incomplete_reason,
                            tool_count,
                        )
                quality_report = artifacts.path(
                    "evidence_quality_report.md"
                ).read_text(encoding="utf-8")
                return guard_final_report(report, quality_report)

            for tool_call in message.tool_calls:
                if tool_count >= self._settings.max_tool_calls:
                    raise ExplorationLimitError(
                        "CODE_EXPLORER_MAX_TOOL_CALLS safety fuse reached"
                    )
                remaining = self._remaining_seconds(started_at)
                tool_count += 1
                result, log_entry = await self._execute_tool(
                    artifacts,
                    tools,
                    task_mode,
                    tool_count,
                    tool_call.id,
                    tool_call.function.name,
                    tool_call.function.arguments,
                    remaining,
                )
                tool_logs.append(log_entry)
                if _is_repeated_tool_call_result(result):
                    consecutive_repeated_tool_calls += 1
                else:
                    consecutive_repeated_tool_calls = 0
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            result,
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )
                self._save_messages(artifacts, messages)
                update_state_from_tool_logs(state, [log_entry])
                update_notebook(state, messages, [log_entry])
                self._save_exploration_artifacts(artifacts, state)
            if (
                task_mode is TaskMode.FEATURE_EXPLORATION
                and consecutive_repeated_tool_calls
                >= _MAX_CONSECUTIVE_REPEATED_TOOL_CALLS
            ):
                stagnation_checkpoint_count += 1
                preliminary_decision = judge_stop(state, "")
                state.stop_decision = preliminary_decision
                update_notebook(state, messages, tool_logs)
                self._save_exploration_artifacts(artifacts, state)
                if preliminary_decision.can_stop:
                    stagnation_prompt = build_stagnation_report_prompt(
                        state,
                        consecutive_repeated_tool_calls,
                    )
                    force_no_tools_report = True
                else:
                    stagnation_prompt = build_anti_stagnation_prompt(
                        preliminary_decision,
                        state,
                        consecutive_repeated_tool_calls,
                        _recent_repeated_calls(tool_logs),
                    )
                    force_no_tools_report = False
                artifacts.write_text(
                    f"stagnation_checkpoint_{stagnation_checkpoint_count}.md",
                    stagnation_prompt,
                )
                messages.append({"role": "user", "content": stagnation_prompt})
                self._save_messages(artifacts, messages)
                consecutive_repeated_tool_calls = 0
                continue
            if (
                task_mode is TaskMode.FEATURE_EXPLORATION
                and not convergence_checkpoint_sent
                and tool_count >= _convergence_checkpoint(self._settings.max_tool_calls)
            ):
                convergence_prompt = build_convergence_checkpoint_prompt(
                    state,
                    tool_count,
                    self._settings.max_tool_calls,
                )
                artifacts.write_text("convergence_checkpoint.md", convergence_prompt)
                messages.append({"role": "user", "content": convergence_prompt})
                self._save_messages(artifacts, messages)
                convergence_checkpoint_sent = True

    async def _rewrite_repo_overview_report(
        self,
        artifacts: RunArtifacts,
        messages: list[dict[str, Any]],
        started_at: float,
        report: str,
    ) -> str:
        """Rewrite one incomplete overview without allowing further tool calls."""

        rewrite_prompt = build_repo_overview_rewrite_prompt(
            _missing_repo_overview_sections(report)
        )
        messages.append({"role": "user", "content": rewrite_prompt})
        self._save_messages(artifacts, messages)
        response = await asyncio.wait_for(
            self._llm_client.create_chat_completion(
                self._memory.prepare(messages),
                None,
            ),
            timeout=self._remaining_seconds(started_at),
        )
        message = response.choices[0].message
        messages.append(_assistant_message_to_dict(message))
        self._save_messages(artifacts, messages)
        if message.tool_calls:
            raise RepoOverviewReportError(
                "repo_overview rewrite returned unexpected tool calls"
            )

        rewritten_report = _sanitize_report(message.content or "")
        missing_sections = _missing_repo_overview_sections(rewritten_report)
        if missing_sections:
            joined = ", ".join(missing_sections)
            raise RepoOverviewReportError(
                f"repo_overview rewrite is missing required sections: {joined}"
            )
        return rewritten_report

    async def _execute_tool(
        self,
        artifacts: RunArtifacts,
        tools: ExplorerTools,
        task_mode: TaskMode,
        sequence: int,
        tool_call_id: str,
        name: str,
        raw_arguments: str,
        remaining_seconds: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started_at = perf_counter()
        started_at_iso = datetime.now(timezone.utc).isoformat()
        log_entry: dict[str, Any] = {
            "timestamp": started_at_iso,
            "started_at": started_at_iso,
            "index": sequence,
            "sequence": sequence,
            "tool_call_id": tool_call_id,
            "tool": name,
            "tool_name": name,
            "mapped_mcp_tool_name": _mapped_mcp_tool_name(name),
            "raw_arguments": raw_arguments,
        }
        try:
            arguments = _decode_arguments(raw_arguments)
            log_entry["arguments"] = arguments
            result = await asyncio.wait_for(
                tools.call(name, arguments, task_mode.value),
                timeout=remaining_seconds,
            )
            wrapped_result = {"ok": True, "result": result}
            log_entry["status"] = "success"
            log_entry["result_preview"] = _preview(result)
        except ToolArgumentValidationError as exc:
            wrapped_result = {
                "ok": False,
                "validation_error": str(exc),
            }
            log_entry["mapped_mcp_tool_name"] = exc.mapped_mcp_tool_name
            log_entry["status"] = "validation_error"
            log_entry["result_preview"] = str(exc)
        except Exception as exc:
            wrapped_result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            log_entry["status"] = "error"
            log_entry["error"] = wrapped_result["error"]
            log_entry["result_preview"] = wrapped_result["error"]
            artifacts.append_error(
                f"tool call {sequence} ({name}) failed: {type(exc).__name__}: {exc}"
            )
        log_entry["duration_ms"] = round((perf_counter() - started_at) * 1000, 3)
        log_entry["response"] = wrapped_result
        artifacts.append_jsonl("tool_calls.jsonl", log_entry)
        return wrapped_result, log_entry

    def _remaining_seconds(self, started_at: float) -> float:
        remaining = self._settings.max_seconds - (monotonic() - started_at)
        if remaining <= 0:
            raise ExplorationLimitError(
                "CODE_EXPLORER_MAX_SECONDS safety fuse reached"
            )
        return remaining

    @staticmethod
    def _save_messages(
        artifacts: RunArtifacts,
        messages: list[dict[str, Any]],
    ) -> None:
        artifacts.write_text("messages.md", _render_messages(messages))

    @staticmethod
    def _save_exploration_artifacts(
        artifacts: RunArtifacts,
        state: ExplorationState,
    ) -> None:
        state_payload = state.to_dict()
        artifacts.write_json("exploration_state.json", state_payload)
        artifacts.write_text("notebook.md", state.notebook_markdown or "")
        evidence_lines = [
            json.dumps(asdict(evidence), ensure_ascii=False, default=str)
            for evidence in state.evidence_items
        ]
        artifacts.write_text(
            "evidence_items.jsonl",
            "\n".join(evidence_lines) + ("\n" if evidence_lines else ""),
        )
        artifacts.write_json(
            "stage_graph.json",
            {
                "stages": state_payload["stages"],
                "edges": state_payload["edges"],
            },
        )
        artifacts.write_text(
            "evidence_quality_report.md",
            render_evidence_quality_report(state, state.stop_decision),
        )

    @classmethod
    def _save_failure_artifacts(
        cls,
        artifacts: RunArtifacts,
        messages: list[dict[str, Any]],
        state: ExplorationState,
        failure: BaseException,
    ) -> None:
        """Persist a truthful non-empty report when a run cannot finish."""

        failure_traceback = traceback.format_exc()
        try:
            update_notebook(state, messages, [])
            cls._save_exploration_artifacts(artifacts, state)
            if not artifacts.path("report.md").read_text(encoding="utf-8").strip():
                artifacts.write_text(
                    "report.md",
                    _build_recovery_report(state, failure),
                )
            artifacts.append_error(failure_traceback)
            cls._save_messages(artifacts, messages)
        except Exception:
            artifacts.append_error(
                "failed to persist recovery artifacts:\n" + traceback.format_exc()
            )


def _assistant_message_to_dict(message: ChatCompletionMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _decode_arguments(raw_arguments: str) -> dict[str, Any]:
    value = json.loads(raw_arguments or "{}")
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")
    return value


def _mapped_mcp_tool_name(explorer_name: str) -> str | None:
    for spec in EXPLORER_TOOL_SPECS:
        if spec.explorer_name == explorer_name:
            return spec.mcp_name
    return None


def _preview(value: object, limit: int = 1_000) -> str:
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    if len(serialized) <= limit:
        return serialized
    return f"{serialized[:limit]}... [truncated]"


def _convergence_checkpoint(max_tool_calls: int) -> int:
    return max(1, max_tool_calls * 3 // 4)


def _is_repeated_tool_call_result(result: dict[str, Any]) -> bool:
    return "REPEATED_TOOL_CALL_WARNING" in json.dumps(
        result,
        ensure_ascii=False,
        default=str,
    )


def _recent_repeated_calls(tool_logs: list[dict[str, Any]]) -> list[str]:
    repeated: list[str] = []
    for log in reversed(tool_logs):
        if "REPEATED_TOOL_CALL_WARNING" not in str(log.get("result_preview", "")):
            if repeated:
                break
            continue
        repeated.append(
            f"{log.get('tool_name', 'tool')}("
            f"{json.dumps(log.get('arguments', {}), ensure_ascii=False, sort_keys=True)})"
        )
    return list(dict.fromkeys(reversed(repeated)))


def _sanitize_report(report: str) -> str:
    normalized = report.strip()
    if not normalized:
        raise ValueError("LLM returned an empty final report")
    normalized = _FENCED_BLOCK.sub("[源码正文已省略]", normalized)
    if "代码探索结果" not in normalized[:120]:
        normalized = f"# 代码探索结果\n\n{normalized}"
    return f"{normalized}\n"


def _build_recovery_report(
    state: ExplorationState,
    failure: BaseException,
) -> str:
    """Render a local summary without pretending an interrupted run completed."""

    return _build_local_state_report(
        state,
        title=(
            "# 代码探索结果（本地恢复摘要）"
            if (state.stop_decision or judge_stop(state, "")).can_stop
            else "# 代码探索结果（未完成）"
        ),
        introduction=(
            "> 最终 LLM 汇总未能完成。以下内容根据已保存的本地探索状态生成，"
            "不是完整最终结论。"
        ),
        termination_reason=f"{type(failure).__name__}: {failure}",
        tool_count=None,
    )


def _build_incomplete_report(
    state: ExplorationState,
    termination_reason: str,
    tool_count: int,
) -> str:
    """Render a usable phase report while preserving CAN_STOP=no."""

    return _build_local_state_report(
        state,
        title="# 代码探索结果（阶段性）",
        introduction=(
            "> 本报告为当前静态探索下的阶段性结论，Evidence Gate 未完全通过。"
            "以下结论只代表已确认 strong evidence 能支撑的部分，未闭合链路已单独列出。"
        ),
        termination_reason=termination_reason,
        tool_count=tool_count,
    )


def _build_local_state_report(
    state: ExplorationState,
    *,
    title: str,
    introduction: str,
    termination_reason: str,
    tool_count: int | None,
) -> str:
    """Render one truthful local report from persisted exploration state."""

    decision = state.stop_decision or judge_stop(state, "")
    gate_status = "passed" if decision.can_stop else "not passed"
    prioritized, _ = _filter_candidates(state)
    lines = [
        title,
        "",
        introduction,
        "",
        "## 运行状态",
        f"- 终止原因：{termination_reason}",
        f"- Evidence Gate: {gate_status}",
        f"- continuation count: {state.continuation_count}",
        f"- tool call count: {tool_count if tool_count is not None else 'unknown'}",
        "",
        "## 已确认主线事实",
        *_markdown_items(
            _deduplicate_strings(
                [
                    fact.claim
                    for fact in state.confirmed_facts
                    if is_strong_evidence(fact)
                ]
            )
        ),
        "",
        "## 已识别业务阶段",
        *_markdown_items(
            [f"{stage.name} ({_stage_report_status(stage)})" for stage in state.stages]
        ),
        "",
        "## 已确认阶段连接",
        *_markdown_items(
            [
                f"{edge.from_stage} -> {edge.to_stage}: {edge.connection_type}"
                for edge in state.edges
                if edge.confidence == "confirmed"
                and any(can_satisfy_stage_edge(item) for item in edge.evidence)
            ]
        ),
        "",
        "## 未闭合的关键断点",
        "### Blocking Gaps",
        *_markdown_items(decision.blocking_gaps),
        "",
        "### Missing Stage Fields",
        *_markdown_items(
            [
                f"{stage}: {', '.join(fields)}"
                for stage, fields in decision.missing_stage_fields.items()
            ]
        ),
        "",
        "### Missing Edges",
        *_markdown_items(decision.missing_edges),
        "",
        "### Evidence Problems",
        *_markdown_items(decision.evidence_problems),
        "",
        "## 不确定项",
        *_markdown_items(state.unknowns + state.open_questions),
        "",
        "## 下一步最有价值探索方向",
        *_markdown_items(prioritized),
        "",
        "## 说明",
        "- 本报告不包含源码正文。",
        "- StopDecision 保持 CAN_STOP=no，阶段性报告不代表 Evidence Gate 已通过。"
        if not decision.can_stop
        else "- Evidence Gate 已通过。",
        "- 完整证据、阶段图和消息记录请查看同目录本地产物。",
    ]
    return "\n".join(lines) + "\n"


def _markdown_items(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- None"]


def _stage_report_status(stage: BusinessStage) -> str:
    if stage.optional:
        return "optional"
    if stage.confidence in {"confirmed", "inferred", "supporting", "unknown"}:
        return stage.confidence
    if any(is_strong_evidence(evidence) for evidence in stage.evidence):
        return "supporting"
    return "unknown"


def _deduplicate_strings(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def guard_final_report(report: str, evidence_quality_report: str) -> str:
    """Downgrade overclaims when the local evidence audit still has gaps."""

    if not _quality_report_has_gaps(evidence_quality_report):
        return report
    guarded = _OVERCLAIM_PHRASES.sub("仍有待确认项", report)
    notice = (
        "> 本报告为当前静态探索下的阶段性结论，仍存在证据缺口。"
        "请结合“不确定项”和本地 evidence_quality_report.md 阅读。"
    )
    if notice in guarded:
        return guarded
    return f"{notice}\n\n{guarded}"


def _quality_report_has_gaps(evidence_quality_report: str) -> bool:
    if re.search(r"^- Can stop:\s*no\s*$", evidence_quality_report, re.MULTILINE):
        return True
    for section in _QUALITY_GAP_SECTIONS:
        match = re.search(
            rf"^## {re.escape(section)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
            evidence_quality_report,
            re.MULTILINE | re.DOTALL,
        )
        if not match:
            continue
        lines = [line.strip() for line in match.group("body").splitlines()]
        if any(line.startswith("- ") and line != "- None" for line in lines):
            return True
    return False


def _needs_repo_overview_rewrite(report: str) -> bool:
    """Return whether an overview report is missing any required section."""

    return bool(_missing_repo_overview_sections(report))


def _missing_repo_overview_sections(report: str) -> list[str]:
    return [
        section for section in _REPO_OVERVIEW_REQUIRED_SECTIONS if section not in report
    ]


def _render_messages(messages: list[dict[str, Any]]) -> str:
    sections = ["# Messages"]
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role", "unknown"))
        serialized = json.dumps(message, ensure_ascii=False, indent=2, default=str)
        sections.extend(
            (
                "",
                f"## {index}. {role}",
                "",
                "````json",
                serialized,
                "````",
            )
        )
    return "\n".join(sections) + "\n"
