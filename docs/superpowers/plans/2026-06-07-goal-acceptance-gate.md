# Goal Acceptance Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a goal-driven acceptance checklist gate that applies to every CodeExplorerAgent task mode before a final report can be returned.

**Architecture:** Add a new `acceptance.py` module for checklist models, normalization, prompt builders, LLM review parsing, and local evidence evaluation. Keep the existing `StopJudge` focused on feature business-chain closure, and fold its result into the new acceptance decision only for `feature_exploration`.

**Tech Stack:** Python 3.12, dataclasses, pytest, OpenAI-compatible Chat Completions, existing CodeGraph MCP tool logs and `ExplorationState` evidence.

---

## File Map

- Create: `app/agents/code_explorer/acceptance.py`
  - Owns `AcceptanceCriterion`, `AcceptanceChecklist`, `AcceptanceDecision`, prompt builders, JSON parsing, append-only checklist normalization, evidence matching, and Markdown rendering.
- Create: `tests/test_acceptance.py`
  - Pure unit tests for checklist normalization, append-only revision rules, parsing, mode-specific evidence policies, and feature StopJudge folding.
- Modify: `app/storage/run_store.py`
  - Adds static and dynamic acceptance artifacts.
- Modify: `tests/test_run_store.py`
  - Verifies new acceptance artifact paths are allowed and created.
- Modify: `app/agents/code_explorer/agent.py`
  - Generates the initial checklist, requests candidate acceptance review before returning, writes acceptance artifacts, continues exploration on unsatisfied blocking items, and includes unmet acceptance criteria in incomplete reports.
- Modify: `tests/test_agent.py`
  - Updates fake LLM flows for checklist generation and review, then tests the new return gate.
- Modify: `tests/test_agent_incomplete_report.py`
  - Verifies incomplete reports include acceptance gate failures.

## Task 1: Acceptance Models And Checklist Normalization

**Files:**
- Create: `app/agents/code_explorer/acceptance.py`
- Test: `tests/test_acceptance.py`

- [ ] **Step 1: Write failing tests for model normalization**

Add `tests/test_acceptance.py` with:

```python
from app.agents.code_explorer.acceptance import (
    AcceptanceChecklistError,
    normalize_checklist_payload,
)


def test_normalize_checklist_repairs_ids_and_requires_blocking() -> None:
    checklist = normalize_checklist_payload(
        {
            "criteria": [
                {
                    "description": "Identify project entry points",
                    "blocking": True,
                    "evidence_policy": "repo_overview",
                    "required_evidence": ["entry", "main"],
                },
                {
                    "id": "nice-to-have",
                    "description": "Mention optional risks",
                    "blocking": False,
                    "evidence_policy": "repo_overview",
                    "required_evidence": ["risk"],
                },
            ]
        },
        goal="summarize architecture",
        task_mode="repo_overview",
    )

    assert checklist.goal == "summarize architecture"
    assert checklist.task_mode == "repo_overview"
    assert checklist.revision == 1
    assert checklist.criteria[0].id.startswith("criterion-")
    assert checklist.criteria[0].blocking is True
    assert checklist.criteria[0].required_evidence == ["entry", "main"]
    assert checklist.criteria[1].id == "nice-to-have"


def test_normalize_checklist_rejects_empty_or_non_blocking_payload() -> None:
    for payload in (
        {"criteria": []},
        {"criteria": [{"description": "Only optional", "blocking": False}]},
    ):
        try:
            normalize_checklist_payload(
                payload,
                goal="summarize architecture",
                task_mode="repo_overview",
            )
        except AcceptanceChecklistError as exc:
            assert "blocking" in str(exc) or "criteria" in str(exc)
        else:
            raise AssertionError("invalid checklist was accepted")


def test_checklist_revision_is_append_only() -> None:
    original = normalize_checklist_payload(
        {
            "criteria": [
                {
                    "id": "entry-points",
                    "description": "Identify project entry points",
                    "blocking": True,
                    "evidence_policy": "repo_overview",
                    "required_evidence": ["entry"],
                }
            ]
        },
        goal="summarize architecture",
        task_mode="repo_overview",
    )

    revised = normalize_checklist_payload(
        {
            "criteria": [
                {
                    "id": "entry-points",
                    "description": "Identify project entry points",
                    "blocking": True,
                    "evidence_policy": "repo_overview",
                    "required_evidence": ["entry"],
                },
                {
                    "id": "module-map",
                    "description": "Map module responsibilities",
                    "blocking": True,
                    "evidence_policy": "repo_overview",
                    "required_evidence": ["module"],
                },
            ]
        },
        goal="summarize architecture",
        task_mode="repo_overview",
        previous=original,
    )

    assert revised.revision == 2
    assert [criterion.id for criterion in revised.criteria] == [
        "entry-points",
        "module-map",
    ]

    for bad_payload in (
        {"criteria": []},
        {
            "criteria": [
                {
                    "id": "entry-points",
                    "description": "Identify project entry points",
                    "blocking": False,
                    "evidence_policy": "repo_overview",
                    "required_evidence": ["entry"],
                }
            ]
        },
    ):
        try:
            normalize_checklist_payload(
                bad_payload,
                goal="summarize architecture",
                task_mode="repo_overview",
                previous=original,
            )
        except AcceptanceChecklistError:
            pass
        else:
            raise AssertionError("invalid checklist revision was accepted")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_acceptance.py -q`

Expected: FAIL during import with `ModuleNotFoundError` for `app.agents.code_explorer.acceptance`.

- [ ] **Step 3: Implement acceptance models and normalization**

Create `app/agents/code_explorer/acceptance.py`:

```python
"""Goal-level acceptance checklist and local completion gate."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


class AcceptanceChecklistError(ValueError):
    """Raised when an acceptance checklist cannot be safely used."""


@dataclass(slots=True)
class EvidenceRef:
    """Pointer from an acceptance item to local tool-backed evidence."""

    evidence_id: str | None = None
    tool_call_index: int | None = None
    file_path: str | None = None
    symbol: str | None = None
    summary: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceRef":
        return cls(
            evidence_id=_optional_string(value.get("evidence_id")),
            tool_call_index=_optional_int(value.get("tool_call_index")),
            file_path=_optional_string(value.get("file_path")),
            symbol=_optional_string(value.get("symbol")),
            summary=str(value.get("summary") or ""),
        )


@dataclass(slots=True)
class AcceptanceCriterion:
    """One user-goal acceptance criterion."""

    id: str
    description: str
    task_mode: str
    blocking: bool
    evidence_policy: str
    required_evidence: list[str] = field(default_factory=list)
    satisfied: bool = False
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, task_mode: str) -> "AcceptanceCriterion":
        description = str(value.get("description") or "").strip()
        if not description:
            raise AcceptanceChecklistError("acceptance criterion description is required")
        criterion_id = str(value.get("id") or _criterion_id(description)).strip()
        blocking = bool(value.get("blocking", True))
        evidence_policy = str(value.get("evidence_policy") or task_mode).strip()
        required_evidence = _string_list(value.get("required_evidence"))
        evidence_refs = [
            EvidenceRef.from_dict(item)
            for item in value.get("evidence_refs", [])
            if isinstance(item, dict)
        ]
        return cls(
            id=criterion_id,
            description=description,
            task_mode=task_mode,
            blocking=blocking,
            evidence_policy=evidence_policy,
            required_evidence=required_evidence,
            satisfied=bool(value.get("satisfied", False)),
            evidence_refs=evidence_refs,
            notes=_string_list(value.get("notes")),
        )


@dataclass(slots=True)
class AcceptanceChecklist:
    """Append-only checklist derived from the user's goal."""

    goal: str
    task_mode: str
    criteria: list[AcceptanceCriterion]
    revision: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Acceptance Checklist",
            "",
            f"- Goal: {self.goal}",
            f"- Task mode: {self.task_mode}",
            f"- Revision: {self.revision}",
            "",
            "## Criteria",
        ]
        for criterion in self.criteria:
            blocking = "blocking" if criterion.blocking else "non-blocking"
            required = ", ".join(criterion.required_evidence) or "tool-backed evidence"
            lines.extend(
                (
                    f"### {criterion.id}",
                    f"- Description: {criterion.description}",
                    f"- Type: {blocking}",
                    f"- Evidence policy: {criterion.evidence_policy}",
                    f"- Required evidence: {required}",
                    "",
                )
            )
        return "\n".join(lines).rstrip() + "\n"


@dataclass(slots=True)
class AcceptanceDecision:
    """Local decision made before a candidate report may be returned."""

    can_return: bool
    reason: str
    satisfied: list[str] = field(default_factory=list)
    unsatisfied_blocking: list[str] = field(default_factory=list)
    weak_or_missing_evidence: list[str] = field(default_factory=list)
    continuation_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"CAN_RETURN: {'yes' if self.can_return else 'no'}",
            "",
            "## Reason",
            self.reason,
            "",
            "## Satisfied Criteria",
            *_markdown_items(self.satisfied),
            "",
            "## Unsatisfied Blocking Criteria",
            *_markdown_items(self.unsatisfied_blocking),
            "",
            "## Weak Or Missing Evidence",
            *_markdown_items(self.weak_or_missing_evidence),
        ]
        return "\n".join(lines) + "\n"


def normalize_checklist_payload(
    payload: dict[str, Any],
    *,
    goal: str,
    task_mode: str,
    previous: AcceptanceChecklist | None = None,
) -> AcceptanceChecklist:
    """Normalize one LLM checklist payload and enforce append-only revisions."""

    raw_criteria = payload.get("criteria")
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise AcceptanceChecklistError("acceptance checklist requires non-empty criteria")

    criteria: list[AcceptanceCriterion] = []
    seen: set[str] = set()
    for item in raw_criteria:
        if not isinstance(item, dict):
            raise AcceptanceChecklistError("acceptance criterion must be an object")
        criterion = AcceptanceCriterion.from_dict(item, task_mode=task_mode)
        if criterion.id in seen:
            raise AcceptanceChecklistError(f"duplicate acceptance criterion id: {criterion.id}")
        seen.add(criterion.id)
        criteria.append(criterion)

    if not any(criterion.blocking for criterion in criteria):
        raise AcceptanceChecklistError("acceptance checklist requires at least one blocking criterion")

    if previous is not None:
        _validate_append_only(previous, criteria)
        revision = previous.revision + 1
    else:
        revision = 1

    return AcceptanceChecklist(
        goal=goal.strip(),
        task_mode=task_mode,
        criteria=criteria,
        revision=revision,
    )


def _validate_append_only(
    previous: AcceptanceChecklist,
    criteria: list[AcceptanceCriterion],
) -> None:
    incoming_by_id = {criterion.id: criterion for criterion in criteria}
    for old in previous.criteria:
        new = incoming_by_id.get(old.id)
        if new is None:
            raise AcceptanceChecklistError(f"cannot remove acceptance criterion: {old.id}")
        if old.blocking and not new.blocking:
            raise AcceptanceChecklistError(f"cannot downgrade blocking criterion: {old.id}")
        if old.description != new.description:
            raise AcceptanceChecklistError(f"cannot rewrite criterion description: {old.id}")


def _criterion_id(description: str) -> str:
    digest = hashlib.sha1(description.encode("utf-8")).hexdigest()[:8]
    return f"criterion-{digest}"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [str(value).strip()] if str(value).strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _markdown_items(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- None"]
```

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run: `pytest tests/test_acceptance.py -q`

Expected: PASS for the three normalization tests.

- [ ] **Step 5: Commit Task 1**

```bash
git add app/agents/code_explorer/acceptance.py tests/test_acceptance.py
git commit -m "feat: add acceptance checklist models"
```

## Task 2: Local Evidence Policy Evaluation

**Files:**
- Modify: `app/agents/code_explorer/acceptance.py`
- Modify: `tests/test_acceptance.py`

- [ ] **Step 1: Add failing evidence policy tests**

Append to `tests/test_acceptance.py`:

```python
from app.agents.code_explorer.acceptance import (
    AcceptanceDecision,
    evaluate_acceptance,
)
from app.agents.code_explorer.exploration_state import Evidence, ExplorationState, StopDecision


def _checklist(task_mode: str, required: list[str]) -> object:
    return normalize_checklist_payload(
        {
            "criteria": [
                {
                    "id": "coverage",
                    "description": "Cover the requested goal",
                    "blocking": True,
                    "evidence_policy": task_mode,
                    "required_evidence": required,
                }
            ]
        },
        goal="goal",
        task_mode=task_mode,
    )


def _state(task_mode: str, evidence: list[Evidence]) -> ExplorationState:
    return ExplorationState(
        goal="goal",
        task_mode=task_mode,
        evidence_items=evidence,
        confirmed_facts=[item for item in evidence if item.strength == "strong"],
    )


def _evidence(
    claim: str,
    *,
    source_type: str = "tool_result",
    strength: str = "weak",
    file_path: str | None = "app/main.py",
    symbol: str | None = "main",
) -> Evidence:
    return Evidence(
        id=claim.lower().replace(" ", "-"),
        claim=claim,
        source_type=source_type,
        summary=claim,
        confidence="confirmed",
        file_path=file_path,
        symbol=symbol,
        strength=strength,  # type: ignore[arg-type]
    )


def test_repo_overview_requires_tool_backed_evidence() -> None:
    checklist = _checklist("repo_overview", ["entry"])
    report_only = _state(
        "repo_overview",
        [_evidence("entry point described in report", source_type="report")],
    )
    with_tool = _state(
        "repo_overview",
        [_evidence("entry point app/main.py found", source_type="file_path")],
    )

    assert not evaluate_acceptance(checklist, report_only, "summary", "repo_overview").can_return
    assert evaluate_acceptance(checklist, with_tool, "summary", "repo_overview").can_return


def test_symbol_investigation_requires_symbol_specific_evidence() -> None:
    checklist = _checklist("symbol_investigation", ["WidgetService"])
    unrelated = _state(
        "symbol_investigation",
        [_evidence("Found symbol OtherService", symbol="OtherService")],
    )
    related = _state(
        "symbol_investigation",
        [_evidence("Found symbol WidgetService", symbol="WidgetService")],
    )

    assert not evaluate_acceptance(checklist, unrelated, "summary", "symbol_investigation").can_return
    assert evaluate_acceptance(checklist, related, "summary", "symbol_investigation").can_return


def test_feature_acceptance_folds_stop_judge_decision() -> None:
    checklist = _checklist("feature_exploration", ["WidgetService"])
    state = _state(
        "feature_exploration",
        [_evidence("WidgetService calls WidgetRepository", strength="strong")],
    )

    blocked = evaluate_acceptance(
        checklist,
        state,
        "summary",
        "feature_exploration",
        feature_decision=StopDecision(can_stop=False, reason="chain open"),
    )
    allowed = evaluate_acceptance(
        checklist,
        state,
        "summary",
        "feature_exploration",
        feature_decision=StopDecision(can_stop=True, reason="chain closed"),
    )

    assert not blocked.can_return
    assert "Feature Evidence Gate" in blocked.unsatisfied_blocking[0]
    assert allowed.can_return
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_acceptance.py -q`

Expected: FAIL with `ImportError` for `evaluate_acceptance`.

- [ ] **Step 3: Implement local evaluation**

Add these imports near the top of `app/agents/code_explorer/acceptance.py`:

```python
from app.agents.code_explorer.exploration_state import Evidence, ExplorationState, StopDecision
```

Add below `normalize_checklist_payload`:

```python
def evaluate_acceptance(
    checklist: AcceptanceChecklist,
    state: ExplorationState,
    candidate_report: str,
    task_mode: str,
    *,
    review: dict[str, Any] | None = None,
    feature_decision: StopDecision | None = None,
) -> AcceptanceDecision:
    """Evaluate whether the candidate report satisfies the goal checklist."""

    del candidate_report
    review = review or {}
    satisfied: list[str] = []
    unsatisfied: list[str] = []
    weak_or_missing: list[str] = []

    for criterion in checklist.criteria:
        matched = _matching_evidence(criterion, state, task_mode)
        review_satisfied = criterion.id in set(_string_list(review.get("satisfied_criteria")))
        criterion.satisfied = bool(matched) and (review_satisfied or not review)
        criterion.evidence_refs = [_evidence_ref(item) for item in matched]
        if criterion.satisfied:
            satisfied.append(criterion.id)
            continue
        reason = f"{criterion.id}: {criterion.description}"
        if criterion.blocking:
            unsatisfied.append(reason)
        weak_or_missing.append(reason)

    if task_mode == "feature_exploration" and feature_decision is not None:
        if not feature_decision.can_stop:
            unsatisfied.append(f"Feature Evidence Gate: {feature_decision.reason}")
            weak_or_missing.extend(feature_decision.blocking_gaps)

    can_return = not unsatisfied
    return AcceptanceDecision(
        can_return=can_return,
        reason=(
            "All blocking acceptance criteria are supported by tool evidence."
            if can_return
            else "Blocking acceptance criteria still lack acceptable tool evidence."
        ),
        satisfied=satisfied,
        unsatisfied_blocking=_deduplicate(unsatisfied),
        weak_or_missing_evidence=_deduplicate(weak_or_missing),
    )


def _matching_evidence(
    criterion: AcceptanceCriterion,
    state: ExplorationState,
    task_mode: str,
) -> list[Evidence]:
    matches: list[Evidence] = []
    for evidence in state.evidence_items:
        if not _tool_backed(evidence):
            continue
        if task_mode == "symbol_investigation" and not _symbol_matches(criterion, evidence):
            continue
        if task_mode == "feature_exploration" and evidence.strength == "noise":
            continue
        if _criterion_matches_evidence(criterion, evidence):
            matches.append(evidence)
    return matches


def _tool_backed(evidence: Evidence) -> bool:
    return evidence.source_type != "report" and evidence.strength != "noise"


def _symbol_matches(criterion: AcceptanceCriterion, evidence: Evidence) -> bool:
    haystack = " ".join(
        item
        for item in (
            evidence.symbol or "",
            evidence.claim,
            evidence.summary,
            evidence.file_path or "",
        )
        if item
    ).casefold()
    required = [item.casefold() for item in criterion.required_evidence]
    return bool(required) and any(item in haystack for item in required)


def _criterion_matches_evidence(
    criterion: AcceptanceCriterion,
    evidence: Evidence,
) -> bool:
    haystack = " ".join(
        item
        for item in (
            evidence.id,
            evidence.claim,
            evidence.summary,
            evidence.file_path or "",
            evidence.symbol or "",
            evidence.source_type,
        )
        if item
    ).casefold()
    required = [item.casefold() for item in criterion.required_evidence]
    if not required:
        return evidence.file_path is not None or evidence.symbol is not None
    return any(item in haystack for item in required)


def _evidence_ref(evidence: Evidence) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence.id,
        file_path=evidence.file_path,
        symbol=evidence.symbol,
        summary=evidence.summary or evidence.claim,
    )


def _deduplicate(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
```

- [ ] **Step 4: Run tests to verify Task 2 passes**

Run: `pytest tests/test_acceptance.py -q`

Expected: PASS for all acceptance tests.

- [ ] **Step 5: Commit Task 2**

```bash
git add app/agents/code_explorer/acceptance.py tests/test_acceptance.py
git commit -m "feat: evaluate acceptance evidence policies"
```

## Task 3: Checklist And Review Prompts

**Files:**
- Modify: `app/agents/code_explorer/acceptance.py`
- Modify: `tests/test_acceptance.py`

- [ ] **Step 1: Add failing prompt and parser tests**

Append to `tests/test_acceptance.py`:

```python
from app.agents.code_explorer.acceptance import (
    build_acceptance_checklist_prompt,
    build_acceptance_review_prompt,
    parse_acceptance_checklist_response,
    parse_acceptance_review_response,
)


def test_checklist_prompt_requires_structured_goal_specific_json() -> None:
    prompt = build_acceptance_checklist_prompt(
        "总结项目架构",
        "repo_overview",
    )

    assert "repo_overview" in prompt
    assert "criteria" in prompt
    assert "blocking" in prompt
    assert "不要写泛泛项" in prompt


def test_parse_checklist_response_accepts_fenced_json() -> None:
    payload = parse_acceptance_checklist_response(
        """```json
        {
          "criteria": [
            {
              "id": "entry",
              "description": "Identify entry points",
              "blocking": true,
              "evidence_policy": "repo_overview",
              "required_evidence": ["entry"]
            }
          ]
        }
        ```"""
    )

    assert payload["criteria"][0]["id"] == "entry"


def test_review_prompt_and_parser_support_append_criteria() -> None:
    checklist = _checklist("repo_overview", ["entry"])
    prompt = build_acceptance_review_prompt(checklist, "# report")

    assert "satisfied_criteria" in prompt
    assert "append_criteria" in prompt

    review = parse_acceptance_review_response(
        """{
          "satisfied_criteria": ["coverage"],
          "append_criteria": [
            {
              "id": "config",
              "description": "Cover runtime configuration",
              "blocking": true,
              "evidence_policy": "repo_overview",
              "required_evidence": ["config"]
            }
          ],
          "notes": ["needs config evidence"]
        }"""
    )

    assert review["satisfied_criteria"] == ["coverage"]
    assert review["append_criteria"][0]["id"] == "config"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_acceptance.py -q`

Expected: FAIL with imports missing for prompt and parser functions.

- [ ] **Step 3: Implement prompt builders and JSON parsers**

Add imports near the top of `app/agents/code_explorer/acceptance.py`:

```python
import json
import re
```

Add below `evaluate_acceptance`:

```python
_JSON_FENCE = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\})\s*```", re.DOTALL)


def build_acceptance_checklist_prompt(goal: str, task_mode: str) -> str:
    """Ask the model for a strict goal-specific acceptance checklist."""

    return f"""\
## 生成目标验收清单

用户目标：
{goal.strip()}

任务模式：
{task_mode}

请只返回 JSON 对象，不要返回 Markdown 说明。JSON schema:
{{
  "criteria": [
    {{
      "id": "short-stable-id",
      "description": "具体、可验证的验收项",
      "blocking": true,
      "evidence_policy": "{task_mode}",
      "required_evidence": ["keyword-or-symbol"]
    }}
  ]
}}

要求：
- 至少 3 个 blocking 验收项。
- 每项必须能通过 CodeGraph 工具证据验证。
- 不要写泛泛项，例如“理解项目”“分析清楚”。
- repo_overview 覆盖入口、模块结构、核心链路、配置依赖、测试或风险。
- symbol_investigation 覆盖定义、职责、上游调用、下游依赖、影响面。
- feature_exploration 覆盖入口、主链阶段、转换、持久化或外部边界、阶段连接。
"""


def build_acceptance_review_prompt(
    checklist: AcceptanceChecklist,
    candidate_report: str,
) -> str:
    """Ask the model to propose criterion satisfaction and append-only additions."""

    return f"""\
## 验收清单自评

请根据候选报告和你已经看到的工具证据，返回 JSON 对象。不要返回 Markdown 说明。

当前验收清单：
{json.dumps(checklist.to_dict(), ensure_ascii=False, indent=2)}

候选报告：
{candidate_report}

JSON schema:
{{
  "satisfied_criteria": ["criterion-id"],
  "append_criteria": [
    {{
      "id": "new-stable-id",
      "description": "新增的隐含验收项",
      "blocking": true,
      "evidence_policy": "{checklist.task_mode}",
      "required_evidence": ["keyword-or-symbol"]
    }}
  ],
  "notes": ["short reason"]
}}

规则：
- 只有确实有工具证据支撑的项才能放入 satisfied_criteria。
- 允许追加用户目标隐含但当前清单遗漏的验收项。
- 不允许删除已有项，不允许把 blocking 项降级。
"""


def parse_acceptance_checklist_response(content: str) -> dict[str, Any]:
    """Parse a checklist JSON object from model content."""

    payload = _parse_json_object(content)
    if "criteria" not in payload:
        raise AcceptanceChecklistError("acceptance checklist response missing criteria")
    return payload


def parse_acceptance_review_response(content: str) -> dict[str, Any]:
    """Parse a model's structured checklist review."""

    payload = _parse_json_object(content)
    satisfied = _string_list(payload.get("satisfied_criteria"))
    append_criteria = payload.get("append_criteria", [])
    if not isinstance(append_criteria, list):
        raise AcceptanceChecklistError("append_criteria must be a list")
    return {
        "satisfied_criteria": satisfied,
        "append_criteria": [item for item in append_criteria if isinstance(item, dict)],
        "notes": _string_list(payload.get("notes")),
    }


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    match = _JSON_FENCE.search(text)
    if match:
        text = match.group("body").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AcceptanceChecklistError(f"invalid acceptance JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AcceptanceChecklistError("acceptance response must be a JSON object")
    return payload
```

- [ ] **Step 4: Run tests to verify Task 3 passes**

Run: `pytest tests/test_acceptance.py -q`

Expected: PASS for all acceptance tests.

- [ ] **Step 5: Commit Task 3**

```bash
git add app/agents/code_explorer/acceptance.py tests/test_acceptance.py
git commit -m "feat: add acceptance prompts and parsers"
```

## Task 4: Acceptance Run Artifacts

**Files:**
- Modify: `app/storage/run_store.py`
- Modify: `tests/test_run_store.py`

- [ ] **Step 1: Write failing artifact tests**

Append to `tests/test_run_store.py`:

```python
def test_run_store_saves_acceptance_artifacts(tmp_path):
    store = RunStore(tmp_path / "runs")
    artifacts = store.create_run("run-acceptance")

    artifacts.write_json("acceptance_checklist.json", {"criteria": []})
    artifacts.write_text("acceptance_checklist.md", "# Acceptance Checklist")
    artifacts.write_text("acceptance_decision_1.md", "CAN_RETURN: no")
    artifacts.write_text("acceptance_continuation_1.md", "continue with tools")

    assert artifacts.path("acceptance_checklist.json").exists()
    assert artifacts.path("acceptance_checklist.md").exists()
    assert artifacts.path("acceptance_decision_1.md").read_text(encoding="utf-8") == "CAN_RETURN: no"
    assert artifacts.path("acceptance_continuation_1.md").read_text(encoding="utf-8") == "continue with tools"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_store.py::test_run_store_saves_acceptance_artifacts -q`

Expected: FAIL with `ValueError: unsupported run artifact`.

- [ ] **Step 3: Add acceptance artifact names**

Modify `app/storage/run_store.py`:

```python
ARTIFACT_NAMES = (
    "input.md",
    "system_prompt.md",
    "task_prompt.md",
    "messages.md",
    "tool_calls.jsonl",
    "report.md",
    "errors.log",
    "exploration_state.json",
    "notebook.md",
    "convergence_checkpoint.md",
    "evidence_items.jsonl",
    "stage_graph.json",
    "evidence_quality_report.md",
    "acceptance_checklist.json",
    "acceptance_checklist.md",
)

_DYNAMIC_ARTIFACT_PATTERN = re.compile(
    r"(?:(?:stop_decision|continuation|stagnation_checkpoint|acceptance_decision|acceptance_continuation)_[1-9][0-9]*\.md)"
)
```

- [ ] **Step 4: Run artifact tests**

Run: `pytest tests/test_run_store.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add app/storage/run_store.py tests/test_run_store.py
git commit -m "feat: persist acceptance artifacts"
```

## Task 5: Agent Checklist Generation And Candidate Review

**Files:**
- Modify: `app/agents/code_explorer/agent.py`
- Modify: `tests/test_agent.py`

- [ ] **Step 1: Add failing repo overview gate test**

Append to `tests/test_agent.py`:

```python
class FakeAcceptanceOverviewLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        self.calls += 1
        if self.calls == 1:
            assert tools is None
            return _response(
                json.dumps(
                    {
                        "criteria": [
                            {
                                "id": "entry",
                                "description": "Identify entry points",
                                "blocking": True,
                                "evidence_policy": "repo_overview",
                                "required_evidence": ["app/main.py"],
                            }
                        ]
                    }
                ),
                None,
            )
        if self.calls == 2:
            assert tools is not None
            return _response(
                "# 代码探索结果\n\n"
                "## 索引状态\n可用。\n\n"
                "## 顶层仓库结构\n结构。\n\n"
                "## 主要模块职责\n职责。\n\n"
                "## 关键启动入口\n入口。\n\n"
                "## 主要业务域\n业务。\n\n"
                "## 建议进一步探索方向\n方向。\n\n"
                "## 不确定项\n无。",
                None,
            )
        if self.calls == 3:
            assert tools is None
            return _response(
                json.dumps(
                    {
                        "satisfied_criteria": [],
                        "append_criteria": [],
                        "notes": ["missing tool evidence"],
                    }
                ),
                None,
            )
        if self.calls == 4:
            assert "Acceptance Gate 未通过" in messages[-1]["content"]
            return _tool_call(
                "call-entry",
                "list_code_files",
                '{"pattern":"app/main.py","format":"flat"}',
            )
        if self.calls == 5:
            assert messages[-1]["role"] == "tool"
            return _response(
                "# 代码探索结果\n\n"
                "## 索引状态\n可用。\n\n"
                "## 顶层仓库结构\n结构。\n\n"
                "## 主要模块职责\n职责。\n\n"
                "## 关键启动入口\napp/main.py。\n\n"
                "## 主要业务域\n业务。\n\n"
                "## 建议进一步探索方向\n方向。\n\n"
                "## 不确定项\n无。",
                None,
            )
        assert tools is None
        return _response(
            json.dumps(
                {
                    "satisfied_criteria": ["entry"],
                    "append_criteria": [],
                    "notes": ["entry covered"],
                }
            ),
            None,
        )


class FakeAcceptanceCodeGraphMCPClient(FakeCodeGraphMCPClient):
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        del name, arguments
        return {"content": [{"type": "text", "text": "app/main.py"}]}


def test_repo_overview_candidate_must_pass_acceptance_gate(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(agent_module, "CodeGraphMCPClient", FakeAcceptanceCodeGraphMCPClient)
    llm_client = FakeAcceptanceOverviewLLMClient()

    result = asyncio.run(
        CodeExplorerAgent(_settings(tmp_path, max_continuations=2), llm_client).run(
            "请总结整个项目的架构框架"
        )
    )

    assert llm_client.calls == 6
    assert "app/main.py" in result.report
    assert result.artifacts.path("acceptance_checklist.json").exists()
    assert result.artifacts.path("acceptance_decision_1.md").exists()
    assert result.artifacts.path("acceptance_continuation_1.md").exists()
    assert "CAN_RETURN: no" in result.artifacts.path("acceptance_decision_1.md").read_text(encoding="utf-8")
    assert "CAN_RETURN: yes" in result.artifacts.path("acceptance_decision_2.md").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py::test_repo_overview_candidate_must_pass_acceptance_gate -q`

Expected: FAIL because no acceptance checklist is generated and no acceptance artifacts are written.

- [ ] **Step 3: Import acceptance helpers in `agent.py`**

Add to `app/agents/code_explorer/agent.py` imports:

```python
from app.agents.code_explorer.acceptance import (
    AcceptanceChecklist,
    AcceptanceChecklistError,
    AcceptanceDecision,
    build_acceptance_checklist_prompt,
    build_acceptance_review_prompt,
    evaluate_acceptance,
    normalize_checklist_payload,
    parse_acceptance_checklist_response,
    parse_acceptance_review_response,
)
```

- [ ] **Step 4: Generate and persist the initial checklist**

In `CodeExplorerAgent.run()`, keep checklist generation inside the existing `try` block so failures still produce recovery artifacts. Add `checklist: AcceptanceChecklist | None = None` before `started_at = monotonic()`:

```python
        checklist: AcceptanceChecklist | None = None
```

Inside the existing `try:`, before `async with CodeGraphMCPClient(...) as codegraph:`, add:

```python
            checklist = await self._generate_acceptance_checklist(
                task.strip(),
                task_mode,
                started_at,
            )
            self._save_acceptance_checklist(artifacts, checklist)
```

Change the `_run_loop` call to pass `checklist`:

```python
                result = await self._run_loop(
                    artifacts,
                    messages,
                    tools,
                    started_at,
                    task_mode,
                    state,
                    checklist,
                )
```

Change `_run_loop` signature:

```python
    async def _run_loop(
        self,
        artifacts: RunArtifacts,
        messages: list[dict[str, Any]],
        tools: ExplorerTools,
        started_at: float,
        task_mode: TaskMode,
        state: ExplorationState,
        checklist: AcceptanceChecklist,
    ) -> str:
```

- [ ] **Step 5: Add checklist and review helper methods**

Add methods inside `CodeExplorerAgent`, before `_run_loop`:

```python
    async def _generate_acceptance_checklist(
        self,
        task: str,
        task_mode: TaskMode,
        started_at: float,
    ) -> AcceptanceChecklist:
        prompt = build_acceptance_checklist_prompt(task, task_mode.value)
        response = await asyncio.wait_for(
            self._llm_client.create_chat_completion(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                None,
            ),
            timeout=self._remaining_seconds(started_at),
        )
        content = response.choices[0].message.content or ""
        payload = parse_acceptance_checklist_response(content)
        return normalize_checklist_payload(
            payload,
            goal=task,
            task_mode=task_mode.value,
        )

    async def _review_acceptance_candidate(
        self,
        checklist: AcceptanceChecklist,
        candidate_report: str,
        started_at: float,
    ) -> dict[str, Any]:
        response = await asyncio.wait_for(
            self._llm_client.create_chat_completion(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_acceptance_review_prompt(
                            checklist,
                            candidate_report,
                        ),
                    },
                ],
                None,
            ),
            timeout=self._remaining_seconds(started_at),
        )
        message = response.choices[0].message
        if message.tool_calls:
            raise ExplorationLimitError("acceptance review returned unexpected tool calls")
        return parse_acceptance_review_response(message.content or "{}")

    @staticmethod
    def _save_acceptance_checklist(
        artifacts: RunArtifacts,
        checklist: AcceptanceChecklist,
    ) -> None:
        artifacts.write_json("acceptance_checklist.json", checklist.to_dict())
        artifacts.write_text("acceptance_checklist.md", checklist.to_markdown())
```

- [ ] **Step 6: Add acceptance decision artifact helpers**

Add methods inside `CodeExplorerAgent`, near `_save_exploration_artifacts`:

```python
    @staticmethod
    def _save_acceptance_decision(
        artifacts: RunArtifacts,
        index: int,
        decision: AcceptanceDecision,
    ) -> None:
        artifacts.write_text(
            f"acceptance_decision_{index}.md",
            decision.to_markdown(),
        )
```

Add module-level continuation builder below `_build_incomplete_report`:

```python
def _build_acceptance_continuation_prompt(
    decision: AcceptanceDecision,
) -> str:
    lines = [
        "## Acceptance Gate 未通过",
        "",
        "当前候选报告还不能返回，请继续使用工具探索。",
        "",
        "### 未满足的 blocking 验收项",
        *_markdown_items(decision.unsatisfied_blocking),
        "",
        "### 缺失或较弱的证据",
        *_markdown_items(decision.weak_or_missing_evidence),
        "",
        "不要输出最终报告，请继续使用工具探索。",
        "优先补齐上述验收项需要的 CodeGraph 工具证据。",
    ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 7: Gate candidate reports in `_run_loop`**

Inside `_run_loop`, initialize a counter next to `stop_decision_count`:

```python
        acceptance_decision_count = 0
```

Replace the `if not message.tool_calls:` block with:

```python
            if not message.tool_calls:
                report = _sanitize_report(message.content or "")
                if (
                    task_mode is TaskMode.REPO_OVERVIEW
                    and _needs_repo_overview_rewrite(report)
                ):
                    report = await self._rewrite_repo_overview_report(
                        artifacts,
                        messages,
                        started_at,
                        report,
                    )

                feature_decision: StopDecision | None = None
                if task_mode is TaskMode.FEATURE_EXPLORATION:
                    update_state_from_candidate_report(state, report)
                    feature_decision = judge_stop(state, report)
                    state.stop_decision = feature_decision
                    update_notebook(state, messages, tool_logs)
                    stop_decision_count += 1
                    artifacts.write_text(
                        f"stop_decision_{stop_decision_count}.md",
                        feature_decision.to_markdown(),
                    )
                    self._save_exploration_artifacts(artifacts, state)

                try:
                    review = await self._review_acceptance_candidate(
                        checklist,
                        report,
                        started_at,
                    )
                    append_criteria = review.get("append_criteria", [])
                    if append_criteria:
                        checklist = normalize_checklist_payload(
                            {
                                "criteria": [
                                    *[asdict(criterion) for criterion in checklist.criteria],
                                    *append_criteria,
                                ]
                            },
                            goal=checklist.goal,
                            task_mode=checklist.task_mode,
                            previous=checklist,
                        )
                        self._save_acceptance_checklist(artifacts, checklist)
                except AcceptanceChecklistError as exc:
                    review = {"satisfied_criteria": [], "notes": [str(exc)]}

                acceptance_decision_count += 1
                acceptance_decision = evaluate_acceptance(
                    checklist,
                    state,
                    report,
                    task_mode.value,
                    review=review,
                    feature_decision=feature_decision,
                )
                self._save_acceptance_decision(
                    artifacts,
                    acceptance_decision_count,
                    acceptance_decision,
                )
                self._save_exploration_artifacts(artifacts, state)

                if acceptance_decision.can_return:
                    quality_report = artifacts.path(
                        "evidence_quality_report.md"
                    ).read_text(encoding="utf-8")
                    return guard_final_report(report, quality_report)

                incomplete_reason = (
                    "CODE_EXPLORER_MAX_CONTINUATIONS safety fuse reached "
                    "before Acceptance Gate passed"
                )
                if state.continuation_count < self._settings.max_continuations:
                    state.continuation_count += 1
                    continuation_prompt = _build_acceptance_continuation_prompt(
                        acceptance_decision,
                    )
                    acceptance_decision.continuation_prompt = continuation_prompt
                    artifacts.write_text(
                        f"acceptance_continuation_{state.continuation_count}.md",
                        continuation_prompt,
                    )
                    messages.append({"role": "user", "content": continuation_prompt})
                    self._save_messages(artifacts, messages)
                    self._save_exploration_artifacts(artifacts, state)
                    continue

                if self._settings.fail_on_incomplete:
                    raise ExplorationLimitError(
                        "exploration remained incomplete after "
                        "CODE_EXPLORER_MAX_CONTINUATIONS safety fuse reached"
                    )
                return _build_incomplete_report(
                    state,
                    incomplete_reason,
                    tool_count,
                    acceptance_decision,
                )
```

- [ ] **Step 8: Run the focused repo overview gate test**

Run: `pytest tests/test_agent.py::test_repo_overview_candidate_must_pass_acceptance_gate -q`

Expected: PASS.

- [ ] **Step 9: Commit Task 5**

```bash
git add app/agents/code_explorer/agent.py tests/test_agent.py
git commit -m "feat: gate candidate reports with acceptance checklist"
```

## Task 6: Incomplete Reports Include Acceptance Gate Failures

**Files:**
- Modify: `app/agents/code_explorer/agent.py`
- Modify: `tests/test_agent_incomplete_report.py`

- [ ] **Step 1: Add failing incomplete report test**

Append to `tests/test_agent_incomplete_report.py`:

```python
from app.agents.code_explorer.acceptance import AcceptanceDecision
from app.agents.code_explorer.agent import _build_incomplete_report
from app.agents.code_explorer.exploration_state import ExplorationState


def test_incomplete_report_lists_unmet_acceptance_items() -> None:
    state = ExplorationState(goal="总结项目架构", task_mode="repo_overview")
    decision = AcceptanceDecision(
        can_return=False,
        reason="Blocking acceptance criteria still lack acceptable tool evidence.",
        unsatisfied_blocking=["entry: Identify project entry points"],
        weak_or_missing_evidence=["entry: Identify project entry points"],
    )

    report = _build_incomplete_report(
        state,
        "continuation fuse reached",
        3,
        decision,
    )

    assert "Acceptance Gate: not passed" in report
    assert "entry: Identify project entry points" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_incomplete_report.py::test_incomplete_report_lists_unmet_acceptance_items -q`

Expected: FAIL because `_build_incomplete_report` does not accept an acceptance decision.

- [ ] **Step 3: Update incomplete report helpers**

Change `_build_incomplete_report` signature in `app/agents/code_explorer/agent.py`:

```python
def _build_incomplete_report(
    state: ExplorationState,
    termination_reason: str,
    tool_count: int,
    acceptance_decision: AcceptanceDecision | None = None,
) -> str:
```

Change its call to `_build_local_state_report`:

```python
    return _build_local_state_report(
        state,
        title="# 代码探索结果（阶段性）",
        introduction=(
            "> 本报告为当前静态探索下的阶段性结论，Acceptance Gate 或 Evidence Gate "
            "未完全通过。以下结论只代表已有工具证据能支撑的部分，未闭合项已单独列出。"
        ),
        termination_reason=termination_reason,
        tool_count=tool_count,
        acceptance_decision=acceptance_decision,
    )
```

Change `_build_local_state_report` signature:

```python
def _build_local_state_report(
    state: ExplorationState,
    *,
    title: str,
    introduction: str,
    termination_reason: str,
    tool_count: int | None,
    acceptance_decision: AcceptanceDecision | None = None,
) -> str:
```

In `_build_recovery_report`, pass `acceptance_decision=None`:

```python
        acceptance_decision=None,
```

In `_build_local_state_report`, add after `gate_status`:

```python
    acceptance_status = (
        "passed"
        if acceptance_decision and acceptance_decision.can_return
        else "not passed"
        if acceptance_decision
        else "unknown"
    )
```

In the `lines` list under `"## 运行状态"`, add:

```python
        f"- Acceptance Gate: {acceptance_status}",
```

In the `lines` list before `"## 不确定项"`, add:

```python
        "",
        "## 未满足的目标验收项",
        *_markdown_items(
            acceptance_decision.unsatisfied_blocking if acceptance_decision else []
        ),
        "",
        "## 目标验收证据缺口",
        *_markdown_items(
            acceptance_decision.weak_or_missing_evidence if acceptance_decision else []
        ),
```

- [ ] **Step 4: Run incomplete report tests**

Run: `pytest tests/test_agent_incomplete_report.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

```bash
git add app/agents/code_explorer/agent.py tests/test_agent_incomplete_report.py
git commit -m "feat: report unmet acceptance criteria"
```

## Task 7: Update Existing Agent Tests For Checklist Calls

**Files:**
- Modify: `tests/test_agent.py`
- Modify: `tests/test_agent_incomplete_report.py`

- [ ] **Step 1: Add reusable checklist and review helpers to `tests/test_agent.py`**

Add near the bottom of `tests/test_agent.py`, before `_settings`:

```python
def _acceptance_checklist_response(
    *,
    task_mode: str = "feature_exploration",
    required: str = "WidgetService",
) -> Any:
    return _response(
        json.dumps(
            {
                "criteria": [
                    {
                        "id": "goal-coverage",
                        "description": "Cover the user goal",
                        "blocking": True,
                        "evidence_policy": task_mode,
                        "required_evidence": [required],
                    }
                ]
            }
        ),
        None,
    )


def _acceptance_review_response(*, satisfied: bool) -> Any:
    return _response(
        json.dumps(
            {
                "satisfied_criteria": ["goal-coverage"] if satisfied else [],
                "append_criteria": [],
                "notes": ["reviewed"],
            }
        ),
        None,
    )
```

- [ ] **Step 2: Make fake MCP responses produce usable evidence**

Update `FakeCodeGraphMCPClient.call_tool` in `tests/test_agent.py`:

```python
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        del name, arguments
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "- codegraph_status (function) - app/main.py\n"
                        "- WidgetService (class) - app/services.py\n"
                        "WidgetService -> WidgetRepository"
                    ),
                }
            ]
        }
```

This keeps existing message assertions valid while giving `extract_evidence_from_tool_result()` enough located evidence for acceptance tests.

- [ ] **Step 3: Update fake LLM clients with explicit acceptance calls**

Use this call sequence table:

| Fake client | New call sequence |
|---|---|
| `FakeLLMClient` | 1 checklist for `symbol_investigation` required `codegraph_status`; 2 existing `check_code_index` tool call; 3 existing final report; 4 acceptance review satisfied |
| `FakeOverviewLLMClient` | 1 checklist for `repo_overview` required `app/main.py`; 2 `list_code_files` tool call; 3 existing incomplete overview; 4 existing rewrite response; 5 acceptance review satisfied |
| `FakeSelfCorrectingLLMClient` | 1 checklist for `symbol_investigation` required `WidgetService`; 2 existing invalid tool call; 3 existing corrected tool call; 4 existing final report; 5 acceptance review satisfied |
| `FakeContinuingLLMClient` | 1 checklist for `feature_exploration` required `WidgetService`; increment existing branches by 1; after each candidate report return acceptance review satisfied |
| `FakeConvergenceCheckpointLLMClient` | 1 checklist for `feature_exploration` required `WidgetService`; increment existing branches by 1; after each candidate report return acceptance review satisfied |
| `FakeStagnatingLLMClient` | 1 checklist for `feature_exploration` required `WidgetService`; increment existing branches by 1; after each candidate report return acceptance review satisfied |
| `FakeFailingLLMClient` | no sequence change; it raises on the checklist-generation call and must still produce recovery artifacts |
| `FakeCancelledLLMClient` | no sequence change; it raises on the checklist-generation call and must still produce recovery artifacts |

For `FakeLLMClient`, use:

```python
        if self.calls == 1:
            return _acceptance_checklist_response(
                task_mode="symbol_investigation",
                required="codegraph_status",
            )
        if self.calls == 2:
            return _response(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        id="call-1",
                        type="function",
                        function=SimpleNamespace(
                            name="check_code_index",
                            arguments="{}",
                        ),
                    )
                ],
            )
        if self.calls == 3:
            assert messages[-1]["role"] == "tool"
            assert len(tools) == 10
            return _response(
                content="# 代码探索结果\n\n索引可用。\n\n```java\nclass Hidden {}\n```",
                tool_calls=None,
            )
        return _acceptance_review_response(satisfied=True)
```

For `FakeOverviewLLMClient`, use this first branch and shift the existing overview/rewrite branches after it:

```python
        if len(self.tool_arguments) == 1:
            assert tools is None
            return _acceptance_checklist_response(
                task_mode="repo_overview",
                required="app/main.py",
            )
        if len(self.tool_arguments) == 2:
            assert tools is not None
            return _tool_call(
                "overview-files",
                "list_code_files",
                '{"pattern":"app/main.py","format":"flat"}',
            )
```

At the end of `FakeOverviewLLMClient.create_chat_completion`, after the rewrite response branch, add:

```python
        assert tools is None
        return _acceptance_review_response(satisfied=True)
```

- [ ] **Step 4: Run full agent tests**

Run: `pytest tests/test_agent.py tests/test_agent_incomplete_report.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 7**

```bash
git add tests/test_agent.py tests/test_agent_incomplete_report.py
git commit -m "test: update agent fakes for acceptance review"
```

## Task 8: Full Verification And Cleanup

**Files:**
- Review only unless tests expose required fixes.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
pytest \
  tests/test_acceptance.py \
  tests/test_run_store.py \
  tests/test_agent.py \
  tests/test_agent_incomplete_report.py \
  tests/test_stop_judge.py \
  tests/test_prompt.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 3: Check git diff for scope**

Run: `git diff --stat HEAD`

Expected: only files from this plan are changed, plus any user-owned dirty files that existed before execution.

- [ ] **Step 4: Commit final fixes if Step 1 or Step 2 required changes**

If verification required small fixes, commit only those files:

```bash
git add app/agents/code_explorer/acceptance.py app/agents/code_explorer/agent.py app/storage/run_store.py tests/test_acceptance.py tests/test_agent.py tests/test_agent_incomplete_report.py tests/test_run_store.py
git commit -m "fix: stabilize acceptance gate tests"
```

If no fixes were needed after Task 7, skip this commit.

## Self-Review Notes

- Spec coverage: Tasks 1-3 cover models, generation prompts, review prompts, append-only rules, and local evidence evaluation. Tasks 4-6 cover artifacts, run-loop gating, continuation, incomplete behavior, and `fail_on_incomplete` integration through the existing continuation fuse path. Task 7 updates existing fake LLM flows. Task 8 verifies the full suite.
- Type consistency: `AcceptanceChecklist`, `AcceptanceDecision`, and helper names are used consistently across plan tasks.
- Scope: The plan leaves `StopJudge` in place and creates a separate acceptance module, matching the approved design.
