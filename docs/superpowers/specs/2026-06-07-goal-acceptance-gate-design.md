# Goal Acceptance Gate Design

Date: 2026-06-07
Status: Draft for user review

## Purpose

CodeExplorerAgent currently decides completion differently by task mode. Feature exploration uses a strong-evidence StopJudge, while repository overview mostly relies on required report sections and rewrite prompts. This design adds a goal-driven acceptance layer for every task mode.

The agent will derive an acceptance checklist from the user's original goal before exploration starts. Whenever the model tries to return a final report, local code evaluates the current evidence against that checklist. If any blocking item is not satisfied, the agent must continue exploration unless a configured safety fuse has been reached.

## Goals

- Generate a concrete, verifiable checklist from the user's task at the start of each run.
- Apply the checklist to all task modes: `repo_overview`, `symbol_investigation`, and `feature_exploration`.
- Let the model propose checklist items and candidate satisfaction, while local code enforces hard completion rules.
- Allow checklist growth during exploration, but never allow deletion or downgrading of existing blocking items.
- Preserve the existing feature Evidence Gate as a hard requirement for feature exploration.
- Persist checklist and decision artifacts so users can audit why the agent continued or returned.

## Non-Goals

- This design does not replace CodeGraph evidence extraction.
- This design does not remove the current `StopJudge` business-chain checks.
- This design does not implement a fully semantic proof system for arbitrary natural-language goals.
- This design does not change the CLI interface. Optional debug flags require a separate design decision.

## Architecture

Add a new module:

`app/agents/code_explorer/acceptance.py`

This module owns goal-level completion logic. It should stay separate from `stop_judge.py`, which continues to own feature business-chain evidence checks.

Core data structures:

```python
AcceptanceCriterion
- id: str
- description: str
- task_mode: str
- blocking: bool
- evidence_policy: str
- required_evidence: list[str]
- satisfied: bool
- evidence_refs: list[EvidenceRef]
- notes: list[str]

EvidenceRef
- evidence_id: str | None
- tool_call_index: int | None
- file_path: str | None
- symbol: str | None
- summary: str

AcceptanceChecklist
- goal: str
- task_mode: str
- criteria: list[AcceptanceCriterion]
- revision: int

AcceptanceDecision
- can_return: bool
- reason: str
- satisfied: list[str]
- unsatisfied_blocking: list[str]
- weak_or_missing_evidence: list[str]
- continuation_prompt: str | None
```

The implementation may use dataclasses to match the rest of the codebase.

## Checklist Generation

At the start of `CodeExplorerAgent.run()`, after task mode detection and initial `ExplorationState` creation, the agent asks the LLM to generate a checklist from the user's task.

The checklist generation prompt must require concrete acceptance items. It should reject vague items such as "understand the project" and instead require observable coverage like "identify entry points", "map module responsibilities", or "trace callers and callees of the target symbol".

Local normalization must enforce:

- The checklist is not empty.
- Every criterion has a stable unique id.
- At least one criterion is blocking.
- Every blocking criterion has an evidence policy.
- Existing blocking criteria cannot be removed or downgraded in later revisions.
- New criteria may be appended when exploration reveals implicit user-goal requirements.

If generation returns invalid data, local code should either repair safe structural issues, such as missing ids, or request regeneration. It must not silently accept a checklist without blocking criteria.

## Run Flow

The existing run loop changes at the point where an assistant message has no tool calls.

1. Create run artifacts and initial `ExplorationState`.
2. Generate and persist `AcceptanceChecklist`.
3. Run the existing tool-enabled exploration loop.
4. Continue updating `ExplorationState`, notebook, evidence, and tool logs as today.
5. When the model produces a no-tool-call report, treat it as a candidate final report.
6. Call `evaluate_acceptance(checklist, state, report, task_mode)`.
7. For `feature_exploration`, also call the existing `judge_stop(state, report)` and fold its result into the acceptance decision.
8. If all blocking acceptance criteria pass, return the guarded final report.
9. If any blocking criterion fails and continuation budget remains, append an acceptance continuation prompt and keep exploring.
10. If a safety fuse is reached before acceptance passes, follow `fail_on_incomplete`:
    - default: return a clearly marked incomplete report that lists unmet acceptance criteria;
    - `fail_on_incomplete=True`: raise an exploration limit error.

## Evidence Policies

The checklist is universal, but completion rules differ by task mode.

### Repository Overview

Blocking criteria usually cover:

- project entry points;
- top-level directories and module responsibilities;
- core runtime or request/workflow path;
- external dependencies and configuration;
- test coverage, operational risks, or important gaps.

Each blocking criterion must be supported by CodeGraph-backed tool evidence, usually from `list_code_files`, `get_code_context`, `explore_symbol`, or `get_symbol_detail`.

The existing fixed repo-overview report sections remain useful as output format guidance, but they are no longer sufficient as the completion standard.

### Symbol Investigation

Blocking criteria usually cover:

- definition location;
- signature and responsibility;
- upstream callers;
- downstream callees or dependencies;
- impact area, boundary, or ambiguity resolution.

The evidence must be tied to the target symbol. If the symbol is missing, ambiguous, or not actually inspected through CodeGraph evidence, blocking criteria cannot pass.

### Feature Exploration

Blocking criteria usually cover:

- trigger or entry point;
- mainline stages;
- important transformation logic;
- persistence or external boundary;
- adjacent stage connections;
- user-requested extras such as error paths, state transitions, or dynamic boundaries.

Feature exploration must satisfy both:

- the goal acceptance checklist;
- the existing strong-evidence feature `StopJudge`.

This preserves current safety around business-stage closure while making user-specific acceptance explicit.

## Continuation Prompts

Add an acceptance continuation prompt builder, either inside `acceptance.py` or a small companion module.

The prompt should include:

- unmet blocking criteria;
- why each criterion failed;
- already accepted evidence references;
- suggested evidence types to collect next;
- a direct instruction to continue using tools and not return a final report yet.

For feature exploration, this prompt should include both acceptance gaps and existing Evidence Gate gaps so the model can prioritize the real blockers.

## Artifacts

Add these run artifacts:

- `acceptance_checklist.json`
- `acceptance_checklist.md`
- `acceptance_decision_N.md`
- `acceptance_continuation_N.md`

The JSON artifact is the machine-readable source of truth. The Markdown artifact is for auditability.

The existing `evidence_quality_report.md`, `notebook.md`, `stop_decision_N.md`, and incomplete report behavior should remain.

## Error Handling

- Invalid initial checklist: regenerate or fail early with a clear configuration/runtime error.
- Invalid appended checklist revision: reject the revision and continue with the previous valid checklist.
- Candidate report without satisfied blocking criteria: continue exploration.
- Safety fuse reached before checklist passes: use `fail_on_incomplete` behavior.
- LLM claims a criterion is satisfied without tool evidence: local gate marks it unsatisfied.

## Testing

Add focused tests for:

- checklist normalization rejects empty criteria and requires blocking items;
- generated ids are stable or repaired safely;
- append-only revision rules allow additions but reject deletion and blocking downgrades;
- repository overview requires evidence for structure, entry points, modules, config, and tests/risks;
- symbol investigation requires symbol-specific definition, caller, callee, and impact evidence;
- feature exploration requires both acceptance checklist pass and existing StopJudge pass;
- `_run_loop` evaluates acceptance before returning a no-tool-call report;
- continuation prompts list unmet acceptance criteria;
- incomplete reports list unmet checklist items;
- `fail_on_incomplete=True` raises when safety fuses are reached before acceptance passes;
- new artifacts are written.

Existing StopJudge tests should continue to pass, with only minimal updates where `feature_exploration` completion now has an additional acceptance layer.

## Open Implementation Notes

- Prefer dataclasses and pure functions so the acceptance gate can be tested without invoking the LLM or MCP.
- Keep LLM-generated checklist JSON small and strict.
- Reuse `ExplorationState.evidence_items`, tool logs, and existing evidence strength helpers where possible.
- Avoid making report prose itself a primary evidence source for blocking criteria.
- Keep `stop_judge.py` focused on business-chain evidence; do not move goal-checklist logic into it.
