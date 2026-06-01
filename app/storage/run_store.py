"""Local artifact storage for CodeExplorerAgent runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


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
)

_DYNAMIC_ARTIFACT_PATTERN = re.compile(
    r"(?:stop_decision|continuation|stagnation_checkpoint)_[1-9][0-9]*\.md"
)


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Paths and append helpers for one local run."""

    run_id: str
    run_dir: Path

    def path(self, name: str) -> Path:
        """Return an artifact path, rejecting unknown artifact names."""

        if name not in ARTIFACT_NAMES and not _DYNAMIC_ARTIFACT_PATTERN.fullmatch(
            name
        ):
            raise ValueError(f"unsupported run artifact: {name}")
        return self.run_dir / name

    def write_text(self, name: str, content: str) -> None:
        """Overwrite a text artifact."""

        self.path(name).write_text(content, encoding="utf-8")

    def append_jsonl(self, name: str, value: dict[str, Any]) -> None:
        """Append a JSON object to a JSONL artifact."""

        line = json.dumps(value, ensure_ascii=False, default=str)
        with self.path(name).open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    def write_json(self, name: str, value: Any) -> None:
        """Overwrite a JSON artifact with readable UTF-8 content."""

        content = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        self.write_text(name, f"{content}\n")

    def append_error(self, error: str) -> None:
        """Append one error line to the local error log."""

        with self.path("errors.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{error.rstrip()}\n")


class RunStore:
    """Creates isolated run directories under the configured root."""

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir

    def create_run(self, run_id: str | None = None) -> RunArtifacts:
        """Create a run directory and all required artifact files."""

        actual_run_id = run_id or _new_run_id()
        run_dir = self._runs_dir / actual_run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        for name in ARTIFACT_NAMES:
            (run_dir / name).touch()
        return RunArtifacts(run_id=actual_run_id, run_dir=run_dir)


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"
