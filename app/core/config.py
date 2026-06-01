"""Environment-backed application settings."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for the CLI and agent."""

    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str
    codegraph_mcp_command: str
    codegraph_mcp_args: tuple[str, ...]
    codegraph_project_path: Path
    max_tool_calls: int
    max_seconds: int
    runs_dir: Path
    max_continuations: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""

        load_dotenv(Path.cwd() / ".env", override=False)
        return cls(
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_model=os.getenv("LLM_MODEL", "qwen-plus"),
            codegraph_mcp_command=os.getenv("CODEGRAPH_MCP_COMMAND", "codegraph"),
            codegraph_mcp_args=tuple(
                shlex.split(os.getenv("CODEGRAPH_MCP_ARGS", "serve --mcp"))
            ),
            codegraph_project_path=Path(
                os.getenv("CODEGRAPH_PROJECT_PATH", Path.cwd())
            ).expanduser(),
            max_tool_calls=_read_positive_int("CODE_EXPLORER_MAX_TOOL_CALLS", 120),
            max_seconds=_read_positive_int("CODE_EXPLORER_MAX_SECONDS", 900),
            runs_dir=Path(os.getenv("RUNS_DIR", "runs")).expanduser(),
            max_continuations=_read_positive_int(
                "CODE_EXPLORER_MAX_CONTINUATIONS", 3
            ),
        )


def _read_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    value = default if raw_value is None else int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
