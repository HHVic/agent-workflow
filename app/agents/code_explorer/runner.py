"""CLI entry point for CodeExplorerAgent."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from app.agents.code_explorer.agent import CodeExplorerAgent
from app.core.config import Settings
from app.core.llm_client import LLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore a CodeGraph-indexed repository with an LLM agent.",
    )
    parser.add_argument("task", help="Natural-language code exploration task")
    parser.add_argument(
        "--project-path",
        type=Path,
        help="CodeGraph-indexed repository path; defaults to CODEGRAPH_PROJECT_PATH or cwd",
    )
    return parser


async def run_agent(task: str, settings: Settings) -> int:
    """Run one CLI task and print its artifact paths."""

    llm_client = LLMClient(settings)
    agent = CodeExplorerAgent(settings, llm_client)
    try:
        result = await agent.run(task)
    except Exception as exc:
        print(f"CodeExplorerAgent failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if agent.last_artifacts is not None:
            _print_artifact_paths(agent.last_artifacts.run_id, agent.last_artifacts.run_dir)
        return 1
    finally:
        await llm_client.close()

    _print_artifact_paths(result.run_id, result.artifacts.run_dir)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the async agent."""

    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.project_path is not None:
        settings = replace(
            settings,
            codegraph_project_path=args.project_path.expanduser().resolve(),
        )
    return asyncio.run(run_agent(args.task, settings))


def _print_artifact_paths(run_id: str, run_dir: Path) -> None:
    print(f"run_id: {run_id}")
    print(f"report.md: {(run_dir / 'report.md').resolve()}")
    print(f"tool_calls.jsonl: {(run_dir / 'tool_calls.jsonl').resolve()}")
    print(f"messages.md: {(run_dir / 'messages.md').resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
