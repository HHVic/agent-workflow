from pathlib import Path

import pytest

from app.core.config import Settings


def test_settings_load_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "CODEGRAPH_MCP_COMMAND",
        "CODEGRAPH_MCP_ARGS",
        "CODEGRAPH_PROJECT_PATH",
        "CODE_EXPLORER_MAX_TOOL_CALLS",
        "CODE_EXPLORER_MAX_SECONDS",
        "CODE_EXPLORER_MAX_CONTINUATIONS",
        "RUNS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    settings = Settings.from_env()

    assert settings.llm_api_key is None
    assert settings.llm_base_url is None
    assert settings.llm_model == "qwen-plus"
    assert settings.codegraph_mcp_command == "codegraph"
    assert settings.codegraph_mcp_args == ("serve", "--mcp")
    assert settings.codegraph_project_path == tmp_path
    assert settings.max_tool_calls == 120
    assert settings.max_seconds == 900
    assert settings.max_continuations == 3
    assert settings.runs_dir == Path("runs")


def test_settings_load_environment_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_path = tmp_path / "source repo"
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("CODEGRAPH_MCP_COMMAND", "/usr/local/bin/codegraph")
    monkeypatch.setenv("CODEGRAPH_MCP_ARGS", 'serve --mcp --path "/tmp/source repo"')
    monkeypatch.setenv("CODEGRAPH_PROJECT_PATH", str(project_path))
    monkeypatch.setenv("CODE_EXPLORER_MAX_TOOL_CALLS", "42")
    monkeypatch.setenv("CODE_EXPLORER_MAX_SECONDS", "75")
    monkeypatch.setenv("CODE_EXPLORER_MAX_CONTINUATIONS", "5")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "logs"))

    settings = Settings.from_env()

    assert settings.llm_api_key == "secret"
    assert settings.llm_base_url == "https://llm.example/v1"
    assert settings.llm_model == "custom-model"
    assert settings.codegraph_mcp_command == "/usr/local/bin/codegraph"
    assert settings.codegraph_mcp_args == (
        "serve",
        "--mcp",
        "--path",
        "/tmp/source repo",
    )
    assert settings.codegraph_project_path == project_path
    assert settings.max_tool_calls == 42
    assert settings.max_seconds == 75
    assert settings.max_continuations == 5
    assert settings.runs_dir == tmp_path / "logs"


def test_settings_load_dotenv_from_current_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "LLM_API_KEY=dotenv-secret\nLLM_BASE_URL=https://dotenv.example/v1\n",
        encoding="utf-8",
    )

    settings = Settings.from_env()

    assert settings.llm_api_key == "dotenv-secret"
    assert settings.llm_base_url == "https://dotenv.example/v1"
