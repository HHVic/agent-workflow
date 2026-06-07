import asyncio
import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.llm_client import (
    MockLLMResponseError,
    ReplayLLMClient,
    create_llm_client,
)


def test_replay_llm_client_reads_compact_jsonl_responses(tmp_path: Path) -> None:
    responses_file = tmp_path / "responses.jsonl"
    responses_file.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "name": "check_code_index",
                                "arguments": {},
                            }
                        ]
                    }
                ),
                json.dumps({"content": "# 代码探索结果\n\nMock report."}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    async def run() -> None:
        client = ReplayLLMClient(responses_file)

        first = await client.create_chat_completion([], [])
        tool_call = first.choices[0].message.tool_calls[0]
        assert tool_call.id == "call-1"
        assert tool_call.function.name == "check_code_index"
        assert tool_call.function.arguments == "{}"

        second = await client.create_chat_completion([], [])
        assert second.choices[0].message.content == "# 代码探索结果\n\nMock report."

        with pytest.raises(MockLLMResponseError, match="responses exhausted"):
            await client.create_chat_completion([], [])

        await client.close()

    asyncio.run(run())


def test_create_llm_client_uses_replay_file_without_api_key(tmp_path: Path) -> None:
    responses_file = tmp_path / "responses.jsonl"
    responses_file.write_text('{"content":"mock"}\n', encoding="utf-8")

    client = create_llm_client(_settings(tmp_path, responses_file))

    assert isinstance(client, ReplayLLMClient)


def test_replay_llm_client_reports_invalid_json_line(tmp_path: Path) -> None:
    responses_file = tmp_path / "responses.jsonl"
    responses_file.write_text('{"content":"ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(MockLLMResponseError, match="line 2"):
        ReplayLLMClient(responses_file)


def _settings(tmp_path: Path, responses_file: Path) -> Settings:
    return Settings(
        llm_api_key=None,
        llm_base_url=None,
        llm_model="unused-in-replay-mode",
        codegraph_mcp_command="codegraph",
        codegraph_mcp_args=("serve", "--mcp"),
        codegraph_project_path=tmp_path,
        max_tool_calls=10,
        max_seconds=30,
        runs_dir=tmp_path / "runs",
        mock_llm_responses_file=responses_file,
    )
