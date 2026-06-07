from pathlib import Path

from app.agents.code_explorer.runner import build_parser


def test_parser_accepts_mock_llm_responses_file(tmp_path: Path) -> None:
    responses_file = tmp_path / "responses.jsonl"

    args = build_parser().parse_args(
        [
            "概述仓库结构",
            "--mock-llm-responses",
            str(responses_file),
        ]
    )

    assert args.mock_llm_responses == responses_file
