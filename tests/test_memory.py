from app.agents.code_explorer.memory import ConversationMemory


def test_memory_truncates_old_tool_results_but_keeps_recent_results() -> None:
    memory = ConversationMemory(
        recent_tool_results=1,
        old_tool_result_chars=12,
        recent_tool_result_chars=100,
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "tool", "tool_call_id": "old", "content": "abcdefghijklmnopqrstuvwxyz"},
        {"role": "tool", "tool_call_id": "new", "content": "recent-result"},
    ]

    prepared = memory.prepare(messages)

    assert prepared[1]["content"].startswith("[历史工具结果已截断")
    assert "abcdefghijkl" in prepared[1]["content"]
    assert prepared[2]["content"] == "recent-result"
    assert messages[1]["content"] == "abcdefghijklmnopqrstuvwxyz"
