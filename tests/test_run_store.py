import json
from pathlib import Path

from app.storage.run_store import RunStore


def test_run_store_saves_run_artifacts(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    artifacts = store.create_run("run-fixed")

    artifacts.write_text("input.md", "探索任务")
    artifacts.write_text("report.md", "# 代码探索结果")
    artifacts.append_jsonl("tool_calls.jsonl", {"tool": "check_code_index"})
    artifacts.append_error("boom")

    assert artifacts.run_id == "run-fixed"
    assert artifacts.run_dir == tmp_path / "run-fixed"
    assert artifacts.path("input.md").read_text(encoding="utf-8") == "探索任务"
    assert artifacts.path("report.md").read_text(encoding="utf-8") == "# 代码探索结果"
    assert json.loads(
        artifacts.path("tool_calls.jsonl").read_text(encoding="utf-8")
    ) == {"tool": "check_code_index"}
    assert artifacts.path("errors.log").read_text(encoding="utf-8") == "boom\n"


def test_run_store_generates_unique_run_ids(tmp_path: Path) -> None:
    store = RunStore(tmp_path)

    first = store.create_run()
    second = store.create_run()

    assert first.run_id != second.run_id


def test_run_store_saves_dynamic_exploration_artifacts(tmp_path: Path) -> None:
    artifacts = RunStore(tmp_path).create_run("run-state")

    artifacts.write_text("notebook.md", "# Exploration Notebook")
    artifacts.write_text("stop_decision_1.md", "CAN_STOP: no")
    artifacts.write_text("continuation_1.md", "continue")
    artifacts.write_text("convergence_checkpoint.md", "review")
    artifacts.write_json("exploration_state.json", {"continuation_count": 1})
    artifacts.append_jsonl("evidence_items.jsonl", {"id": "tool-1-edge-1"})
    artifacts.write_json("stage_graph.json", {"stages": [], "edges": []})
    artifacts.write_text("evidence_quality_report.md", "# Evidence Quality")

    assert artifacts.path("notebook.md").read_text(encoding="utf-8") == "# Exploration Notebook"
    assert artifacts.path("stop_decision_1.md").read_text(encoding="utf-8") == "CAN_STOP: no"
    assert artifacts.path("continuation_1.md").read_text(encoding="utf-8") == "continue"
    assert artifacts.path("convergence_checkpoint.md").read_text(encoding="utf-8") == "review"
    assert json.loads(
        artifacts.path("evidence_items.jsonl").read_text(encoding="utf-8")
    ) == {"id": "tool-1-edge-1"}
    assert json.loads(
        artifacts.path("stage_graph.json").read_text(encoding="utf-8")
    ) == {"stages": [], "edges": []}
    assert (
        artifacts.path("evidence_quality_report.md").read_text(encoding="utf-8")
        == "# Evidence Quality"
    )
    assert json.loads(
        artifacts.path("exploration_state.json").read_text(encoding="utf-8")
    ) == {"continuation_count": 1}
