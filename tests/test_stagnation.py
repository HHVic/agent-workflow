from app.agents.code_explorer.continuation import build_anti_stagnation_prompt
from app.agents.code_explorer.exploration_state import ExplorationState
from app.agents.code_explorer.stop_judge import judge_stop


def test_unclosed_state_uses_anti_stagnation_continuation_not_candidate_report() -> None:
    state = ExplorationState(goal="explore flow", task_mode="feature_exploration")
    decision = judge_stop(state, "")

    prompt = build_anti_stagnation_prompt(
        decision,
        state,
        5,
        ['get_symbol_detail({"symbol":"WidgetService"})'],
    )

    assert not decision.can_stop
    assert "反停滞继续探索" in prompt
    assert "不要输出最终报告" in prompt
    assert "禁止重复相同工具和参数" in prompt
    assert "WidgetService" in prompt
    assert decision.blocking_gaps[0] in prompt
