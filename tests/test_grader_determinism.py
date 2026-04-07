from __future__ import annotations

from envs.experiment_rescue_lab.models import Action, ActionType, FaultType
from envs.experiment_rescue_lab.server.environment import make_environment
from envs.experiment_rescue_lab.server.grader import make_grader_context, score_task_1, score_task_2, score_task_3


def _run_episode(task_id: str, difficulty: str, seed: int):
    env = make_environment(task_id=task_id, difficulty=difficulty, seed=seed)
    env.reset(seed=seed)

    action_trace = [
        Action(type=ActionType.INSPECT, args={}),
        Action(type=ActionType.RUN_DIAGNOSTIC_PROBE, args={}),
        Action(type=ActionType.CALIBRATE_SENSOR, args={}),
        Action(type=ActionType.ADJUST_PARAM_A, args={}),
    ]

    last_result = None
    for action in action_trace:
        last_result = env.step(action)
        if last_result.done:
            break

    assert last_result is not None
    final_state = env.hidden
    assert final_state is not None

    return env, action_trace, final_state


def test_task_1_grader_is_deterministic():
    env, action_trace, final_state = _run_episode("task_1", "easy", 101)
    ctx = make_grader_context(
        episode_seed=101,
        task_id="task_1",
        difficulty="easy",
        hidden_fault=final_state.fault_type,
        hidden_faults=[final_state.fault_type],
        final_hidden_state=final_state,
    )

    s1 = score_task_1(ctx, action_trace)
    s2 = score_task_1(ctx, action_trace)

    assert s1.model_dump() == s2.model_dump()
    assert 0.0 <= s1.total_score <= 1.0


def test_task_2_grader_is_deterministic():
    env, action_trace, final_state = _run_episode("task_2", "medium", 202)
    ctx = make_grader_context(
        episode_seed=202,
        task_id="task_2",
        difficulty="medium",
        hidden_fault=final_state.fault_type,
        hidden_faults=[final_state.fault_type],
        final_hidden_state=final_state,
    )

    s1 = score_task_2(ctx, action_trace)
    s2 = score_task_2(ctx, action_trace)

    assert s1.model_dump() == s2.model_dump()
    assert 0.0 <= s1.total_score <= 1.0


def test_task_3_grader_is_deterministic():
    env, action_trace, final_state = _run_episode("task_3", "hard", 303)
    ctx = make_grader_context(
        episode_seed=303,
        task_id="task_3",
        difficulty="hard",
        hidden_fault=final_state.fault_type,
        hidden_faults=[final_state.fault_type],
        final_hidden_state=final_state,
    )

    s1 = score_task_3(ctx, action_trace)
    s2 = score_task_3(ctx, action_trace)

    assert s1.model_dump() == s2.model_dump()
    assert 0.0 <= s1.total_score <= 1.0


def test_grader_scores_stay_in_unit_interval():
    _, action_trace, final_state = _run_episode("task_3", "hard", 404)
    ctx = make_grader_context(
        episode_seed=404,
        task_id="task_3",
        difficulty="hard",
        hidden_fault=final_state.fault_type,
        hidden_faults=[final_state.fault_type],
        final_hidden_state=final_state,
    )

    score = score_task_3(ctx, action_trace)

    assert 0.0 <= score.diagnosis_score <= 1.0
    assert 0.0 <= score.recovery_score <= 1.0
    assert 0.0 <= score.efficiency_score <= 1.0
    assert 0.0 <= score.safety_score <= 1.0
    assert 0.0 <= score.final_quality_score <= 1.0
    assert 0.0 <= score.total_score <= 1.0