from __future__ import annotations

from models import Action, ActionType, Stage, TerminalStatus
from server.environment import make_environment


def test_step_advances_environment_state():
    env = make_environment(task_id="task_1", difficulty="easy", seed=123)
    obs0 = env.reset(seed=123)
    state0 = env.state()

    result = env.step(Action(type=ActionType.INSPECT, args={}))

    assert result.observation.task_id == "task_1"
    assert result.reward <= 1.5
    assert result.observation.previous_action == ActionType.INSPECT
    assert result.observation.last_reward == result.reward
    assert result.observation.steps_remaining == obs0.steps_remaining - 1
    assert env.state().step_count == state0.step_count + 1
    assert env.state().terminal_status in {
        TerminalStatus.ACTIVE,
        TerminalStatus.RECOVERED,
        TerminalStatus.FAILED,
        TerminalStatus.TRUNCATED,
    }


def test_step_returns_valid_observation_ranges():
    env = make_environment(task_id="task_2", difficulty="medium", seed=456)
    env.reset(seed=456)

    result = env.step(Action(type=ActionType.RUN_DIAGNOSTIC_PROBE, args={}))

    obs = result.observation
    assert 0.0 <= obs.sensor_a <= 1.0
    assert 0.0 <= obs.sensor_b <= 1.0
    assert 0.0 <= obs.sensor_c <= 1.0
    assert 0.0 <= obs.rolling_mean <= 1.0
    assert 0.0 <= obs.rolling_slope <= 1.0
    assert 0.0 <= obs.volatility <= 1.0
    assert 0.0 <= obs.anomaly_score <= 1.0
    assert 0.0 <= obs.diagnosis_uncertainty <= 1.0
    assert isinstance(obs.log_events, list)
    assert len(obs.available_actions) > 0


def test_step_is_deterministic_given_same_seed_and_action():
    env1 = make_environment(task_id="task_3", difficulty="hard", seed=777)
    env2 = make_environment(task_id="task_3", difficulty="hard", seed=777)

    env1.reset(seed=777)
    env2.reset(seed=777)

    action = Action(type=ActionType.ENABLE_SAFE_MODE, args={})

    res1 = env1.step(action)
    res2 = env2.step(action)

    assert res1.model_dump() == res2.model_dump()


def test_multiple_steps_do_not_break_state():
    env = make_environment(task_id="task_1", difficulty="easy", seed=999)
    env.reset(seed=999)

    actions = [
        Action(type=ActionType.INSPECT, args={}),
        Action(type=ActionType.RUN_DIAGNOSTIC_PROBE, args={}),
        Action(type=ActionType.CALIBRATE_SENSOR, args={}),
    ]

    last_result = None
    for action in actions:
        last_result = env.step(action)
        assert env.state().terminal_status in {
            TerminalStatus.ACTIVE,
            TerminalStatus.RECOVERED,
            TerminalStatus.FAILED,
            TerminalStatus.TRUNCATED,
        }
        if last_result.done:
            break

    assert last_result is not None
    assert env.state().stage in {
        Stage.INITIALIZATION,
        Stage.MONITORING,
        Stage.DEGRADED,
        Stage.RECOVERY,
        Stage.TERMINAL,
    }