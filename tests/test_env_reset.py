from __future__ import annotations

from envs.experiment_rescue_lab.server.environment import make_environment
from envs.experiment_rescue_lab.models import Stage, TerminalStatus


def test_reset_returns_valid_observation():
    env = make_environment(task_id="task_1", difficulty="easy", seed=123)
    obs = env.reset()

    assert obs.task_id == "task_1"
    assert obs.stage == Stage.INITIALIZATION
    assert 0.0 <= obs.sensor_a <= 1.0
    assert 0.0 <= obs.sensor_b <= 1.0
    assert 0.0 <= obs.sensor_c <= 1.0
    assert obs.steps_remaining == 12
    assert obs.budget_remaining == 6
    assert len(obs.available_actions) > 0
    assert obs.last_reward == 0.0


def test_reset_is_deterministic_for_same_seed():
    env1 = make_environment(task_id="task_2", difficulty="medium", seed=999)
    env2 = make_environment(task_id="task_2", difficulty="medium", seed=999)

    obs1 = env1.reset(seed=999)
    obs2 = env2.reset(seed=999)

    assert obs1.model_dump() == obs2.model_dump()
    assert env1.state().model_dump() == env2.state().model_dump()


def test_reset_changes_episode_state_after_new_seed():
    env = make_environment(task_id="task_3", difficulty="hard", seed=7)

    first = env.reset(seed=7)
    first_state = env.state()

    second = env.reset(seed=8)
    second_state = env.state()

    # Different seeds should produce different episodes in general.
    assert first.model_dump() != second.model_dump()
    assert first_state.episode_id != second_state.episode_id


def test_reset_clears_terminal_status():
    env = make_environment(task_id="task_1", difficulty="easy", seed=42)
    env.reset(seed=42)

    state = env.state()
    assert state.terminal_status == TerminalStatus.ACTIVE
    assert state.step_count == 0