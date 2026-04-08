from __future__ import annotations

import os
from typing import Optional

from client import ExperimentRescueClient
from models import Action, ActionType, Observation, StateSnapshot


ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://deleteduser-meta-openenv-hackathon.hf.space")
TASK_NAME = os.getenv("TASK_NAME", "task_1")
DIFFICULTY = os.getenv("DIFFICULTY", "easy")
SEED = int(os.getenv("SEED", "52"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))


def format_action(action: Action) -> str:
    if action.args:
        args = ", ".join(f"{k}={v}" for k, v in action.args.items())
        return f"{action.type.value}({args})"
    return action.type.value


def heuristic_policy(observation: Observation, last_action: Optional[ActionType]) -> ActionType:
    """Small deterministic policy for the demo.

    This is intentionally simple and readable. It is not the evaluation policy.
    """
    logs = set(observation.log_events)

    if "possible_contamination" in logs:
        return ActionType.DISCARD_SAMPLE

    if observation.diagnosis_uncertainty > 0.65:
        return ActionType.INSPECT

    if observation.anomaly_score > 0.70:
        return ActionType.RUN_DIAGNOSTIC_PROBE

    if observation.volatility > 0.55:
        return ActionType.ENABLE_SAFE_MODE

    if observation.sensor_a < 0.40:
        return ActionType.ADJUST_PARAM_A

    if observation.sensor_b < 0.40:
        return ActionType.ADJUST_PARAM_B

    if observation.sensor_c < 0.55:
        return ActionType.CALIBRATE_SENSOR

    if last_action == ActionType.RUN_DIAGNOSTIC_PROBE:
        return ActionType.CALIBRATE_SENSOR

    return ActionType.CONTINUE_PROCESS


def print_observation(prefix: str, obs: Observation) -> None:
    print(prefix)
    print(f"  task_id: {obs.task_id}")
    print(f"  stage: {obs.stage.value}")
    print(
        f"  sensors: a={obs.sensor_a:.3f}, b={obs.sensor_b:.3f}, c={obs.sensor_c:.3f}"
    )
    print(
        f"  trends: mean={obs.rolling_mean:.3f}, slope={obs.rolling_slope:.3f}, volatility={obs.volatility:.3f}"
    )
    print(
        f"  anomaly_score={obs.anomaly_score:.3f}, diagnosis_uncertainty={obs.diagnosis_uncertainty:.3f}"
    )
    print(f"  logs: {', '.join(obs.log_events) if obs.log_events else 'none'}")
    print(f"  steps_remaining: {obs.steps_remaining}")
    print(f"  budget_remaining: {obs.budget_remaining}")
    print(f"  available_actions: {[a.value for a in obs.available_actions]}")
    if obs.previous_action is not None:
        print(f"  previous_action: {obs.previous_action.value}")
    print(f"  last_reward: {obs.last_reward:.2f}")


def print_state(prefix: str, state: StateSnapshot) -> None:
    print(prefix)
    print(f"  episode_id: {state.episode_id}")
    print(f"  task_id: {state.task_id}")
    print(f"  step_count: {state.step_count}")
    print(f"  budget_remaining: {state.budget_remaining}")
    print(f"  stage: {state.stage.value}")
    print(f"  terminal_status: {state.terminal_status.value}")
    print(f"  visible_metrics: {state.visible_metrics}")
    print(f"  action_history: {[a.value for a in state.action_history]}")


def main() -> int:
    client = ExperimentRescueClient(base_url=ENV_BASE_URL)
    last_action: Optional[ActionType] = None
    total_reward = 0.0
    total_steps = 0

    print("=== Autonomous Experiment Rescue Lab Demo ===")
    print(f"Server URL: {ENV_BASE_URL}")
    print(f"Task: {TASK_NAME}")
    print(f"Difficulty: {DIFFICULTY}")
    print(f"Seed: {SEED}")
    print()

    try:
        print("Connecting to environment...")
        print("Health:", client.health())
        print()

        print("Resetting episode...")
        observation, state = client.reset(task_id=TASK_NAME, difficulty=DIFFICULTY, seed=SEED)
        print_state("Initial state:", state)
        print()
        print_observation("Initial observation:", observation)
        print()

        for step in range(1, MAX_STEPS + 1):
            action_type = heuristic_policy(observation, last_action)
            action = Action(type=action_type, args={})
            total_steps += 1

            print(f"Step {step} chosen action: {format_action(action)}")

            try:
                result = client.step(action)
                observation = result.observation
                total_reward += float(result.reward or 0.0)
                last_action = action_type

                print(
                    f"  reward={result.reward:.2f}, done={str(result.done).lower()}, error=null"
                )
                print_observation("  next observation:", observation)
                print()

                if result.done:
                    break

            except Exception as exc:
                print(f"  reward=0.00, done=false, error={exc}")
                print()
                break

        final_state = client.state()
        print_state("Final state:", final_state)
        print()
        print(f"Average reward: {(total_reward/total_steps):.2f}")
        print("Demo finished successfully.")
        return 0

    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
