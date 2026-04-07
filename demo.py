from __future__ import annotations

import os
from typing import Optional

from envs.experiment_rescue_lab.client import ExperimentRescueClient
from envs.experiment_rescue_lab.models import Action, ActionType, Observation


ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
TASK_NAME = os.getenv("TASK_NAME", "task_3")
DIFFICULTY = os.getenv("DIFFICULTY", "medium")
SEED = int(os.getenv("SEED", "42"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))


def heuristic_policy(observation: Observation, last_action: Optional[ActionType]) -> ActionType:
    sensor_a = observation.sensor_a
    sensor_b = observation.sensor_b
    sensor_c = observation.sensor_c
    anomaly = observation.anomaly_score
    uncertainty = observation.diagnosis_uncertainty
    slope = observation.rolling_slope
    volatility = observation.volatility
    logs = set(observation.log_events)

    if "possible_contamination" in logs:
        return ActionType.DISCARD_SAMPLE
    if "safe_mode_active" not in logs and (sensor_b < 0.45 or volatility > 0.55):
        return ActionType.ENABLE_SAFE_MODE
    if anomaly > 0.70 and uncertainty > 0.50:
        return ActionType.RUN_DIAGNOSTIC_PROBE
    if uncertainty > 0.65:
        return ActionType.INSPECT
    if sensor_a < 0.40 and slope > 0.20:
        return ActionType.ADJUST_PARAM_A
    if sensor_b < 0.40 and volatility > 0.35:
        return ActionType.ADJUST_PARAM_B
    if sensor_c < 0.55 and anomaly > 0.45:
        return ActionType.CALIBRATE_SENSOR
    if slope > 0.25 and sensor_a < 0.55:
        return ActionType.ADJUST_PARAM_A
    if sensor_b < 0.55 and sensor_c < 0.60:
        return ActionType.PAUSE_PROCESS
    if anomaly < 0.30 and uncertainty < 0.35:
        return ActionType.CONTINUE_PROCESS
    if last_action == ActionType.RUN_DIAGNOSTIC_PROBE:
        return ActionType.CALIBRATE_SENSOR
    if last_action == ActionType.CALIBRATE_SENSOR:
        return ActionType.ADJUST_PARAM_A
    return ActionType.INSPECT


def format_action(action: Action) -> str:
    if action.args:
        args_str = ",".join(f"{k}={v}" for k, v in action.args.items())
        return f"{action.type.value}({args_str})"
    return action.type.value


def main() -> int:
    client = ExperimentRescueClient(base_url=ENV_BASE_URL)

    print("=== Autonomous Experiment Rescue Lab Demo ===")
    print(f"Server: {ENV_BASE_URL}")
    print(f"Task: {TASK_NAME}")
    print(f"Difficulty: {DIFFICULTY}")
    print(f"Seed: {SEED}")

    try:
        print("Health:", client.health())
        observation, state = client.reset(task_id=TASK_NAME, difficulty=DIFFICULTY, seed=SEED)
        print("Initial state:", state.model_dump())
        print("Initial observation:", observation.model_dump())

        total_reward = 0.0
        last_action: Optional[ActionType] = None

        for step in range(1, MAX_STEPS + 1):
            action_type = heuristic_policy(observation, last_action)
            action = Action(type=action_type, args={})
            result = client.step(action)
            observation = result.observation
            total_reward += float(result.reward or 0.0)
            last_action = action_type

            print(
                f"Step {step}: action={format_action(action)} "
                f"reward={result.reward:.2f} done={str(result.done).lower()}"
            )

            if result.done:
                break

        final_state = client.state()
        print("Final state:", final_state.model_dump())
        print(f"Total normalized reward: {total_reward:.2f}")
        print("Demo complete.")
        return 0

    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
