from __future__ import annotations

import asyncio
import os
import textwrap
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI
from client import ExperimentRescueClient
from models import Action, ActionType, Observation, TerminalStatus


# -----------------------------------------------------------------------------
# Required configuration
# -----------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

HF_TOKEN_RAW = os.getenv("HF_TOKEN")
HF_TOKEN = HF_TOKEN_RAW.strip() if HF_TOKEN_RAW is not None else None

# Environment server URL for the OpenEnv HTTP endpoint.
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://deleteduser-meta-openenv-hackathon.hf.space")

TASK_NAME = os.getenv("TASK_NAME", "task_3")
BENCHMARK = os.getenv("BENCHMARK", "autonomous_experiment_rescue_lab")
DIFFICULTY = os.getenv("DIFFICULTY", "hard")
SEED = int(os.getenv("SEED", "52"))

MAX_STEPS = 12
USE_LLM_POLICY = os.getenv("USE_LLM_POLICY", "0") == "1"
SUCCESS_SCORE_THRESHOLD = 0.60


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a partially observed experiment-rescue benchmark.
    The environment exposes noisy sensor readings, trend features, and logs.
    Your job is to diagnose faults and recover the experiment safely.

    If asked to propose an action, respond with exactly one token from:
    inspect
    run_diagnostic_probe
    calibrate_sensor
    adjust_param_a
    adjust_param_b
    enable_safe_mode
    pause_process
    continue_process
    discard_sample
    restart_substage

    Return only one token.
    """
).strip()


@dataclass
class EpisodeResult:
    task_id: str
    difficulty: str
    total_reward: float
    steps: int
    score: float
    success: bool


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error is not None else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def action_to_string(action: Action) -> str:
    if action.args:
        args_str = ",".join(f"{k}={v}" for k, v in action.args.items())
        return f"{action.type.value}({args_str})"
    return action.type.value


def _action(action_type: ActionType) -> Action:
    return Action(type=action_type, args={})


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


def llm_policy_action(
    client: OpenAI,
    step: int,
    observation: Observation,
    last_reward: float,
    history: List[str],
) -> ActionType:
    user_prompt = textwrap.dedent(
        f"""
        Step: {step}
        Last reward: {last_reward:.2f}
        Task: {observation.task_id}
        Stage: {observation.stage.value}

        Sensors:
          sensor_a={observation.sensor_a:.3f}
          sensor_b={observation.sensor_b:.3f}
          sensor_c={observation.sensor_c:.3f}

        Trends:
          rolling_mean={observation.rolling_mean:.3f}
          rolling_slope={observation.rolling_slope:.3f}
          volatility={observation.volatility:.3f}

        Uncertainty:
          anomaly_score={observation.anomaly_score:.3f}
          diagnosis_uncertainty={observation.diagnosis_uncertainty:.3f}

        Logs: {observation.log_events}
        Available actions: {[a.value for a in observation.available_actions]}
        History:
        {chr(10).join(history[-4:]) if history else "None"}

        Return exactly one action token.
        """
    ).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=16,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip().lower()
        token = text.split()[0] if text else ""
    except Exception:
        token = ""

    mapping = {
        "inspect": ActionType.INSPECT,
        "run_diagnostic_probe": ActionType.RUN_DIAGNOSTIC_PROBE,
        "calibrate_sensor": ActionType.CALIBRATE_SENSOR,
        "adjust_param_a": ActionType.ADJUST_PARAM_A,
        "adjust_param_b": ActionType.ADJUST_PARAM_B,
        "enable_safe_mode": ActionType.ENABLE_SAFE_MODE,
        "pause_process": ActionType.PAUSE_PROCESS,
        "continue_process": ActionType.CONTINUE_PROCESS,
        "discard_sample": ActionType.DISCARD_SAMPLE,
        "restart_substage": ActionType.RESTART_SUBSTAGE,
    }
    return mapping.get(token, ActionType.INSPECT)


async def run_episode(
    env_client: ExperimentRescueClient,
    llm_client: Optional[OpenAI],
    task_id: str,
    difficulty: str,
    seed: int,
) -> EpisodeResult:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    total_reward = 0.0
    score = 0.0
    success = False
    last_action: Optional[ActionType] = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation, _state = env_client.reset(task_id=task_id, difficulty=difficulty, seed=seed)
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if USE_LLM_POLICY and llm_client is not None:
                action_type = llm_policy_action(llm_client, step, observation, last_reward, history)
            else:
                action_type = heuristic_policy(observation, last_action)

            action = _action(action_type)

            try:
                result = env_client.step(action)
                obs = result.observation
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                error = None
            except Exception as exc:
                obs = observation
                reward = 0.0
                done = False
                error = str(exc)

            rewards.append(reward)
            total_reward += reward
            steps_taken = step
            last_reward = reward
            last_action = action_type

            log_step(step=step, action=action_to_string(action), reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_to_string(action)} -> reward {reward:.2f}, "
                f"anomaly={obs.anomaly_score:.3f}, uncertainty={obs.diagnosis_uncertainty:.3f}"
            )

            observation = obs

            if error is not None or done:
                break

        # Rewards are normalized to [0, 1], so the mean is a valid score.
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return EpisodeResult(
        task_id=task_id,
        difficulty=difficulty,
        total_reward=total_reward,
        steps=steps_taken,
        score=score,
        success=success,
    )


async def main() -> None:
    llm_client = None
    if USE_LLM_POLICY:
        if HF_TOKEN is None or not HF_TOKEN.strip():
            print("Warning: HF_TOKEN is missing but USE_LLM_POLICY=1. Attempting without token.", flush=True)
            hf_token = ""
        else:
            hf_token = HF_TOKEN.strip()
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=hf_token)

    env_client = ExperimentRescueClient(base_url=ENV_BASE_URL)

    try:
        try:
            _ = env_client.health()
        except Exception:
            pass

        await run_episode(
            env_client=env_client,
            llm_client=llm_client,
            task_id=TASK_NAME,
            difficulty=DIFFICULTY,
            seed=SEED,
        )
    finally:
        env_client.close()


if __name__ == "__main__":
    asyncio.run(main())