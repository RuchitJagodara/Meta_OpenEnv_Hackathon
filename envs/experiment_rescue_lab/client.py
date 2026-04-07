from __future__ import annotations

from typing import Optional

import requests

from envs.experiment_rescue_lab.models import (
    Action,
    Observation,
    StateSnapshot,
    StepResult,
)


class ExperimentRescueClient:
    """Thin HTTP client for the Autonomous Experiment Rescue Lab server.

    This wrapper keeps the baseline/inference code simple and ensures that
    the project has a clean separation between:
    - transport logic
    - environment logic
    - agent logic
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def metadata(self) -> dict:
        response = requests.get(f"{self.base_url}/metadata", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def reset(
        self,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> tuple[Observation, StateSnapshot]:
        payload = {
            "task_id": task_id,
            "difficulty": difficulty,
            "seed": seed,
        }
        # Remove nulls so the API receives a compact request.
        payload = {k: v for k, v in payload.items() if v is not None}

        response = requests.post(
            f"{self.base_url}/reset",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        observation = Observation.model_validate(data["observation"])
        state = StateSnapshot.model_validate(data["state"])
        return observation, state

    def step(self, action: Action) -> StepResult:
        payload = {"action": action.model_dump()}
        response = requests.post(
            f"{self.base_url}/step",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return StepResult.model_validate(data["result"])

    def state(self) -> StateSnapshot:
        response = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        return StateSnapshot.model_validate(response.json())

    def close(self) -> None:
        self._session.close()


def make_client(base_url: str, timeout: float = 30.0) -> ExperimentRescueClient:
    return ExperimentRescueClient(base_url=base_url, timeout=timeout)