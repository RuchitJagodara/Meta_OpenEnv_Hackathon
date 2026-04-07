from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from envs.experiment_rescue_lab.models import (
    Action,
    Observation,
    StateSnapshot,
    StepResult,
)
from envs.experiment_rescue_lab.server.environment import (
    ExperimentRescueEnvironment,
    make_environment,
)


app = FastAPI(
    title="Autonomous Experiment Rescue Lab",
    version="0.1.0",
    description=(
        "A seeded, partially observable benchmark where an agent diagnoses "
        "a hidden fault and rescues a failing experiment under constraints."
    ),
)

# Single environment instance for the Space/server process.
# The environment remains deterministic under seeds and is safe for a hackathon-grade demo.
ENV = make_environment()


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default=None, description="task_1, task_2, or task_3")
    difficulty: Optional[str] = Field(default=None, description="easy, medium, or hard")
    seed: Optional[int] = Field(default=None, description="Optional deterministic seed")


class StepRequest(BaseModel):
    action: Action


class ResetResponse(BaseModel):
    observation: Observation
    state: StateSnapshot


class StepResponse(BaseModel):
    result: StepResult
    state: StateSnapshot


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "name": "Autonomous Experiment Rescue Lab",
        "version": "0.1.0",
        "task_id": ENV.task_id,
        "difficulty": ENV.difficulty,
    }


@app.post("/reset", response_model=ResetResponse)
def reset_environment(request: ResetRequest) -> ResetResponse:
    try:
        if request.task_id is not None:
            ENV.task_id = request.task_id
        if request.difficulty is not None:
            ENV.difficulty = request.difficulty

        observation = ENV.reset(seed=request.seed)
        state = ENV.state()
        return ResetResponse(observation=observation, state=state)
    except Exception as exc:  # pragma: no cover - surfaced to Space logs
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step_environment(request: StepRequest) -> StepResponse:
    try:
        result = ENV.step(request.action)
        state = ENV.state()
        return StepResponse(result=result, state=state)
    except Exception as exc:  # pragma: no cover - surfaced to Space logs
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state", response_model=StateSnapshot)
def get_state() -> StateSnapshot:
    try:
        return ENV.state()
    except Exception as exc:  # pragma: no cover - surfaced to Space logs
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def root() -> dict:
    return {
        "message": "Autonomous Experiment Rescue Lab is running.",
        "routes": ["/health", "/metadata", "/reset", "/step", "/state"],
    }


# Optional local development entrypoint:
#   uvicorn envs.experiment_rescue_lab.server.app:app --host 0.0.0.0 --port 7860