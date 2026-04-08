from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from models import (
    Action,
    Observation,
    StateSnapshot,
    StepResult,
)
from server.environment import (
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
        "routes": ["/health", "/metadata", "/reset", "/step", "/state", "/web"],
    }


_WEB_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Autonomous Experiment Rescue Lab</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 60px auto; padding: 0 20px; color: #222; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
    p  {{ color: #555; margin-top: 0.25rem; }}
    ul {{ line-height: 2; }}
    code {{ background: #f3f3f3; padding: 2px 6px; border-radius: 4px; }}
    .badge {{ display: inline-block; background: #22c55e; color: #fff; border-radius: 99px; padding: 2px 12px; font-size: .8rem; margin-left: 8px; vertical-align: middle; }}
  </style>
</head>
<body>
  <h1>Autonomous Experiment Rescue Lab <span class="badge">online</span></h1>
  <p>A seeded, partially observable benchmark where an AI agent diagnoses a hidden fault and rescues a failing experiment.</p>
  <h2>API Endpoints</h2>
  <ul>
    <li><code>GET  /health</code> — Liveness probe</li>
    <li><code>GET  /metadata</code> — Environment metadata</li>
    <li><code>POST /reset</code> — Start a new episode</li>
    <li><code>POST /step</code> — Take one action</li>
    <li><code>GET  /state</code> — Current episode state</li>
    <li><code>GET  /docs</code> — Interactive OpenAPI docs</li>
  </ul>
  <h2>Quick start</h2>
  <pre><code>curl -X POST {base_url}/reset \\
  -H "Content-Type: application/json" \\
  -d '{{"task_id":"task_3","difficulty":"medium","seed":42}}'</code></pre>
</body>
</html>"""


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
@app.get("/web/", response_class=HTMLResponse, include_in_schema=False)
def web_interface(logs: Optional[str] = None) -> HTMLResponse:
    """Serve a simple web status page (used by HuggingFace Spaces and openenv-core)."""
    return HTMLResponse(content=_WEB_HTML.format(base_url=""), status_code=200)


# Optional local development entrypoint:
#   uvicorn server.app:app --host 0.0.0.0 --port 7860

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()