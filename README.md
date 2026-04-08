---
title: Meta OpenEnv Hackathon
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🔬 Autonomous Experiment Rescue Lab

> A seeded, partially observable benchmark where an AI agent diagnoses a hidden fault in a failing scientific experiment and recovers the run under strict time and budget constraints.

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-compliant-green)](https://github.com/openenv/openenv)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](./Dockerfile)
[![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen)](#testing)
[![Tasks](https://img.shields.io/badge/tasks-3%20with%203%20difficulties-orange)](#tasks)

---

## Why This Benchmark Matters

Modern AI agents are increasingly deployed in high-stakes operational settings — **laboratory automation, manufacturing quality control, robotics calibration, and chemical process monitoring**. All of these domains share a critical failure mode: experiments degrade silently, faults compound, and agents must act under uncertainty before total failure.

Existing benchmarks rarely model:
- **Diagnosis under noisy, partial signals** (you see sensors, not the root cause)
- **Budget-constrained interventions** (every costly action must count)
- **Recovery under time pressure** (12 steps to save a failing run)
- **Multi-cause failures** (hard mode introduces compound faults)

This benchmark fills that gap by modeling the complete **fault → diagnose → intervene → recover** cycle that appears constantly in real-world scientific and industrial systems.

**Who benefits:**
- **RL researchers** gain a structured, shaped reward environment with non-trivial exploration
- **LLM agent researchers** get a benchmark where chain-of-thought reasoning, tool use, and uncertainty management are rewarded
- **Industry teams** get a proxy task matching real operational decision-making

---

## Environment Overview

Each episode represents a single failing experiment. A **hidden fault** is seeded into the environment — the agent cannot see it directly. It must:

1. **Observe** noisy sensor telemetry and partial process signals  
2. **Diagnose** the fault using inspection and probing actions  
3. **Intervene** with the right corrective action before degradation becomes irreversible  
4. **Recover** the experiment to an acceptable quality level within the budget

The environment is **fully deterministic under seed** and **varied across resets**, making it ideal for reproducible evaluation and fair agent comparison.

---

## Tasks

### Task 1 — Fault Identification (`task_1`)

**Objective:** Infer the hidden fault type from noisy telemetry alone.

The agent must distinguish between 5 fault types using diagnostic actions:
- `drift` — gradual parameter drift
- `contamination` — sample or reagent contamination
- `overheating` — thermal runaway
- `miscalibration` — sensor offset/gain error
- `resource_depletion` — consumable exhaustion

**Grader emphasis (60%):** Diagnosis quality — did the agent correctly identify the fault, in how many steps, and with what confidence?

---

### Task 2 — Single Intervention Recovery (`task_2`)

**Objective:** Choose the single best corrective action to stabilize the experiment.

Given partial telemetry, the agent must apply the most effective intervention. The fault-to-action mapping is non-trivial and context-dependent:

| Fault | Optimal Action |
|---|---|
| Drift | `adjust_param_a` |
| Contamination | `discard_sample` |
| Overheating | `enable_safe_mode` |
| Miscalibration | `calibrate_sensor` |
| Resource Depletion | `pause_process` |

**Grader emphasis:** Recovery score (35%) + intervention bonus (15% if the last action is optimal for the fault type).

---

### Task 3 — Full Rescue Under Constraints (`task_3`)

**Objective:** Diagnose, probe, and recover the experiment using a limited 5–6 action budget across 12 steps.

This is the full benchmark task. It introduces:
- **Multi-fault episodes** (35% probability in hard mode)
- **Stricter budget** (only 5 interventions in hard)
- **Delayed effects** (agent must sequence actions correctly)
- **Compounding degradation** (each step the experiment drifts further)

**Grader weights:** Diagnosis 25% · Recovery 30% · Efficiency 15% · Safety 10% · Final Quality 20%.

---

## Difficulty Levels

| Parameter | Easy | Medium | Hard |
|---|---|---|---|
| Noise scale | 0.03–0.10 | 0.08–0.18 | 0.12–0.28 |
| Fault severity | 0.25–0.45 | 0.35–0.65 | 0.55–0.90 |
| Latent quality start | 0.70–0.90 | 0.55–0.82 | 0.35–0.72 |
| Stability margin start | 0.65–0.90 | 0.40–0.70 | 0.15–0.55 |
| Contamination level | 0.00–0.20 | 0.05–0.35 | 0.10–0.55 |
| Multi-fault prob | 0% | 10% | 35% |
| Action budget | 6 | 6 | 5 |

- **Easy:** Clear signals, single dominant fault, generous budget — ideal for initial agent development
- **Medium:** Meaningful noise, occasional distractors, resource tradeoffs — tests robust strategies
- **Hard:** Heavy partial observability, compound faults, high fault severity, minimal budget — challenges frontier models

---

## Action Space

10 typed actions with distinct cost, risk, and benefit profiles:

| Action | Budget Cost | Primary Effect | Best For |
|---|---|---|---|
| `inspect` | Free | Reduces diagnosis uncertainty | Any fault — safe, low-info gather |
| `run_diagnostic_probe` | 1 | Strong uncertainty reduction | When anomaly is high but cause is unclear |
| `calibrate_sensor` | 1 | Improves stability + quality | Miscalibration faults |
| `adjust_param_a` | 1 | Quality + stability boost (channel A) | Drift, Resource Depletion |
| `adjust_param_b` | 1 | Quality + stability boost (channel B) | Overheating, Contamination |
| `enable_safe_mode` | 1 | Lowers drift rate, safer dynamics | Overheating, volatile episodes |
| `pause_process` | 1 | Stability recovery | Resource depletion, tight margins |
| `continue_process` | Free | Quality gain if stable, risk if not | Only safe when stability > 0.70 |
| `discard_sample` | 1 | Removes contamination | Contamination faults |
| `restart_substage` | 1 | Quality + stability reset | Multi-fault or deeply degraded episodes |

**Key design choices:**
- `inspect` and `continue_process` are free — deliberation is always affordable
- Costly actions have fault-dependent outcomes — wrong intervention can hurt
- `continue_process` when stability is low accelerates fault severity — risky

---

## Observation Space

The agent receives a structured observation at each step:

```python
class Observation(BaseModel):
    task_id: str               # which task is running
    stage: Stage               # initialization → monitoring → degraded → recovery → terminal
    sensor_a: float            # primary quality proxy (0.0–1.0)
    sensor_b: float            # stability proxy (0.0–1.0)
    sensor_c: float            # contamination inverse proxy (0.0–1.0)
    rolling_mean: float        # mean of all sensors
    rolling_slope: float       # process trend (rising = degrading)
    volatility: float          # inter-sensor variance
    anomaly_score: float       # derived fault severity signal (0.0–1.0)
    diagnosis_uncertainty: float  # how confident the system is (lower = more certain)
    log_events: List[str]      # discrete events: anomaly_detected, needs_more_diagnostics, etc.
    steps_remaining: int       # episode time left
    budget_remaining: int      # interventions left
    available_actions: List[ActionType]
    previous_action: Optional[ActionType]
    last_reward: float
```

**Partial observability:** Sensors are noisy, fault severity and contamination levels are hidden. The agent cannot directly see the ground truth — it must infer from the observation structure.

---

## Reward Design

The reward signal is **dense and shaped** to support both RL training and LLM scoring:

```
reward = -0.01                           (per-step cost)
       + 0.50 × quality_delta            (reward improvement)
       + 0.45 × stability_delta          (reward stabilization)
       - 0.35 × severity_delta           (penalize worsening fault)
       - 0.25 × contamination_delta      (penalize spreading contamination)
       + 0.10 × [correct fault action]   (bonus for matching fault-optimal action)
       + 0.01 × [safe_mode_active]       (small safety bonus)
       + 1.25                            (terminal: RECOVERED)
       - 1.50                            (terminal: FAILED)
```

All rewards are normalized to `[0.0, 1.0]` via `(raw + 1.5) / 3.0`.

**Why this design works:**
- Dense signal prevents sparse-reward exploration failure
- Action correctness bonus guides agents toward fault-optimal choices
- Large terminal bonuses create strong intrinsic motivation
- Step cost discourages wasted inspections without progress
- Catastrophic failure penalty is asymmetric — safety matters

---

## Grading

All graders produce scores in `[0.0, 1.0]` and are **fully deterministic under seed**.

### GraderScore components

| Component | Weight in Task 3 | Description |
|---|---|---|
| Diagnosis score | 25% | Probe use, correct fault inference, confidence |
| Recovery score | 30% | Final quality, stability, fault severity reduction |
| Efficiency score | 15% | Steps used, budget remaining, repetition penalty |
| Safety score | 10% | Safe vs risky actions, avoidance of catastrophic failure |
| Final quality score | 20% | Latent quality + stability at episode end |

Task 1 reweights to 60% diagnosis. Task 2 adds a 15% intervention bonus. Task 3 uses the full balanced composition.

### Determinism guarantee

```bash
# Same seed + same policy = same score
python -c "
from server.environment import make_environment
from models import Action, ActionType

for seed in [42, 100, 999]:
    env = make_environment(task_id='task_3', difficulty='hard', seed=seed)
    obs = env.reset(seed=seed)
    result = env.step(Action(type=ActionType.INSPECT, args={}))
    print(f'seed={seed} reward={result.reward:.4f}')
"
```

---

## Experiment Types

The environment spans **5 real-world experiment domains**, selected probabilistically based on the fault type:

| Experiment Type | Associated Faults | Hidden Params |
|---|---|---|
| `chemistry` | Overheating, Contamination, Multi-fault | temperature, pressure, mixing |
| `materials` | Overheating, Contamination | heat, alignment, density |
| `calibration` | Miscalibration, Drift | offset, gain, drift_rate |
| `manufacturing` | Drift, Contamination, Resource Depletion | throughput, defect_rate, cycle_time |
| `robotics_sim` | Resource Depletion, Miscalibration | stability, precision, latency |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe → `{"status": "ok"}` |
| `GET` | `/metadata` | Environment version and current config |
| `POST` | `/reset` | Start episode: `{task_id, difficulty, seed}` |
| `POST` | `/step` | Take action: `{action: {type, args}}` |
| `GET` | `/state` | Machine-readable episode snapshot |
| `GET` | `/web` | Web UI status page |
| `GET` | `/docs` | Auto-generated OpenAPI docs |

---

## Repository Structure

```text
autonomous-experiment-rescue-lab/
├── README.md                          ← This file
├── openenv.yaml                       ← OpenEnv spec
├── pyproject.toml                     ← Package config + scripts
├── Dockerfile                         ← Root-level build
├── requirements.txt                   ← Runtime dependencies
├── inference.py                       ← Baseline agent (heuristic + LLM)
├── demo.py                            ← Interactive demo script
├── models.py                          ← Typed Pydantic models (Action, Observation, etc.)
├── client.py                          ← HTTP client wrapper
├── __init__.py                        ← Top-level package exports
├── tests/
│   ├── test_env_reset.py              ← Reset correctness
│   ├── test_env_step.py               ← Step transitions
│   ├── test_grader_determinism.py     ← Score reproducibility
│   ├── test_package_imports.py        ← Module importability
│   └── test_inference_smoke.py        ← Baseline script smoke test
└── server/
    ├── app.py                         ← FastAPI application
    ├── environment.py                 ← Core simulation engine (692 lines)
    ├── scenarios.py                   ← Seeded scenario generation
    ├── grader.py                      ← Deterministic scoring
    ├── reward.py                      ← Shaped reward functions
    ├── config.py                      ← Server configuration
    ├── Dockerfile                     ← Server-specific Docker build
    ├── requirements.txt               ← Server dependencies
    └── __init__.py
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/RuchitJagodara/Meta_OpenEnv_Hackathon.git
cd Meta_OpenEnv_Hackathon

# Using uv (recommended)
uv sync

# Or pip
pip install -r requirements.txt
```

### Run the server locally

```bash
# Via uv script entrypoint
uv run server

# Or directly
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Health check

```bash
curl http://localhost:7860/health
# → {"status": "ok"}
```

### Reset an episode

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_3", "difficulty": "hard", "seed": 42}'
```

### Take a step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"type": "inspect", "args": {}}}'
```

### Run the demo

```bash
# Runs a full episode with the heuristic policy and pretty-prints each step
python demo.py
```

### Run the baseline inference script

```bash
export HF_TOKEN="your_hf_token"
export ENV_BASE_URL="http://localhost:7860"
export MODEL_NAME="gpt-4.1-mini"    # or any OpenAI-compatible model

# Heuristic policy (no LLM required)
python inference.py

# LLM policy
USE_LLM_POLICY=1 python inference.py
```

---

## Baseline Agent

`inference.py` implements two policies:

### Heuristic Policy (default)
A hand-crafted rule-based agent that demonstrates correct reasoning patterns:
- Activates safe mode when sensor_b is critically low or volatility is high
- Uses diagnostic probes when anomaly is high but uncertainty remains high
- Selects parameter adjustment channel based on sensor signal patterns
- Falls back to inspection when uncertain

### LLM Policy (`USE_LLM_POLICY=1`)
An LLM-powered agent that receives a structured observation prompt and selects one of 10 action tokens. Supports any OpenAI-compatible endpoint via `API_BASE_URL`.

**Expected output format:**
```
[START] task=task_3 env=autonomous_experiment_rescue_lab model=gpt-4.1-mini
[STEP] step=1 action=enable_safe_mode reward=0.52 done=false error=null
[STEP] step=2 action=adjust_param_b reward=0.57 done=false error=null
...
[END] success=false steps=12 rewards=0.52,0.57,0.49,...
```

---

## Testing

```bash
# Run all 17 tests
pytest -v

# Individual test files
pytest tests/test_grader_determinism.py -v   # Reproducibility
pytest tests/test_env_reset.py -v            # State initialization
pytest tests/test_env_step.py -v             # Transition logic
pytest tests/test_package_imports.py -v      # Import coverage
```

All 17 tests pass. Grader determinism is explicitly validated — same seed always produces identical scores.

---

## Docker

```bash
# Build
docker build -t experiment-rescue-lab .

# Run
docker run -p 7860:7860 experiment-rescue-lab

# Verify
curl http://localhost:7860/health
```

---

## Validate Submission

```bash
./validate-submission.sh https://your-username-meta-openenv-hackathon.hf.space
```

Checks:
1. HF Space is live and responds to `/reset`
2. Docker image builds within timeout
3. `openenv validate` passes

---

## Environment Configuration

Key limits defined in `openenv.yaml`:

```yaml
limits:
  max_steps: 12
  initial_budget: 6
  runtime_seconds: 1200
  cpu_cores: 2
  memory_gb: 8
```

---

## Design Goals

| Goal | How it's achieved |
|---|---|
| **Deterministic under seed** | All RNG seeded from episode seed; same seed + same actions = identical episode |
| **Varied across resets** | Seed-derived scenario generation spans 5 experiment types × 6 fault types × 3 difficulties |
| **Dense, informative reward** | Shaped reward with 7 components; no sparse terminal-only signal |
| **Real-world grounding** | Observation names, fault types, and intervention names match real lab/industrial terminology |
| **Difficulty is meaningful** | Noise, severity, budget, and multi-fault probability all scale together; hard is genuinely hard |
| **Fair evaluation** | Deterministic graders, reproducible seeds, explicit score decomposition |
| **Agent-friendly** | 10-action discrete space, structured observations, log event strings for LLM context |

---

## What Makes This Novel

Most OpenEnv benchmarks model **task completion** — finish a coding problem, write a document, navigate a webpage. This benchmark instead models **failure recovery** — a fundamentally different challenge that real operational AI systems face every day.

**Unique mechanics:**
1. **Hidden fault type** — the problem statement is partially unobservable; the agent must diagnose before acting
2. **Fault-action coupling** — the *same action* has different outcomes depending on the hidden fault; naïve heuristics fail
3. **Budget-step duality** — time (steps) and cost (budget) are separate constraints, creating interesting tradeoffs
4. **Monotonic degradation** — the environment drifts toward failure on its own; inaction is costly
5. **Multi-fault mode** — hard difficulty can inject secondary faults, requiring agents that don't fixate on a single diagnosis
6. **5 experiment domains** — chemistry, materials, calibration, manufacturing, robotics all use the same interface, testing generalization

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENV_BASE_URL` | `https://...hf.space` | The environment server URL |
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | `gpt-4.1-mini` | Model to use for LLM policy |
| `HF_TOKEN` | *(required)* | HuggingFace token for LLM API authentication |
| `TASK_NAME` | `task_3` | Which task to run |
| `DIFFICULTY` | `hard` | Episode difficulty level |
| `SEED` | `52` | Episode seed |
| `USE_LLM_POLICY` | `0` | Set to `1` to enable LLM policy |
| `SUCCESS_SCORE_THRESHOLD` | `0.60` | Minimum average reward for `success=true` |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for the Meta × Hugging Face OpenEnv Hackathon 2026*
