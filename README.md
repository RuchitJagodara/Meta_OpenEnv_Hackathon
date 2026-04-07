---
title: Meta OpenEnv Hackathon
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Autonomous Experiment Rescue Lab

A seeded, partially observable benchmark where an AI agent diagnoses a hidden fault in a failing experiment, chooses the right interventions, and recovers the run under time and budget constraints.

## Why this environment exists

This benchmark is designed to test a capability gap that matters in real systems:

- diagnosing failures from noisy signals,
- choosing interventions under uncertainty,
- recovering an ongoing process safely,
- balancing speed, cost, and risk.

The goal is to approximate real operational decision-making in a compact, deterministic environment.

---

## Core idea

Each episode represents one experiment run.

The agent observes partial telemetry and must decide whether to:
- inspect,
- probe,
- calibrate,
- adjust parameters,
- pause,
- continue,
- discard a sample,
- restart a substage,
- or switch to safe mode.

Behind the scenes, the environment contains a hidden fault and latent process dynamics. The agent must infer what is happening and act before the experiment becomes unusable.

---

## Tasks

### Task 1 — Fault Identification
Infer the hidden fault from noisy telemetry.

Primary objective:
- identify the correct failure mode.

### Task 2 — Single Intervention Recovery
Choose the best corrective action to stabilize the experiment.

Primary objective:
- pick the intervention that most improves process stability and quality.

### Task 3 — Full Rescue Under Constraints
Diagnose, probe, and recover the experiment using limited time and budget.

Primary objective:
- maximize final quality while avoiding catastrophic failure.

---

## Difficulty levels

### Easy
- clear signals,
- one dominant fault,
- low noise,
- generous budget.

### Medium
- noisier signals,
- some distractors,
- tighter budget,
- meaningful tradeoffs.

### Hard
- partial observability,
- multi-cause failures,
- delayed effects,
- risky interventions,
- stricter budget.

---

## Action space

The environment supports the following actions:

- `inspect`
- `run_diagnostic_probe`
- `calibrate_sensor`
- `adjust_param_a`
- `adjust_param_b`
- `enable_safe_mode`
- `pause_process`
- `continue_process`
- `discard_sample`
- `restart_substage`

Each action has a different cost, risk profile, and likely benefit depending on the hidden fault.

---

## Observation space

The agent receives:

- sensor readings,
- rolling trend summaries,
- anomaly score,
- diagnosis uncertainty,
- recent log events,
- remaining steps,
- remaining budget,
- available actions,
- previous action,
- last reward.

The observation is intentionally informative but not enough to directly reveal the hidden ground truth.

---

## Reward design

The reward is shaped to encourage learning rather than sparse trial-and-error only.

Typical components include:

- small step cost,
- reward for reducing uncertainty,
- reward for choosing a helpful intervention,
- reward for improving stability and quality,
- penalty for risky or harmful actions,
- large terminal bonus for successful recovery,
- severe penalty for catastrophic failure.

This makes the environment useful for both RL and agent evaluation.

---

## Grading

The benchmark uses deterministic graders that score episodes in the range `[0.0, 1.0]`.

The grading focuses on:

- diagnosis quality,
- recovery success,
- efficiency,
- safety,
- final quality.

Each task gets a different emphasis, with Task 3 acting as the full benchmark.

---

## Repository structure

```text
autonomous-experiment-rescue-lab/
├── README.md
├── requirements.txt
├── openenv.yaml
├── inference.py
├── Dockerfile
├── tests/
│   ├── test_env_reset.py
│   ├── test_env_step.py
│   ├── test_grader_determinism.py
│   └── test_inference_smoke.py
└── envs/
    └── experiment_rescue_lab/
        ├── __init__.py
        ├── models.py
        ├── client.py
        ├── README.md
        └── server/
            ├── __init__.py
            ├── environment.py
            ├── app.py
            ├── scenarios.py
            ├── reward.py
            ├── grader.py
            └── config.py

```

## Quickstart

### Installation

```bash
pip install -r requirements.txt
```

### Run locally (server)

```bash
uvicorn envs.experiment_rescue_lab.server.app:app --host 0.0.0.0 --port 7860
```

Health check:

```bash
curl http://localhost:7860/health
```

Reset an episode:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_3","difficulty":"medium","seed":42}'
```

Take a step:

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"type":"inspect","args":{}}}'
```

## Baseline inference (`inference.py`)

The root-level `inference.py` script is expected to:

- read `API_BASE_URL`
- read `MODEL_NAME`
- read `HF_TOKEN` (if needed)
- connect to the environment
- run each task
- print reproducible scores

## Design goals

- deterministic under seed
- varied across resets
- easy to validate (tests + deterministic grading)
- lightweight (modest CPU/memory)
- useful for both training and evaluation
- novel, failure-focused benchmark for hackathon settings

## What makes it different

Most benchmarks emphasize:

- question answering
- generic planning/scheduling
- general assistant behavior

This benchmark emphasizes:

- diagnosis under partial observability
- recovery via safe interventions
- robust decision-making under uncertainty
- cost/risk tradeoffs in a failing, ongoing process
