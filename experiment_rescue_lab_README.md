# Autonomous Experiment Rescue Lab

This environment is a **seeded, partially observable benchmark** for diagnosing a hidden fault in a failing experiment and recovering the run under **time and budget constraints**.

## Purpose

The agent is expected to:

- identify the most likely fault from noisy signals,
- choose useful diagnostic actions,
- apply corrective interventions,
- preserve safety and efficiency,
- maximize the final recovered experiment quality.

This environment aims to model a real operational workflow rather than a toy puzzle.

## Supported tasks

### Task 1 — Fault identification
Infer the hidden fault from telemetry and log events.

### Task 2 — Single-intervention recovery
Choose the best corrective action to stabilize the process.

### Task 3 — Full rescue under constraints
Diagnose, probe, and recover the experiment using limited steps and budget.

## Difficulty levels

### Easy
- clearer signal separation
- lower noise
- fewer hidden interactions
- more generous budget

### Medium
- moderate noise
- meaningful trade-offs
- tighter budget

### Hard
- partial observability
- multi-cause faults
- delayed effects
- stricter intervention budget

## Observation space

The agent may observe:

- `sensor_a`
- `sensor_b`
- `sensor_c`
- `rolling_mean`
- `rolling_slope`
- `volatility`
- `anomaly_score`
- `diagnosis_uncertainty`
- `log_events`
- `steps_remaining`
- `budget_remaining`
- `available_actions`
- `previous_action`
- `last_reward`

Observations are informative, but do not directly reveal the hidden fault.

## Action space

Supported actions:

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

## Reward structure

The reward combines:

- a small per-step cost,
- rewards for improved diagnosis,
- rewards for helpful interventions,
- penalties for risky or harmful actions,
- a terminal bonus for successful recovery,
- a terminal penalty for catastrophic failure.

## Determinism

Episodes are deterministic under a fixed seed:

- same seed + same actions → same episode
- different seed → different generated scenario

This supports reproducible grading, debugging, and fair evaluation.

## Package layout

- `models.py` — typed data models
- `client.py` — HTTP client wrapper
- `server/` — environment server implementation

## Running locally

Start the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Check health:

```bash
curl http://localhost:7860/health
```

Reset an episode:

```bash
curl -X POST http://localhost:7860/reset \
    -H "Content-Type: application/json" \
    -d '{"task_id":"task_3","difficulty":"medium","seed":42}'
```

## Design principles

This environment is designed to be:

- deterministic
- testable
- lightweight
- useful for agent evaluation
- easy to deploy
- novel enough for hackathon benchmarking

The benchmark emphasizes **diagnosis and recovery** rather than generic planning.
