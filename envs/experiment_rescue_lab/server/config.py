from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class EnvironmentLimits:
    max_steps: int = 12
    initial_budget: int = 6
    cpu_cores: int = 2
    memory_gb: int = 8
    runtime_seconds: int = 1200


@dataclass(frozen=True)
class RewardWeights:
    step_cost: float = -0.01
    inspect_cost: float = -0.02
    diagnostic_probe_cost: float = -0.05
    calibration_cost: float = -0.04
    safe_mode_cost: float = -0.03
    discard_sample_cost: float = -0.08
    restart_substage_cost: float = -0.06
    irreversible_failure_penalty: float = -1.50
    successful_recovery_bonus: float = 1.25


@dataclass(frozen=True)
class ScoreWeights:
    diagnosis_score: float = 0.25
    recovery_score: float = 0.30
    efficiency_score: float = 0.15
    safety_score: float = 0.10
    final_quality_score: float = 0.20


TASK_SPECS: Dict[str, Dict[str, object]] = {
    "task_1": {
        "name": "fault_identification",
        "description": "Identify the hidden fault from noisy telemetry.",
        "primary_metric": "diagnosis_score",
    },
    "task_2": {
        "name": "single_intervention_recovery",
        "description": "Choose the best intervention to stabilize the process.",
        "primary_metric": "recovery_score",
    },
    "task_3": {
        "name": "full_rescue_under_constraints",
        "description": "Diagnose, probe, and recover the experiment under budget and time pressure.",
        "primary_metric": "total_score",
    },
}

DIFFICULTY_ORDER: List[str] = ["easy", "medium", "hard"]

SUPPORTED_FAULTS: List[str] = [
    "drift",
    "contamination",
    "overheating",
    "miscalibration",
    "resource_depletion",
    "multi_fault",
]

SUPPORTED_EXPERIMENT_TYPES: List[str] = [
    "chemistry",
    "materials",
    "calibration",
    "manufacturing",
    "robotics_sim",
]

SUPPORTED_ACTIONS: List[str] = [
    "inspect",
    "run_diagnostic_probe",
    "calibrate_sensor",
    "adjust_param_a",
    "adjust_param_b",
    "enable_safe_mode",
    "pause_process",
    "continue_process",
    "discard_sample",
    "restart_substage",
]


def default_limits() -> EnvironmentLimits:
    return EnvironmentLimits()


def default_reward_weights() -> RewardWeights:
    return RewardWeights()


def default_score_weights() -> ScoreWeights:
    return ScoreWeights()