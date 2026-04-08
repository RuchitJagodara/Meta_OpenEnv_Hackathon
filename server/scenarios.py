from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Tuple

from models import (
    ExperimentType,
    FaultType,
    HiddenState,
    ScenarioSpec,
)


TASK_LIBRARY: Dict[str, Dict[str, object]] = {
    "task_1": {
        "name": "fault_identification",
        "allowed_faults": [
            FaultType.DRIFT,
            FaultType.MISCALIBRATION,
            FaultType.OVERHEATING,
            FaultType.CONTAMINATION,
            FaultType.RESOURCE_DEPLETION,
        ],
        "primary_goal": "diagnose",
    },
    "task_2": {
        "name": "single_intervention_recovery",
        "allowed_faults": [
            FaultType.DRIFT,
            FaultType.MISCALIBRATION,
            FaultType.OVERHEATING,
            FaultType.CONTAMINATION,
            FaultType.RESOURCE_DEPLETION,
        ],
        "primary_goal": "recover",
    },
    "task_3": {
        "name": "full_rescue_under_constraints",
        "allowed_faults": [
            FaultType.DRIFT,
            FaultType.MISCALIBRATION,
            FaultType.OVERHEATING,
            FaultType.CONTAMINATION,
            FaultType.RESOURCE_DEPLETION,
            FaultType.MULTI_FAULT,
        ],
        "primary_goal": "diagnose_and_recover",
    },
}


DIFFICULTY_SETTINGS: Dict[str, Dict[str, object]] = {
    "easy": {
        "max_steps": 12,
        "budget": 8,
        "noise_scale": (0.01, 0.05),
        "fault_severity": (0.10, 0.25),
        "latent_quality": (0.85, 0.98),
        "contamination_level": (0.00, 0.10),
        "stability_margin": (0.75, 0.95),
        "multi_fault_prob": 0.00,
    },
    "medium": {
        "max_steps": 12,
        "budget": 6,
        "noise_scale": (0.05, 0.15),
        "fault_severity": (0.35, 0.60),
        "latent_quality": (0.55, 0.80),
        "contamination_level": (0.05, 0.35),
        "stability_margin": (0.40, 0.70),
        "multi_fault_prob": 0.10,
    },
    "hard": {
        "max_steps": 12,
        "budget": 4,
        "noise_scale": (0.15, 0.40),
        "fault_severity": (0.70, 0.95),
        "latent_quality": (0.15, 0.45),
        "contamination_level": (0.30, 0.70),
        "stability_margin": (0.05, 0.35),
        "multi_fault_prob": 0.50,
    },
}


EXPERIMENT_TYPES_BY_FAULT: Dict[FaultType, List[ExperimentType]] = {
    FaultType.DRIFT: [ExperimentType.MANUFACTURING, ExperimentType.CALIBRATION, ExperimentType.ROBOTICS_SIM],
    FaultType.MISCALIBRATION: [ExperimentType.CALIBRATION, ExperimentType.ROBOTICS_SIM, ExperimentType.MANUFACTURING],
    FaultType.OVERHEATING: [ExperimentType.CHEMISTRY, ExperimentType.MATERIALS, ExperimentType.MANUFACTURING],
    FaultType.CONTAMINATION: [ExperimentType.CHEMISTRY, ExperimentType.MATERIALS, ExperimentType.MANUFACTURING],
    FaultType.RESOURCE_DEPLETION: [ExperimentType.ROBOTICS_SIM, ExperimentType.MANUFACTURING, ExperimentType.CALIBRATION],
    FaultType.MULTI_FAULT: [ExperimentType.CHEMISTRY, ExperimentType.MANUFACTURING, ExperimentType.ROBOTICS_SIM],
}


def _stable_episode_id(seed: int, task_id: str, difficulty: str) -> str:
    raw = f"{seed}:{task_id}:{difficulty}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return f"exp_{digest[:16]}"


def _rand_range(rng: random.Random, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return rng.uniform(lo, hi)


def _choose_experiment_type(rng: random.Random, fault_type: FaultType) -> ExperimentType:
    choices = EXPERIMENT_TYPES_BY_FAULT[fault_type]
    return choices[rng.randrange(len(choices))]


def _choose_hidden_params(rng: random.Random, experiment_type: ExperimentType) -> Dict[str, float]:
    if experiment_type == ExperimentType.CHEMISTRY:
        return {
            "temperature_target": rng.uniform(35.0, 85.0),
            "pressure_target": rng.uniform(0.9, 2.2),
            "mixing_target": rng.uniform(0.4, 0.95),
        }
    if experiment_type == ExperimentType.MATERIALS:
        return {
            "heat_target": rng.uniform(120.0, 260.0),
            "alignment_target": rng.uniform(0.2, 0.85),
            "density_target": rng.uniform(0.45, 0.95),
        }
    if experiment_type == ExperimentType.CALIBRATION:
        return {
            "offset_target": rng.uniform(-0.8, 0.8),
            "gain_target": rng.uniform(0.75, 1.25),
            "drift_target": rng.uniform(0.0, 0.2),
        }
    if experiment_type == ExperimentType.MANUFACTURING:
        return {
            "throughput_target": rng.uniform(0.55, 0.95),
            "defect_target": rng.uniform(0.0, 0.15),
            "cycle_time_target": rng.uniform(0.3, 0.9),
        }
    return {
        "stability_target": rng.uniform(0.70, 0.98),
        "precision_target": rng.uniform(0.60, 0.95),
        "latency_target": rng.uniform(0.10, 0.55),
    }


def _choose_noise_profile(rng: random.Random, noise_scale: float) -> Dict[str, float]:
    return {
        "sensor_a": max(0.01, noise_scale * rng.uniform(0.9, 1.3)),
        "sensor_b": max(0.01, noise_scale * rng.uniform(0.8, 1.4)),
        "sensor_c": max(0.01, noise_scale * rng.uniform(1.0, 1.6)),
        "log_noise": max(0.01, noise_scale * rng.uniform(0.7, 1.1)),
        "trend_noise": max(0.01, noise_scale * rng.uniform(0.6, 1.2)),
    }


def _maybe_add_secondary_fault(rng: random.Random, difficulty: str, primary: FaultType) -> List[FaultType]:
    if difficulty != "hard":
        return [primary]

    if rng.random() > 0.5:
        return [primary]

    candidates = [
        FaultType.DRIFT,
        FaultType.MISCALIBRATION,
        FaultType.OVERHEATING,
        FaultType.CONTAMINATION,
        FaultType.RESOURCE_DEPLETION,
    ]
    candidates = [f for f in candidates if f != primary]
    secondary = candidates[rng.randrange(len(candidates))]
    return [primary, secondary]


def build_scenario_spec(seed: int, task_id: str, difficulty: str) -> ScenarioSpec:
    if task_id not in TASK_LIBRARY:
        raise ValueError(f"Unknown task_id: {task_id}")
    if difficulty not in DIFFICULTY_SETTINGS:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    # Seed incorporates task_id and difficulty so each combination generates a totally unique initial state
    hashed_seed = int(hashlib.sha256(f"{seed}:{task_id}:{difficulty}".encode()).hexdigest()[:8], 16)
    rng = random.Random(hashed_seed)
    task_info = TASK_LIBRARY[task_id]
    difficulty_info = DIFFICULTY_SETTINGS[difficulty]

    allowed_faults: List[FaultType] = list(task_info["allowed_faults"])  # type: ignore[index]
    primary_fault = allowed_faults[rng.randrange(len(allowed_faults))]

    max_steps = int(difficulty_info["max_steps"])  # type: ignore[index]
    budget = int(difficulty_info["budget"])  # type: ignore[index]
    noise_scale = _rand_range(rng, difficulty_info["noise_scale"])  # type: ignore[index]
    target_quality = rng.uniform(0.78, 0.95) if difficulty == "easy" else rng.uniform(0.65, 0.90)

    hidden_faults = _maybe_add_secondary_fault(rng, difficulty, primary_fault)

    experiment_type = _choose_experiment_type(rng, primary_fault)

    return ScenarioSpec(
        task_id=task_id,
        difficulty=difficulty,
        experiment_type=experiment_type,
        fault_type=primary_fault,
        max_steps=max_steps,
        budget=budget,
        hidden_faults=hidden_faults,
        noise_scale=noise_scale,
        target_quality=target_quality,
    )


def build_hidden_state(seed: int, spec: ScenarioSpec) -> HiddenState:
    hashed_seed = int(hashlib.sha256(f"{seed}:{spec.task_id}:{spec.difficulty}".encode()).hexdigest()[:8], 16)
    rng = random.Random(hashed_seed + 9173)

    fault_severity_bounds = DIFFICULTY_SETTINGS[spec.difficulty]["fault_severity"]  # type: ignore[index]
    latent_quality_bounds = DIFFICULTY_SETTINGS[spec.difficulty]["latent_quality"]  # type: ignore[index]
    contamination_bounds = DIFFICULTY_SETTINGS[spec.difficulty]["contamination_level"]  # type: ignore[index]
    stability_bounds = DIFFICULTY_SETTINGS[spec.difficulty]["stability_margin"]  # type: ignore[index]

    fault_severity = _rand_range(rng, fault_severity_bounds)  # type: ignore[arg-type]
    latent_quality = _rand_range(rng, latent_quality_bounds)  # type: ignore[arg-type]
    contamination_level = _rand_range(rng, contamination_bounds)  # type: ignore[arg-type]
    stability_margin = _rand_range(rng, stability_bounds)  # type: ignore[arg-type]

    # Multi-fault episodes are harder and start slightly less stable.
    if len(spec.hidden_faults) > 1:
        fault_severity = min(1.0, fault_severity + rng.uniform(0.10, 0.20))
        latent_quality = max(0.0, latent_quality - rng.uniform(0.05, 0.12))
        stability_margin = max(0.0, stability_margin - rng.uniform(0.05, 0.15))

    hidden_state = HiddenState(
        episode_id=_stable_episode_id(seed, spec.task_id, spec.difficulty),
        seed=seed,
        experiment_type=spec.experiment_type,
        fault_type=spec.fault_type,
        fault_severity=fault_severity,
        latent_quality=latent_quality,
        true_target_params=_choose_hidden_params(rng, spec.experiment_type),
        noise_profile=_choose_noise_profile(rng, spec.noise_scale),
        contamination_level=contamination_level,
        stability_margin=stability_margin,
        step_count=0,
        max_steps=spec.max_steps,
        budget_remaining=spec.budget,
        initial_budget=spec.budget,
        safe_mode_enabled=False,
    )

    return hidden_state


def list_supported_tasks() -> List[str]:
    return list(TASK_LIBRARY.keys())


def list_supported_difficulties() -> List[str]:
    return list(DIFFICULTY_SETTINGS.keys())


def scenario_summary(spec: ScenarioSpec) -> Dict[str, object]:
    return {
        "task_id": spec.task_id,
        "difficulty": spec.difficulty,
        "experiment_type": spec.experiment_type.value,
        "fault_type": spec.fault_type.value,
        "hidden_faults": [fault.value for fault in spec.hidden_faults],
        "max_steps": spec.max_steps,
        "budget": spec.budget,
        "noise_scale": spec.noise_scale,
        "target_quality": spec.target_quality,
    }