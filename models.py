from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ExperimentType(str, Enum):
    CHEMISTRY = "chemistry"
    MATERIALS = "materials"
    CALIBRATION = "calibration"
    MANUFACTURING = "manufacturing"
    ROBOTICS_SIM = "robotics_sim"


class FaultType(str, Enum):
    DRIFT = "drift"
    CONTAMINATION = "contamination"
    OVERHEATING = "overheating"
    MISCALIBRATION = "miscalibration"
    RESOURCE_DEPLETION = "resource_depletion"
    MULTI_FAULT = "multi_fault"


class Stage(str, Enum):
    INITIALIZATION = "initialization"
    MONITORING = "monitoring"
    DEGRADED = "degraded"
    RECOVERY = "recovery"
    TERMINAL = "terminal"


class TerminalStatus(str, Enum):
    ACTIVE = "active"
    RECOVERED = "recovered"
    FAILED = "failed"
    TRUNCATED = "truncated"


class ActionType(str, Enum):
    INSPECT = "inspect"
    RUN_DIAGNOSTIC_PROBE = "run_diagnostic_probe"
    CALIBRATE_SENSOR = "calibrate_sensor"
    ADJUST_PARAM_A = "adjust_param_a"
    ADJUST_PARAM_B = "adjust_param_b"
    ENABLE_SAFE_MODE = "enable_safe_mode"
    PAUSE_PROCESS = "pause_process"
    CONTINUE_PROCESS = "continue_process"
    DISCARD_SAMPLE = "discard_sample"
    RESTART_SUBSTAGE = "restart_substage"


class Action(BaseModel):
    """A single environment action.

    The environment keeps the action space intentionally small so that the
    benchmark remains easy to validate, easy to grade, and meaningful to learn.
    """

    type: ActionType
    args: Dict[str, float] = Field(default_factory=dict)


class Observation(BaseModel):
    """What the agent can observe at each step."""

    task_id: str
    stage: Stage
    sensor_a: float
    sensor_b: float
    sensor_c: float
    rolling_mean: float
    rolling_slope: float
    volatility: float
    anomaly_score: float
    diagnosis_uncertainty: float
    log_events: List[str] = Field(default_factory=list)
    steps_remaining: int
    budget_remaining: int
    available_actions: List[ActionType] = Field(default_factory=list)
    previous_action: Optional[ActionType] = None
    last_reward: float = 0.0


class HiddenState(BaseModel):
    """Internal environment state used by the simulator and graders.

    This model is not fully exposed to the agent. It exists to support seeded
    scenario generation, deterministic transitions, and reproducible scoring.
    """

    episode_id: str
    seed: int
    experiment_type: ExperimentType
    fault_type: FaultType
    fault_severity: float = Field(ge=0.0, le=1.0)
    latent_quality: float = Field(ge=0.0, le=1.0)
    true_target_params: Dict[str, float] = Field(default_factory=dict)
    noise_profile: Dict[str, float] = Field(default_factory=dict)
    contamination_level: float = Field(ge=0.0, le=1.0)
    stability_margin: float = Field(ge=0.0, le=1.0)
    step_count: int = 0
    max_steps: int = 12
    budget_remaining: int = 6
    initial_budget: int = 6
    safe_mode_enabled: bool = False
    terminal_status: TerminalStatus = TerminalStatus.ACTIVE
    inferred_fault: Optional[FaultType] = None
    inferred_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    current_stage: Stage = Stage.INITIALIZATION
    action_history: List[ActionType] = Field(default_factory=list)
    diagnostic_history: List[str] = Field(default_factory=list)

    def diagnosis_history_complexity(self) -> float:
        """Return a small proxy for how diagnostically complex the episode is."""
        complexity = 0.0
        if self.contamination_level > 0.25:
            complexity += 0.20
        if self.fault_severity > 0.65:
            complexity += 0.20
        if self.safe_mode_enabled:
            complexity += 0.05
        if len(self.diagnostic_history) >= 2:
            complexity += 0.10
        if len(self.diagnostic_history) >= 4:
            complexity += 0.10
        if self.inferred_fault is not None and self.inferred_fault != self.fault_type:
            complexity += 0.10
        return min(1.0, complexity)


class StateSnapshot(BaseModel):
    """A machine-readable snapshot for debugging and validation."""

    episode_id: str
    task_id: str
    step_count: int
    budget_remaining: int
    stage: Stage
    visible_metrics: Dict[str, float] = Field(default_factory=dict)
    action_history: List[ActionType] = Field(default_factory=list)
    terminal_status: TerminalStatus


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, object] = Field(default_factory=dict)


class GraderScore(BaseModel):
    diagnosis_score: float = Field(ge=0.0, le=1.0)
    recovery_score: float = Field(ge=0.0, le=1.0)
    efficiency_score: float = Field(ge=0.0, le=1.0)
    safety_score: float = Field(ge=0.0, le=1.0)
    final_quality_score: float = Field(ge=0.0, le=1.0)
    total_score: float = Field(ge=0.0, le=1.0)


class ScenarioSpec(BaseModel):
    """Seeded scenario template used to generate one episode."""

    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    experiment_type: ExperimentType
    fault_type: FaultType
    max_steps: int = 12
    budget: int = 6
    hidden_faults: List[FaultType] = Field(default_factory=list)
    noise_scale: float = Field(default=0.1, ge=0.0, le=1.0)
    target_quality: float = Field(default=0.85, ge=0.0, le=1.0)


class EnvironmentMetadata(BaseModel):
    """Top-level metadata useful for documentation and UI display."""

    name: str = "Autonomous Experiment Rescue Lab"
    version: str = "0.1.0"
    description: str = (
        "A seeded, partially observable benchmark where an agent diagnoses "
        "a hidden fault and rescues a failing experiment under constraints."
    )
    tasks: List[str] = Field(default_factory=lambda: ["task_1", "task_2", "task_3"])
