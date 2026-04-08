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
    """A single programmatic environment action.

    Action definitions intentionally mimic discrete operations in real-world
    experiment pipelines (triage, inspection, parameter tuning, sensor calibration).
    """

    type: ActionType = Field(description="The primary discrete categoric action type to execute.")
    args: Dict[str, float] = Field(
        default_factory=dict, 
        description="Optional continuous arguments to pass alongside the discrete action (e.g., target ranges)."
    )


class Observation(BaseModel):
    """The rich, dynamic state vector the agent observes at each discrete step.
    
    Contains multimodal signals including raw sensor floats, calculated trend heuristics,
    and unstructured log diagnostics to heavily test LLM synthesis capabilities.
    """

    task_id: str = Field(description="The active hackathon task constraint identifier.")
    stage: Stage = Field(description="The current operational stage of the experiment lifecycle.")
    sensor_a: float = Field(description="Noisy primary scalar metric (e.g., temperature/offset).")
    sensor_b: float = Field(description="Noisy secondary scalar metric (e.g., pressure/gain).")
    sensor_c: float = Field(description="Noisy tertiary scalar metric (e.g., density/latency).")
    rolling_mean: float = Field(description="Aggregated 3-step trailing average of sensor stability.")
    rolling_slope: float = Field(description="Calculated velocity (first derivative) of recent sensor deviations.")
    volatility: float = Field(description="Empirical variance calculation showing system turbulence.")
    anomaly_score: float = Field(description="An estimated AI-driven heuristic indicating presence of a latent fault.")
    diagnosis_uncertainty: float = Field(description="Confidence entropy associated with the anomaly classifier.")
    log_events: List[str] = Field(default_factory=list, description="Text-based discrete event logs acting as auxiliary signals.")
    sensors_degraded: bool = Field(default=False, description="True if severe fault cascades have permanently impaired sensor reliability.")
    steps_remaining: int = Field(description="Number of discrete steps remaining before forced truncation.")
    budget_remaining: int = Field(description="Operational credit limit. Actions consume budget.")
    available_actions: List[ActionType] = Field(default_factory=list, description="Currently permitted actions in this state.")
    previous_action: Optional[ActionType] = Field(default=None, description="The action taken in the immediately preceding transition.")
    last_reward: float = Field(default=0.0, description="The reward delta returned from the previous transition.")


class HiddenState(BaseModel):
    """Internal environment state used by the simulator and graders.

    This model is not fully exposed to the agent. It exists to support seeded
    scenario generation, deterministic transitions, and reproducible scoring.
    """

    episode_id: str
    seed: int
    experiment_type: ExperimentType
    fault_type: FaultType
    fault_severity: float = Field(ge=0.0, le=1.0, description="Hidden magnitude scalar of the active catastrophic fault.")
    latent_quality: float = Field(ge=0.0, le=1.0, description="The true unobserved sample quality score driving the primary reward.")
    true_target_params: Dict[str, float] = Field(default_factory=dict)
    noise_profile: Dict[str, float] = Field(default_factory=dict)
    contamination_level: float = Field(ge=0.0, le=1.0, description="Accumulated system contamination that worsens exponentially if unchecked.")
    stability_margin: float = Field(ge=0.0, le=1.0, description="System resilience buffer. If it reaches zero, a cascading failure occurs.")
    sensors_degraded: bool = Field(default=False, description="Whether cascading failures have permanently worsened noise profiles.")
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
    probes_used: int = Field(default=0, description="Tracks unique diagnostic milestones uncovered by the agent.")

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
