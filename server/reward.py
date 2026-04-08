from __future__ import annotations

from typing import Dict, Tuple

from models import Action, ActionType, FaultType, HiddenState


def diagnosis_history_complexity(hidden: HiddenState) -> float:
    """Return a small complexity measure based on how much diagnostic evidence exists.

    Lower values mean the diagnosis is easier / more obvious.
    Higher values mean the episode has more competing signals.
    """
    complexity = 0.0

    if hidden.contamination_level > 0.25:
        complexity += 0.20
    if hidden.fault_severity > 0.65:
        complexity += 0.20
    if hidden.safe_mode_enabled:
        complexity += 0.05
    if len(hidden.diagnostic_history) >= 2:
        complexity += 0.10
    if len(hidden.diagnostic_history) >= 4:
        complexity += 0.10
    if hidden.inferred_fault is not None and hidden.inferred_fault != hidden.fault_type:
        complexity += 0.10

    return min(1.0, complexity)


def best_action_for_fault(fault: FaultType) -> ActionType:
    """Return the most likely best action for a given fault type."""
    if fault == FaultType.MISCALIBRATION:
        return ActionType.CALIBRATE_SENSOR
    if fault == FaultType.CONTAMINATION:
        return ActionType.DISCARD_SAMPLE
    if fault == FaultType.OVERHEATING:
        return ActionType.ENABLE_SAFE_MODE
    if fault == FaultType.DRIFT:
        return ActionType.ADJUST_PARAM_A
    if fault == FaultType.RESOURCE_DEPLETION:
        return ActionType.PAUSE_PROCESS
    return ActionType.RUN_DIAGNOSTIC_PROBE


def action_effect_profile(action: ActionType, hidden: HiddenState) -> Dict[str, float]:
    """Return a lightweight, deterministic estimate of the action's effect.

    The environment core can use this to keep reward shaping and transitions consistent.
    """
    profile = {
        "quality_delta": 0.0,
        "stability_delta": 0.0,
        "severity_delta": 0.0,
        "contamination_delta": 0.0,
        "budget_cost": 0.0,
        "risk_penalty": 0.0,
    }

    if action == ActionType.INSPECT:
        profile["budget_cost"] = 0.0
    elif action == ActionType.RUN_DIAGNOSTIC_PROBE:
        profile["budget_cost"] = 1.0
        profile["risk_penalty"] = 0.01
    elif action == ActionType.CALIBRATE_SENSOR:
        profile["budget_cost"] = 1.0
        if hidden.fault_type == FaultType.MISCALIBRATION:
            profile["quality_delta"] = 0.12
            profile["stability_delta"] = 0.10
            profile["severity_delta"] = -0.15
        else:
            profile["quality_delta"] = 0.03
            profile["stability_delta"] = 0.04
    elif action == ActionType.ADJUST_PARAM_A:
        profile["budget_cost"] = 1.0
        if hidden.fault_type in {FaultType.DRIFT, FaultType.RESOURCE_DEPLETION}:
            profile["quality_delta"] = 0.10
            profile["stability_delta"] = 0.08
            profile["severity_delta"] = -0.08
        else:
            profile["quality_delta"] = -0.03
            profile["stability_delta"] = -0.02
            profile["risk_penalty"] = 0.02
    elif action == ActionType.ADJUST_PARAM_B:
        profile["budget_cost"] = 1.0
        if hidden.fault_type in {FaultType.OVERHEATING, FaultType.CONTAMINATION}:
            profile["quality_delta"] = 0.10
            profile["stability_delta"] = 0.07
            profile["severity_delta"] = -0.08
        else:
            profile["quality_delta"] = -0.03
            profile["stability_delta"] = -0.02
            profile["risk_penalty"] = 0.02
    elif action == ActionType.ENABLE_SAFE_MODE:
        profile["budget_cost"] = 1.0
        profile["stability_delta"] = 0.10
        profile["risk_penalty"] = -0.03
    elif action == ActionType.PAUSE_PROCESS:
        profile["budget_cost"] = 1.0
        profile["stability_delta"] = 0.06
    elif action == ActionType.CONTINUE_PROCESS:
        profile["risk_penalty"] = 0.01
        if hidden.stability_margin > 0.70:
            profile["quality_delta"] = 0.03
        else:
            profile["quality_delta"] = -0.05
            profile["severity_delta"] = 0.04
    elif action == ActionType.DISCARD_SAMPLE:
        profile["budget_cost"] = 1.0
        profile["contamination_delta"] = -0.20
        profile["quality_delta"] = 0.05
        profile["severity_delta"] = -0.05
    elif action == ActionType.RESTART_SUBSTAGE:
        profile["budget_cost"] = 1.0
        profile["quality_delta"] = 0.08
        profile["stability_delta"] = 0.10
        profile["severity_delta"] = -0.08
        profile["contamination_delta"] = -0.08

    return profile


def shaped_reward(
    hidden: HiddenState,
    action: ActionType,
    previous_quality: float,
    previous_stability: float,
    previous_severity: float,
    previous_contamination: float,
) -> Tuple[float, Dict[str, float]]:
    """Compute a shaped reward for a transition.

    This should remain deterministic and easy to interpret.
    """
    profile = action_effect_profile(action, hidden)

    quality_delta = hidden.latent_quality - previous_quality
    stability_delta = hidden.stability_margin - previous_stability
    severity_delta = hidden.fault_severity - previous_severity
    contamination_delta = hidden.contamination_level - previous_contamination

    reward = -0.01  # per-step cost

    reward += max(-0.10, min(0.20, 0.50 * quality_delta))
    reward += max(-0.08, min(0.18, 0.45 * stability_delta))
    reward -= max(0.0, 0.35 * severity_delta)
    reward -= max(0.0, 0.25 * contamination_delta)

    if action == best_action_for_fault(hidden.fault_type):
        reward += 0.10

    if hidden.safe_mode_enabled:
        reward += 0.01

    if hidden.terminal_status.value == "recovered":
        reward += 1.25
    elif hidden.terminal_status.value == "failed":
        reward -= 1.50

    return reward, profile