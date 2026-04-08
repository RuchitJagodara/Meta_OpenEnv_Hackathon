from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from models import (
    Action,
    ActionType,
    FaultType,
    GraderScore,
    HiddenState,
    TerminalStatus,
)
from server.reward import best_action_for_fault


@dataclass(frozen=True)
class GraderContext:
    episode_seed: int
    task_id: str
    difficulty: str
    hidden_fault: FaultType
    hidden_faults: List[FaultType]
    final_hidden_state: HiddenState


def _clamp01(value: float) -> float:
    return max(0.001, min(0.999, float(value)))


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.001
    return _clamp01((value - lo) / (hi - lo))


def _last_action(action_trace: Sequence[Action]) -> Optional[ActionType]:
    if not action_trace:
        return None
    return action_trace[-1].type


def _count_action(action_trace: Sequence[Action], action_type: ActionType) -> int:
    return sum(1 for action in action_trace if action.type == action_type)


def _diagnosis_score_from_trace(ctx: GraderContext, action_trace: Sequence[Action]) -> float:
    """
    Deterministic task-agnostic diagnosis score.

    We infer diagnosis quality from the trace:
    - more diagnostic actions improve the score,
    - repeated probing without useful intervention hurts efficiency,
    - the trace is scored more generously when the final state is stable.
    """
    probes = _count_action(action_trace, ActionType.RUN_DIAGNOSTIC_PROBE)
    inspections = _count_action(action_trace, ActionType.INSPECT)
    calibration = _count_action(action_trace, ActionType.CALIBRATE_SENSOR)

    base = 0.20
    base += 0.18 * min(2, probes)
    base += 0.08 * min(2, inspections)
    base += 0.06 * min(1, calibration)

    if ctx.final_hidden_state.inferred_fault == ctx.hidden_fault:
        base += 0.35

    if ctx.final_hidden_state.inferred_confidence > 0.75:
        base += 0.10

    if ctx.final_hidden_state.terminal_status == TerminalStatus.RECOVERED:
        base += 0.08

    return _clamp01(base)


def _recovery_score_from_final_state(ctx: GraderContext) -> float:
    """
    Score how well the episode was recovered.
    """
    h = ctx.final_hidden_state
    if h.terminal_status == TerminalStatus.FAILED:
        return 0.001

    quality_term = _normalize(h.latent_quality, 0.35, 0.95)
    stability_term = _normalize(h.stability_margin, 0.20, 0.90)
    severity_term = 1.0 - _normalize(h.fault_severity, 0.15, 0.95)
    contamination_term = 1.0 - _normalize(h.contamination_level, 0.10, 0.80)

    score = 0.40 * quality_term
    score += 0.25 * stability_term
    score += 0.20 * severity_term
    score += 0.15 * contamination_term

    if h.terminal_status == TerminalStatus.RECOVERED:
        score += 0.10

    return _clamp01(score)


def _efficiency_score_from_trace(ctx: GraderContext, action_trace: Sequence[Action]) -> float:
    """
    Reward shorter, cleaner traces that still accomplish the task.
    """
    h = ctx.final_hidden_state

    max_steps = max(1, h.max_steps)
    initial_budget = max(1, h.initial_budget)

    steps_used = len(action_trace)
    budget_used = max(0, h.initial_budget - h.budget_remaining)

    step_eff = 1.0 - _normalize(steps_used, 0, max_steps)
    budget_eff = 1.0 - _normalize(budget_used, 0, initial_budget)

    repeated_probes = max(0, _count_action(action_trace, ActionType.RUN_DIAGNOSTIC_PROBE) - 1)
    repetition_penalty = min(0.25, 0.08 * repeated_probes)

    score = 0.65 * step_eff + 0.35 * budget_eff
    score -= repetition_penalty

    return _clamp01(score)


def _safety_score_from_trace(ctx: GraderContext, action_trace: Sequence[Action]) -> float:
    """
    Penalize harmful or overly risky behavior.
    """
    h = ctx.final_hidden_state

    harmful_actions = [
        ActionType.CONTINUE_PROCESS,
        ActionType.RESTART_SUBSTAGE,
        ActionType.DISCARD_SAMPLE,
    ]

    risky_count = sum(1 for action in action_trace if action.type in harmful_actions)
    safe_count = sum(
        1 for action in action_trace
        if action.type in {ActionType.PAUSE_PROCESS, ActionType.ENABLE_SAFE_MODE, ActionType.INSPECT, ActionType.RUN_DIAGNOSTIC_PROBE}
    )

    base = 0.55 + 0.06 * min(3, safe_count)
    base -= 0.07 * min(3, risky_count)

    if h.terminal_status == TerminalStatus.FAILED:
        base -= 0.35

    if h.safe_mode_enabled:
        base += 0.06

    return _clamp01(base)


def _final_quality_score(ctx: GraderContext) -> float:
    h = ctx.final_hidden_state

    if h.terminal_status == TerminalStatus.FAILED:
        return 0.001

    quality = _normalize(h.latent_quality, 0.30, 0.95)
    stability = _normalize(h.stability_margin, 0.25, 0.95)

    score = 0.70 * quality + 0.30 * stability

    if h.terminal_status == TerminalStatus.RECOVERED:
        score += 0.10

    return _clamp01(score)


def _compose_total(
    diagnosis_score: float,
    recovery_score: float,
    efficiency_score: float,
    safety_score: float,
    final_quality_score: float,
) -> float:
    total = (
        0.25 * diagnosis_score
        + 0.30 * recovery_score
        + 0.15 * efficiency_score
        + 0.10 * safety_score
        + 0.20 * final_quality_score
    )
    return _clamp01(total)


def score_task(ctx: GraderContext, action_trace: Sequence[Action]) -> GraderScore:
    """
    Deterministic scoring entrypoint for all tasks.

    The final state and trace are enough to produce reproducible scores.
    """
    diagnosis_score = _diagnosis_score_from_trace(ctx, action_trace)
    recovery_score = _recovery_score_from_final_state(ctx)
    efficiency_score = _efficiency_score_from_trace(ctx, action_trace)
    safety_score = _safety_score_from_trace(ctx, action_trace)
    final_quality_score = _final_quality_score(ctx)
    total_score = _compose_total(
        diagnosis_score,
        recovery_score,
        efficiency_score,
        safety_score,
        final_quality_score,
    )

    return GraderScore(
        diagnosis_score=diagnosis_score,
        recovery_score=recovery_score,
        efficiency_score=efficiency_score,
        safety_score=safety_score,
        final_quality_score=final_quality_score,
        total_score=total_score,
    )


def score_task_1(ctx: GraderContext, action_trace: Sequence[Action]) -> GraderScore:
    """
    Task 1 focuses most heavily on diagnosis quality.
    """
    base = score_task(ctx, action_trace)
    total = _clamp01(
        0.60 * base.diagnosis_score
        + 0.15 * base.efficiency_score
        + 0.10 * base.safety_score
        + 0.15 * base.final_quality_score
    )
    return base.model_copy(update={"total_score": total})


def score_task_2(ctx: GraderContext, action_trace: Sequence[Action]) -> GraderScore:
    """
    Task 2 emphasizes choosing the correct intervention.
    """
    base = score_task(ctx, action_trace)
    best_action = best_action_for_fault(ctx.hidden_fault)
    last_action = _last_action(action_trace)

    intervention_bonus = 0.0
    if last_action == best_action:
        intervention_bonus = 0.15
    elif last_action in {
        ActionType.CALIBRATE_SENSOR,
        ActionType.ENABLE_SAFE_MODE,
        ActionType.DISCARD_SAMPLE,
        ActionType.PAUSE_PROCESS,
    }:
        intervention_bonus = 0.08

    total = _clamp01(
        0.20 * base.diagnosis_score
        + 0.35 * base.recovery_score
        + 0.15 * base.efficiency_score
        + 0.10 * base.safety_score
        + 0.20 * base.final_quality_score
        + intervention_bonus
    )
    return base.model_copy(update={"total_score": total})


def score_task_3(ctx: GraderContext, action_trace: Sequence[Action]) -> GraderScore:
    """
    Task 3 is the full rescue benchmark and uses the main score composition.
    """
    return score_task(ctx, action_trace)


def make_grader_context(
    episode_seed: int,
    task_id: str,
    difficulty: str,
    hidden_fault: FaultType,
    hidden_faults: List[FaultType],
    final_hidden_state: HiddenState,
) -> GraderContext:
    return GraderContext(
        episode_seed=episode_seed,
        task_id=task_id,
        difficulty=difficulty,
        hidden_fault=hidden_fault,
        hidden_faults=hidden_faults,
        final_hidden_state=final_hidden_state,
    )