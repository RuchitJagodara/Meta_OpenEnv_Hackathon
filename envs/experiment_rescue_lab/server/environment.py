from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from envs.experiment_rescue_lab.models import (
    Action,
    ActionType,
    FaultType,
    HiddenState,
    Observation,
    ScenarioSpec,
    Stage,
    StateSnapshot,
    StepResult,
    TerminalStatus,
)
from envs.experiment_rescue_lab.server.scenarios import (
    build_hidden_state,
    build_scenario_spec,
)


@dataclass
class _InternalBelief:
    anomaly_score: float = 0.0
    diagnosis_uncertainty: float = 1.0
    guessed_fault: Optional[FaultType] = None
    guessed_confidence: float = 0.0


class ExperimentRescueEnvironment:
    """Seeded, partially observable experiment-control environment."""

    def __init__(
        self,
        task_id: str = "task_3",
        difficulty: str = "medium",
        seed: Optional[int] = None,
    ) -> None:
        self.task_id = task_id
        self.difficulty = difficulty
        self.master_seed = 0 if seed is None else int(seed)

        self._episode_index = 0
        self._rng = random.Random(self.master_seed)

        self.spec: Optional[ScenarioSpec] = None
        self.hidden: Optional[HiddenState] = None
        self._belief = _InternalBelief()
        self._last_reward = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Observation:
        """Start a new episode and return the initial observation."""
        if seed is not None:
            episode_seed = int(seed)
        else:
            episode_seed = self._derive_episode_seed()

        self._rng = random.Random(episode_seed)

        self.spec = build_scenario_spec(
            seed=episode_seed,
            task_id=self.task_id,
            difficulty=self.difficulty,
        )
        self.hidden = build_hidden_state(seed=episode_seed, spec=self.spec)
        self._belief = _InternalBelief()
        self._last_reward = 0.0

        self.hidden.current_stage = Stage.INITIALIZATION
        self.hidden.terminal_status = TerminalStatus.ACTIVE
        self.hidden.step_count = 0
        self.hidden.action_history.clear()
        self.hidden.diagnostic_history.clear()

        return self._make_observation(previous_action=None, last_reward=0.0)

    def step(self, action: Action) -> StepResult:
        """Apply one action and advance the simulation by one step."""
        self._ensure_ready()
        assert self.hidden is not None
        assert self.spec is not None

        if self.hidden.terminal_status != TerminalStatus.ACTIVE:
            obs = self._make_observation(previous_action=action.type, last_reward=0.0)
            return StepResult(
                observation=obs,
                reward=0.0,
                done=True,
                info={"terminal_status": self.hidden.terminal_status.value},
            )

        self.hidden.step_count += 1
        self.hidden.action_history.append(action.type)

        reward, info = self._apply_action(action)

        self._update_process_dynamics(action)

        done = self._check_terminal_conditions()
        if done:
            reward += self._terminal_reward()

        reward = self._normalize_reward(reward)
        info["terminal_status"] = self.hidden.terminal_status.value

        self._last_reward = reward
        obs = self._make_observation(previous_action=action.type, last_reward=reward)

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> StateSnapshot:
        """Return a safe machine-readable snapshot for debugging and validation."""
        self._ensure_ready()
        assert self.hidden is not None
        assert self.spec is not None

        # Keep this intentionally non-leaky: no latent quality, fault severity,
        # contamination level, or stability margin here.
        visible_metrics = {
            "steps_remaining": float(max(0, self.hidden.max_steps - self.hidden.step_count)),
            "budget_remaining": float(self.hidden.budget_remaining),
            "safe_mode_enabled": 1.0 if self.hidden.safe_mode_enabled else 0.0,
            "diagnostic_count": float(len(self.hidden.diagnostic_history)),
            "action_count": float(len(self.hidden.action_history)),
        }

        return StateSnapshot(
            episode_id=self.hidden.episode_id,
            task_id=self.task_id,
            step_count=self.hidden.step_count,
            budget_remaining=self.hidden.budget_remaining,
            stage=self.hidden.current_stage,
            visible_metrics=visible_metrics,
            action_history=list(self.hidden.action_history),
            terminal_status=self.hidden.terminal_status,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_ready(self) -> None:
        if self.spec is None or self.hidden is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

    def _derive_episode_seed(self) -> int:
        self._episode_index += 1
        return self.master_seed + 10_000 * self._episode_index + self._rng.randrange(1_000_000)

    def _make_observation(
        self,
        previous_action: Optional[ActionType],
        last_reward: float,
    ) -> Observation:
        assert self.hidden is not None
        assert self.spec is not None

        sensor_a, sensor_b, sensor_c = self._simulate_sensors()
        rolling_mean, rolling_slope, volatility = self._simulate_trends(sensor_a, sensor_b, sensor_c)
        anomaly_score = self._compute_anomaly_score(sensor_a, sensor_b, sensor_c, volatility)
        diagnosis_uncertainty = self._compute_diagnosis_uncertainty(anomaly_score)

        log_events = self._make_log_events(anomaly_score, diagnosis_uncertainty)
        available_actions = self._available_actions()

        return Observation(
            task_id=self.task_id,
            stage=self.hidden.current_stage,
            sensor_a=sensor_a,
            sensor_b=sensor_b,
            sensor_c=sensor_c,
            rolling_mean=rolling_mean,
            rolling_slope=rolling_slope,
            volatility=volatility,
            anomaly_score=anomaly_score,
            diagnosis_uncertainty=diagnosis_uncertainty,
            log_events=log_events,
            steps_remaining=max(0, self.hidden.max_steps - self.hidden.step_count),
            budget_remaining=self.hidden.budget_remaining,
            available_actions=available_actions,
            previous_action=previous_action,
            last_reward=last_reward,
        )

    def _simulate_sensors(self) -> Tuple[float, float, float]:
        assert self.hidden is not None
        assert self.spec is not None

        rng = self._step_rng()

        fault = self.hidden.fault_type
        severity = self.hidden.fault_severity
        quality = self.hidden.latent_quality
        contamination = self.hidden.contamination_level
        stability = self.hidden.stability_margin

        noise_a = self.hidden.noise_profile["sensor_a"]
        noise_b = self.hidden.noise_profile["sensor_b"]
        noise_c = self.hidden.noise_profile["sensor_c"]

        sensor_a = quality
        sensor_b = stability
        sensor_c = 1.0 - contamination

        if fault == FaultType.DRIFT:
            sensor_a -= 0.25 * severity
            sensor_b -= 0.10 * severity
            sensor_c += 0.05 * severity
        elif fault == FaultType.CONTAMINATION:
            sensor_a -= 0.35 * severity
            sensor_b -= 0.15 * severity
            sensor_c -= 0.40 * severity
        elif fault == FaultType.OVERHEATING:
            sensor_a -= 0.20 * severity
            sensor_b -= 0.40 * severity
            sensor_c -= 0.20 * severity
        elif fault == FaultType.MISCALIBRATION:
            sensor_a += 0.10 * severity
            sensor_b -= 0.20 * severity
            sensor_c += 0.08 * severity
        elif fault == FaultType.RESOURCE_DEPLETION:
            depletion_factor = 0.20 + 0.50 * severity
            sensor_a -= depletion_factor * 0.30
            sensor_b -= depletion_factor * 0.35
            sensor_c -= depletion_factor * 0.20
        elif fault == FaultType.MULTI_FAULT:
            sensor_a -= 0.28 * severity
            sensor_b -= 0.28 * severity
            sensor_c -= 0.25 * severity

        if len(self.spec.hidden_faults) > 1:
            secondary = self.spec.hidden_faults[1]
            sensor_a, sensor_b, sensor_c = self._apply_secondary_fault(
                secondary, severity, sensor_a, sensor_b, sensor_c
            )

        if self.hidden.safe_mode_enabled:
            sensor_b += 0.05
            sensor_c += 0.03
            sensor_a -= 0.02

        sensor_a += rng.uniform(-noise_a, noise_a)
        sensor_b += rng.uniform(-noise_b, noise_b)
        sensor_c += rng.uniform(-noise_c, noise_c)

        sensor_a = self._clamp01(sensor_a)
        sensor_b = self._clamp01(sensor_b)
        sensor_c = self._clamp01(sensor_c)

        return sensor_a, sensor_b, sensor_c

    def _apply_secondary_fault(
        self,
        fault: FaultType,
        severity: float,
        sensor_a: float,
        sensor_b: float,
        sensor_c: float,
    ) -> Tuple[float, float, float]:
        if fault == FaultType.DRIFT:
            sensor_a -= 0.10 * severity
            sensor_b -= 0.05 * severity
        elif fault == FaultType.CONTAMINATION:
            sensor_a -= 0.15 * severity
            sensor_c -= 0.15 * severity
        elif fault == FaultType.OVERHEATING:
            sensor_b -= 0.15 * severity
            sensor_c -= 0.08 * severity
        elif fault == FaultType.MISCALIBRATION:
            sensor_a += 0.05 * severity
            sensor_b -= 0.10 * severity
        elif fault == FaultType.RESOURCE_DEPLETION:
            sensor_a -= 0.08 * severity
            sensor_b -= 0.08 * severity
        return sensor_a, sensor_b, sensor_c

    def _simulate_trends(self, sensor_a: float, sensor_b: float, sensor_c: float) -> Tuple[float, float, float]:
        assert self.hidden is not None
        rng = self._step_rng()

        rolling_mean = (sensor_a + sensor_b + sensor_c) / 3.0
        base_slope = 0.15 * (1.0 - self.hidden.stability_margin) + 0.20 * self.hidden.fault_severity

        if self.hidden.safe_mode_enabled:
            base_slope *= 0.65

        rolling_slope = base_slope + rng.uniform(
            -self.hidden.noise_profile["trend_noise"],
            self.hidden.noise_profile["trend_noise"],
        )

        volatility = (
            abs(sensor_a - sensor_b)
            + abs(sensor_b - sensor_c)
            + abs(sensor_c - sensor_a)
        ) / 3.0

        volatility += 0.35 * self.hidden.contamination_level
        volatility += 0.20 * max(0.0, self.hidden.fault_severity - self.hidden.stability_margin)
        volatility += rng.uniform(
            -self.hidden.noise_profile["trend_noise"],
            self.hidden.noise_profile["trend_noise"],
        )
        volatility = max(0.0, volatility)

        return (
            self._clamp01(rolling_mean),
            self._clamp01(rolling_slope),
            self._clamp01(volatility),
        )

    def _compute_anomaly_score(self, sensor_a: float, sensor_b: float, sensor_c: float, volatility: float) -> float:
        assert self.hidden is not None
        baseline = 1.0 - self.hidden.latent_quality
        sensor_gap = (abs(sensor_a - sensor_b) + abs(sensor_b - sensor_c) + abs(sensor_c - sensor_a)) / 3.0
        score = 0.40 * baseline + 0.35 * sensor_gap + 0.25 * volatility
        if self.hidden.safe_mode_enabled:
            score *= 0.90
        return self._clamp01(score)

    def _compute_diagnosis_uncertainty(self, anomaly_score: float) -> float:
        assert self.hidden is not None
        uncertainty = 1.0 - anomaly_score
        uncertainty += 0.15 * (1.0 - self.hidden.stability_margin)
        if self.hidden.safe_mode_enabled:
            uncertainty *= 0.95
        return self._clamp01(uncertainty)

    def _make_log_events(self, anomaly_score: float, diagnosis_uncertainty: float) -> List[str]:
        assert self.hidden is not None

        events: List[str] = []
        if anomaly_score > 0.70:
            events.append("anomaly_detected")
        if diagnosis_uncertainty > 0.60:
            events.append("needs_more_diagnostics")
        if self.hidden.safe_mode_enabled:
            events.append("safe_mode_active")
        if self.hidden.contamination_level > 0.35:
            events.append("possible_contamination")
        if self.hidden.fault_severity > 0.65:
            events.append("process_degrading")
        if not events:
            events.append("within_normal_bounds")
        return events[:4]

    def _available_actions(self) -> List[ActionType]:
        assert self.hidden is not None

        actions = [
            ActionType.INSPECT,
            ActionType.CONTINUE_PROCESS,
            ActionType.PAUSE_PROCESS,
            ActionType.RUN_DIAGNOSTIC_PROBE,
        ]

        if self.hidden.budget_remaining > 0:
            actions.extend(
                [
                    ActionType.CALIBRATE_SENSOR,
                    ActionType.ADJUST_PARAM_A,
                    ActionType.ADJUST_PARAM_B,
                    ActionType.ENABLE_SAFE_MODE,
                    ActionType.DISCARD_SAMPLE,
                    ActionType.RESTART_SUBSTAGE,
                ]
            )

        if self.hidden.budget_remaining <= 1:
            actions = [a for a in actions if a not in {ActionType.RESTART_SUBSTAGE}]

        seen = set()
        out: List[ActionType] = []
        for a in actions:
            if a not in seen:
                out.append(a)
                seen.add(a)
        return out

    def _apply_action(self, action: Action) -> Tuple[float, Dict[str, object]]:
        assert self.hidden is not None
        assert self.spec is not None

        reward = -0.01
        info: Dict[str, object] = {"action": action.type.value}

        _ = action.args

        if action.type == ActionType.INSPECT:
            reward -= 0.02
            self._belief.anomaly_score = min(1.0, self._belief.anomaly_score + 0.10)
            self._belief.diagnosis_uncertainty = max(0.0, self._belief.diagnosis_uncertainty - 0.08)
            self.hidden.diagnostic_history.append("inspect")
            info["diagnostic_gain"] = 0.10

        elif action.type == ActionType.RUN_DIAGNOSTIC_PROBE:
            if self.hidden.budget_remaining > 0:
                self.hidden.budget_remaining -= 1
                reward -= 0.05
                gain = 0.16 if self.spec.difficulty == "easy" else 0.12 if self.spec.difficulty == "medium" else 0.08
                self._belief.anomaly_score = min(1.0, self._belief.anomaly_score + gain)
                self._belief.diagnosis_uncertainty = max(0.0, self._belief.diagnosis_uncertainty - gain)
                self.hidden.diagnostic_history.append("probe")
                reward += 0.05
                info["diagnostic_gain"] = gain
            else:
                reward -= 0.10
                info["failed_reason"] = "no_budget"

        elif action.type == ActionType.CALIBRATE_SENSOR:
            reward += self._apply_calibration()
            info["intervention"] = "calibration"

        elif action.type == ActionType.ADJUST_PARAM_A:
            reward += self._apply_param_adjustment(channel="a")
            info["intervention"] = "adjust_param_a"

        elif action.type == ActionType.ADJUST_PARAM_B:
            reward += self._apply_param_adjustment(channel="b")
            info["intervention"] = "adjust_param_b"

        elif action.type == ActionType.ENABLE_SAFE_MODE:
            if self.hidden.budget_remaining > 0:
                self.hidden.budget_remaining -= 1
                self.hidden.safe_mode_enabled = True
                reward += 0.02
                info["intervention"] = "safe_mode"
            else:
                reward -= 0.10
                info["failed_reason"] = "no_budget"

        elif action.type == ActionType.PAUSE_PROCESS:
            reward += self._apply_pause()
            info["intervention"] = "pause"

        elif action.type == ActionType.CONTINUE_PROCESS:
            reward += self._apply_continue()
            info["intervention"] = "continue"

        elif action.type == ActionType.DISCARD_SAMPLE:
            if self.hidden.budget_remaining > 0:
                self.hidden.budget_remaining -= 1
                reward += self._apply_discard()
                info["intervention"] = "discard"
            else:
                reward -= 0.10
                info["failed_reason"] = "no_budget"

        elif action.type == ActionType.RESTART_SUBSTAGE:
            if self.hidden.budget_remaining > 0:
                self.hidden.budget_remaining -= 1
                reward += self._apply_restart_substage()
                info["intervention"] = "restart_substage"
            else:
                reward -= 0.10
                info["failed_reason"] = "no_budget"

        else:
            raise ValueError(f"Unsupported action: {action.type}")

        diag_bonus = self._diagnosis_bonus()
        reward += diag_bonus
        info["diagnosis_bonus"] = diag_bonus

        instability_penalty = max(0.0, 0.25 - self.hidden.stability_margin) * 0.05
        reward -= instability_penalty
        info["instability_penalty"] = instability_penalty

        return reward, info

    def _apply_calibration(self) -> float:
        assert self.hidden is not None

        if self.hidden.budget_remaining > 0:
            self.hidden.budget_remaining -= 1

        if self.hidden.fault_type == FaultType.MISCALIBRATION:
            self.hidden.stability_margin = min(1.0, self.hidden.stability_margin + 0.20)
            self.hidden.latent_quality = min(1.0, self.hidden.latent_quality + 0.08)
            self.hidden.fault_severity = max(0.0, self.hidden.fault_severity - 0.18)
            return 0.20

        self.hidden.stability_margin = min(1.0, self.hidden.stability_margin + 0.05)
        self.hidden.latent_quality = min(1.0, self.hidden.latent_quality + 0.02)
        return 0.04

    def _apply_param_adjustment(self, channel: str) -> float:
        assert self.hidden is not None
        if self.hidden.budget_remaining > 0:
            self.hidden.budget_remaining -= 1

        correct_channel = self._best_channel_for_fault(self.hidden.fault_type)
        if channel == correct_channel:
            self.hidden.latent_quality = min(1.0, self.hidden.latent_quality + 0.12)
            self.hidden.stability_margin = min(1.0, self.hidden.stability_margin + 0.10)
            self.hidden.fault_severity = max(0.0, self.hidden.fault_severity - 0.10)
            return 0.18

        self.hidden.latent_quality = max(0.0, self.hidden.latent_quality - 0.03)
        self.hidden.stability_margin = max(0.0, self.hidden.stability_margin - 0.02)
        return -0.06

    def _apply_pause(self) -> float:
        assert self.hidden is not None
        if self.hidden.budget_remaining > 0:
            self.hidden.budget_remaining -= 1

        self.hidden.stability_margin = min(1.0, self.hidden.stability_margin + 0.08)
        self.hidden.fault_severity = max(0.0, self.hidden.fault_severity - 0.03)
        return 0.05

    def _apply_continue(self) -> float:
        assert self.hidden is not None

        if self.hidden.stability_margin > 0.70:
            self.hidden.latent_quality = min(1.0, self.hidden.latent_quality + 0.04)
            return 0.03

        self.hidden.latent_quality = max(0.0, self.hidden.latent_quality - 0.05)
        self.hidden.fault_severity = min(1.0, self.hidden.fault_severity + 0.04)
        return -0.08

    def _apply_discard(self) -> float:
        assert self.hidden is not None

        self.hidden.contamination_level = max(0.0, self.hidden.contamination_level - 0.22)
        self.hidden.latent_quality = min(1.0, self.hidden.latent_quality + 0.06)
        self.hidden.fault_severity = max(0.0, self.hidden.fault_severity - 0.05)
        return 0.10

    def _apply_restart_substage(self) -> float:
        assert self.hidden is not None

        self.hidden.latent_quality = min(1.0, self.hidden.latent_quality + 0.10)
        self.hidden.stability_margin = min(1.0, self.hidden.stability_margin + 0.12)
        self.hidden.fault_severity = max(0.0, self.hidden.fault_severity - 0.08)
        self.hidden.contamination_level = max(0.0, self.hidden.contamination_level - 0.10)
        return 0.12

    def _diagnosis_bonus(self) -> float:
        assert self.hidden is not None

        inferred = self._infer_fault_from_state()
        confidence = self._estimate_confidence(inferred)

        self._belief.guessed_fault = inferred
        self._belief.guessed_confidence = confidence

        if inferred == self.hidden.fault_type:
            bonus = 0.06 * confidence
            self.hidden.inferred_fault = inferred
            self.hidden.inferred_confidence = confidence
            return bonus

        return -0.02 * (1.0 - confidence)

    def _infer_fault_from_state(self) -> FaultType:
        assert self.hidden is not None

        if self.hidden.contamination_level > 0.40:
            return FaultType.CONTAMINATION
        if self.hidden.safe_mode_enabled and self.hidden.stability_margin < 0.55:
            return FaultType.OVERHEATING
        if self.hidden.fault_severity > 0.75 and len(self.spec.hidden_faults) > 1:
            return FaultType.MULTI_FAULT
        if self.hidden.fault_type == FaultType.MISCALIBRATION or self.hidden.current_stage == Stage.INITIALIZATION:
            return FaultType.MISCALIBRATION
        if self.hidden.stability_margin < 0.35:
            return FaultType.OVERHEATING
        if self.hidden.latent_quality < 0.50:
            return FaultType.DRIFT
        return self.hidden.fault_type

    def _estimate_confidence(self, inferred: FaultType) -> float:
        assert self.hidden is not None

        base = 0.55 + 0.20 * (1.0 - self.hidden.diagnosis_history_complexity())
        if inferred == self.hidden.fault_type:
            base += 0.20
        if self.hidden.safe_mode_enabled:
            base -= 0.05
        return self._clamp01(base)

    def _update_process_dynamics(self, action: Action) -> None:
        assert self.hidden is not None

        drift_rate = 0.02 + 0.08 * self.hidden.fault_severity

        if self.hidden.safe_mode_enabled:
            drift_rate *= 0.65

        self.hidden.latent_quality = max(0.0, self.hidden.latent_quality - drift_rate)
        self.hidden.stability_margin = max(0.0, self.hidden.stability_margin - drift_rate * 0.75)

        if self.hidden.fault_type == FaultType.DRIFT:
            self.hidden.fault_severity = min(1.0, self.hidden.fault_severity + 0.03)
        elif self.hidden.fault_type == FaultType.CONTAMINATION:
            self.hidden.contamination_level = min(1.0, self.hidden.contamination_level + 0.02)
        elif self.hidden.fault_type == FaultType.OVERHEATING:
            self.hidden.stability_margin = max(0.0, self.hidden.stability_margin - 0.03)
        elif self.hidden.fault_type == FaultType.RESOURCE_DEPLETION:
            self.hidden.fault_severity = min(1.0, self.hidden.fault_severity + 0.025)

        if self.hidden.fault_type == FaultType.MULTI_FAULT or len(self.spec.hidden_faults) > 1:
            self.hidden.fault_severity = min(1.0, self.hidden.fault_severity + 0.015)

        if action.type == ActionType.CONTINUE_PROCESS and self.hidden.stability_margin < 0.40:
            self.hidden.fault_severity = min(1.0, self.hidden.fault_severity + 0.05)

        if action.type == ActionType.PAUSE_PROCESS:
            self.hidden.stability_margin = min(1.0, self.hidden.stability_margin + 0.03)

        if self.hidden.step_count == 1:
            self.hidden.current_stage = Stage.MONITORING
        elif self.hidden.stability_margin < 0.40:
            self.hidden.current_stage = Stage.DEGRADED
        elif self.hidden.stability_margin < 0.65:
            self.hidden.current_stage = Stage.RECOVERY
        else:
            self.hidden.current_stage = Stage.MONITORING

        if self.hidden.latent_quality <= 0.02 or self.hidden.fault_severity >= 0.98:
            self.hidden.terminal_status = TerminalStatus.FAILED

    def _check_terminal_conditions(self) -> bool:
        assert self.hidden is not None

        if self.hidden.terminal_status == TerminalStatus.FAILED:
            self.hidden.current_stage = Stage.TERMINAL
            return True

        if self.hidden.step_count >= self.hidden.max_steps:
            if self.hidden.latent_quality >= 0.80 and self.hidden.stability_margin >= 0.65:
                self.hidden.terminal_status = TerminalStatus.RECOVERED
            else:
                self.hidden.terminal_status = TerminalStatus.TRUNCATED
            self.hidden.current_stage = Stage.TERMINAL
            return True

        if self.hidden.latent_quality >= 0.88 and self.hidden.stability_margin >= 0.75 and self.hidden.fault_severity <= 0.20:
            self.hidden.terminal_status = TerminalStatus.RECOVERED
            self.hidden.current_stage = Stage.TERMINAL
            return True

        return False

    def _terminal_reward(self) -> float:
        assert self.hidden is not None

        if self.hidden.terminal_status == TerminalStatus.RECOVERED:
            quality_bonus = self._clamp01(self.hidden.latent_quality)
            safety_bonus = self._clamp01(self.hidden.stability_margin)
            return 1.25 + 0.50 * quality_bonus + 0.25 * safety_bonus

        if self.hidden.terminal_status == TerminalStatus.FAILED:
            return -1.50

        quality_bonus = 0.25 * self._clamp01(self.hidden.latent_quality)
        return quality_bonus

    def _best_channel_for_fault(self, fault: FaultType) -> str:
        if fault in {FaultType.MISCALIBRATION}:
            return "a"
        if fault in {FaultType.DRIFT, FaultType.RESOURCE_DEPLETION}:
            return "a"
        if fault in {FaultType.OVERHEATING, FaultType.CONTAMINATION}:
            return "b"
        return "b"

    def _step_rng(self) -> random.Random:
        assert self.hidden is not None
        return random.Random(self.hidden.seed + 7919 * (self.hidden.step_count + 1))

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _normalize_reward(self, raw_reward: float) -> float:
        # Maps a reasonable raw reward range roughly from [-1.5, +1.5] to [0.0, 1.0]
        return self._clamp01((raw_reward + 1.5) / 3.0)


def make_environment(
    task_id: str = "task_3",
    difficulty: str = "medium",
    seed: Optional[int] = None,
) -> ExperimentRescueEnvironment:
    return ExperimentRescueEnvironment(task_id=task_id, difficulty=difficulty, seed=seed)