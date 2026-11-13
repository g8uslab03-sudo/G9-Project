"""核心編排器：驅動自主學習 / 檢測 / 修復 / 進化的閉環迴圈。"""

from __future__ import annotations

import time
from typing import Any

from autonomous_ai.config import AutonomousConfig
from autonomous_ai.detector import Detector
from autonomous_ai.evolver import Evolver
from autonomous_ai.guardrails import Guardrails
from autonomous_ai.learner import Learner
from autonomous_ai.memory import MemoryEntry, MemoryStore


class AutonomousCore:
    """全自主系統核心。

    迴圈：
    1. 觀察環境 / 內部狀態。
    2. 學習：以經驗歸納技能，以湯普森採樣選擇動作。
    3. 執行動作（守護在護欄內）。
    4. 檢測：評估健康、偵測異常。
    5. 修復：針對偵測訊號自診斷並修復。
    6. 進化：定期淘汰低效技能、調整參數。
    7. 持久化狀態，進入下一週期。
    """

    def __init__(self, config: AutonomousConfig | None = None) -> None:
        self.config = config or AutonomousConfig()
        self.memory = MemoryStore(self.config.memory_path, self.config.state_path)
        self.learner = Learner(exploration_rate=self.config.exploration_rate,
                               min_confidence=self.config.min_confidence_to_apply)
        self.detector = Detector(anomaly_threshold=self.config.anomaly_threshold,
                                 health_interval=self.config.health_check_interval_seconds)
        self.healer = Healer(max_attempts=self.config.max_heal_attempts_per_issue,
                             rollback_on_failure=self.config.rollback_on_heal_failure)
        self.evolver = Evolver(retire_threshold=self.config.skill_retire_threshold,
                               cycle_interval=self.config.evolution_cycle_interval)
        self.guardrails = Guardrails(
            forbidden_operations=self.config.forbidden_operations,
            max_state_mutation_per_cycle=self.config.max_state_mutation_per_cycle,
        )
        self._running = False
        self._cycle = 0
        self._start_time = time.time()

    def observe_environment(self) -> dict[str, Any]:
        """自主觀察：收集系統與記憶中的訊號作為下一輪決策依據。"""
        state = self.memory.state
        recent = self.memory.recent_memories(limit=50)
        failures = [m for m in recent if m.outcome == "failure"]
        avg_conf = (
            sum(m.confidence for m in recent) / len(recent) if recent else 1.0
        )
        return {
            "cycle": self._cycle,
            "uptime": time.time() - self._start_time,
            "health_score": state.health_score,
            "recent_failures": len(failures),
            "avg_confidence": avg_conf,
            "total_observations": state.total_observations,
            "total_actions": state.total_actions,
            "skills": len(self.learner.all_skills()),
        }

    def run_cycle(self) -> dict[str, Any]:
        """執行單一自主週期。"""
        self.guardrails.begin_cycle()
        now = time.time()
        context = self.observe_environment()

        # 1. 學習與選擇動作
        action_taken = False
        if self.config.learning_enabled:
            skill = self.learner.select_skill(context)
            if skill is not None:
                action_taken = True
                operation = f"apply_skill:{skill.skill_id}"
                if self.guardrails.check_operation(operation, {"skill": skill.name}):
                    result = {"applied": True, "skill": skill.skill_id}
                    self._record_memory("observation", context, skill.action, result,
                                        skill.confidence, "success",
                                        ["learning", "action"])
                    self.learner.record_outcome(skill.skill_id, True)
                else:
                    self._record_memory("observation", context, skill.action, None,
                                        0.0, "blocked", ["guardrail", "blocked"])

        # 2. 檢測
        if self.config.detection_enabled:
            self.detector.check_health(
                self.memory.state, context["recent_failures"], context["avg_confidence"], now)
            self.detector.scan_anomalies(self.memory.recent_memories(limit=100), now)

        # 3. 修復
        if self.config.healing_enabled:
            for signal in self.detector.recent_signals(limit=10):
                attempts = self.healer.heal(signal, self.memory.state,
                                            self.learner, self.memory)
                if attempts:
                    self.memory.update_state(
                        total_heals=self.memory.state.total_heals + len(attempts))

        # 4. 進化
        if self.config.evolution_enabled and self.evolver.should_evolve(self._cycle):
            events = self.evolver.evolve(self._cycle, self.learner, self.config, self.memory)
            self.memory.update_state(
                total_evolutions=self.memory.state.total_evolutions + len(events))

        # 5. 狀態更新與持久化
        self._cycle += 1
        uptime = time.time() - self._start_time
        self.memory.update_state(
            cycle=self._cycle,
            uptime_seconds=uptime,
            health_score=self.memory.state.health_score,
            active_skills=[s.skill_id for s in self.learner.all_skills()],
            known_issues=[s.signal_type for s in self.detector.recent_signals(limit=5)],
        )
        self.memory.save()

        return {
            "cycle": self._cycle,
            "context": context,
            "action_taken": action_taken,
            "detections": len(self.detector.recent_signals()),
            "heals": len(self.healer.recent_attempts()),
            "evolutions": len(self.evolver.recent_events()),
            "health": self.memory.state.health_score,
            "guardrail_violations": self.guardrails.violation_count(),
        }

    def _record_memory(self, source: str, observation: dict[str, Any],
                       action: dict[str, Any] | None, result: dict[str, Any] | None,
                       confidence: float, outcome: str, tags: list[str]) -> None:
        entry = MemoryEntry(
            timestamp=time.time(), source=source, observation=observation,
            action=action, result=result, confidence=confidence,
            outcome=outcome, tags=tags)
        self.memory.add_memory(entry)
        self.learner.observe(observation, action, result, confidence, outcome, tags)

    def run(self, max_cycles: int | None = None) -> None:
        """啟動自主迴圈（阻塞）。"""
        self._running = True
        limit = max_cycles if max_cycles is not None else self.config.max_cycles
        try:
            while self._running:
                if limit is not None and self._cycle >= limit:
                    break
                self.run_cycle()
                time.sleep(self.config.cycle_interval_seconds)
        finally:
            self._running = False
            self.memory.save()

    def stop(self) -> None:
        self._running = False

    def status(self) -> dict[str, Any]:
        state = self.memory.state
        return {
            "cycle": self._cycle,
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time,
            "health_score": state.health_score,
            "total_observations": state.total_observations,
            "total_actions": state.total_actions,
            "total_heals": state.total_heals,
            "total_evolutions": state.total_evolutions,
            "skills": len(self.learner.all_skills()),
            "detections": len(self.detector.signals),
            "heal_attempts": len(self.healer.attempts),
            "evolution_events": len(self.evolver.events),
            "guardrail_violations": self.guardrails.violation_count(),
        }
