"""自主進化模組：隨時間不斷成長、精進技能、調整策略與系統自身。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvolutionEvent:
    """單次進化事件紀錄。"""

    timestamp: float
    cycle: int
    event_type: str  # skill_added / skill_retired / param_adjusted / strategy_shift
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cycle": self.cycle,
            "event_type": self.event_type,
            "detail": self.detail,
        }


class Evolver:
    """自主進化引擎。

    職責：
    - 週期性檢視技能表現，淘汰低效技能，保留高效技能。
    - 調整學習參數（探索率、信心度門檻等）以適應環境。
    - 記錄進化軌跡，形成自我成長的閉環。
    """

    def __init__(self, retire_threshold: float = 0.3,
                 cycle_interval: int = 10) -> None:
        self.retire_threshold = retire_threshold
        self.cycle_interval = cycle_interval
        self.events: list[EvolutionEvent] = []
        self._last_evolution_cycle = 0

    def should_evolve(self, cycle: int) -> bool:
        return cycle - self._last_evolution_cycle >= self.cycle_interval

    def evolve(self, cycle: int, learner: Any, config: Any,
               memory: Any) -> list[EvolutionEvent]:
        """執行一次進化週期，回傳發生的變更。"""
        events: list[EvolutionEvent] = []
        now = time.time()

        # 淘汰低效技能
        retired = learner.retire_low_performers(self.retire_threshold)
        for sid in retired:
            events.append(EvolutionEvent(
                timestamp=now, cycle=cycle, event_type="skill_retired",
                detail={"skill_id": sid, "reason": "low_performance"}))
            self._log_memory(memory, f"retired skill {sid} due to low performance",
                             ["evolution", "retire"])

        # 調整探索率：若整體信心度偏低，提高探索；偏高則降低探索
        skills = learner.all_skills()
        if skills:
            avg_conf = sum(s.confidence for s in skills) / len(skills)
            if avg_conf < 0.4 and config.exploration_rate < 0.5:
                config.exploration_rate = round(config.exploration_rate + 0.05, 3)
                events.append(EvolutionEvent(
                    timestamp=now, cycle=cycle, event_type="param_adjusted",
                    detail={"param": "exploration_rate", "value": config.exploration_rate}))
            elif avg_conf > 0.8 and config.exploration_rate > 0.05:
                config.exploration_rate = round(config.exploration_rate - 0.05, 3)
                events.append(EvolutionEvent(
                    timestamp=now, cycle=cycle, event_type="param_adjusted",
                    detail={"param": "exploration_rate", "value": config.exploration_rate}))

        self._last_evolution_cycle = cycle
        self.events.extend(events)
        return events

    def record_shift(self, cycle: int, description: str,
                     detail: dict[str, Any], memory: Any) -> EvolutionEvent:
        now = time.time()
        event = EvolutionEvent(timestamp=now, cycle=cycle,
                               event_type="strategy_shift", detail={
                                   "description": description, **detail})
        self.events.append(event)
        self._log_memory(memory, description, ["evolution", "shift"])
        return event

    def _log_memory(self, memory: Any, message: str, tags: list[str]) -> None:
        if memory is None:
            return
        from autonomous_ai.memory import MemoryEntry
        entry = MemoryEntry(
            timestamp=time.time(), source="evolver",
            observation={"message": message}, action=None, result=None,
            confidence=1.0, outcome="info", tags=tags)
        memory.add_memory(entry)

    def recent_events(self, limit: int = 50) -> list[EvolutionEvent]:
        return list(self.events[-limit:])
