"""自主學習模組：從觀察與結果中歸納規則，建立可執行的技能原型。"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillPrototype:
    """從經驗中歸納出的技能候選。"""

    skill_id: str
    name: str
    condition: dict[str, Any]  # 觸發條件
    action: dict[str, Any]  # 執行動作
    expected_outcome: dict[str, Any]
    confidence: float
    usage_count: int = 0
    success_count: int = 0
    created_at: float = field(default_factory=lambda: __import__("time").time())
    updated_at: float = field(default_factory=lambda: __import__("time").time())

    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "condition": self.condition,
            "action": self.action,
            "expected_outcome": self.expected_outcome,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillPrototype":
        return cls(**data)


class Learner:
    """自主學習引擎。

    核心機制：
    1. 觀察環境狀態與動作結果。
    2. 以湯普森採樣評估技能信心度，決定是否採用。
    3. 從成功/失敗經驗中歸納新技能原型，並持續精進。
    """

    def __init__(self, exploration_rate: float = 0.15, min_confidence: float = 0.7) -> None:
        self.exploration_rate = exploration_rate
        self.min_confidence = min_confidence
        self.skills: dict[str, SkillPrototype] = {}
        self._rng = random.Random()

    def observe(self, observation: dict[str, Any], action: dict[str, Any] | None,
                result: dict[str, Any] | None, confidence: float, outcome: str,
                tags: list[str]) -> None:
        """記錄單筆學習樣本，並嘗試歸納新技能。"""
        if action is not None and result is not None:
            self._update_skill_from_experience(observation, action, result, outcome)
        if outcome == "success" and action is not None:
            self._maybe_derive_skill(observation, action, result, confidence, tags)

    def select_skill(self, context: dict[str, Any]) -> SkillPrototype | None:
        """以湯普森採樣從適用技能中選一個。"""
        applicable = [s for s in self.skills.values() if self._matches(context, s.condition)]
        if not applicable:
            return None
        # 探索與利用：以 Beta 分佈採樣信心度
        best: SkillPrototype | None = None
        best_score = -1.0
        for skill in applicable:
            alpha = skill.success_count + 1
            beta = skill.usage_count - skill.success_count + 1
            score = self._rng.betavariate(alpha, beta)
            if self._rng.random() < self.exploration_rate:
                score += self._rng.uniform(-0.1, 0.1)
            if score > best_score:
                best_score = score
                best = skill
        if best is not None and best.confidence >= self.min_confidence:
            best.usage_count += 1
            best.updated_at = __import__("time").time()
        return best if best is not None and best.confidence >= self.min_confidence else None

    def record_outcome(self, skill_id: str, success: bool) -> None:
        skill = self.skills.get(skill_id)
        if skill is None:
            return
        skill.usage_count += 1
        if success:
            skill.success_count += 1
        skill.confidence = self._compute_confidence(skill)
        skill.updated_at = __import__("time").time()

    def _compute_confidence(self, skill: SkillPrototype) -> float:
        if skill.usage_count == 0:
            return skill.confidence
        rate = skill.success_rate()
        # 信心度 = 成功率 × 樣本可信度（樣本越多越穩）
        reliability = min(1.0, skill.usage_count / 10.0)
        return round(rate * 0.7 + skill.confidence * 0.3 * reliability, 4)

    def _matches(self, context: dict[str, Any], condition: dict[str, Any]) -> bool:
        for key, value in condition.items():
            if key not in context:
                return False
            if context[key] != value:
                return False
        return True

    def _update_skill_from_experience(self, observation: dict[str, Any],
                                      action: dict[str, Any], result: dict[str, Any],
                                      outcome: str) -> None:
        skill_id = self._skill_id_from_action(action)
        skill = self.skills.get(skill_id)
        if skill is None:
            skill = SkillPrototype(
                skill_id=skill_id,
                name=f"skill_{skill_id}",
                condition=observation,
                action=action,
                expected_outcome=result,
                confidence=0.5,
            )
            self.skills[skill_id] = skill
        self.record_outcome(skill_id, outcome == "success")

    def _maybe_derive_skill(self, observation: dict[str, Any], action: dict[str, Any],
                            result: dict[str, Any], confidence: float,
                            tags: list[str]) -> None:
        skill_id = self._skill_id_from_action(action)
        if skill_id in self.skills:
            return
        if confidence < self.min_confidence:
            return
        self.skills[skill_id] = SkillPrototype(
            skill_id=skill_id,
            name=f"derived_{skill_id}",
            condition=observation,
            action=action,
            expected_outcome=result,
            confidence=round(confidence, 4),
        )

    def _skill_id_from_action(self, action: dict[str, Any]) -> str:
        raw = json_dumps_sorted(action)
        return "skill_" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    def retire_low_performers(self, threshold: float) -> list[str]:
        retired = [sid for sid, s in self.skills.items()
                   if s.usage_count >= 5 and s.confidence < threshold]
        for sid in retired:
            del self.skills[sid]
        return retired

    def all_skills(self) -> list[SkillPrototype]:
        return list(self.skills.values())


def json_dumps_sorted(obj: Any) -> str:
    import json
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)
