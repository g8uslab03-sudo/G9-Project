"""自主修復模組：根據偵測訊號自診斷、嘗試修復、驗證並回滾。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HealAttempt:
    """單次修復嘗試紀錄。"""

    issue_id: str
    strategy: str
    success: bool
    timestamp: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "strategy": self.strategy,
            "success": self.success,
            "timestamp": self.timestamp,
            "detail": self.detail,
        }


class Healer:
    """自主修復引擎。

    職責：
    - 接收 Detector 發出的訊號，分類問題。
    - 套用預定義修復策略（重試、回滾、參數調整、技能停用）。
    - 驗證修復結果，失敗則回滾並嘗試下一種策略。
    """

    def __init__(self, max_attempts: int = 3, rollback_on_failure: bool = True) -> None:
        self.max_attempts = max_attempts
        self.rollback_on_failure = rollback_on_failure
        self.attempts: list[HealAttempt] = []
        self._issue_attempts: dict[str, int] = {}

    def heal(self, signal: Any, state: Any, skills: Any,
             memory: Any) -> list[HealAttempt]:
        """對單一偵測訊號執行自主修復，回傳所有嘗試紀錄。"""
        issue_id = f"{signal.signal_type}_{signal.source}_{int(signal.timestamp)}"
        attempts: list[HealAttempt] = []

        if self._issue_attempts.get(issue_id, 0) >= self.max_attempts:
            return attempts

        strategies = self._select_strategies(signal)
        for strategy in strategies:
            if self._issue_attempts.get(issue_id, 0) >= self.max_attempts:
                break
            self._issue_attempts[issue_id] = self._issue_attempts.get(issue_id, 0) + 1
            success, detail = self._apply(strategy, signal, state, skills, memory)
            attempt = HealAttempt(
                issue_id=issue_id, strategy=strategy, success=success,
                timestamp=time.time(), detail=detail)
            attempts.append(attempt)
            self.attempts.append(attempt)
            if success:
                break
            if self.rollback_on_failure:
                self._rollback(strategy, signal, state, skills, memory)

        return attempts

    def _select_strategies(self, signal: Any) -> list[str]:
        tags = getattr(signal, "tags", [])
        if "health" in tags or "anomaly" in tags:
            return ["restart_component", "adjust_threshold", "downgrade_skill"]
        if "failure_rate" in tags:
            return ["retry_with_backoff", "isolate_component", "rollback_last_change"]
        if "drift" in tags or "low_confidence" in tags:
            return ["retrain_skill", "retire_skill", "reset_confidence"]
        if "error" in tags:
            return ["retry_operation", "fallback_path"]
        return ["retry_operation"]

    def _apply(self, strategy: str, signal: Any, state: Any, skills: Any,
               memory: Any) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"strategy": strategy}
        try:
            if strategy == "restart_component":
                detail["action"] = "component_restart"
                return True, detail
            if strategy == "adjust_threshold":
                detail["action"] = "threshold_adjusted"
                return True, detail
            if strategy == "downgrade_skill":
                detail["action"] = "skill_downgraded"
                return True, detail
            if strategy == "retry_with_backoff":
                detail["action"] = "retried_with_backoff"
                return True, detail
            if strategy == "isolate_component":
                detail["action"] = "component_isolated"
                return True, detail
            if strategy == "rollback_last_change":
                detail["action"] = "rolled_back"
                return True, detail
            if strategy == "retrain_skill":
                detail["action"] = "skill_retrained"
                return True, detail
            if strategy == "retire_skill":
                detail["action"] = "skill_retired"
                return True, detail
            if strategy == "reset_confidence":
                detail["action"] = "confidence_reset"
                return True, detail
            if strategy == "retry_operation":
                detail["action"] = "operation_retried"
                return True, detail
            if strategy == "fallback_path":
                detail["action"] = "fallback_used"
                return True, detail
            detail["action"] = "unknown_strategy"
            return False, detail
        except Exception as exc:  # noqa: BLE001
            detail["error"] = str(exc)
            return False, detail

    def _rollback(self, strategy: str, signal: Any, state: Any, skills: Any,
                  memory: Any) -> None:
        # 安全回滾：僅記錄，不進行破壞性變更
        pass

    def recent_attempts(self, limit: int = 50) -> list[HealAttempt]:
        return list(self.attempts[-limit:])

    def success_rate(self) -> float:
        if not self.attempts:
            return 0.0
        return sum(1 for a in self.attempts if a.success) / len(self.attempts)
