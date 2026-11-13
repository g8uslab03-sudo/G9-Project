"""安全護欄：約束全自主系統的行為邊界，防止越界變更。"""

from __future__ import annotations

from typing import Any


class Guardrails:
    """安全護欄。

    原則：
    - 禁止破壞性操作（刪除檔案、建立程序、對外寫入網路）。
    - 每個週期狀態變更數量上限。
    - 所有變更皆可回溯與審計。
    """

    def __init__(self, forbidden_operations: list[str],
                 max_state_mutation_per_cycle: int) -> None:
        self.forbidden_operations = set(forbidden_operations)
        self.max_state_mutation_per_cycle = max_state_mutation_per_cycle
        self._mutations_this_cycle = 0
        self.violations: list[dict[str, Any]] = []

    def begin_cycle(self) -> None:
        self._mutations_this_cycle = 0

    def check_operation(self, operation: str, detail: dict[str, Any]) -> bool:
        if operation in self.forbidden_operations:
            self._record_violation(operation, detail, "forbidden_operation")
            return False
        if self._mutations_this_cycle >= self.max_state_mutation_per_cycle:
            self._record_violation(operation, detail, "mutation_limit")
            return False
        self._mutations_this_cycle += 1
        return True

    def _record_violation(self, operation: str, detail: dict[str, Any],
                          reason: str) -> None:
        self.violations.append({
            "operation": operation,
            "detail": detail,
            "reason": reason,
            "timestamp": __import__("time").time(),
        })

    def violation_count(self) -> int:
        return len(self.violations)
