"""Autonomous AI Private System - configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AutonomousConfig:
    """全自主系統設定。所有行為皆由內部迴圈自行驅動，無需人工介入。"""

    # 核心迴圈
    cycle_interval_seconds: float = float(os.getenv("AUTO_CYCLE_INTERVAL", "60"))
    max_cycles: int | None = None  # None = 無限運行
    max_concurrent_skills: int = int(os.getenv("AUTO_MAX_CONCURRENT_SKILLS", "4"))

    # 學習
    learning_enabled: bool = os.getenv("AUTO_LEARNING_ENABLED", "1") == "1"
    exploration_rate: float = float(os.getenv("AUTO_EXPLORATION_RATE", "0.15"))
    min_confidence_to_apply: float = float(os.getenv("AUTO_MIN_CONFIDENCE", "0.7"))

    # 檢測
    detection_enabled: bool = os.getenv("AUTO_DETECTION_ENABLED", "1") == "1"
    anomaly_threshold: float = float(os.getenv("AUTO_ANOMALY_THRESHOLD", "0.65"))
    health_check_interval_seconds: float = float(os.getenv("AUTO_HEALTH_INTERVAL", "30"))

    # 修復
    healing_enabled: bool = os.getenv("AUTO_HEALING_ENABLED", "1") == "1"
    max_heal_attempts_per_issue: int = int(os.getenv("AUTO_MAX_HEAL_ATTEMPTS", "3"))
    rollback_on_heal_failure: bool = os.getenv("AUTO_ROLLBACK_ON_FAILURE", "1") == "1"

    # 進化
    evolution_enabled: bool = os.getenv("AUTO_EVOLUTION_ENABLED", "1") == "1"
    evolution_cycle_interval: int = int(os.getenv("AUTO_EVOLUTION_CYCLE", "10"))  # 每 N 個週期進化一次
    skill_retire_threshold: float = float(os.getenv("AUTO_SKILL_RETIRE_THRESHOLD", "0.3"))

    # 儲存
    data_dir: str = os.getenv("AUTO_DATA_DIR", "./data")
    memory_path: str = os.getenv("AUTO_MEMORY_PATH", "./data/memory.json")
    state_path: str = os.getenv("AUTO_STATE_PATH", "./data/state.json")
    log_path: str = os.getenv("AUTO_LOG_PATH", "./data/autonomous.log")

    # 安全護欄
    max_state_mutation_per_cycle: int = int(os.getenv("AUTO_MAX_STATE_MUTATION", "50"))
    forbidden_operations: list[str] = field(
        default_factory=lambda: ["network_write", "file_delete", "process_spawn"]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_interval_seconds": self.cycle_interval_seconds,
            "max_cycles": self.max_cycles,
            "max_concurrent_skills": self.max_concurrent_skills,
            "learning_enabled": self.learning_enabled,
            "exploration_rate": self.exploration_rate,
            "min_confidence_to_apply": self.min_confidence_to_apply,
            "detection_enabled": self.detection_enabled,
            "anomaly_threshold": self.anomaly_threshold,
            "health_check_interval_seconds": self.health_check_interval_seconds,
            "healing_enabled": self.healing_enabled,
            "max_heal_attempts_per_issue": self.max_heal_attempts_per_issue,
            "rollback_on_heal_failure": self.rollback_on_heal_failure,
            "evolution_enabled": self.evolution_enabled,
            "evolution_cycle_interval": self.evolution_cycle_interval,
            "skill_retire_threshold": self.skill_retire_threshold,
            "data_dir": self.data_dir,
            "memory_path": self.memory_path,
            "state_path": self.state_path,
            "log_path": self.log_path,
            "max_state_mutation_per_cycle": self.max_state_mutation_per_cycle,
            "forbidden_operations": self.forbidden_operations,
        }
