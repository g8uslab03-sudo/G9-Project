"""Autonomous AI Private System - 持久化記憶與狀態儲存。"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """單筆記憶：觀察、動作、結果與信心度。"""

    timestamp: float
    source: str
    observation: dict[str, Any]
    action: dict[str, Any] | None
    result: dict[str, Any] | None
    confidence: float
    outcome: str  # success / failure / partial / unknown
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SystemState:
    """系統整體狀態快照。"""

    cycle: int = 0
    uptime_seconds: float = 0.0
    total_observations: int = 0
    total_actions: int = 0
    total_heals: int = 0
    total_evolutions: int = 0
    health_score: float = 1.0
    active_skills: list[str] = field(default_factory=list)
    known_issues: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryStore:
    """線程安全的記憶與狀態儲存，支援原子寫入與重建。"""

    def __init__(self, memory_path: str, state_path: str) -> None:
        self.memory_path = memory_path
        self.state_path = state_path
        self._lock = threading.RLock()
        self._memories: list[MemoryEntry] = []
        self._state = SystemState()
        self._ensure_dirs()
        self._load()

    def _ensure_dirs(self) -> None:
        for path in (self.memory_path, self.state_path):
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

    def _atomic_write(self, path: str, payload: str) -> None:
        directory = os.path.dirname(path) or "."
        tmp = os.path.join(directory, f".{os.path.basename(path)}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def _load(self) -> None:
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                self._memories = [MemoryEntry(**m) for m in data.get("memories", [])]
            except (json.JSONDecodeError, TypeError, OSError):
                self._memories = []
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                self._state = SystemState(**data)
            except (json.JSONDecodeError, TypeError, OSError):
                self._state = SystemState()

    def save(self) -> None:
        with self._lock:
            payload = json.dumps(
                {"memories": [m.to_dict() for m in self._memories]},
                ensure_ascii=False,
                indent=2,
            )
            self._atomic_write(self.memory_path, payload)
            state_payload = json.dumps(self._state.to_dict(), ensure_ascii=False, indent=2)
            self._atomic_write(self.state_path, state_payload)

    def add_memory(self, entry: MemoryEntry) -> None:
        with self._lock:
            self._memories.append(entry)
            self._state.total_observations += 1
            if entry.action is not None:
                self._state.total_actions += 1
            self._state.last_updated = time.time()

    def recent_memories(self, limit: int = 100) -> list[MemoryEntry]:
        with self._lock:
            return list(self._memories[-limit:])

    def memories_by_tag(self, tag: str, limit: int = 50) -> list[MemoryEntry]:
        with self._lock:
            return [m for m in self._memories if tag in m.tags][-limit:]

    @property
    def state(self) -> SystemState:
        with self._lock:
            return SystemState(**self._state.to_dict())

    def update_state(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
            self._state.last_updated = time.time()
