"""自主檢測模組：持續監測系統健康、偵測異常與偏差。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionSignal:
    """單次檢測訊號。"""

    timestamp: float
    signal_type: str  # health / anomaly / drift / error
    severity: float  # 0.0 ~ 1.0
    source: str
    detail: dict[str, Any]
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "signal_type": self.signal_type,
            "severity": self.severity,
            "source": self.source,
            "detail": self.detail,
            "tags": self.tags,
        }


class Detector:
    """自主偵測引擎。

    職責：
    - 定期評估系統健康分數。
    - 偵測記憶中連續失敗、信心度下降、狀態漂移等異常。
    - 發出可供修復模組消費的偵測訊號。
    """

    def __init__(self, anomaly_threshold: float = 0.65,
                 health_interval: float = 30.0) -> None:
        self.anomaly_threshold = anomaly_threshold
        self.health_interval = health_interval
        self._last_health_check = 0.0
        self.signals: list[DetectionSignal] = []

    def check_health(self, state: Any, recent_failures: int,
                     avg_confidence: float, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        health = 1.0
        penalties = 0.0

        if recent_failures > 0:
            penalties += min(0.5, recent_failures * 0.1)
        if avg_confidence < 0.5:
            penalties += 0.3
        if getattr(state, "health_score", 1.0) < 0.5:
            penalties += 0.2

        health = max(0.0, min(1.0, health - penalties))
        if hasattr(state, "health_score"):
            state.health_score = health

        self._last_health_check = now
        if health < self.anomaly_threshold:
            self._emit("health", severity=1.0 - health, source="health_check",
                       detail={"health": health, "recent_failures": recent_failures,
                               "avg_confidence": avg_confidence},
                       tags=["health", "anomaly"])
        return health

    def scan_anomalies(self, memories: list[Any], now: float | None = None) -> list[DetectionSignal]:
        now = now if now is not None else time.time()
        new_signals: list[DetectionSignal] = []
        if len(memories) < 5:
            return new_signals

        recent = memories[-20:]
        failures = [m for m in recent if getattr(m, "outcome", "") == "failure"]
        failure_rate = len(failures) / len(recent)
        if failure_rate > self.anomaly_threshold:
            sig = DetectionSignal(
                timestamp=now, signal_type="anomaly", severity=failure_rate,
                source="memory_scan", detail={"failure_rate": failure_rate,
                                              "window": len(recent)},
                tags=["anomaly", "failure_rate"])
            new_signals.append(sig)
            self.signals.append(sig)

        low_conf = [m for m in recent if getattr(m, "confidence", 1.0) < 0.4]
        if len(low_conf) > len(recent) * 0.5:
            sig = DetectionSignal(
                timestamp=now, signal_type="drift", severity=0.7,
                source="memory_scan",
                detail={"low_confidence_count": len(low_conf), "window": len(recent)},
                tags=["drift", "low_confidence"])
            new_signals.append(sig)
            self.signals.append(sig)

        return new_signals

    def emit_error(self, source: str, detail: dict[str, Any],
                   severity: float = 0.8, tags: list[str] | None = None) -> DetectionSignal:
        sig = DetectionSignal(
            timestamp=time.time(), signal_type="error", severity=severity,
            source=source, detail=detail, tags=tags or ["error"])
        self.signals.append(sig)
        return sig

    def recent_signals(self, limit: int = 50) -> list[DetectionSignal]:
        return list(self.signals[-limit:])

    def clear_signals(self, older_than: float | None = None) -> int:
        if older_than is None:
            count = len(self.signals)
            self.signals = []
            return count
        before = len(self.signals)
        self.signals = [s for s in self.signals if s.timestamp >= older_than]
        return before - len(self.signals)
