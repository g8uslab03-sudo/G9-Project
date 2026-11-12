"""
Clock abstraction for time-dependent operations.
Enables testing with fixed time and timezone handling.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Optional
import time


class Clock(ABC):
    """Abstract clock for time operations."""
    
    @abstractmethod
    def now(self) -> datetime:
        pass
    
    @abstractmethod
    def now_utc(self) -> datetime:
        pass
    
    @abstractmethod
    def timestamp(self) -> float:
        pass
    
    def today(self) -> datetime:
        return self.now().replace(hour=0, minute=0, second=0, microsecond=0)


class SystemClock(Clock):
    """System clock using actual system time."""
    
    def __init__(self, timezone: Optional[timezone] = None):
        self._timezone = timezone or timezone.utc
    
    def now(self) -> datetime:
        return datetime.now(self._timezone)
    
    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)
    
    def timestamp(self) -> float:
        return time.time()


class FixedClock(Clock):
    """Fixed clock for testing."""
    
    def __init__(self, fixed_time: datetime):
        self._fixed_time = fixed_time
    
    def now(self) -> datetime:
        return self._fixed_time
    
    def now_utc(self) -> datetime:
        if self._fixed_time.tzinfo is None:
            return self._fixed_time.replace(tzinfo=timezone.utc)
        return self._fixed_time.astimezone(timezone.utc)
    
    def timestamp(self) -> float:
        return self._fixed_time.timestamp()
    
    def advance(self, delta: timedelta) -> None:
        self._fixed_time += delta


class MockClock(Clock):
    """Mock clock that can be controlled programmatically."""
    
    def __init__(self, start_time: Optional[datetime] = None):
        self._current_time = start_time or datetime.now(timezone.utc)
        self._auto_advance = timedelta(0)
    
    def now(self) -> datetime:
        return self._current_time
    
    def now_utc(self) -> datetime:
        return self._current_time.astimezone(timezone.utc)
    
    def timestamp(self) -> float:
        return self._current_time.timestamp()
    
    def set_time(self, time: datetime) -> None:
        self._current_time = time
    
    def advance(self, delta: timedelta) -> None:
        self._current_time += delta
    
    def set_auto_advance(self, delta: timedelta) -> None:
        self._auto_advance = delta
    
    def tick(self) -> datetime:
        current = self._current_time
        self._current_time += self._auto_advance
        return current


# Global clock instance (can be replaced for testing)
_clock: Clock = SystemClock()


def get_clock() -> Clock:
    return _clock


def set_clock(clock: Clock) -> None:
    global _clock
    _clock = clock