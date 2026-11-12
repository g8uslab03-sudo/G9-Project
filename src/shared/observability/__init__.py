"""
Observability - Logging, Metrics, Tracing, Health Checks.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from uuid import UUID, uuid4
from enum import Enum
import logging
import time
import asyncio
from contextlib import asynccontextmanager
from ..kernel.clock import get_clock


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    level: LogLevel
    message: str
    timestamp: datetime
    logger_name: str
    correlation_id: Optional[UUID] = None
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Exception] = None


class Logger(ABC):
    """Abstract logger."""
    
    @abstractmethod
    def log(self, entry: LogEntry) -> None:
        pass
    
    def debug(self, message: str, **context) -> None:
        self.log(LogEntry(LogLevel.DEBUG, message, get_clock().now_utc(), self.name, context=context))
    
    def info(self, message: str, **context) -> None:
        self.log(LogEntry(LogLevel.INFO, message, get_clock().now_utc(), self.name, context=context))
    
    def warning(self, message: str, **context) -> None:
        self.log(LogEntry(LogLevel.WARNING, message, get_clock().now_utc(), self.name, context=context))
    
    def error(self, message: str, exception: Optional[Exception] = None, **context) -> None:
        self.log(LogEntry(LogLevel.ERROR, message, get_clock().now_utc(), self.name, context=context, exception=exception))
    
    def critical(self, message: str, exception: Optional[Exception] = None, **context) -> None:
        self.log(LogEntry(LogLevel.CRITICAL, message, get_clock().now_utc(), self.name, context=context, exception=exception))
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class StructuredLogger(Logger):
    """Structured JSON logger."""
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self._name = name
        self._level = level
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.value))
        
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
    
    @property
    def name(self) -> str:
        return self._name
    
    def log(self, entry: LogEntry) -> None:
        if entry.level.value < self._level.value:
            return
        
        extra = {
            "correlation_id": str(entry.correlation_id) if entry.correlation_id else None,
            "context": entry.context,
        }
        
        if entry.exception:
            self._logger.log(
                getattr(logging, entry.level.value),
                entry.message,
                extra=extra,
                exc_info=entry.exception
            )
        else:
            self._logger.log(
                getattr(logging, entry.level.value),
                entry.message,
                extra=extra
            )


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: get_clock().now_utc())


class MetricsCollector(ABC):
    """Abstract metrics collector."""
    
    @abstractmethod
    def record(self, metric: Metric) -> None:
        pass
    
    def counter(self, name: str, value: float = 1, **labels) -> None:
        self.record(Metric(name, MetricType.COUNTER, value, labels))
    
    def gauge(self, name: str, value: float, **labels) -> None:
        self.record(Metric(name, MetricType.GAUGE, value, labels))
    
    def histogram(self, name: str, value: float, **labels) -> None:
        self.record(Metric(name, MetricType.HISTOGRAM, value, labels))
    
    def timing(self, name: str, duration_ms: float, **labels) -> None:
        self.histogram(f"{name}_duration_ms", duration_ms, **labels)


class InMemoryMetricsCollector(MetricsCollector):
    """In-memory metrics collector for development."""
    
    def __init__(self):
        self._metrics: List[Metric] = []
        self._lock = asyncio.Lock()
    
    def record(self, metric: Metric) -> None:
        self._metrics.append(metric)
    
    def get_metrics(self, name: Optional[str] = None) -> List[Metric]:
        if name:
            return [m for m in self._metrics if m.name == name]
        return self._metrics.copy()
    
    def clear(self) -> None:
        self._metrics.clear()


class PrometheusMetricsCollector(MetricsCollector):
    """Prometheus-compatible metrics collector."""
    
    def __init__(self):
        try:
            from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
            self._registry = CollectorRegistry()
            self._counters: Dict[str, Counter] = {}
            self._gauges: Dict[str, Gauge] = {}
            self._histograms: Dict[str, Histogram] = {}
            self._enabled = True
        except ImportError:
            self._enabled = False
    
    def record(self, metric: Metric) -> None:
        if not self._enabled:
            return
        
        labels = metric.labels
        
        if metric.type == MetricType.COUNTER:
            if metric.name not in self._counters:
                self._counters[metric.name] = Counter(metric.name, "", labelnames=list(labels.keys()))
            self._counters[metric.name].labels(**labels).inc(metric.value)
        
        elif metric.type == MetricType.GAUGE:
            if metric.name not in self._gauges:
                self._gauges[metric.name] = Gauge(metric.name, "", labelnames=list(labels.keys()))
            self._gauges[metric.name].labels(**labels).set(metric.value)
        
        elif metric.type == MetricType.HISTOGRAM:
            if metric.name not in self._histograms:
                self._histograms[metric.name] = Histogram(metric.name, "", labelnames=list(labels.keys()))
            self._histograms[metric.name].labels(**labels).observe(metric.value)


@dataclass
class Span:
    """Distributed tracing span."""
    trace_id: UUID
    span_id: UUID
    parent_span_id: Optional[UUID]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    
    def finish(self, status: str = "OK") -> None:
        self.end_time = get_clock().now_utc()
        self.status = status
    
    def log(self, message: str, **fields) -> None:
        self.logs.append({
            "timestamp": get_clock().now_utc().isoformat(),
            "message": message,
            **fields
        })
    
    def tag(self, key: str, value: Any) -> None:
        self.tags[key] = value
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return (get_clock().now_utc() - self.start_time).total_seconds() * 1000


class Tracer(ABC):
    """Abstract distributed tracer."""
    
    @abstractmethod
    @asynccontextmanager
    async def start_span(self, operation_name: str, parent: Optional[Span] = None) -> Span:
        pass
    
    @abstractmethod
    def inject(self, span: Span, carrier: Dict[str, str]) -> None:
        pass
    
    @abstractmethod
    def extract(self, carrier: Dict[str, str]) -> Optional[Span]:
        pass


class NoOpTracer(Tracer):
    """No-op tracer for when tracing is disabled."""
    
    @asynccontextmanager
    async def start_span(self, operation_name: str, parent: Optional[Span] = None) -> Span:
        span = Span(
            trace_id=uuid4(),
            span_id=uuid4(),
            parent_span_id=parent.span_id if parent else None,
            operation_name=operation_name,
            start_time=get_clock().now_utc()
        )
        try:
            yield span
        finally:
            span.finish()
    
    def inject(self, span: Span, carrier: Dict[str, str]) -> None:
        pass
    
    def extract(self, carrier: Dict[str, str]) -> Optional[Span]:
        return None


@dataclass
class HealthCheckResult:
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class HealthCheck(ABC):
    """Abstract health check."""
    
    @abstractmethod
    async def check(self) -> HealthCheckResult:
        pass


class HealthCheckRegistry:
    """Registry of health checks."""
    
    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
    
    def register(self, name: str, check: HealthCheck) -> None:
        self._checks[name] = check
    
    async def run_all(self) -> Dict[str, HealthCheckResult]:
        results = {}
        for name, check in self._checks.items():
            try:
                results[name] = await check.check()
            except Exception as e:
                results[name] = HealthCheckResult(
                    component=name,
                    status="unhealthy",
                    message=str(e),
                    timestamp=get_clock().now_utc()
                )
        return results
    
    async def run(self, name: str) -> Optional[HealthCheckResult]:
        check = self._checks.get(name)
        if check:
            return await check.check()
        return None


# Global instances
_logger_registry: Dict[str, Logger] = {}
_metrics_collector: MetricsCollector = InMemoryMetricsCollector()
_tracer: Tracer = NoOpTracer()
_health_registry = HealthCheckRegistry()


def get_logger(name: str) -> Logger:
    if name not in _logger_registry:
        _logger_registry[name] = StructuredLogger(name)
    return _logger_registry[name]


def set_metrics_collector(collector: MetricsCollector) -> None:
    global _metrics_collector
    _metrics_collector = collector


def get_metrics() -> MetricsCollector:
    return _metrics_collector


def set_tracer(tracer: Tracer) -> None:
    global _tracer
    _tracer = tracer


def get_tracer() -> Tracer:
    return _tracer


def get_health_registry() -> HealthCheckRegistry:
    return _health_registry


# Decorators for observability
def traced(operation_name: Optional[str] = None):
    """Decorator to trace function execution."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = operation_name or f"{func.__module__}.{func.__name__}"
            async with tracer.start_span(name) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.tag("success", True)
                    return result
                except Exception as e:
                    span.tag("success", False)
                    span.tag("error", str(e))
                    span.log("Function failed", error=str(e))
                    raise
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't use async context manager
            # This is a simplified version
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def metered(metric_name: str, metric_type: MetricType = MetricType.HISTOGRAM):
    """Decorator to record metrics for function execution."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            metrics = get_metrics()
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                if metric_type == MetricType.HISTOGRAM:
                    metrics.timing(metric_name, duration)
                elif metric_type == MetricType.COUNTER:
                    metrics.counter(metric_name, 1, success="true")
                return result
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                if metric_type == MetricType.HISTOGRAM:
                    metrics.timing(metric_name, duration)
                elif metric_type == MetricType.COUNTER:
                    metrics.counter(metric_name, 1, success="false")
                raise
        
        def sync_wrapper(*args, **kwargs):
            metrics = get_metrics()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                if metric_type == MetricType.HISTOGRAM:
                    metrics.timing(metric_name, duration)
                elif metric_type == MetricType.COUNTER:
                    metrics.counter(metric_name, 1, success="true")
                return result
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                if metric_type == MetricType.HISTOGRAM:
                    metrics.timing(metric_name, duration)
                elif metric_type == MetricType.COUNTER:
                    metrics.counter(metric_name, 1, success="false")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator