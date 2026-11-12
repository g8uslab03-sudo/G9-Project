"""
Shared exceptions for the system.
"""
from __future__ import annotations
from typing import Any, Optional


class SystemException(Exception):
    """Base system exception."""
    
    def __init__(self, message: str, code: str = "SYSTEM_ERROR", context: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context or {}


class ConfigurationError(SystemException):
    """Configuration error."""
    pass


class ValidationError(SystemException):
    """Validation error."""
    pass


class NotFoundError(SystemException):
    """Resource not found error."""
    pass


class ConcurrencyError(SystemException):
    """Optimistic concurrency error."""
    pass


class BusinessRuleError(SystemException):
    """Business rule violation."""
    pass


class ExternalServiceError(SystemException):
    """External service error."""
    pass


class ResourceExhaustedError(SystemException):
    """Resource exhausted (VRAM, disk, memory, etc.)."""
    pass


class ModelLoadError(SystemException):
    """Model loading error."""
    pass


class TrainingError(SystemException):
    """Training error."""
    pass


class InferenceError(SystemException):
    """Inference error."""
    pass


class CompressionError(SystemException):
    """Compression error."""
    pass


class MemoryError(SystemException):
    """Memory system error."""
    pass


class SkillError(SystemException):
    """Skill execution error."""
    pass


class OrchestrationError(SystemException):
    """Orchestration error."""
    pass


class ResearchError(SystemException):
    """Research module error."""
    pass


class DreamError(SystemException):
    """Dream module error."""
    pass


class AuditError(SystemException):
    """Audit module error."""
    pass


class MagpieError(SystemException):
    """Magpie training error."""
    pass


class SandboxError(SystemException):
    """Sandbox execution error."""
    pass


class NetworkError(SystemException):
    """Network error."""
    pass


class TimeoutError(SystemException):
    """Operation timeout error."""
    pass


class CircuitBreakerOpenError(SystemException):
    """Circuit breaker is open."""
    pass


class RateLimitError(SystemException):
    """Rate limit exceeded."""
    pass


class AuthenticationError(SystemException):
    """Authentication error."""
    pass


class AuthorizationError(SystemException):
    """Authorization error."""
    pass