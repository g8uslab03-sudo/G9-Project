"""
Result types for functional error handling.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Optional
from uuid import UUID, uuid4

T = TypeVar('T')
E = TypeVar('E', bound=Exception)


class Result(Generic[T], ABC):
    """Base result type - either Success or Failure."""
    
    @abstractmethod
    def is_success(self) -> bool:
        pass
    
    @abstractmethod
    def is_failure(self) -> bool:
        pass
    
    @abstractmethod
    def unwrap(self) -> T:
        pass
    
    @abstractmethod
    def unwrap_err(self) -> Exception:
        pass
    
    @abstractmethod
    def map(self, func: callable) -> 'Result':
        pass
    
    @abstractmethod
    def map_err(self, func: callable) -> 'Result':
        pass
    
    @abstractmethod
    def and_then(self, func: callable) -> 'Result':
        pass


@dataclass
class Success(Result[T]):
    """Successful result with value."""
    value: T
    
    def is_success(self) -> bool:
        return True
    
    def is_failure(self) -> bool:
        return False
    
    def unwrap(self) -> T:
        return self.value
    
    def unwrap_err(self) -> Exception:
        raise RuntimeError("Cannot unwrap error from Success")
    
    def map(self, func: callable) -> 'Result':
        try:
            return Success(func(self.value))
        except Exception as e:
            return Failure(e)
    
    def map_err(self, func: callable) -> 'Result':
        return self
    
    def and_then(self, func: callable) -> 'Result':
        try:
            result = func(self.value)
            return result if isinstance(result, Result) else Success(result)
        except Exception as e:
            return Failure(e)


@dataclass
class Failure(Result[T]):
    """Failed result with error."""
    error: Exception
    error_code: str = "UNKNOWN_ERROR"
    context: dict = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
    
    def is_success(self) -> bool:
        return False
    
    def is_failure(self) -> bool:
        return True
    
    def unwrap(self) -> T:
        raise self.error
    
    def unwrap_err(self) -> Exception:
        return self.error
    
    def map(self, func: callable) -> 'Result':
        return self
    
    def map_err(self, func: callable) -> 'Result':
        try:
            new_error = func(self.error)
            return Failure(new_error, self.error_code, self.context)
        except Exception as e:
            return Failure(e, "MAP_ERROR_FAILED", self.context)
    
    def and_then(self, func: callable) -> 'Result':
        return self


def success(value: T) -> Success[T]:
    return Success(value)


def failure(error: Exception, error_code: str = "UNKNOWN_ERROR", context: dict = None) -> Failure[T]:
    return Failure(error, error_code, context or {})


# Common error types
class DomainError(Exception):
    """Base domain error."""
    pass


class ValidationError(DomainError):
    """Validation error."""
    pass


class NotFoundError(DomainError):
    """Entity not found error."""
    pass


class ConcurrencyError(DomainError):
    """Optimistic concurrency error."""
    pass


class BusinessRuleError(DomainError):
    """Business rule violation error."""
    pass


class ExternalServiceError(DomainError):
    """External service error."""
    pass


class ResourceExhaustedError(DomainError):
    """Resource exhausted error (VRAM, disk, etc.)."""
    pass