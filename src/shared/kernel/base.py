"""
Base classes for Domain-Driven Design.
Implements Entity, AggregateRoot, ValueObject with proper equality and immutability.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar, List, Optional
from uuid import UUID, uuid4
import copy


T = TypeVar('T')
TId = TypeVar('TId', bound='Identifier')


class Identifier(ABC):
    """Base class for all identifiers."""
    
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass


class UUIDIdentifier(Identifier):
    """UUID-based identifier."""
    
    def __init__(self, value: Optional[UUID] = None):
        self._value = value or uuid4()
    
    @property
    def value(self) -> UUID:
        return self._value
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UUIDIdentifier):
            return False
        return self._value == other._value
    
    def __hash__(self) -> int:
        return hash(self._value)
    
    def __str__(self) -> str:
        return str(self._value)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value})"


class Entity(Generic[TId]):
    """Base entity with identity and version for optimistic concurrency."""
    
    def __init__(self, id: TId, version: int = 0):
        self._id = id
        self._version = version
        self._created_at = datetime.utcnow()
        self._updated_at = self._created_at
    
    @property
    def id(self) -> TId:
        return self._id
    
    @property
    def version(self) -> int:
        return self._version
    
    @property
    def created_at(self) -> datetime:
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        return self._updated_at
    
    def _increment_version(self) -> None:
        self._version += 1
        self._updated_at = datetime.utcnow()
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, version={self.version})"


class AggregateRoot(Entity[TId]):
    """Aggregate root with domain event publishing."""
    
    def __init__(self, id: TId, version: int = 0):
        super().__init__(id, version)
        self._domain_events: List[DomainEvent] = []
    
    @property
    def domain_events(self) -> List[DomainEvent]:
        return self._domain_events.copy()
    
    def add_domain_event(self, event: DomainEvent) -> None:
        self._domain_events.append(event)
    
    def clear_domain_events(self) -> None:
        self._domain_events.clear()
    
    def mark_changes(self) -> None:
        self._increment_version()


class ValueObject(ABC):
    """Immutable value object with structural equality."""
    
    @abstractmethod
    def _equality_components(self) -> tuple:
        """Return components that define equality."""
        pass
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValueObject):
            return False
        if not isinstance(other, self.__class__):
            return False
        return self._equality_components() == other._equality_components()
    
    def __hash__(self) -> int:
        return hash(self._equality_components())
    
    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise AttributeError("Value objects are immutable")
        super().__setattr__(name, value)
    
    def with_changes(self, **changes: Any) -> 'ValueObject':
        """Create a new instance with changes."""
        current = self.__dict__.copy()
        current.update(changes)
        new_obj = self.__class__.__new__(self.__class__)
        new_obj.__dict__.update(current)
        return new_obj


@dataclass(frozen=True)
class DomainEvent:
    """Base domain event with metadata."""
    event_id: UUID = field(default_factory=uuid4)
    aggregate_id: UUID = field(default_factory=uuid4)
    aggregate_type: str = ""
    version: int = 1
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[UUID] = None
    causation_id: Optional[UUID] = None
    metadata: dict = field(default_factory=dict)
    
    @property
    def event_type(self) -> str:
        return self.__class__.__name__
    
    def to_dict(self) -> dict:
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type,
            "aggregate_id": str(self.aggregate_id),
            "aggregate_type": self.aggregate_type,
            "version": self.version,
            "occurred_at": self.occurred_at.isoformat(),
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "causation_id": str(self.causation_id) if self.causation_id else None,
            "metadata": self.metadata,
            "payload": self._payload(),
        }
    
    @abstractmethod
    def _payload(self) -> dict:
        """Return event-specific payload."""
        pass


class DomainEventMixin:
    """Mixin for aggregates to handle domain events."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._domain_events: List[DomainEvent] = []
    
    def add_domain_event(self, event: DomainEvent) -> None:
        self._domain_events.append(event)
    
    def get_domain_events(self) -> List[DomainEvent]:
        return self._domain_events.copy()
    
    def clear_domain_events(self) -> None:
        self._domain_events.clear()