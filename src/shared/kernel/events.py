"""
Event Sourcing infrastructure - Event Store, Snapshots, Projections.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar, List, Optional, Dict, AsyncIterator
from uuid import UUID, uuid4
import json
import asyncio
from ..kernel.base import DomainEvent, AggregateRoot
from ..kernel.result import Result, success, failure, NotFoundError, ConcurrencyError


TAggregate = TypeVar('TAggregate', bound=AggregateRoot)
TEvent = TypeVar('TEvent', bound=DomainEvent)


@dataclass
class StoredEvent:
    """Persisted event with metadata."""
    event_id: UUID
    event_type: str
    aggregate_id: UUID
    aggregate_type: str
    version: int
    occurred_at: datetime
    correlation_id: Optional[UUID]
    causation_id: Optional[UUID]
    metadata: dict
    payload: dict
    
    def to_domain_event(self) -> DomainEvent:
        # This would need a registry of event types
        raise NotImplementedError("Use EventDeserializer")
    
    @classmethod
    def from_domain_event(cls, event: DomainEvent) -> 'StoredEvent':
        return cls(
            event_id=event.event_id,
            event_type=event.event_type,
            aggregate_id=event.aggregate_id,
            aggregate_type=event.aggregate_type,
            version=event.version,
            occurred_at=event.occurred_at,
            correlation_id=event.correlation_id,
            causation_id=event.causation_id,
            metadata=event.metadata,
            payload=event._payload(),
        )


@dataclass
class Snapshot(Generic[TAggregate]):
    """Aggregate snapshot for performance."""
    aggregate_id: UUID
    aggregate_type: str
    version: int
    state: dict
    created_at: datetime = field(default_factory=datetime.utcnow)


class EventStore(ABC, Generic[TEvent]):
    """Abstract event store for Event Sourcing."""
    
    @abstractmethod
    async def append(self, events: List[TEvent], expected_version: Optional[int] = None) -> Result[None]:
        pass
    
    @abstractmethod
    async def get_events(self, aggregate_id: UUID, from_version: int = 0) -> Result[List[TEvent]]:
        pass
    
    @abstractmethod
    async def get_events_by_correlation(self, correlation_id: UUID) -> Result[List[TEvent]]:
        pass
    
    @abstractmethod
    async def get_all_events(self, from_position: int = 0, batch_size: int = 100) -> AsyncIterator[TEvent]:
        pass


class SnapshotStore(ABC, Generic[TAggregate]):
    """Abstract snapshot store."""
    
    @abstractmethod
    async def save(self, snapshot: Snapshot[TAggregate]) -> Result[None]:
        pass
    
    @abstractmethod
    async def get(self, aggregate_id: UUID) -> Result[Optional[Snapshot[TAggregate]]]:
        pass
    
    @abstractmethod
    async def delete(self, aggregate_id: UUID) -> Result[None]:
        pass


class EventSerializer(ABC):
    """Serialize/deserialize domain events."""
    
    @abstractmethod
    def serialize(self, event: DomainEvent) -> StoredEvent:
        pass
    
    @abstractmethod
    def deserialize(self, stored: StoredEvent) -> DomainEvent:
        pass


class JsonEventSerializer(EventSerializer):
    """JSON-based event serializer with type registry."""
    
    def __init__(self):
        self._event_types: Dict[str, type] = {}
    
    def register_event_type(self, event_type: type) -> None:
        self._event_types[event_type.__name__] = event_type
    
    def serialize(self, event: DomainEvent) -> StoredEvent:
        return StoredEvent.from_domain_event(event)
    
    def deserialize(self, stored: StoredEvent) -> DomainEvent:
        event_class = self._event_types.get(stored.event_type)
        if not event_class:
            raise ValueError(f"Unknown event type: {stored.event_type}")
        # Reconstruct event - this is simplified
        return event_class(
            event_id=stored.event_id,
            aggregate_id=stored.aggregate_id,
            aggregate_type=stored.aggregate_type,
            version=stored.version,
            occurred_at=stored.occurred_at,
            correlation_id=stored.correlation_id,
            causation_id=stored.causation_id,
            metadata=stored.metadata,
            **stored.payload
        )


class InMemoryEventStore(EventStore[DomainEvent]):
    """In-memory event store for testing and development."""
    
    def __init__(self):
        self._events: Dict[UUID, List[StoredEvent]] = {}
        self._all_events: List[StoredEvent] = []
        self._lock = asyncio.Lock()
    
    async def append(self, events: List[DomainEvent], expected_version: Optional[int] = None) -> Result[None]:
        async with self._lock:
            if not events:
                return success(None)
            
            aggregate_id = events[0].aggregate_id
            existing = self._events.get(aggregate_id, [])
            
            if expected_version is not None and len(existing) != expected_version:
                return failure(ConcurrencyError(f"Expected version {expected_version}, got {len(existing)}"))
            
            stored_events = [StoredEvent.from_domain_event(e) for e in events]
            self._events.setdefault(aggregate_id, []).extend(stored_events)
            self._all_events.extend(stored_events)
            return success(None)
    
    async def get_events(self, aggregate_id: UUID, from_version: int = 0) -> Result[List[DomainEvent]]:
        events = self._events.get(aggregate_id, [])
        return success([e for i, e in enumerate(events) if i >= from_version])
    
    async def get_events_by_correlation(self, correlation_id: UUID) -> Result[List[DomainEvent]]:
        events = [e for e in self._all_events if e.correlation_id == correlation_id]
        return success(events)
    
    async def get_all_events(self, from_position: int = 0, batch_size: int = 100) -> AsyncIterator[DomainEvent]:
        for i in range(from_position, len(self._all_events), batch_size):
            batch = self._all_events[i:i+batch_size]
            for event in batch:
                yield event


class InMemorySnapshotStore(SnapshotStore[TAggregate]):
    """In-memory snapshot store."""
    
    def __init__(self):
        self._snapshots: Dict[UUID, Snapshot] = {}
        self._lock = asyncio.Lock()
    
    async def save(self, snapshot: Snapshot[TAggregate]) -> Result[None]:
        async with self._lock:
            self._snapshots[snapshot.aggregate_id] = snapshot
            return success(None)
    
    async def get(self, aggregate_id: UUID) -> Result[Optional[Snapshot[TAggregate]]]:
        return success(self._snapshots.get(aggregate_id))
    
    async def delete(self, aggregate_id: UUID) -> Result[None]:
        async with self._lock:
            self._snapshots.pop(aggregate_id, None)
            return success(None)


class AggregateRepository(Generic[TAggregate]):
    """Repository for loading/saving aggregates with event sourcing."""
    
    def __init__(
        self,
        event_store: EventStore,
        snapshot_store: SnapshotStore[TAggregate],
        serializer: EventSerializer,
        aggregate_factory: callable,
        snapshot_threshold: int = 100
    ):
        self._event_store = event_store
        self._snapshot_store = snapshot_store
        self._serializer = serializer
        self._aggregate_factory = aggregate_factory
        self._snapshot_threshold = snapshot_threshold
    
    async def get(self, aggregate_id: UUID) -> Result[TAggregate]:
        # Try to load from snapshot first
        snapshot_result = await self._snapshot_store.get(aggregate_id)
        if snapshot_result.is_failure():
            return failure(snapshot_result.unwrap_err())
        
        snapshot = snapshot_result.unwrap()
        aggregate = self._aggregate_factory()
        
        if snapshot:
            aggregate._load_from_snapshot(snapshot.state, snapshot.version)
            from_version = snapshot.version
        else:
            from_version = 0
        
        # Load events after snapshot
        events_result = await self._event_store.get_events(aggregate_id, from_version)
        if events_result.is_failure():
            return failure(events_result.unwrap_err())
        
        for event in events_result.unwrap():
            aggregate._apply_event(event)
        
        return success(aggregate)
    
    async def save(self, aggregate: TAggregate) -> Result[None]:
        events = aggregate.get_domain_events()
        if not events:
            return success(None)
        
        expected_version = aggregate.version - len(events)
        append_result = await self._event_store.append(events, expected_version)
        if append_result.is_failure():
            return failure(append_result.unwrap_err())
        
        # Check if snapshot needed
        if aggregate.version % self._snapshot_threshold == 0:
            snapshot = Snapshot(
                aggregate_id=aggregate.id,
                aggregate_type=type(aggregate).__name__,
                version=aggregate.version,
                state=aggregate._get_snapshot_state()
            )
            await self._snapshot_store.save(snapshot)
        
        aggregate.clear_domain_events()
        return success(None)


class Projection(ABC):
    """Base class for read model projections."""
    
    @abstractmethod
    async def project(self, event: DomainEvent) -> None:
        pass
    
    @abstractmethod
    async def reset(self) -> None:
        pass


class ProjectionManager:
    """Manages multiple projections."""
    
    def __init__(self, event_store: EventStore):
        self._event_store = event_store
        self._projections: List[Projection] = []
        self._checkpoints: Dict[str, int] = {}
    
    def register(self, projection: Projection, name: str) -> None:
        self._projections.append(projection)
        self._checkpoints[name] = 0
    
    async def run(self, batch_size: int = 100) -> None:
        for projection in self._projections:
            name = type(projection).__name__
            position = self._checkpoints.get(name, 0)
            
            async for event in self._event_store.get_all_events(position, batch_size):
                await projection.project(event)
                position += 1
            
            self._checkpoints[name] = position
    
    async def rebuild_all(self) -> None:
        for projection in self._projections:
            await projection.reset()
        self._checkpoints.clear()
        await self.run()