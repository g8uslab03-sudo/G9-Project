"""
CQRS Commands and Queries with handlers.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar, Optional
from uuid import UUID, uuid4
from ..kernel.base import Command, Query, CommandHandler, QueryHandler


TCommand = TypeVar('TCommand', bound=Command)
TQuery = TypeVar('TQuery', bound=Query)
TResult = TypeVar('TResult')


@dataclass
class Command(Generic[TResult]):
    """Base command for write operations."""
    command_id: UUID = field(default_factory=uuid4)
    correlation_id: Optional[UUID] = None
    causation_id: Optional[UUID] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    
    @property
    def command_type(self) -> str:
        return self.__class__.__name__


@dataclass
class Query(Generic[TResult]):
    """Base query for read operations."""
    query_id: UUID = field(default_factory=uuid4)
    correlation_id: Optional[UUID] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    
    @property
    def query_type(self) -> str:
        return self.__class__.__name__


class CommandHandler(ABC, Generic[TCommand, TResult]):
    """Base command handler."""
    
    @abstractmethod
    async def handle(self, command: TCommand) -> TResult:
        pass
    
    @property
    @abstractmethod
    def command_type(self) -> type:
        pass


class QueryHandler(ABC, Generic[TQuery, TResult]):
    """Base query handler."""
    
    @abstractmethod
    async def handle(self, query: TQuery) -> TResult:
        pass
    
    @property
    @abstractmethod
    def query_type(self) -> type:
        pass


class CommandBus:
    """Command bus for dispatching commands to handlers."""
    
    def __init__(self):
        self._handlers: dict[type, CommandHandler] = {}
        self._middleware: list = []
    
    def register(self, handler: CommandHandler) -> None:
        self._handlers[handler.command_type] = handler
    
    def add_middleware(self, middleware: callable) -> None:
        self._middleware.append(middleware)
    
    async def dispatch(self, command: Command) -> Any:
        handler = self._handlers.get(type(command))
        if not handler:
            raise ValueError(f"No handler registered for command: {command.command_type}")
        
        # Apply middleware
        async def execute(cmd: Command) -> Any:
            return await handler.handle(cmd)
        
        for middleware in reversed(self._middleware):
            execute = middleware(execute)
        
        return await execute(command)


class QueryBus:
    """Query bus for dispatching queries to handlers."""
    
    def __init__(self):
        self._handlers: dict[type, QueryHandler] = {}
        self._middleware: list = []
    
    def register(self, handler: QueryHandler) -> None:
        self._handlers[handler.query_type] = handler
    
    def add_middleware(self, middleware: callable) -> None:
        self._middleware.append(middleware)
    
    async def dispatch(self, query: Query) -> Any:
        handler = self._handlers.get(type(query))
        if not handler:
            raise ValueError(f"No handler registered for query: {query.query_type}")
        
        async def execute(q: Query) -> Any:
            return await handler.handle(q)
        
        for middleware in reversed(self._middleware):
            execute = middleware(execute)
        
        return await execute(query)