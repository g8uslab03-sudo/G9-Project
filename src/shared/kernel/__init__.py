"""
Shared Kernel - Core building blocks for the entire system.
Implements CQRS, Event Sourcing, and base classes for Clean/Hexagonal Architecture.
"""
from .base import *
from .events import *
from .commands import *
from .queries import *
from .result import *
from .clock import *
from .id_generator import *

__all__ = [
    "Entity",
    "AggregateRoot",
    "ValueObject",
    "DomainEvent",
    "Command",
    "Query",
    "CommandHandler",
    "QueryHandler",
    "Result",
    "Success",
    "Failure",
    "Clock",
    "SystemClock",
    "IdGenerator",
    "UUIDGenerator",
]