"""
ID Generator for creating unique identifiers.
Supports UUID, ULID, and custom generators.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID, uuid4
import time
import random
import string


class IdGenerator(ABC):
    """Abstract ID generator."""
    
    @abstractmethod
    def generate(self) -> str:
        pass
    
    @abstractmethod
    def generate_uuid(self) -> UUID:
        pass


class UUIDGenerator(IdGenerator):
    """Standard UUID v4 generator."""
    
    def generate(self) -> str:
        return str(uuid4())
    
    def generate_uuid(self) -> UUID:
        return uuid4()


class ULIDGenerator(IdGenerator):
    """Universally Unique Lexicographically Sortable Identifier generator.
    ULIDs are sortable by creation time.
    """
    
    _ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford's Base32
    
    def __init__(self):
        self._last_time = 0
        self._random = random.SystemRandom()
    
    def generate(self) -> str:
        return self._encode_time(time.time() * 1000) + self._encode_random(80)
    
    def generate_uuid(self) -> UUID:
        # Convert ULID to UUID (not standard but unique)
        ulid = self.generate()
        # Use first 32 chars as UUID-like
        return UUID(ulid[:8] + "-" + ulid[8:12] + "-" + ulid[12:16] + "-" + ulid[16:20] + "-" + ulid[20:32])
    
    def _encode_time(self, timestamp_ms: int) -> str:
        if timestamp_ms < self._last_time:
            timestamp_ms = self._last_time
        self._last_time = timestamp_ms
        return self._encode_int(timestamp_ms, 10)
    
    def _encode_random(self, bits: int) -> str:
        bytes_needed = (bits + 7) // 8
        random_bytes = self._random.randbytes(bytes_needed)
        random_int = int.from_bytes(random_bytes, 'big')
        return self._encode_int(random_int, 16)
    
    def _encode_int(self, value: int, length: int) -> str:
        encoded = []
        for _ in range(length):
            encoded.append(self._ENCODING[value & 31])
            value >>= 5
        return ''.join(reversed(encoded))


class CompositeIdGenerator(IdGenerator):
    """Composite generator with prefix support."""
    
    def __init__(self, generator: IdGenerator, prefix: str = ""):
        self._generator = generator
        self._prefix = prefix
    
    def generate(self) -> str:
        return f"{self._prefix}{self._generator.generate()}"
    
    def generate_uuid(self) -> UUID:
        return self._generator.generate_uuid()


class SnowflakeIdGenerator(IdGenerator):
    """Twitter Snowflake-inspired ID generator.
    64-bit: 41 bits timestamp | 10 bits machine | 12 bits sequence
    """
    
    def __init__(self, machine_id: int = 0, epoch: int = 1609459200000):  # 2021-01-01
        if machine_id < 0 or machine_id > 1023:
            raise ValueError("Machine ID must be between 0 and 1023")
        self._machine_id = machine_id
        self._epoch = epoch
        self._sequence = 0
        self._last_timestamp = -1
    
    def generate(self) -> str:
        return str(self._next_id())
    
    def generate_uuid(self) -> UUID:
        # Convert to UUID-like format
        id_val = self._next_id()
        bytes_val = id_val.to_bytes(8, 'big') + b'\x00' * 8
        return UUID(bytes=bytes_val)
    
    def _next_id(self) -> int:
        timestamp = self._current_timestamp()
        
        if timestamp < self._last_timestamp:
            raise RuntimeError("Clock moved backwards")
        
        if timestamp == self._last_timestamp:
            self._sequence = (self._sequence + 1) & 0xFFF
            if self._sequence == 0:
                timestamp = self._wait_next_ms()
        else:
            self._sequence = 0
        
        self._last_timestamp = timestamp
        
        return ((timestamp - self._epoch) << 22) | (self._machine_id << 12) | self._sequence
    
    def _current_timestamp(self) -> int:
        return int(time.time() * 1000)
    
    def _wait_next_ms(self) -> int:
        timestamp = self._current_timestamp()
        while timestamp <= self._last_timestamp:
            timestamp = self._current_timestamp()
        return timestamp


# Global ID generator
_id_generator: IdGenerator = UUIDGenerator()


def get_id_generator() -> IdGenerator:
    return _id_generator


def set_id_generator(generator: IdGenerator) -> None:
    global _id_generator
    _id_generator = generator