"""
Shared utilities.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, TypeVar, Callable, Awaitable
from functools import wraps
import asyncio
import hashlib
import json
import pickle
import gzip
import lz4.frame
import zstandard as zstd
from datetime import datetime, timedelta
from uuid import UUID
import inspect

T = TypeVar('T')


def generate_hash(data: Any, algorithm: str = "sha256") -> str:
    """Generate hash of data."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    elif isinstance(data, dict):
        data = json.dumps(data, sort_keys=True).encode('utf-8')
    elif isinstance(data, (list, tuple)):
        data = json.dumps(data).encode('utf-8')
    else:
        data = pickle.dumps(data)
    
    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def compress_data(data: bytes, algorithm: str = "zstd") -> bytes:
    """Compress data using specified algorithm."""
    if algorithm == "gzip":
        return gzip.compress(data)
    elif algorithm == "lz4":
        return lz4.frame.compress(data)
    elif algorithm == "zstd":
        cctx = zstd.ZstdCompressor(level=3)
        return cctx.compress(data)
    else:
        raise ValueError(f"Unknown compression algorithm: {algorithm}")


def decompress_data(data: bytes, algorithm: str = "zstd") -> bytes:
    """Decompress data."""
    if algorithm == "gzip":
        return gzip.decompress(data)
    elif algorithm == "lz4":
        return lz4.frame.decompress(data)
    elif algorithm == "zstd":
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    else:
        raise ValueError(f"Unknown compression algorithm: {algorithm}")


def serialize_json(data: Any) -> str:
    """Serialize to JSON with custom handlers."""
    def default_handler(obj):
        if isinstance(obj, (datetime, UUID)):
            return str(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    
    return json.dumps(data, default=default_handler, ensure_ascii=False)


def deserialize_json(json_str: str, target_type: type = None) -> Any:
    """Deserialize from JSON."""
    data = json.loads(json_str)
    if target_type and hasattr(target_type, 'from_dict'):
        return target_type.from_dict(data)
    return data


async def retry_async(
    func: Callable[..., Awaitable[T]],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
) -> T:
    """Retry async function with exponential backoff."""
    import random
    
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt == max_attempts - 1:
                break
            
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            if jitter:
                delay *= (0.5 + random.random())
            
            await asyncio.sleep(delay)
    
    raise last_exception


def retry_sync(
    func: Callable[..., T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
) -> T:
    """Retry sync function with exponential backoff."""
    import random
    import time
    
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt == max_attempts - 1:
                break
            
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            if jitter:
                delay *= (0.5 + random.random())
            
            time.sleep(delay)
    
    raise last_exception


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        half_open_requests: int = 3
    ):
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._timeout = timeout
        self._half_open_requests = half_open_requests
        
        self._state = "closed"  # closed, open, half_open
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_successes = 0
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> str:
        if self._state == "open":
            if self._last_failure_time and \
               datetime.utcnow() - self._last_failure_time > timedelta(seconds=self._timeout):
                return "half_open"
        return self._state
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        async with self._lock:
            current_state = self.state
            
            if current_state == "open":
                raise Exception("Circuit breaker is open")
            
            if current_state == "half_open" and self._half_open_successes >= self._half_open_requests:
                self._state = "closed"
                self._failure_count = 0
                self._success_count = 0
                self._half_open_successes = 0
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == "half_open":
                self._half_open_successes += 1
                if self._half_open_successes >= self._success_threshold:
                    self._state = "closed"
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_successes = 0
            else:
                self._success_count += 1
                self._failure_count = 0
    
    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = datetime.utcnow()
            
            if self._state == "half_open":
                self._state = "open"
                self._half_open_successes = 0
            elif self._failure_count >= self._failure_threshold:
                self._state = "open"


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, burst: int):
        self._rate = rate  # tokens per second
        self._burst = burst
        self._tokens = float(burst)
        self._last_update = datetime.utcnow()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        async with self._lock:
            now = datetime.utcnow()
            elapsed = (now - self._last_update).total_seconds()
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_update = now
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            
            # Wait for tokens
            wait_time = (tokens - self._tokens) / self._rate
            self._tokens = 0
        
        await asyncio.sleep(wait_time)


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """Split list into chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_function_signature(func: Callable) -> Dict[str, Any]:
    """Get function signature info."""
    sig = inspect.signature(func)
    return {
        "name": func.__name__,
        "module": func.__module__,
        "parameters": {
            name: {
                "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                "default": param.default if param.default != inspect.Parameter.empty else None,
                "kind": param.kind.name
            }
            for name, param in sig.parameters.items()
        },
        "return_annotation": str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else None
    }


class LRUCache:
    """Thread-safe LRU cache."""
    
    def __init__(self, max_size: int = 1000):
        self._max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self._max_size:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]
            
            self._cache[key] = value
            self._access_order.append(key)
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)
                return True
            return False
    
    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()