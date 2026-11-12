"""Deterministic JSON conversion helpers.

The research record is hash-chained, so the canonical JSON form must be stable
across Python processes and platforms.  ``jsonable`` deliberately normalises a
few values that ``json.dumps`` would otherwise distinguish:

* integer-valued floats (``1.0``) become ``int`` (``1``)
* tuples/sets become JSON arrays (sets are sorted when possible)
* enums become their ``.value``
* datetimes become UTC ISO-8601 strings
* dataclass instances become dictionaries
* bytes become base64 strings
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as _dt
import json
from enum import Enum
from typing import Any, Mapping


def jsonable(value: Any) -> Any:
    """Return a JSON-serialisable, canonical representation of *value*."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return value.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [jsonable(item) for item in value]
        try:
            return sorted(items)
        except TypeError:
            return items
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite floats are not valid JSON")
        return int(value) if value.is_integer() else value
    return value


def canonical_json(value: Any) -> str:
    """Return canonical JSON text (sorted keys, compact separators)."""

    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def to_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        sort_keys=True,
    )


def from_json(text: str | bytes) -> Any:
    return json.loads(text)