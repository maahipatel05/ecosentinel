"""
EcoSentinel in-memory TTL cache.
Stores API responses so repeated queries for the same city within the TTL
window return instantly instead of hitting external APIs again.
All entries expire automatically — no manual cleanup needed.
"""

import time
from typing import Any

_STORE: dict[str, tuple[Any, float]] = {}

# TTL constants (seconds) — tuned to each data source's update frequency
TTL_GEOCODE   = 86_400  # 24 h  — city coordinates never change
TTL_SENSOR_ID = 21_600  # 6 h   — sensor assignments are stable within a session
TTL_HISTORY   = 21_600  # 6 h   — historical PM2.5 / wind data doesn't change mid-session
TTL_CURRENT   =    900  # 15 min — live readings update at most hourly
TTL_FIRE      =  1_800  # 30 min — satellite passes every ~90 min
TTL_WIND      =  1_800  # 30 min — archived wind data is stable within a session
TTL_TOOL      =    900  # 15 min — full MCP tool output (air quality, weather, wildfires)


def get(key: str) -> Any | None:
    """Return cached value if present and not yet expired, else None."""
    entry = _STORE.get(key)
    if entry is None:
        return None
    value, expiry = entry
    if time.time() >= expiry:
        del _STORE[key]
        return None
    return value


def set(key: str, value: Any, ttl_seconds: int) -> None:
    """Store a value that expires ttl_seconds from now."""
    _STORE[key] = (value, time.time() + ttl_seconds)


def clear() -> None:
    """Evict all entries. Called in tests to start from a clean state."""
    _STORE.clear()


def size() -> int:
    """Return the number of entries that have not yet expired."""
    now = time.time()
    return sum(1 for _, (_, exp) in _STORE.items() if exp > now)
