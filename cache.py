"""Simple JSON-file cache for API responses, keyed by city+date.

Cache directory: ~/.cache/meteo_avg/
TTL: 30 minutes (configurable).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "meteo_avg"
DEFAULT_TTL = 30 * 60  # 30 minutes in seconds


def _cache_path(city: str, date: str) -> Path:
    safe_key = f"{city.lower().replace(' ', '_')}_{date}"
    return CACHE_DIR / f"{safe_key}.json"


def get(city: str, date: str, ttl: int = DEFAULT_TTL) -> dict | None:
    """Return cached data if it exists and is fresh, else None."""
    path = _cache_path(city, date)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    stored_at = data.get("_cached_at", 0)
    if time.time() - stored_at > ttl:
        return None

    return data


def put(city: str, date: str, payload: dict) -> None:
    """Write data to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(city, date)
    payload["_cached_at"] = time.time()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
