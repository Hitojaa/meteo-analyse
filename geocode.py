"""Geocoding with Nominatim (primary) and Open-Meteo (fallback)."""

from __future__ import annotations

import logging

import httpx
from timezonefinder import TimezoneFinder

from models import GeoLocation

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
USER_AGENT = "meteo_avg/1.0 (https://github.com/meteo-avg; weather aggregation CLI)"
_tf = TimezoneFinder()
log = logging.getLogger(__name__)


def _resolve_timezone(lat: float, lon: float) -> str:
    tz = _tf.timezone_at(lat=lat, lng=lon)
    return tz if tz else "UTC"


def _geocode_nominatim(city: str) -> GeoLocation:
    """Try Nominatim first (richer metadata)."""
    params = {
        "q": city,
        "format": "jsonv2",
        "limit": 5,
        "addressdetails": 1,
        "accept-language": "en",
    }
    headers = {"User-Agent": USER_AGENT}

    with httpx.Client(timeout=10) as client:
        resp = client.get(NOMINATIM_URL, params=params, headers=headers)
        resp.raise_for_status()

    results = resp.json()
    if not results:
        raise ValueError(f"City not found via Nominatim: {city!r}")

    best = results[0]
    lat = float(best["lat"])
    lon = float(best["lon"])
    address = best.get("address", {})

    return GeoLocation(
        name=best.get("name", city),
        display_name=best.get("display_name", city),
        lat=lat,
        lon=lon,
        country=address.get("country", ""),
        timezone=_resolve_timezone(lat, lon),
    )


def _geocode_open_meteo(city: str) -> GeoLocation:
    """Fallback: Open-Meteo geocoding API (no key, no strict User-Agent)."""
    params = {"name": city, "count": 5, "language": "en", "format": "json"}

    with httpx.Client(timeout=10) as client:
        resp = client.get(OPEN_METEO_GEO_URL, params=params)
        resp.raise_for_status()

    data = resp.json()
    results = data.get("results")
    if not results:
        raise ValueError(f"City not found via Open-Meteo geocoding: {city!r}")

    best = results[0]
    lat = best["latitude"]
    lon = best["longitude"]
    tz = best.get("timezone", _resolve_timezone(lat, lon))

    country = best.get("country", "")
    admin1 = best.get("admin1", "")
    name = best.get("name", city)
    parts = [p for p in [name, admin1, country] if p]
    display = ", ".join(parts)

    return GeoLocation(
        name=name,
        display_name=display,
        lat=lat,
        lon=lon,
        country=country,
        timezone=tz,
    )


def geocode(city: str) -> GeoLocation:
    """Resolve a city name to geographic coordinates.

    Tries Nominatim first, falls back to Open-Meteo geocoding API.
    """
    try:
        return _geocode_nominatim(city)
    except Exception as exc:
        log.info("Nominatim failed (%s), falling back to Open-Meteo geocoding", exc)

    return _geocode_open_meteo(city)
