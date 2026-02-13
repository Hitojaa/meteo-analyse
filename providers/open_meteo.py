"""Open-Meteo provider (no API key required).

Docs: https://open-meteo.com/en/docs
Endpoint returns daily min/max directly.
"""

from __future__ import annotations

import httpx

from models import DataQuality, ProviderResult

BASE_URL = "https://api.open-meteo.com/v1/forecast"
PROVIDER_NAME = "Open-Meteo"


def fetch(lat: float, lon: float, date: str, timezone: str) -> ProviderResult:
    """Fetch daily min/max temperature from Open-Meteo.

    Args:
        lat: Latitude.
        lon: Longitude.
        date: Target date as YYYY-MM-DD.
        timezone: IANA timezone string (e.g. "Europe/Paris").
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": timezone,
        "start_date": date,
        "end_date": date,
    }

    with httpx.Client(timeout=10) as client:
        resp = client.get(BASE_URL, params=params)
        resp.raise_for_status()

    data = resp.json()
    daily = data["daily"]

    tmin = daily["temperature_2m_min"][0]
    tmax = daily["temperature_2m_max"][0]

    return ProviderResult(
        provider_name=PROVIDER_NAME,
        tmin_c=tmin,
        tmax_c=tmax,
        date=date,
        timezone=timezone,
        quality=DataQuality.DAILY_DIRECT,
    )
