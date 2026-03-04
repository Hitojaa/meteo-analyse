"""OpenWeatherMap provider (requires OPENWEATHER_KEY env var).

Strategy: Use the "5 day / 3 hour" forecast endpoint (free tier).
This gives 3-hour intervals; we filter to the target local date,
then compute tmin/tmax from those values.

Docs: https://openweathermap.org/forecast5
"""

from __future__ import annotations

import os
from datetime import datetime

import httpx
from zoneinfo import ZoneInfo

from models import DataQuality, ProviderResult

BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"
PROVIDER_NAME = "OpenWeatherMap"


def fetch(lat: float, lon: float, date: str, timezone: str) -> ProviderResult:
    """Fetch temperature forecast from OpenWeatherMap 5-day/3h endpoint.

    Computes tmin/tmax from 3-hour intervals falling on the target date
    in the city's local timezone.

    Raises:
        RuntimeError: If OPENWEATHER_KEY is not set or no data for the date.
    """
    api_key = os.environ.get("OPENWEATHER_KEY")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError("OPENWEATHER_KEY environment variable not set – skipping OpenWeatherMap")

    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
    }

    with httpx.Client(timeout=10) as client:
        resp = client.get(BASE_URL, params=params)
        resp.raise_for_status()

    data = resp.json()
    tz = ZoneInfo(timezone)
    target_date = datetime.strptime(date, "%Y-%m-%d").date()

    temps: list[float] = []
    for entry in data.get("list", []):
        dt_utc = datetime.fromtimestamp(entry["dt"], tz=ZoneInfo("UTC"))
        dt_local = dt_utc.astimezone(tz)
        if dt_local.date() == target_date:
            temps.append(entry["main"]["temp"])

    if not temps:
        raise RuntimeError(
            f"No forecast data from OpenWeatherMap for {date} "
            f"(timezone {timezone}). The date may be out of range."
        )

    return ProviderResult(
        provider_name=PROVIDER_NAME,
        tmin_c=min(temps),
        tmax_c=max(temps),
        date=date,
        timezone=timezone,
        quality=DataQuality.COMPUTED_FROM_HOURLY,
    )
