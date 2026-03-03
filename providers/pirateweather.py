"""Pirate Weather provider (requires PIRATEWEATHER_KEY env var).

Docs: https://pirateweather.net/en/latest/
Free tier: 2000 calls/day, returns daily high/low in the target timezone.

Pirate Weather uses HRRR, NBM, and GFS models. The NBM (National Blend
of Models) is itself an ensemble of 13+ NWP models optimized for US
locations, making this provider especially valuable for US cities.
"""

from __future__ import annotations

import os

import httpx

from models import DataQuality, ProviderResult

BASE_URL = "https://api.pirateweather.net/forecast"
PROVIDER_NAME = "Pirate Weather (NBM)"


def fetch(lat: float, lon: float, date: str, timezone: str) -> ProviderResult:
    """Fetch daily min/max temperature from Pirate Weather.

    Uses the time machine endpoint for specific dates.

    Raises:
        RuntimeError: If PIRATEWEATHER_KEY is not set or no data returned.
    """
    api_key = os.environ.get("PIRATEWEATHER_KEY")
    if not api_key:
        raise RuntimeError(
            "PIRATEWEATHER_KEY environment variable not set – skipping Pirate Weather"
        )

    # Pirate Weather uses Unix timestamp for time machine requests
    # We pass the date as a string in ISO format
    url = f"{BASE_URL}/{api_key}/{lat},{lon},{date}T12:00:00"

    params = {
        "units": "si",  # Celsius
        "exclude": "minutely,alerts,flags",
    }

    with httpx.Client(timeout=10) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()

    data = resp.json()
    daily = data.get("daily", {}).get("data", [])

    if not daily:
        raise RuntimeError(f"Pirate Weather returned no daily data for {date}")

    day = daily[0]
    tmin = day.get("temperatureLow") or day.get("temperatureMin")
    tmax = day.get("temperatureHigh") or day.get("temperatureMax")

    if tmin is None or tmax is None:
        raise RuntimeError(f"Pirate Weather: missing temperature data for {date}")

    return ProviderResult(
        provider_name=PROVIDER_NAME,
        tmin_c=round(tmin, 1),
        tmax_c=round(tmax, 1),
        date=date,
        timezone=timezone,
        quality=DataQuality.DAILY_DIRECT,
    )
