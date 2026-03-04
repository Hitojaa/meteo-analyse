"""Pirate Weather provider (requires PIRATEWEATHER_KEY env var).

Docs: https://docs.pirateweather.net/en/latest/API/
Free tier: 10,000 calls/month, returns daily high/low.

Pirate Weather uses HRRR, NBM, and GFS models. The NBM (National Blend
of Models) is itself an ensemble of 13+ NWP models optimized for US
locations, making this provider especially valuable for US cities.

API is Dark Sky-compatible:
  - Forecast: GET /forecast/{key}/{lat},{lon}
  - Time Machine: GET /forecast/{key}/{lat},{lon},{unix_timestamp}
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx

from models import DataQuality, ProviderResult

BASE_URL = "https://api.pirateweather.net/forecast"
PROVIDER_NAME = "Pirate Weather (NBM)"


def fetch(lat: float, lon: float, date: str, tz: str) -> ProviderResult:
    """Fetch daily min/max temperature from Pirate Weather.

    For future dates (within 7 days): uses the forecast endpoint and
    finds the matching day in the daily array.
    For past/specific dates: uses the time machine endpoint with a
    Unix timestamp.

    Raises:
        RuntimeError: If PIRATEWEATHER_KEY is not set or no data returned.
    """
    api_key = os.environ.get("PIRATEWEATHER_KEY")
    if not api_key:
        raise RuntimeError(
            "PIRATEWEATHER_KEY environment variable not set – skipping Pirate Weather"
        )

    target = datetime.fromisoformat(date).date()
    now = datetime.now().date()
    days_ahead = (target - now).days

    if days_ahead >= 0 and days_ahead <= 7:
        # Future date within forecast range — use forecast endpoint
        url = f"{BASE_URL}/{api_key}/{lat},{lon}"
    else:
        # Past or far-future date — use time machine with Unix timestamp
        # Set to noon UTC on the target date
        dt = datetime(target.year, target.month, target.day, 12, 0, 0,
                      tzinfo=timezone.utc)
        unix_ts = int(dt.timestamp())
        url = f"{BASE_URL}/{api_key}/{lat},{lon},{unix_ts}"

    params = {
        "units": "si",  # Celsius
        "exclude": "minutely,alerts,flags",
    }

    with httpx.Client(timeout=10) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()

    data = resp.json()
    daily_list = data.get("daily", {}).get("data", [])

    if not daily_list:
        raise RuntimeError(f"Pirate Weather returned no daily data for {date}")

    # Find the day matching our target date
    target_str = target.strftime("%Y-%m-%d")
    day = None
    for d in daily_list:
        day_ts = d.get("time", 0)
        day_date = datetime.fromtimestamp(day_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if day_date == target_str:
            day = d
            break

    # Fallback: if no exact match, use the first day (time machine returns 1 day)
    if day is None:
        day = daily_list[0]

    tmin = day.get("temperatureLow") or day.get("temperatureMin")
    tmax = day.get("temperatureHigh") or day.get("temperatureMax")

    if tmin is None or tmax is None:
        raise RuntimeError(f"Pirate Weather: missing temperature data for {date}")

    return ProviderResult(
        provider_name=PROVIDER_NAME,
        tmin_c=round(tmin, 1),
        tmax_c=round(tmax, 1),
        date=date,
        timezone=tz,
        quality=DataQuality.DAILY_DIRECT,
    )
