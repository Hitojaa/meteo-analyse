"""WeatherAPI.com provider (requires WEATHERAPI_KEY env var).

Docs: https://www.weatherapi.com/docs/
Free tier: forecast up to 3 days, returns daily min/max in °C.
"""

from __future__ import annotations

import os

import httpx

from models import DataQuality, ProviderResult

BASE_URL = "https://api.weatherapi.com/v1/forecast.json"
PROVIDER_NAME = "WeatherAPI"


def fetch(lat: float, lon: float, date: str, timezone: str) -> ProviderResult:
    """Fetch daily min/max temperature from WeatherAPI.com.

    Raises:
        RuntimeError: If WEATHERAPI_KEY is not set.
    """
    api_key = os.environ.get("WEATHERAPI_KEY")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError("WEATHERAPI_KEY environment variable not set – skipping WeatherAPI")

    params = {
        "key": api_key,
        "q": f"{lat},{lon}",
        "dt": date,
        "aqi": "no",
        "alerts": "no",
    }

    with httpx.Client(timeout=10) as client:
        resp = client.get(BASE_URL, params=params)
        resp.raise_for_status()

    data = resp.json()
    day = data["forecast"]["forecastday"][0]["day"]

    tmin = day["mintemp_c"]
    tmax = day["maxtemp_c"]

    return ProviderResult(
        provider_name=PROVIDER_NAME,
        tmin_c=tmin,
        tmax_c=tmax,
        date=date,
        timezone=timezone,
        quality=DataQuality.DAILY_DIRECT,
    )
