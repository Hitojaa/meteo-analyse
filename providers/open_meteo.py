"""Open-Meteo provider – multi-model (no API key required).

Docs: https://open-meteo.com/en/docs
Queries multiple NWP models in a single request via the &models= parameter.
Each model returns its own daily min/max, giving us many independent sources.
"""

from __future__ import annotations

import logging

import httpx

from models import DataQuality, ProviderResult

BASE_URL = "https://api.open-meteo.com/v1/forecast"
log = logging.getLogger(__name__)

# Available models on Open-Meteo with human-readable labels.
# These mirror what Windy shows (ECMWF, GFS, ICON, Météo-France, etc.)
MODELS: dict[str, str] = {
    "best_match": "Open-Meteo (Best Match)",
    "ecmwf_ifs025": "ECMWF IFS 0.25°",
    "gfs_seamless": "NOAA GFS",
    "icon_seamless": "DWD ICON",
    "meteofrance_seamless": "Météo-France",
    "gem_seamless": "Canadian GEM",
    "jma_seamless": "JMA (Japan)",
    "ukmo_seamless": "UK Met Office",
    "metno_seamless": "MET Norway",
    "knmi_seamless": "KNMI (Netherlands)",
    "dmi_seamless": "DMI (Denmark)",
    "arpae_cosmo_seamless": "ARPAE COSMO (Italy)",
}


def fetch(lat: float, lon: float, date: str, timezone: str) -> list[ProviderResult]:
    """Fetch daily min/max from Open-Meteo for all available NWP models.

    Returns one ProviderResult per model.
    """
    model_keys = list(MODELS.keys())
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": timezone,
        "start_date": date,
        "end_date": date,
        "models": ",".join(model_keys),
    }

    with httpx.Client(timeout=15) as client:
        resp = client.get(BASE_URL, params=params)
        resp.raise_for_status()

    data = resp.json()
    daily = data.get("daily", {})

    results: list[ProviderResult] = []
    for model_id, label in MODELS.items():
        # When models are specified, Open-Meteo suffixes keys with _modelname
        # e.g. temperature_2m_max_ecmwf_ifs025
        # The "best_match" model uses the base keys (no suffix).
        if model_id == "best_match":
            tmin_key = "temperature_2m_min"
            tmax_key = "temperature_2m_max"
        else:
            tmin_key = f"temperature_2m_min_{model_id}"
            tmax_key = f"temperature_2m_max_{model_id}"

        tmin_vals = daily.get(tmin_key)
        tmax_vals = daily.get(tmax_key)

        if not tmin_vals or not tmax_vals:
            log.debug("Model %s: no data returned (keys %s/%s missing)", model_id, tmin_key, tmax_key)
            continue

        tmin = tmin_vals[0]
        tmax = tmax_vals[0]

        if tmin is None or tmax is None:
            log.debug("Model %s: null values for %s", model_id, date)
            continue

        results.append(
            ProviderResult(
                provider_name=label,
                tmin_c=tmin,
                tmax_c=tmax,
                date=date,
                timezone=timezone,
                quality=DataQuality.DAILY_DIRECT,
            )
        )

    if not results:
        raise RuntimeError("Open-Meteo returned no usable data for any model")

    return results
