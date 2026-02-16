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

# Core models (well-tested, guaranteed by Open-Meteo)
CORE_MODELS: dict[str, str] = {
    "best_match": "Open-Meteo (Best Match)",
    "ecmwf_ifs025": "ECMWF IFS 0.25°",
    "gfs_seamless": "NOAA GFS",
    "icon_seamless": "DWD ICON",
    "meteofrance_seamless": "Météo-France",
    "gem_seamless": "Canadian GEM",
    "jma_seamless": "JMA (Japan)",
    "ukmo_seamless": "UK Met Office",
    "metno_seamless": "MET Norway",
}

# Extra regional models (may not be available for all locations)
EXTRA_MODELS: dict[str, str] = {
    "knmi_seamless": "KNMI (Netherlands)",
    "dmi_seamless": "DMI (Denmark)",
}

MODELS: dict[str, str] = {**CORE_MODELS, **EXTRA_MODELS}


def _query_models(
    client: httpx.Client,
    models: dict[str, str],
    lat: float,
    lon: float,
    date: str,
    timezone: str,
) -> dict:
    """Query Open-Meteo with given models, return the daily dict."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": timezone,
        "start_date": date,
        "end_date": date,
        "models": ",".join(models.keys()),
    }
    resp = client.get(BASE_URL, params=params)
    resp.raise_for_status()
    return resp.json().get("daily", {})


def fetch(lat: float, lon: float, date: str, timezone: str) -> list[ProviderResult]:
    """Fetch daily min/max from Open-Meteo for all available NWP models.

    Tries all models first; if the API rejects the request (e.g. an
    unsupported model name), falls back to core models only.

    Returns one ProviderResult per model.
    """
    with httpx.Client(timeout=15) as client:
        try:
            daily = _query_models(client, MODELS, lat, lon, date, timezone)
            active_models = MODELS
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:
                log.warning(
                    "Open-Meteo rejected full model list, retrying with core models only"
                )
                daily = _query_models(client, CORE_MODELS, lat, lon, date, timezone)
                active_models = CORE_MODELS
            else:
                raise

    results: list[ProviderResult] = []
    for model_id, label in active_models.items():
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
