"""Open-Meteo provider – multi-model (no API key required).

Docs: https://open-meteo.com/en/docs
Queries multiple NWP models in a single request via the &models= parameter.
Each model returns its own daily min/max, giving us many independent sources.

Also fetches hourly temperature profiles for the best_match model to get
a more precise tmax (max of hourly values), which better matches what
weather stations actually record.
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
    "ecmwf_aifs025": "ECMWF AIFS 0.25° (AI)",
    "gfs_seamless": "NOAA GFS",
    "gfs_graphcast025": "GFS GraphCast (AI)",
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
    "bom_access_global": "BOM ACCESS (Australia)",
    "cma_grapes_global": "CMA GRAPES (China)",
    "arpae_cosmo_seamless": "ARPAE COSMO (Italy)",
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


def _fetch_hourly_peak(
    client: httpx.Client,
    lat: float,
    lon: float,
    date: str,
    timezone: str,
) -> tuple[float | None, float | None]:
    """Fetch hourly temperature profile and return (min, max).

    The hourly max is typically more accurate than daily tmax for
    predicting what a weather station will actually record, as it
    captures the actual peak rather than a smoothed daily estimate.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "timezone": timezone,
        "start_date": date,
        "end_date": date,
    }
    try:
        resp = client.get(BASE_URL, params=params)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        valid = [t for t in temps if t is not None]
        if valid:
            return min(valid), max(valid)
    except Exception as exc:
        log.debug("Hourly fetch failed: %s", exc)
    return None, None


def _fetch_instability(
    client: httpx.Client,
    lat: float,
    lon: float,
    date: str,
    timezone: str,
) -> dict:
    """Fetch weather instability indicators (wind, precipitation, cloud cover).

    Returns a dict with instability metrics that can be used to widen
    the uncertainty envelope when weather conditions are volatile.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": (
            "windspeed_10m_max,windgusts_10m_max,"
            "precipitation_sum,precipitation_probability_max"
        ),
        "timezone": timezone,
        "start_date": date,
        "end_date": date,
    }
    result: dict = {}
    try:
        resp = client.get(BASE_URL, params=params)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})

        def _first(key: str) -> float | None:
            vals = daily.get(key, [])
            return vals[0] if vals and vals[0] is not None else None

        result["wind_max_kmh"] = _first("windspeed_10m_max")
        result["wind_gust_kmh"] = _first("windgusts_10m_max")
        result["precip_mm"] = _first("precipitation_sum")
        result["precip_prob_pct"] = _first("precipitation_probability_max")
    except Exception as exc:
        log.debug("Instability fetch failed: %s", exc)
    return result


def fetch(lat: float, lon: float, date: str, timezone: str) -> list[ProviderResult]:
    """Fetch daily min/max from Open-Meteo for all available NWP models.

    Tries all models first; if the API rejects the request (e.g. an
    unsupported model name), falls back to core models only.

    Also fetches hourly temperatures to add a more precise "hourly peak"
    provider result, and instability data stored on the results.

    Returns one ProviderResult per model, plus an hourly-peak result.
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

        # Fetch hourly peak temperature (more precise than daily tmax)
        hourly_min, hourly_max = _fetch_hourly_peak(
            client, lat, lon, date, timezone
        )

        # Fetch instability indicators
        instability = _fetch_instability(client, lat, lon, date, timezone)

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

    # Add hourly peak as an extra high-weight provider
    if hourly_min is not None and hourly_max is not None:
        results.append(
            ProviderResult(
                provider_name="Open-Meteo (Hourly Peak)",
                tmin_c=round(hourly_min, 1),
                tmax_c=round(hourly_max, 1),
                date=date,
                timezone=timezone,
                quality=DataQuality.HOURLY_PEAK,
            )
        )

    if not results:
        raise RuntimeError("Open-Meteo returned no usable data for any model")

    # Attach instability data to all results for downstream use
    for r in results:
        r.instability = instability

    return results
