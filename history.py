"""Historical temperature climatology via Open-Meteo Archive API.

Fetches past temperature data to establish climatological normals
for a given location and date. Used for:
  - Bayesian anchoring of forecasts toward climate norms
  - Skewness direction validation
  - Confidence scoring

The Archive API is free and requires no API key.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, stdev

import httpx

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE_DIR = Path.home() / ".cache" / "meteo_avg" / "history"
CACHE_TTL = 30 * 24 * 3600  # 30 days – historical data doesn't change

log = logging.getLogger(__name__)


@dataclass
class HistoricalStats:
    """Climatological statistics for a location and date window."""

    tmin_mean: float
    tmin_std: float
    tmax_mean: float
    tmax_std: float
    sample_size: int
    years_covered: int


def _cache_key(lat: float, lon: float, month: int, day: int) -> str:
    return f"hist_{lat:.2f}_{lon:.2f}_{month:02d}_{day:02d}"


def _get_cached(key: str) -> HistoricalStats | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - data.get("_cached_at", 0) > CACHE_TTL:
        return None
    return HistoricalStats(
        tmin_mean=data["tmin_mean"],
        tmin_std=data["tmin_std"],
        tmax_mean=data["tmax_mean"],
        tmax_std=data["tmax_std"],
        sample_size=data["sample_size"],
        years_covered=data["years_covered"],
    )


def _put_cached(key: str, stats: HistoricalStats) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    data = {
        "tmin_mean": stats.tmin_mean,
        "tmin_std": stats.tmin_std,
        "tmax_mean": stats.tmax_mean,
        "tmax_std": stats.tmax_std,
        "sample_size": stats.sample_size,
        "years_covered": stats.years_covered,
        "_cached_at": time.time(),
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def fetch_historical(
    lat: float,
    lon: float,
    target_date: str,
    timezone: str,
    years: int = 5,
    window_days: int = 5,
) -> HistoricalStats | None:
    """Fetch historical temperature stats for the same period across past years.

    Queries the Open-Meteo Archive API for a date range spanning `years`
    years back, then filters for a ±`window_days` window around the target
    date's day-of-year. Returns mean and std for tmin and tmax.

    Results are cached for 30 days (historical data doesn't change).
    """
    target = date.fromisoformat(target_date)
    cache_key = _cache_key(lat, lon, target.month, target.day)

    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    # Archive API data is available up to ~5 days ago
    end = min(target - timedelta(days=1), date.today() - timedelta(days=5))
    start = date(target.year - years, 1, 1)

    if start >= end:
        return None

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": timezone,
        }

        with httpx.Client(timeout=20) as client:
            resp = client.get(ARCHIVE_URL, params=params)
            resp.raise_for_status()

        data = resp.json()
        daily = data.get("daily", {})
        times = daily.get("time", [])
        tmins_raw = daily.get("temperature_2m_min", [])
        tmaxs_raw = daily.get("temperature_2m_max", [])
    except Exception as exc:
        log.warning("Historical data fetch failed: %s", exc)
        return None

    # Filter for the target day-of-year ± window
    target_doy = target.timetuple().tm_yday
    current_year = target.year

    filtered_tmins: list[float] = []
    filtered_tmaxs: list[float] = []
    filtered_weights: list[float] = []
    years_seen: set[int] = set()

    for i, date_str in enumerate(times):
        d = date.fromisoformat(date_str)
        doy = d.timetuple().tm_yday

        # Handle year boundary (e.g., Jan 2 vs Dec 30)
        doy_diff = min(abs(doy - target_doy), 365 - abs(doy - target_doy))

        if doy_diff <= window_days:
            tmin_val = tmins_raw[i] if i < len(tmins_raw) else None
            tmax_val = tmaxs_raw[i] if i < len(tmaxs_raw) else None

            if tmin_val is not None and tmax_val is not None:
                filtered_tmins.append(tmin_val)
                filtered_tmaxs.append(tmax_val)
                years_seen.add(d.year)

                # Exponential recency weight: recent years count more
                # year_gap=0 → weight=1.0, gap=1 → 0.85, gap=2 → 0.72
                year_gap = max(0, current_year - d.year)
                filtered_weights.append(0.85 ** year_gap)

    if len(filtered_tmins) < 5:
        log.info("Insufficient historical data: %d samples", len(filtered_tmins))
        return None

    # Weighted mean and std — recent years weighted more heavily
    total_w = sum(filtered_weights)
    w_tmin_mean = sum(v * w for v, w in zip(filtered_tmins, filtered_weights)) / total_w
    w_tmax_mean = sum(v * w for v, w in zip(filtered_tmaxs, filtered_weights)) / total_w

    if len(filtered_tmins) > 1:
        w_tmin_var = sum(
            w * (v - w_tmin_mean) ** 2
            for v, w in zip(filtered_tmins, filtered_weights)
        ) / total_w
        w_tmax_var = sum(
            w * (v - w_tmax_mean) ** 2
            for v, w in zip(filtered_tmaxs, filtered_weights)
        ) / total_w
        w_tmin_std = w_tmin_var ** 0.5
        w_tmax_std = w_tmax_var ** 0.5
    else:
        w_tmin_std = 0.0
        w_tmax_std = 0.0

    stats = HistoricalStats(
        tmin_mean=round(w_tmin_mean, 1),
        tmin_std=round(w_tmin_std, 1),
        tmax_mean=round(w_tmax_mean, 1),
        tmax_std=round(w_tmax_std, 1),
        sample_size=len(filtered_tmins),
        years_covered=len(years_seen),
    )

    _put_cached(cache_key, stats)
    return stats
