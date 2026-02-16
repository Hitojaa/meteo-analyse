"""Self-learning forecast verification system.

Stores past forecast snapshots and verifies them against observed
temperatures from the Open-Meteo Archive API. Computes per-provider
accuracy statistics (bias, MAE) for each location, enabling:

  - Bias correction: subtract systematic errors before aggregation
  - Accuracy-based weighting: give more weight to reliable providers
  - Confidence improvement: verified accuracy increases trust

The system improves over time – every run stores predictions and
verifies old ones, building a provider accuracy profile per location.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx

FORECAST_DIR = Path.home() / ".cache" / "meteo_avg" / "forecasts"
ACCURACY_DIR = Path.home() / ".cache" / "meteo_avg" / "accuracy"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

MIN_VERIFY_AGE_DAYS = 5  # Archive API data lag
MAX_ERRORS_STORED = 30  # Rolling window per provider

log = logging.getLogger(__name__)


@dataclass
class ProviderAccuracy:
    """Accuracy stats for a single provider at a specific location."""

    n: int  # number of verified forecasts
    tmin_bias: float  # mean(forecast - actual), positive = warm bias
    tmax_bias: float
    tmin_mae: float  # mean absolute error
    tmax_mae: float


def _location_key(lat: float, lon: float) -> str:
    return f"{lat:.2f}_{lon:.2f}"


# ── Save forecasts ──────────────────────────────────────────────────


def save_forecast(
    lat: float,
    lon: float,
    date_str: str,
    timezone: str,
    providers: list,
    aggregated_tmin: float,
    aggregated_tmax: float,
) -> None:
    """Save forecast snapshot for later verification.

    Only saves if no snapshot exists for this location+date (idempotent).
    """
    key = f"{_location_key(lat, lon)}_{date_str}"
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    path = FORECAST_DIR / f"{key}.json"

    if path.exists():
        return  # already saved

    data = {
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "date": date_str,
        "timezone": timezone,
        "providers": [
            {"name": p.provider_name, "tmin_c": p.tmin_c, "tmax_c": p.tmax_c}
            for p in providers
            if p.tmin_c is not None and p.tmax_c is not None
        ],
        "aggregated": {"tmin_c": aggregated_tmin, "tmax_c": aggregated_tmax},
        "saved_at": time.time(),
        "verified": False,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Verification ────────────────────────────────────────────────────


def _find_verifiable(lat: float, lon: float) -> list[tuple[Path, dict]]:
    """Find past forecasts old enough to verify (>5 days)."""
    if not FORECAST_DIR.exists():
        return []

    prefix = _location_key(lat, lon)
    cutoff = date.today() - timedelta(days=MIN_VERIFY_AGE_DAYS)
    results = []

    for path in FORECAST_DIR.glob(f"{prefix}_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("verified"):
            continue
        forecast_date = date.fromisoformat(data["date"])
        if forecast_date <= cutoff:
            results.append((path, data))

    return results


def _fetch_actuals(
    lat: float,
    lon: float,
    dates: list[str],
    timezone: str,
) -> dict[str, tuple[float, float]]:
    """Fetch actual observed tmin/tmax for given dates from Archive API."""
    if not dates:
        return {}

    sorted_dates = sorted(dates)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": sorted_dates[0],
        "end_date": sorted_dates[-1],
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": timezone,
    }

    with httpx.Client(timeout=20) as client:
        resp = client.get(ARCHIVE_URL, params=params)
        resp.raise_for_status()

    daily = resp.json().get("daily", {})
    times = daily.get("time", [])
    tmins = daily.get("temperature_2m_min", [])
    tmaxs = daily.get("temperature_2m_max", [])

    target_set = set(dates)
    result = {}
    for i, t in enumerate(times):
        if t in target_set:
            tmin_val = tmins[i] if i < len(tmins) else None
            tmax_val = tmaxs[i] if i < len(tmaxs) else None
            if tmin_val is not None and tmax_val is not None:
                result[t] = (tmin_val, tmax_val)

    return result


# ── Accuracy stats persistence ──────────────────────────────────────


def _accuracy_path(lat: float, lon: float) -> Path:
    return ACCURACY_DIR / f"{_location_key(lat, lon)}.json"


def _load_raw(lat: float, lon: float) -> dict | None:
    path = _accuracy_path(lat, lon)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_raw(lat: float, lon: float, data: dict) -> None:
    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)
    path = _accuracy_path(lat, lon)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _compute_stats(raw: dict) -> dict[str, ProviderAccuracy]:
    """Compute ProviderAccuracy from raw error lists."""
    result = {}
    for name, entry in raw.get("providers", {}).items():
        tmin_errs = entry.get("tmin_errors", [])
        tmax_errs = entry.get("tmax_errors", [])
        n = min(len(tmin_errs), len(tmax_errs))
        if n < 1:
            continue
        result[name] = ProviderAccuracy(
            n=n,
            tmin_bias=round(sum(tmin_errs) / n, 2),
            tmax_bias=round(sum(tmax_errs) / n, 2),
            tmin_mae=round(sum(abs(e) for e in tmin_errs) / n, 2),
            tmax_mae=round(sum(abs(e) for e in tmax_errs) / n, 2),
        )
    return result


# ── Main API ────────────────────────────────────────────────────────


def verify_and_update(
    lat: float,
    lon: float,
    timezone: str,
) -> dict[str, ProviderAccuracy] | None:
    """Verify pending forecasts and update accuracy stats.

    Returns the (possibly updated) accuracy stats for this location,
    or None if no data exists yet.
    """
    verifiable = _find_verifiable(lat, lon)

    # If nothing to verify, just return existing stats
    if not verifiable:
        return load_accuracy(lat, lon)

    dates = [data["date"] for _, data in verifiable]

    try:
        actuals = _fetch_actuals(lat, lon, dates, timezone)
    except Exception as exc:
        log.warning("Verification fetch failed: %s", exc)
        return load_accuracy(lat, lon)

    # Load or init raw accuracy data
    raw = _load_raw(lat, lon) or {"providers": {}, "updated_at": 0}

    verified_count = 0
    for path, forecast in verifiable:
        actual = actuals.get(forecast["date"])
        if actual is None:
            continue

        actual_tmin, actual_tmax = actual

        for p in forecast["providers"]:
            name = p["name"]
            tmin_err = round(p["tmin_c"] - actual_tmin, 2)
            tmax_err = round(p["tmax_c"] - actual_tmax, 2)

            if name not in raw["providers"]:
                raw["providers"][name] = {"tmin_errors": [], "tmax_errors": []}

            entry = raw["providers"][name]
            entry["tmin_errors"].append(tmin_err)
            entry["tmax_errors"].append(tmax_err)

            # Rolling window: keep last N errors
            entry["tmin_errors"] = entry["tmin_errors"][-MAX_ERRORS_STORED:]
            entry["tmax_errors"] = entry["tmax_errors"][-MAX_ERRORS_STORED:]

        # Mark as verified
        forecast["verified"] = True
        try:
            path.write_text(json.dumps(forecast, indent=2), encoding="utf-8")
        except OSError:
            pass
        verified_count += 1

    if verified_count > 0:
        raw["updated_at"] = time.time()
        _save_raw(lat, lon, raw)
        log.info("Verified %d forecast(s) for %s", verified_count, _location_key(lat, lon))

    return _compute_stats(raw)


def load_accuracy(
    lat: float,
    lon: float,
) -> dict[str, ProviderAccuracy] | None:
    """Load accuracy stats for a location (without running verification)."""
    raw = _load_raw(lat, lon)
    if raw is None:
        return None
    stats = _compute_stats(raw)
    return stats if stats else None
