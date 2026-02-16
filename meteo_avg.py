#!/usr/bin/env python3
"""meteo_avg – Aggregate weather forecasts from multiple providers.

Usage:
    python meteo_avg.py "Paris"
    python meteo_avg.py "Lyon" --date 2025-06-15
    python meteo_avg.py "Marseille" --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any

from zoneinfo import ZoneInfo

import cache
from aggregate import aggregate
from geocode import geocode
from history import HistoricalStats, fetch_historical
from models import (
    AggregatedResult,
    ForecastReport,
    GeoLocation,
    ProviderError,
    ProviderResult,
)
from providers import open_meteo, openweather, weatherapi

# Providers that return a single ProviderResult
SINGLE_PROVIDERS = [
    ("WeatherAPI", weatherapi.fetch),
    ("OpenWeatherMap", openweather.fetch),
]

logging.basicConfig(
    format="%(levelname)s: %(message)s",
    level=logging.WARNING,
)
log = logging.getLogger("meteo_avg")


def _today_in_tz(tz_name: str) -> str:
    return datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")


def _fetch_all(
    loc: GeoLocation, date: str
) -> tuple[list[ProviderResult], list[ProviderError]]:
    results: list[ProviderResult] = []
    errors: list[ProviderError] = []

    # Open-Meteo multi-model (returns a list of results)
    try:
        results.extend(open_meteo.fetch(loc.lat, loc.lon, date, loc.timezone))
    except Exception as exc:
        log.warning("Open-Meteo: %s", exc)
        errors.append(ProviderError(provider_name="Open-Meteo", error=str(exc)))

    # Single-result providers (WeatherAPI, OpenWeatherMap)
    for name, fetch_fn in SINGLE_PROVIDERS:
        try:
            result = fetch_fn(loc.lat, loc.lon, date, loc.timezone)
            results.append(result)
        except Exception as exc:
            log.warning("%s: %s", name, exc)
            errors.append(ProviderError(provider_name=name, error=str(exc)))

    return results, errors


def _build_report(
    loc: GeoLocation,
    date: str,
    results: list[ProviderResult],
    errors: list[ProviderError],
    historical: HistoricalStats | None = None,
) -> ForecastReport:
    agg = aggregate(results, historical=historical)
    return ForecastReport(
        location=loc,
        date=date,
        timezone=loc.timezone,
        aggregated=agg,
        per_provider=results,
        errors=errors,
    )


def _report_to_dict(report: ForecastReport) -> dict[str, Any]:
    agg = report.aggregated
    return {
        "location": {
            "name": report.location.name,
            "display_name": report.location.display_name,
            "lat": report.location.lat,
            "lon": report.location.lon,
            "country": report.location.country,
            "timezone": report.location.timezone,
        },
        "date": report.date,
        "timezone": report.timezone,
        "aggregated": {
            "tmin_c": agg.tmin_c,
            "tmax_c": agg.tmax_c,
            "tmin_f": _c_to_f(agg.tmin_c),
            "tmax_f": _c_to_f(agg.tmax_c),
            "tmin_range": list(agg.tmin_range),
            "tmax_range": list(agg.tmax_range),
            "sources_used": agg.sources_used,
            "confidence": agg.confidence,
            "skewness_tmin": agg.skewness_tmin,
            "skewness_tmax": agg.skewness_tmax,
            "warning": agg.warning,
            "historical": {
                "tmin_mean": agg.hist_tmin_mean,
                "tmax_mean": agg.hist_tmax_mean,
                "tmin_std": agg.hist_tmin_std,
                "tmax_std": agg.hist_tmax_std,
                "sample_size": agg.hist_sample_size,
            }
            if agg.hist_tmin_mean is not None
            else None,
        },
        "per_provider": [
            {
                "provider_name": p.provider_name,
                "tmin_c": p.tmin_c,
                "tmax_c": p.tmax_c,
                "tmin_f": _c_to_f(p.tmin_c) if p.tmin_c is not None else None,
                "tmax_f": _c_to_f(p.tmax_c) if p.tmax_c is not None else None,
                "date": p.date,
                "timezone": p.timezone,
                "quality": p.quality.value,
            }
            for p in report.per_provider
        ],
        "errors": [
            {"provider_name": e.provider_name, "error": e.error}
            for e in report.errors
        ],
    }


def _c_to_f(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def _skew_label(skew: float) -> str:
    """Human-readable skewness interpretation."""
    if abs(skew) < 0.3:
        return "symmetric"
    elif skew > 0:
        return f"+{skew:.2f} (high bias)"
    else:
        return f"{skew:.2f} (low bias)"


def _print_table(report: ForecastReport) -> None:
    loc = report.location
    agg = report.aggregated

    print(f"\n{'=' * 76}")
    print(f"  Location : {loc.display_name}")
    print(f"  Date     : {report.date}")
    print(f"  Timezone : {report.timezone}")
    print(f"{'=' * 76}")

    if agg.warning:
        print(f"  ⚠  {agg.warning}")

    print(f"\n  Aggregated forecast:")
    print(
        f"    Tmin : {agg.tmin_c:>6.1f} °C / {_c_to_f(agg.tmin_c):>6.1f} °F"
        f"   (range: {agg.tmin_range[0]:.1f} – {agg.tmin_range[1]:.1f} °C)"
    )
    print(
        f"    Tmax : {agg.tmax_c:>6.1f} °C / {_c_to_f(agg.tmax_c):>6.1f} °F"
        f"   (range: {agg.tmax_range[0]:.1f} – {agg.tmax_range[1]:.1f} °C)"
    )
    print(f"    Sources used: {agg.sources_used}")
    print(f"    Confidence: {agg.confidence:.0f}%")
    print(
        f"    Skewness: Tmin {_skew_label(agg.skewness_tmin)}"
        f"  |  Tmax {_skew_label(agg.skewness_tmax)}"
    )

    # Historical context
    if agg.hist_tmin_mean is not None:
        print(f"\n  Historical context (5-year climatology ±5 days, n={agg.hist_sample_size}):")
        print(
            f"    Tmin avg: {agg.hist_tmin_mean:>5.1f} °C"
            f" (σ {agg.hist_tmin_std:.1f}°C)"
            f"  |  Tmax avg: {agg.hist_tmax_mean:>5.1f} °C"
            f" (σ {agg.hist_tmax_std:.1f}°C)"
        )
        # Show anomaly
        tmin_anom = agg.tmin_c - agg.hist_tmin_mean
        tmax_anom = agg.tmax_c - agg.hist_tmax_mean
        tmin_sign = "+" if tmin_anom >= 0 else ""
        tmax_sign = "+" if tmax_anom >= 0 else ""
        print(
            f"    Anomaly : Tmin {tmin_sign}{tmin_anom:.1f}°C vs normal"
            f"  |  Tmax {tmax_sign}{tmax_anom:.1f}°C vs normal"
        )

    # Provider table – wider to fit more models
    print(f"\n  {'Provider':<25} {'Tmin':>14} {'Tmax':>14} {'Quality':<22}")
    print(f"  {'-' * 75}")
    for p in report.per_provider:
        if p.tmin_c is not None and p.tmax_c is not None:
            tmin_s = f"{p.tmin_c:.1f}°C/{_c_to_f(p.tmin_c):.1f}°F"
            tmax_s = f"{p.tmax_c:.1f}°C/{_c_to_f(p.tmax_c):.1f}°F"
        else:
            tmin_s = "N/A"
            tmax_s = "N/A"
        print(
            f"  {p.provider_name:<25} {tmin_s:>14} {tmax_s:>14} {p.quality.value:<22}"
        )

    if report.errors:
        print(f"\n  Errors:")
        for e in report.errors:
            print(f"    - {e.provider_name}: {e.error}")

    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate weather forecasts from multiple providers."
    )
    parser.add_argument("city", help="City name (e.g. 'Paris', 'Lyon')")
    parser.add_argument(
        "--date",
        default=None,
        help="Target date YYYY-MM-DD (default: today in city timezone)",
    )
    parser.add_argument(
        "--units",
        default="c",
        choices=["c"],
        help="Temperature units (only Celsius supported)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output structured JSON",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        dest="no_history",
        help="Skip historical data fetch (faster, less accurate)",
    )
    args = parser.parse_args(argv)

    # --- Geocode ---
    try:
        loc = geocode(args.city)
    except Exception as exc:
        print(f"Error: could not geocode '{args.city}': {exc}", file=sys.stderr)
        sys.exit(1)

    date = args.date or _today_in_tz(loc.timezone)

    # --- Check cache ---
    cached = cache.get(args.city, date)
    if cached:
        cached.pop("_cached_at", None)
        if args.json_output:
            print(json.dumps(cached, indent=2, ensure_ascii=False))
        else:
            # Rebuild report from cache for display
            report = _rebuild_report_from_cache(cached)
            _print_table(report)
        return

    # --- Fetch historical data ---
    historical: HistoricalStats | None = None
    if not args.no_history:
        try:
            historical = fetch_historical(loc.lat, loc.lon, date, loc.timezone)
        except Exception as exc:
            log.warning("Historical data: %s", exc)

    # --- Fetch from providers ---
    results, errors = _fetch_all(loc, date)

    if not results:
        print("Error: all providers failed.", file=sys.stderr)
        for e in errors:
            print(f"  - {e.provider_name}: {e.error}", file=sys.stderr)
        sys.exit(1)

    # --- Build report ---
    try:
        report = _build_report(loc, date, results, errors, historical=historical)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Cache result ---
    report_dict = _report_to_dict(report)
    cache.put(args.city, date, report_dict)

    # --- Output ---
    if args.json_output:
        print(json.dumps(report_dict, indent=2, ensure_ascii=False))
    else:
        _print_table(report)


def _rebuild_report_from_cache(data: dict) -> ForecastReport:
    """Reconstruct a ForecastReport from cached dict."""
    from models import DataQuality

    loc_d = data["location"]
    loc = GeoLocation(
        name=loc_d["name"],
        display_name=loc_d["display_name"],
        lat=loc_d["lat"],
        lon=loc_d["lon"],
        country=loc_d["country"],
        timezone=loc_d["timezone"],
    )
    providers = [
        ProviderResult(
            provider_name=p["provider_name"],
            tmin_c=p["tmin_c"],
            tmax_c=p["tmax_c"],
            date=p["date"],
            timezone=p["timezone"],
            quality=DataQuality(p["quality"]),
        )
        for p in data.get("per_provider", [])
    ]
    errs = [
        ProviderError(provider_name=e["provider_name"], error=e["error"])
        for e in data.get("errors", [])
    ]
    agg_d = data["aggregated"]
    hist_d = agg_d.get("historical")
    agg = AggregatedResult(
        tmin_c=agg_d["tmin_c"],
        tmax_c=agg_d["tmax_c"],
        tmin_range=tuple(agg_d["tmin_range"]),
        tmax_range=tuple(agg_d["tmax_range"]),
        sources_used=agg_d["sources_used"],
        warning=agg_d.get("warning"),
        skewness_tmin=agg_d.get("skewness_tmin", 0.0),
        skewness_tmax=agg_d.get("skewness_tmax", 0.0),
        confidence=agg_d.get("confidence", 0.0),
        hist_tmin_mean=hist_d["tmin_mean"] if hist_d else None,
        hist_tmax_mean=hist_d["tmax_mean"] if hist_d else None,
        hist_tmin_std=hist_d["tmin_std"] if hist_d else None,
        hist_tmax_std=hist_d["tmax_std"] if hist_d else None,
        hist_sample_size=hist_d["sample_size"] if hist_d else None,
    )
    return ForecastReport(
        location=loc,
        date=data["date"],
        timezone=data["timezone"],
        aggregated=agg,
        per_provider=providers,
        errors=errs,
    )


if __name__ == "__main__":
    main()
