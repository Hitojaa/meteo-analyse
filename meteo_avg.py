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
from models import (
    AggregatedResult,
    ForecastReport,
    GeoLocation,
    ProviderError,
    ProviderResult,
)
from providers import open_meteo, openweather, weatherapi

PROVIDERS = [
    ("Open-Meteo", open_meteo.fetch),
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

    for name, fetch_fn in PROVIDERS:
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
) -> ForecastReport:
    agg = aggregate(results)
    return ForecastReport(
        location=loc,
        date=date,
        timezone=loc.timezone,
        aggregated=agg,
        per_provider=results,
        errors=errors,
    )


def _report_to_dict(report: ForecastReport) -> dict[str, Any]:
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
            "tmin_c": report.aggregated.tmin_c,
            "tmax_c": report.aggregated.tmax_c,
            "tmin_range": list(report.aggregated.tmin_range),
            "tmax_range": list(report.aggregated.tmax_range),
            "sources_used": report.aggregated.sources_used,
            "warning": report.aggregated.warning,
        },
        "per_provider": [
            {
                "provider_name": p.provider_name,
                "tmin_c": p.tmin_c,
                "tmax_c": p.tmax_c,
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


def _print_table(report: ForecastReport) -> None:
    loc = report.location
    agg = report.aggregated

    print(f"\n{'=' * 60}")
    print(f"  Location : {loc.display_name}")
    print(f"  Date     : {report.date}")
    print(f"  Timezone : {report.timezone}")
    print(f"{'=' * 60}")

    if agg.warning:
        print(f"  ⚠  {agg.warning}")

    print(f"\n  Aggregated forecast (°C):")
    print(f"    Tmin : {agg.tmin_c:>6.1f} °C   (range: {agg.tmin_range[0]:.1f} – {agg.tmin_range[1]:.1f})")
    print(f"    Tmax : {agg.tmax_c:>6.1f} °C   (range: {agg.tmax_range[0]:.1f} – {agg.tmax_range[1]:.1f})")
    print(f"    Sources used: {agg.sources_used}")

    print(f"\n  {'Provider':<20} {'Tmin (°C)':>10} {'Tmax (°C)':>10} {'Quality':<22}")
    print(f"  {'-' * 62}")
    for p in report.per_provider:
        tmin_s = f"{p.tmin_c:.1f}" if p.tmin_c is not None else "N/A"
        tmax_s = f"{p.tmax_c:.1f}" if p.tmax_c is not None else "N/A"
        print(f"  {p.provider_name:<20} {tmin_s:>10} {tmax_s:>10} {p.quality.value:<22}")

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

    # --- Fetch from providers ---
    results, errors = _fetch_all(loc, date)

    if not results:
        print("Error: all providers failed.", file=sys.stderr)
        for e in errors:
            print(f"  - {e.provider_name}: {e.error}", file=sys.stderr)
        sys.exit(1)

    # --- Build report ---
    try:
        report = _build_report(loc, date, results, errors)
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
    agg = AggregatedResult(
        tmin_c=agg_d["tmin_c"],
        tmax_c=agg_d["tmax_c"],
        tmin_range=tuple(agg_d["tmin_range"]),
        tmax_range=tuple(agg_d["tmax_range"]),
        sources_used=agg_d["sources_used"],
        warning=agg_d.get("warning"),
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
