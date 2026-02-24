#!/usr/bin/env python3
"""meteo_avg – Intelligent multi-source weather forecast aggregation.

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
from polymarket import analyze as polymarket_analyze, format_analysis as polymarket_format
from polymarket_api import search_temperature_market, fetch_live_prices
from providers import open_meteo, openweather, weatherapi
from verification import ProviderAccuracy, save_forecast, verify_and_update

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
    provider_accuracy: dict[str, ProviderAccuracy] | None = None,
) -> ForecastReport:
    agg = aggregate(results, historical=historical, provider_accuracy=provider_accuracy)
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


def _print_table(
    report: ForecastReport,
    provider_accuracy: dict[str, ProviderAccuracy] | None = None,
) -> None:
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
        print(
            f"\n  Historical context (5-year climatology ±5 days, n={agg.hist_sample_size}):"
        )
        print(
            f"    Tmin avg: {agg.hist_tmin_mean:>5.1f} °C"
            f" (σ {agg.hist_tmin_std:.1f}°C)"
            f"  |  Tmax avg: {agg.hist_tmax_mean:>5.1f} °C"
            f" (σ {agg.hist_tmax_std:.1f}°C)"
        )
        tmin_anom = agg.tmin_c - agg.hist_tmin_mean
        tmax_anom = agg.tmax_c - agg.hist_tmax_mean
        tmin_sign = "+" if tmin_anom >= 0 else ""
        tmax_sign = "+" if tmax_anom >= 0 else ""
        print(
            f"    Anomaly : Tmin {tmin_sign}{tmin_anom:.1f}°C vs normal"
            f"  |  Tmax {tmax_sign}{tmax_anom:.1f}°C vs normal"
        )

    # Self-learning section
    if provider_accuracy:
        verified_providers = [
            (name, acc) for name, acc in provider_accuracy.items() if acc.n >= 3
        ]
        if verified_providers:
            total_verified = max(acc.n for _, acc in verified_providers)
            print(f"\n  Self-learning ({total_verified} verified forecast(s) for this location):")
            # Top 3 most accurate providers by combined MAE
            ranked = sorted(
                verified_providers,
                key=lambda x: (x[1].tmin_mae + x[1].tmax_mae) / 2,
            )
            top = ranked[:3]
            top_str = ", ".join(
                f"{name} (MAE {(a.tmin_mae + a.tmax_mae) / 2:.1f}°C)"
                for name, a in top
            )
            print(f"    Best providers: {top_str}")
            bias_count = sum(1 for _, a in verified_providers if abs(a.tmin_bias) > 0.1 or abs(a.tmax_bias) > 0.1)
            if bias_count > 0:
                print(f"    Bias corrections applied: {bias_count} provider(s) adjusted")

    # Provider table
    has_mae = provider_accuracy and any(
        provider_accuracy.get(p.provider_name) and provider_accuracy[p.provider_name].n >= 3
        for p in report.per_provider
    )

    if has_mae:
        print(
            f"\n  {'Provider':<27} {'Tmin':>14} {'Tmax':>14} {'MAE':>6} {'Quality':<16}"
        )
        print(f"  {'-' * 77}")
    else:
        print(f"\n  {'Provider':<27} {'Tmin':>14} {'Tmax':>14} {'Quality':<16}")
        print(f"  {'-' * 71}")

    for p in report.per_provider:
        if p.tmin_c is not None and p.tmax_c is not None:
            tmin_s = f"{p.tmin_c:.1f}°C/{_c_to_f(p.tmin_c):.1f}°F"
            tmax_s = f"{p.tmax_c:.1f}°C/{_c_to_f(p.tmax_c):.1f}°F"
        else:
            tmin_s = "N/A"
            tmax_s = "N/A"

        if has_mae:
            acc = provider_accuracy.get(p.provider_name) if provider_accuracy else None
            mae_s = (
                f"{(acc.tmin_mae + acc.tmax_mae) / 2:.1f}°"
                if acc and acc.n >= 3
                else "  -"
            )
            print(
                f"  {p.provider_name:<27} {tmin_s:>14} {tmax_s:>14} {mae_s:>6} {p.quality.value:<16}"
            )
        else:
            print(
                f"  {p.provider_name:<27} {tmin_s:>14} {tmax_s:>14} {p.quality.value:<16}"
            )

    if report.errors:
        print(f"\n  Errors:")
        for e in report.errors:
            print(f"    - {e.provider_name}: {e.error}")

    print()


def _run_polymarket_analysis(
    loc: GeoLocation,
    date: str,
    agg: AggregatedResult,
    providers: list[ProviderResult],
    budget: float = 10.0,
) -> tuple | None:
    """Print model-based betting analysis with optional live hedging strategy.

    Returns (market, betting, picks) tuple for use by --buy, or None.
    picks is a list of dicts with keys: label, token_id, price, size, alloc.
    """
    city_short = loc.display_name.split(",")[0].strip()

    # Try to fetch live Polymarket market data for hedging
    market = None
    price_source = "snapshot"
    try:
        market = search_temperature_market(city_short, date)
        if market and market.outcomes:
            # Save Gamma snapshot prices for comparison
            gamma_prices = {o.token_id: o.price for o in market.outcomes if o.token_id}

            # Refresh with live CLOB prices
            live_prices = fetch_live_prices(market)
            n_total = len([o for o in market.outcomes if o.token_id])
            n_live = 0
            if live_prices:
                for o in market.outcomes:
                    if o.token_id in live_prices:
                        o.price = live_prices[o.token_id]
                        n_live += 1

            # Print diagnostic comparison: Gamma snapshot vs CLOB
            print(f"\n  --- Price source diagnostic ---")
            print(f"  {'Outcome':<18} {'Gamma':>7} {'CLOB':>7} {'Δ':>7}")
            for o in market.outcomes:
                gp = gamma_prices.get(o.token_id, 0)
                cp = live_prices.get(o.token_id, 0) if live_prices else 0
                delta = (cp - gp) * 100 if cp > 0 else 0
                src = "CLOB" if o.token_id in (live_prices or {}) else "Gamma"
                print(f"  {o.label:<18} {gp*100:>6.1f}¢ {cp*100:>6.1f}¢ {delta:>+6.1f}¢ [{src}]")
            print()

            if n_live > 0:
                price_source = "live"
            else:
                log.warning(
                    "Polymarket: CLOB returned 0 live prices – using Gamma snapshot (may be stale)"
                )
    except Exception as exc:
        log.warning("Polymarket market fetch: %s", exc)
        market = None

    betting = polymarket_analyze(
        city=city_short,
        date=date,
        country=loc.country,
        agg=agg,
        providers=providers,
        market=market,
        budget=budget,
    )
    # Propagate price source to hedging strategy
    if betting.hedging is not None:
        betting.hedging.price_source = price_source
    print(polymarket_format(betting))

    # Build picks for --buy: extract the top-2 tradeable outcomes
    picks = _extract_buy_picks(betting, market, budget)
    return market, betting, picks


def _extract_buy_picks(betting, market, budget: float = 10.0) -> list[dict]:
    """Extract the top-2 tradeable picks from a BettingAnalysis.

    Returns a list of dicts: [{label, token_id, price, size, alloc}, ...]
    matching the ALLOCATION logic in polymarket.format_analysis.
    """
    h = betting.hedging
    if h is None or not h.bets or market is None:
        return []

    sorted_bins = sorted(betting.bins, key=lambda b: b.prob, reverse=True)
    bankroll = budget

    # Collect top-2 tradeable outcomes by model probability
    raw_picks = []
    for cand in sorted_bins:
        if len(raw_picks) >= 2:
            break
        cand_bet = next((b for b in h.bets if b.label == cand.label), None)
        if cand_bet and cand_bet.market_price > 0:
            raw_picks.append((cand.label, cand.prob, cand_bet.market_price))

    if len(raw_picks) < 2:
        return []

    c1, c2 = raw_picks[0][2], raw_picks[1][2]
    sum_prices = c1 + c2
    shares = bankroll / sum_prices

    picks = []
    for label, prob, price in raw_picks:
        alloc = bankroll * price / sum_prices
        # Find token_id from market outcomes
        token_id = ""
        for o in market.outcomes:
            if o.label == label and o.token_id:
                token_id = o.token_id
                break
        picks.append({
            "label": label,
            "token_id": token_id,
            "price": price,
            "size": round(shares, 1),
            "alloc": round(alloc, 2),
        })

    return picks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Intelligent multi-source weather forecast aggregation."
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
    parser.add_argument(
        "--no-polymarket",
        action="store_true",
        dest="no_polymarket",
        help="Skip Polymarket betting analysis section",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=10.0,
        dest="budget",
        help="Budget in $ for Polymarket betting allocation (default: 10)",
    )
    parser.add_argument(
        "--buy",
        action="store_true",
        dest="buy",
        help="After analysis, prompt to place orders on Polymarket",
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
            report = _rebuild_report_from_cache(cached)
            _print_table(report)

            # Polymarket analysis (also from cache)
            picks = []
            if not args.no_polymarket:
                try:
                    result = _run_polymarket_analysis(loc, date, report.aggregated, report.per_provider, budget=args.budget)
                    if result:
                        _, _, picks = result
                except Exception as exc:
                    log.warning("Polymarket analysis: %s", exc)

            if args.buy:
                _prompt_and_buy(picks)
        return

    # --- Self-learning: verify past predictions & load accuracy ---
    provider_accuracy: dict[str, ProviderAccuracy] | None = None
    try:
        provider_accuracy = verify_and_update(loc.lat, loc.lon, loc.timezone)
    except Exception as exc:
        log.warning("Verification: %s", exc)

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
        report = _build_report(
            loc, date, results, errors,
            historical=historical,
            provider_accuracy=provider_accuracy,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Self-learning: save this forecast for future verification ---
    try:
        save_forecast(
            loc.lat, loc.lon, date, loc.timezone,
            results, report.aggregated.tmin_c, report.aggregated.tmax_c,
        )
    except Exception as exc:
        log.warning("Forecast save: %s", exc)

    # --- Cache result ---
    report_dict = _report_to_dict(report)
    cache.put(args.city, date, report_dict)

    # --- Output ---
    if args.json_output:
        print(json.dumps(report_dict, indent=2, ensure_ascii=False))
    else:
        _print_table(report, provider_accuracy=provider_accuracy)

        # --- Polymarket betting analysis ---
        picks = []
        if not args.no_polymarket:
            try:
                result = _run_polymarket_analysis(loc, date, report.aggregated, results, budget=args.budget)
                if result:
                    _, _, picks = result
            except Exception as exc:
                log.warning("Polymarket analysis: %s", exc)

        if args.buy:
            _prompt_and_buy(picks)


def _prompt_and_buy(picks: list[dict]) -> None:
    """Ask for confirmation and place orders on Polymarket."""
    if not picks:
        print("\n  No tradeable picks found — nothing to buy.")
        return

    # Check for missing token_ids
    missing = [p for p in picks if not p.get("token_id")]
    if missing:
        print("\n  Cannot place orders: missing token IDs for:")
        for p in missing:
            print(f"    - {p['label']}")
        print("  (Market outcomes did not match model bins)")
        return

    print()
    print("  " + "=" * 50)
    print("  ORDER CONFIRMATION")
    print("  " + "=" * 50)
    total = sum(p["alloc"] for p in picks)
    for p in picks:
        print(f"  BUY {p['label']:<10}  {p['size']:.1f} shares @ {p['price'] * 100:.0f}¢"
              f"  (${p['alloc']:.2f})")
    print(f"  {'─' * 50}")
    print(f"  Total cost: ${total:.2f}")
    print()

    try:
        answer = input("  Place these orders? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return

    if answer not in ("y", "yes", "oui", "o"):
        print("  Cancelled.")
        return

    print()
    print("  Placing orders...")

    try:
        from polymarket_buy import place_orders
        results = place_orders(picks)
    except RuntimeError as exc:
        print(f"\n  Error: {exc}", file=sys.stderr)
        return
    except Exception as exc:
        print(f"\n  Unexpected error: {exc}", file=sys.stderr)
        return

    print()
    for r in results:
        if r.success:
            print(f"  {r.label:<10}  OK — order {r.order_id}")
        else:
            print(f"  {r.label:<10}  FAILED — {r.error}")

    ok = sum(1 for r in results if r.success)
    print(f"\n  {ok}/{len(results)} orders placed successfully.")


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
