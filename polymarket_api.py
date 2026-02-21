"""Polymarket API client for temperature market data.

Uses the Gamma API (public, no auth required) for market discovery
and the CLOB API for real-time prices.

Endpoints:
  - Gamma search: GET https://gamma-api.polymarket.com/public-search
  - CLOB prices:  GET https://clob.polymarket.com/price
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime

import httpx

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketOutcome:
    """A single outcome in a Polymarket temperature market."""
    label: str          # e.g. "44-45" or "8" or "41 or less"
    price: float        # implied probability 0.0–1.0
    token_id: str       # CLOB token ID


@dataclass
class TemperatureMarket:
    """A Polymarket temperature market with outcomes and prices."""
    title: str
    slug: str
    url: str
    volume: float
    outcomes: list[MarketOutcome]
    unit: str           # "°F" or "°C"
    active: bool
    end_date: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_date_search(date: str) -> str:
    """'2026-02-17' → 'February 17'."""
    d = datetime.fromisoformat(date)
    return d.strftime("%B %-d") if hasattr(d, "strftime") else d.strftime("%B %d").lstrip("0").replace(" 0", " ")


def _format_date_search_safe(date: str) -> str:
    """Cross-platform date formatting: '2026-02-17' → 'February 17'."""
    d = datetime.fromisoformat(date)
    # %#d on Windows, %-d on Unix — just use %d and strip leading zero
    month = d.strftime("%B")
    day = str(d.day)
    return f"{month} {day}"


def _detect_unit(title: str, description: str = "") -> str:
    """Detect °F or °C from market title/description."""
    text = (title + " " + description).lower()
    if "fahrenheit" in text or "°f" in text:
        return "°F"
    if "celsius" in text or "°c" in text:
        return "°C"
    # Default: if outcomes have 2-value ranges it's likely °F
    return "°F"


def _city_matches(title: str, city: str) -> bool:
    """Check if the market title mentions our city."""
    title_lower = title.lower()
    city_lower = city.lower().strip()

    # Direct match
    if city_lower in title_lower:
        return True

    # Common aliases
    aliases = {
        "new york": ["new york", "nyc", "new-york"],
        "los angeles": ["los angeles", "la", "los-angeles"],
        "london": ["london"],
        "paris": ["paris"],
    }
    for canonical, names in aliases.items():
        if city_lower == canonical or city_lower in names:
            for name in names:
                if name in title_lower:
                    return True

    return False


def _date_matches(title: str, date: str) -> bool:
    """Check if the market title/slug matches our target date."""
    date_str = _format_date_search_safe(date)
    title_lower = title.lower()

    # "February 17" in title
    if date_str.lower() in title_lower:
        return True

    # "Feb 17" in title
    d = datetime.fromisoformat(date)
    short = f"{d.strftime('%b')} {d.day}"
    if short.lower() in title_lower:
        return True

    return False


def normalize_outcome_key(label: str) -> str:
    """Normalize a Polymarket outcome label for matching.

    Examples:
        "44-45"      → "44-45"
        "44-45°F"    → "44-45"
        "8°C"        → "8"
        "8"          → "8"
        "41 or less" → "le41"
        "41°F or less" → "le41"
        "56 or more" → "ge56"
        "56°F or more" → "ge56"
    """
    s = label.strip()
    # Remove unit markers
    s = re.sub(r"\s*[°]?[FCfc]\s*", " ", s).strip()

    low = s.lower()
    if "or less" in low or "ou moins" in low or "or lower" in low:
        nums = re.findall(r"-?\d+", s)
        return f"le{nums[-1]}" if nums else s
    if "or more" in low or "ou plus" in low or "or higher" in low:
        nums = re.findall(r"-?\d+", s)
        return f"ge{nums[0]}" if nums else s

    # Range like "44-45" or "44 - 45" (two numbers separated by hyphen/dash)
    range_match = re.match(r"^\s*(-?\d+)\s*[-–]\s*(\d+)\s*$", s)
    if range_match:
        return f"{range_match.group(1)}-{range_match.group(2)}"

    # Single value like "8" or "-5"
    single_match = re.match(r"^\s*(-?\d+)\s*$", s)
    if single_match:
        return single_match.group(1)

    return s


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def search_temperature_market(
    city: str, date: str, timeout: float = 10.0
) -> TemperatureMarket | None:
    """Search Polymarket for a temperature market matching city and date.

    Uses the Gamma public-search endpoint. Returns None if no match found.
    """
    date_str = _format_date_search_safe(date)
    query = f"highest temperature {city} {date_str}"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                f"{GAMMA_URL}/public-search",
                params={
                    "q": query,
                    "events_status": "active",
                    "limit_per_type": 10,
                },
            )
            resp.raise_for_status()
    except Exception as exc:
        log.warning("Polymarket search failed: %s", exc)
        return None

    data = resp.json()
    events = data.get("events", [])

    if not events:
        # Try broader search without date
        return _fallback_search(city, date, timeout)

    return _find_best_match(events, city, date, timeout)


def _fallback_search(
    city: str, date: str, timeout: float
) -> TemperatureMarket | None:
    """Broader search if specific query returns nothing."""
    query = f"temperature {city}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                f"{GAMMA_URL}/public-search",
                params={
                    "q": query,
                    "events_status": "active",
                    "limit_per_type": 20,
                },
            )
            resp.raise_for_status()
    except Exception as exc:
        log.warning("Polymarket fallback search failed: %s", exc)
        return None

    data = resp.json()
    events = data.get("events", [])
    return _find_best_match(events, city, date, timeout) if events else None


def _fetch_event_markets(slug: str, timeout: float = 10.0) -> list[dict] | None:
    """Fetch all markets for an event by slug.

    The /public-search endpoint may only return 1 market per event.
    This fetches the full event to get all sub-markets.
    Tries slug-based lookup first, then falls back to the /events/{slug} path.
    """
    with httpx.Client(timeout=timeout) as client:
        # Try 1: GET /events?slug={slug}
        try:
            resp = client.get(
                f"{GAMMA_URL}/events",
                params={"slug": slug},
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                markets = data[0].get("markets", [])
                if markets:
                    return markets
            elif isinstance(data, dict):
                markets = data.get("markets", [])
                if markets:
                    return markets
        except Exception as exc:
            log.debug("Gamma /events?slug= failed: %s", exc)

        # Try 2: GET /events/{slug}
        try:
            resp = client.get(f"{GAMMA_URL}/events/{slug}")
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                markets = data.get("markets", [])
                if markets:
                    return markets
        except Exception as exc:
            log.debug("Gamma /events/{slug} failed: %s", exc)

    log.warning("Could not fetch full event markets for slug: %s", slug)
    return None


def _find_best_match(
    events: list[dict], city: str, date: str, timeout: float = 10.0
) -> TemperatureMarket | None:
    """Find the best matching temperature market from event list."""
    for event in events:
        title = event.get("title", "")

        # Must contain "temperature" and match city + date
        if "temperature" not in title.lower():
            continue
        if not _city_matches(title, city):
            continue
        if not _date_matches(title, date):
            continue

        markets = event.get("markets", [])
        if not markets:
            continue

        slug = event.get("slug", "")

        # The /public-search endpoint often returns only 1 market per event.
        # Fetch the full event to get ALL sub-markets (negRisk events have
        # one binary Yes/No market per temperature range).
        if len(markets) <= 1 and slug:
            full_markets = _fetch_event_markets(slug, timeout)
            if full_markets and len(full_markets) > len(markets):
                log.info("Fetched %d markets for event %s (search had %d)",
                         len(full_markets), slug, len(markets))
                markets = full_markets

        description = event.get("description", "")
        unit = _detect_unit(title, description)

        # NegRisk multi-outcome events: each sub-market is a binary Yes/No
        # for one temperature range. Aggregate them into a single market.
        if len(markets) > 1:
            result = _parse_neg_risk_event(markets, title, slug, unit)
            if result is not None:
                return result
            # Fall through to single-market parsing if negRisk parse failed

        # Single market with multiple outcomes (less common)
        market = markets[0]

        try:
            outcomes_raw = json.loads(market.get("outcomes", "[]"))
            prices_raw = json.loads(market.get("outcomePrices", "[]"))
            token_ids_raw = json.loads(market.get("clobTokenIds", "[]"))
        except (json.JSONDecodeError, TypeError):
            continue

        if not outcomes_raw or not prices_raw:
            continue

        # Build outcomes
        outcomes: list[MarketOutcome] = []
        for i, label in enumerate(outcomes_raw):
            price = float(prices_raw[i]) if i < len(prices_raw) else 0.0
            tid = str(token_ids_raw[i]) if i < len(token_ids_raw) else ""
            outcomes.append(MarketOutcome(
                label=str(label),
                price=price,
                token_id=tid,
            ))

        volume = float(market.get("volume", 0) or 0)
        end_date = market.get("endDate")

        return TemperatureMarket(
            title=title,
            slug=slug,
            url=f"https://polymarket.com/event/{slug}",
            volume=volume,
            outcomes=outcomes,
            unit=unit,
            active=bool(market.get("active", False)),
            end_date=end_date,
        )

    return None


def _parse_neg_risk_event(
    markets: list[dict],
    title: str,
    slug: str,
    unit: str,
) -> TemperatureMarket | None:
    """Parse a negRisk event where each sub-market is Yes/No for one outcome.

    Polymarket temperature events are typically structured as multiple binary
    markets, one per temperature range (e.g. "42-43°F Yes/No"). We aggregate
    them: use groupItemTitle as the label and the Yes price as the probability.
    """
    outcomes: list[MarketOutcome] = []
    total_volume = 0.0
    any_active = False
    end_date = None

    for mkt in markets:
        # Each sub-market should have binary Yes/No outcomes
        try:
            outcomes_raw = json.loads(mkt.get("outcomes", "[]"))
            prices_raw = json.loads(mkt.get("outcomePrices", "[]"))
            token_ids_raw = json.loads(mkt.get("clobTokenIds", "[]"))
        except (json.JSONDecodeError, TypeError):
            continue

        if not outcomes_raw or not prices_raw:
            continue

        # Get the outcome label from groupItemTitle or question
        label = mkt.get("groupItemTitle", "").strip()
        if not label:
            # Try to extract from question: "Will the highest temp be 42-43°F?"
            question = mkt.get("question", "")
            label = _extract_range_from_question(question)
        if not label:
            continue

        # Find the "Yes" price and token
        yes_price = 0.0
        yes_token = ""
        for i, o in enumerate(outcomes_raw):
            if str(o).lower() == "yes":
                if i < len(prices_raw):
                    yes_price = float(prices_raw[i])
                if i < len(token_ids_raw):
                    yes_token = str(token_ids_raw[i])
                break

        outcomes.append(MarketOutcome(
            label=label,
            price=yes_price,
            token_id=yes_token,
        ))

        total_volume += float(mkt.get("volume", 0) or 0)
        if mkt.get("active", False):
            any_active = True
        if end_date is None:
            end_date = mkt.get("endDate")

    if not outcomes:
        return None

    # Refine unit detection from outcome labels (may contain °C or °F)
    for o in outcomes:
        if "°C" in o.label or "celsius" in o.label.lower():
            unit = "°C"
            break
        if "°F" in o.label or "fahrenheit" in o.label.lower():
            unit = "°F"
            break

    return TemperatureMarket(
        title=title,
        slug=slug,
        url=f"https://polymarket.com/event/{slug}",
        volume=total_volume,
        outcomes=outcomes,
        unit=unit,
        active=any_active,
        end_date=end_date,
    )


def _extract_range_from_question(question: str) -> str:
    """Extract temperature range from a market question.

    E.g. "Will the highest temperature be 42-43°F?" → "42-43"
    """
    # Match patterns like "42-43°F", "42-43", "46°F or higher", "38 or less"
    m = re.search(r"(-?\d+(?:\s*[-–]\s*\d+)?)\s*(?:°[FC])?\s*(?:or (?:higher|more|less|lower))?", question)
    if m:
        result = m.group(0).strip().rstrip("?").strip()
        # Remove unit for consistency (normalize_outcome_key will handle it)
        return result
    return ""


def fetch_live_prices(
    market: TemperatureMarket, timeout: float = 10.0
) -> dict[str, float]:
    """Fetch real-time CLOB prices for each outcome.

    Returns a dict mapping token_id → live price.
    Falls back to Gamma snapshot prices on failure.
    """
    token_ids = [o.token_id for o in market.outcomes if o.token_id]
    if not token_ids:
        return {}

    prices: dict[str, float] = {}

    try:
        # Batch POST endpoint: side must be uppercase "BUY"
        payload = [{"token_id": tid, "side": "BUY"} for tid in token_ids]
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{CLOB_URL}/prices", json=payload)
            resp.raise_for_status()

            data = resp.json()
            log.debug("CLOB batch response keys: %s", list(data.keys()) if isinstance(data, dict) else type(data).__name__)
            for tid in token_ids:
                entry = data.get(tid)
                if entry is None:
                    log.debug("CLOB batch: no entry for token %s…%s", tid[:8], tid[-6:])
                    continue
                # Response format: { "tid": { "BUY": "0.58" } }
                if isinstance(entry, dict):
                    buy_price = entry.get("BUY") or entry.get("buy")
                    if buy_price is not None:
                        prices[tid] = float(buy_price)
                    else:
                        log.debug("CLOB batch: entry for %s…%s has no BUY key: %s", tid[:8], tid[-6:], entry)
                # Could also be a direct string/float
                elif isinstance(entry, (str, int, float)):
                    prices[tid] = float(entry)

    except Exception as exc:
        log.warning("CLOB batch price fetch failed: %s", exc)

    # Fallback: individual GET requests for any missing tokens
    missing = [tid for tid in token_ids if tid not in prices]
    if missing:
        log.info("CLOB batch missed %d/%d tokens, trying individual GET…", len(missing), len(token_ids))
        try:
            with httpx.Client(timeout=timeout) as client:
                for tid in missing:
                    try:
                        resp = client.get(
                            f"{CLOB_URL}/price",
                            params={"token_id": tid, "side": "BUY"},
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        # Response: { "price": "0.58" }
                        if "price" in data:
                            prices[tid] = float(data["price"])
                        else:
                            log.warning("CLOB GET /price for %s…%s: unexpected response: %s", tid[:8], tid[-6:], data)
                    except Exception as exc:
                        log.warning("CLOB price for %s…%s failed: %s", tid[:8], tid[-6:], exc)
        except Exception as exc:
            log.warning("CLOB individual price fetch failed: %s", exc)

    log.info("CLOB prices: fetched %d/%d live prices", len(prices), len(token_ids))
    return prices
