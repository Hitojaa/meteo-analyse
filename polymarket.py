"""Polymarket temperature betting analysis.

Uses the multi-model forecast distribution to compute probability
for each temperature range and recommend optimal bets.

When a live Polymarket market is found, compares our model's
probabilities with market prices to identify value bets (edge > 0).

Polymarket temperature markets resolve to the highest temperature
recorded at a specific station, measured in whole degrees:
  - US cities  → °F, bins of 2°F  (e.g. 44-45°F)
  - Non-US     → °C, bins of 1°C  (e.g. 8°C)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from models import AggregatedResult, ProviderResult
from polymarket_api import TemperatureMarket, normalize_outcome_key


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BettingBin:
    """A temperature range bin with its probability."""
    label: str
    prob: float
    market_price: float | None = None   # Polymarket price (0.0-1.0) if found
    edge: float | None = None           # our_prob - market_price
    is_best: bool = False
    is_value_bet: bool = False          # edge > threshold


@dataclass
class BettingAnalysis:
    """Complete betting analysis for a Polymarket temperature market."""
    city: str
    date: str
    unit: str           # "°F" or "°C"
    predicted: float    # predicted tmax in market unit
    sigma: float        # uncertainty (1σ) in market unit
    bins: list[BettingBin]
    skewness: float
    confidence: float
    market_found: bool = False
    market_url: str | None = None
    market_volume: float | None = None
    value_bets: list[BettingBin] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """CDF of Normal(mu, sigma) evaluated at x."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


# ---------------------------------------------------------------------------
# Country detection
# ---------------------------------------------------------------------------

_US_NAMES = {
    "united states", "united states of america",
    "usa", "us", "états-unis", "etats-unis",
}


def _is_us(country: str) -> bool:
    return country.lower().strip() in _US_NAMES


# ---------------------------------------------------------------------------
# Market matching
# ---------------------------------------------------------------------------

# Minimum edge to flag as value bet (5 percentage points)
VALUE_BET_THRESHOLD = 0.05


def match_with_market(
    bins: list[BettingBin],
    market: TemperatureMarket,
) -> None:
    """Match our bins with Polymarket outcomes and compute edge.

    Modifies bins in-place: sets market_price, edge, is_value_bet.
    """
    # Build lookup: normalized key → market price
    market_lookup: dict[str, float] = {}
    for outcome in market.outcomes:
        key = normalize_outcome_key(outcome.label)
        market_lookup[key] = outcome.price

    for b in bins:
        our_key = normalize_outcome_key(b.label)

        if our_key in market_lookup:
            b.market_price = market_lookup[our_key]
            b.edge = b.prob - b.market_price
            if b.edge >= VALUE_BET_THRESHOLD - 1e-9:
                b.is_value_bet = True
        else:
            # Try fuzzy: match by checking if our range overlaps
            b.market_price = _fuzzy_match(our_key, market_lookup)
            if b.market_price is not None:
                b.edge = b.prob - b.market_price
                if b.edge >= VALUE_BET_THRESHOLD - 1e-9:
                    b.is_value_bet = True


def _fuzzy_match(key: str, lookup: dict[str, float]) -> float | None:
    """Try to match a bin key with close market outcomes."""
    # For tail bins, try slight variations
    if key.startswith("le") or key.startswith("ge"):
        # Try exact match with ± 1
        prefix = key[:2]
        try:
            num = int(key[2:])
        except ValueError:
            return None
        for delta in [0, -1, 1, -2, 2]:
            test_key = f"{prefix}{num + delta}"
            if test_key in lookup:
                return lookup[test_key]
    return None


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze(
    city: str,
    date: str,
    country: str,
    agg: AggregatedResult,
    providers: list[ProviderResult],
    market: TemperatureMarket | None = None,
) -> BettingAnalysis:
    """Build a probability distribution over Polymarket temperature bins.

    The distribution is modeled as Normal(center, sigma) where:
      - center = aggregated tmax (bias-corrected, climate-anchored) + skewness nudge
      - sigma  = blended provider ensemble spread & historical variability

    Bins match Polymarket conventions:
      - US: 2°F bins aligned to even integers (42-43, 44-45, …)
      - Non-US: 1°C bins (7, 8, 9, …)

    If a live TemperatureMarket is provided, matches bins with market
    outcomes and computes edge (our_prob - market_price).
    """
    use_f = _is_us(country)
    unit = "°F" if use_f else "°C"

    # If market provides different unit, use market's unit
    if market is not None:
        if market.unit == "°F":
            use_f = True
            unit = "°F"
        elif market.unit == "°C":
            use_f = False
            unit = "°C"

    # --- Collect provider tmax in °C ---
    tmax_c = [
        p.tmax_c for p in providers
        if p.tmax_c is not None and not math.isnan(p.tmax_c)
    ]
    if not tmax_c:
        raise ValueError("No valid tmax data for betting analysis")

    center_c = agg.tmax_c

    # --- Compute uncertainty in °C ---
    if len(tmax_c) > 1:
        pstd = (sum((v - center_c) ** 2 for v in tmax_c) / len(tmax_c)) ** 0.5
    else:
        pstd = 1.5

    sigma_c = pstd * 1.2

    if agg.hist_tmax_std is not None and agg.hist_tmax_std > 0:
        sigma_c = max(sigma_c, agg.hist_tmax_std * 0.45)

    sigma_c = max(sigma_c, 0.8)

    # --- Convert to market unit ---
    if use_f:
        center = _c_to_f(center_c)
        sigma = sigma_c * 1.8
        step = 2
    else:
        center = center_c
        sigma = sigma_c
        step = 1

    skew = agg.skewness_tmax
    center += skew * sigma * 0.10

    # --- Generate bins ---
    # If market exists, generate bins matching market outcomes
    if market is not None and market.outcomes:
        bins = _bins_from_market(market, center, sigma, use_f, unit)
    else:
        bins = _bins_from_model(center, sigma, use_f, unit, step)

    # Mark the best bin (highest our-model probability)
    if bins:
        best = max(bins, key=lambda b: b.prob)
        best.is_best = True

    # Match with market if available
    if market is not None:
        match_with_market(bins, market)

    # Find value bets
    value_bets = [b for b in bins if b.is_value_bet]

    return BettingAnalysis(
        city=city,
        date=date,
        unit=unit,
        predicted=round(center, 1),
        sigma=round(sigma, 1),
        bins=bins,
        skewness=skew,
        confidence=agg.confidence,
        market_found=market is not None,
        market_url=market.url if market else None,
        market_volume=market.volume if market else None,
        value_bets=value_bets,
    )


def _bins_from_market(
    market: TemperatureMarket,
    center: float,
    sigma: float,
    use_f: bool,
    unit: str,
) -> list[BettingBin]:
    """Generate bins matching the exact Polymarket market outcomes."""
    bins: list[BettingBin] = []

    for outcome in market.outcomes:
        label = outcome.label
        # Add unit to label if not present
        if "°" not in label and ("or" not in label.lower()):
            label = f"{label}{unit}"
        elif "or" in label.lower() and "°" not in label:
            label = label.replace(" or ", f"{unit} or ")

        # Compute our probability for this exact bin
        prob = _prob_for_outcome(outcome.label, center, sigma, use_f)
        bins.append(BettingBin(label=label, prob=prob))

    return bins


def _prob_for_outcome(
    label: str, center: float, sigma: float, use_f: bool,
) -> float:
    """Compute our model's probability for a Polymarket outcome."""
    import re
    low_text = label.lower()

    # Tail: "X or less" / "X or lower"
    if "or less" in low_text or "or lower" in low_text or "ou moins" in low_text:
        nums = re.findall(r"-?\d+", label)
        if nums:
            upper = int(nums[-1])
            return _normal_cdf(upper + 0.5, center, sigma)
        return 0.0

    # Tail: "X or more" / "X or higher"
    if "or more" in low_text or "or higher" in low_text or "ou plus" in low_text:
        nums = re.findall(r"-?\d+", label)
        if nums:
            lower = int(nums[0])
            return 1.0 - _normal_cdf(lower - 0.5, center, sigma)
        return 0.0

    # Range: "44-45" or "44-45°F"
    range_match = re.match(r".*?(-?\d+)\s*[-–]\s*(\d+)", label)
    if range_match:
        lo_val = int(range_match.group(1))
        hi_val = int(range_match.group(2))
        return _normal_cdf(hi_val + 0.5, center, sigma) - _normal_cdf(lo_val - 0.5, center, sigma)

    # Single value: "8" or "8°C"
    single_match = re.search(r"(-?\d+)", label)
    if single_match:
        val = int(single_match.group(1))
        return _normal_cdf(val + 0.5, center, sigma) - _normal_cdf(val - 0.5, center, sigma)

    return 0.0


def _bins_from_model(
    center: float,
    sigma: float,
    use_f: bool,
    unit: str,
    step: int,
) -> list[BettingBin]:
    """Generate bins from our model (no market data)."""
    center_int = round(center)
    if use_f:
        base = (center_int // 2) * 2
    else:
        base = center_int

    half_range = max(3, math.ceil(3.5 * sigma / step)) * step
    lo = base - half_range
    hi = base + half_range

    raw_bins: list[tuple[int, str, float]] = []
    v = lo
    while v <= hi:
        if use_f:
            p = _normal_cdf(v + 1.5, center, sigma) - _normal_cdf(v - 0.5, center, sigma)
            label = f"{v}-{v + 1}{unit}"
        else:
            p = _normal_cdf(v + 0.5, center, sigma) - _normal_cdf(v - 0.5, center, sigma)
            label = f"{v}{unit}"
        raw_bins.append((v, label, p))
        v += step

    min_p = 0.005

    first_sig = 0
    for i, (_, _, p) in enumerate(raw_bins):
        if p >= min_p:
            first_sig = i
            break

    last_sig = len(raw_bins) - 1
    for i in range(len(raw_bins) - 1, -1, -1):
        if raw_bins[i][2] >= min_p:
            last_sig = i
            break

    bins: list[BettingBin] = []

    lower_bound = raw_bins[first_sig][0]
    p_low = _normal_cdf(lower_bound - 0.5, center, sigma)
    bins.append(BettingBin(f"{lower_bound - 1}{unit} or less", p_low))

    for i in range(first_sig, last_sig + 1):
        bins.append(BettingBin(raw_bins[i][1], raw_bins[i][2]))

    last_v = raw_bins[last_sig][0]
    if use_f:
        upper_bound = last_v + 2
        p_high = 1.0 - _normal_cdf(last_v + 1.5, center, sigma)
    else:
        upper_bound = last_v + 1
        p_high = 1.0 - _normal_cdf(last_v + 0.5, center, sigma)
    bins.append(BettingBin(f"{upper_bound}{unit} or more", p_high))

    return bins


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_analysis(analysis: BettingAnalysis) -> str:
    """Render the betting analysis as a pretty-printed terminal string."""
    W = 72
    lines: list[str] = []

    lines.append("")
    lines.append(f"  {'=' * W}")
    lines.append(f"  POLYMARKET BETTING ANALYSIS")
    lines.append(f"  {'=' * W}")
    lines.append(f"  Market     : Highest temperature in {analysis.city} on {analysis.date}")
    lines.append(f"  Prediction : {analysis.predicted}{analysis.unit}  (σ ±{analysis.sigma}{analysis.unit})")
    lines.append(f"  Confidence : {analysis.confidence:.0f}%")

    if analysis.market_found:
        vol_str = f"${analysis.market_volume:,.0f}" if analysis.market_volume else "N/A"
        lines.append(f"  Live market: YES — Volume: {vol_str}")
        lines.append(f"  Link       : {analysis.market_url}")
    else:
        lines.append(f"  Live market: not found — showing model probabilities only")

    lines.append("")

    # --- Table ---
    max_label = max(len(b.label) for b in analysis.bins)
    has_market = analysis.market_found and any(b.market_price is not None for b in analysis.bins)

    if has_market:
        header = f"  {'Range':<{max_label}}  {'Model':>6}  {'Market':>7}  {'Edge':>6}  {'Signal':>10}"
        lines.append(header)
        lines.append(f"  {'─' * (max_label + 38)}")

        for b in analysis.bins:
            our_pct = b.prob * 100
            if b.market_price is not None:
                mkt_pct = f"{b.market_price * 100:.1f}%"
                edge_pct = b.edge * 100 if b.edge is not None else 0
                edge_str = f"{edge_pct:+.1f}%"
                if b.is_value_bet:
                    signal = "← VALUE"
                elif b.is_best:
                    signal = "← BEST"
                else:
                    signal = ""
            else:
                mkt_pct = "  -"
                edge_str = "  -"
                signal = "← BEST" if b.is_best else ""

            lines.append(
                f"  {b.label:<{max_label}}  {our_pct:>5.1f}%  {mkt_pct:>7}  {edge_str:>6}  {signal:>10}"
            )
    else:
        # No market data — show model only with bar chart
        bar_width = 25
        header = f"  {'Range':<{max_label}}  {'Prob':>6}  {'':^{bar_width}}  {'Fair':>5}"
        lines.append(header)
        lines.append(f"  {'─' * (max_label + bar_width + 18)}")

        max_prob = max(b.prob for b in analysis.bins) if analysis.bins else 1.0

        for b in analysis.bins:
            pct = b.prob * 100
            bar_len = round(b.prob / max_prob * bar_width) if max_prob > 0 else 0
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            price = f"{pct:.0f}¢" if pct >= 1 else "<1¢"
            marker = "  ← BEST" if b.is_best else ""
            lines.append(
                f"  {b.label:<{max_label}}  {pct:>5.1f}%  {bar}  {price:>5}{marker}"
            )

    # --- Recommendation ---
    lines.append("")
    lines.append(f"  {'─' * W}")
    lines.append(f"  RECOMMENDATION")
    lines.append(f"  {'─' * W}")

    if has_market and analysis.value_bets:
        # We found value bets — recommend them
        vb_sorted = sorted(analysis.value_bets, key=lambda b: b.edge or 0, reverse=True)
        top = vb_sorted[0]
        edge_pct = (top.edge or 0) * 100
        mkt_pct = (top.market_price or 0) * 100
        our_pct = top.prob * 100

        lines.append(
            f"  → BUY \"{top.label}\" at {mkt_pct:.0f}¢ — "
            f"our model: {our_pct:.1f}% → edge +{edge_pct:.1f}%"
        )

        if len(vb_sorted) > 1:
            second = vb_sorted[1]
            s_edge = (second.edge or 0) * 100
            s_mkt = (second.market_price or 0) * 100
            lines.append(
                f"  → Also: \"{second.label}\" at {s_mkt:.0f}¢ "
                f"(edge +{s_edge:.1f}%)"
            )

        # Volume warning
        if analysis.market_volume is not None and analysis.market_volume < 10000:
            lines.append(f"  → WARNING: Low volume (${analysis.market_volume:,.0f}) — thin market, slippage risk")

    elif has_market:
        # Market found but no value bets
        best = next((b for b in analysis.bins if b.is_best), analysis.bins[0])
        our_pct = best.prob * 100
        mkt_pct = (best.market_price or 0) * 100

        lines.append(f"  → No strong value bet found — market prices are efficient")
        lines.append(f"  → Best range: \"{best.label}\" (model {our_pct:.1f}% vs market {mkt_pct:.1f}%)")
        lines.append(f"  → Consider waiting for better odds or checking other markets")
    else:
        # No market — model-only recommendation
        best = next((b for b in analysis.bins if b.is_best), analysis.bins[0])
        best_pct = best.prob * 100
        lines.append(f"  → BUY \"{best.label}\" — our model gives {best_pct:.1f}% probability")
        lines.append(f"    If Polymarket price < {best_pct:.0f}¢, this is a VALUE BET")

        sorted_bins = sorted(analysis.bins, key=lambda b: b.prob, reverse=True)
        if len(sorted_bins) > 1:
            second = sorted_bins[1]
            lines.append(f"  → Also consider: \"{second.label}\" ({second.prob * 100:.1f}%)")

    # Risk assessment
    if analysis.confidence >= 70:
        risk_label = "LOW"
        risk_note = "High model agreement + historical consistency"
    elif analysis.confidence >= 50:
        risk_label = "MEDIUM"
        risk_note = "Moderate model agreement"
    else:
        risk_label = "HIGH"
        risk_note = "Low model agreement — consider smaller position"

    lines.append(f"  → Risk: {risk_label} — {risk_note}")

    if abs(analysis.skewness) > 0.3:
        direction = "higher" if analysis.skewness > 0 else "lower"
        lines.append(f"  → Skew alert: some models lean {direction} than consensus")

    lines.append(f"  {'=' * W}")
    lines.append("")

    return "\n".join(lines)
