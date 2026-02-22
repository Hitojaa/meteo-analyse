"""Polymarket temperature betting analysis.

Uses the multi-model forecast distribution to compute probability
for each temperature range and recommend optimal bets.

When a live Polymarket market is found, computes a value-betting
strategy that targets mispriced outcomes where our model probability
exceeds the market ask price.  Positions are sized using the Kelly
criterion for optimal growth.

Polymarket temperature markets resolve to the highest temperature
recorded at a specific station, measured in whole degrees:
  - US cities  → °F, bins of 2°F  (e.g. 44-45°F)
  - Non-US     → °C, bins of 1°C  (e.g. 8°C)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from models import AggregatedResult, ProviderResult
from polymarket_api import TemperatureMarket


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BettingBin:
    """A temperature range bin with its probability."""
    label: str
    prob: float
    is_best: bool = False


@dataclass
class HedgeBet:
    """A single value bet within a strategy."""
    label: str
    market_price: float    # ask price (0.0–1.0)
    model_prob: float      # our model's probability
    edge: float            # model_prob - market_price
    stake: float           # $ allocated to this range
    shares: float          # shares = stake / price
    payout: float          # payout if this range wins = shares


@dataclass
class HedgingStrategy:
    """A value-betting strategy targeting mispriced outcomes."""
    budget: float
    bets: list[HedgeBet]
    total_staked: float          # sum of stakes (≈ budget)
    model_prob_covered: float    # sum of model probs for bet outcomes
    expected_value: float        # EV = Σ(prob × payout) − total_staked
    num_outcomes: int = 0        # total outcomes in market
    market_url: str | None = None
    market_volume: float | None = None
    price_source: str = "snapshot"  # "live" or "snapshot"


@dataclass
class BettingAnalysis:
    """Betting analysis based on model probability distribution."""
    city: str
    date: str
    unit: str           # "°F" or "°C"
    predicted: float    # predicted tmax in market unit
    sigma: float        # uncertainty (1σ) in market unit
    bins: list[BettingBin]
    skewness: float
    confidence: float
    hedging: HedgingStrategy | None = None


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
# Model probability for a market outcome
# ---------------------------------------------------------------------------

def _prob_for_outcome(
    label: str, center: float, sigma: float,
) -> float:
    """Compute our model's probability for a Polymarket outcome label."""
    import re
    low = label.lower()

    # Tail: "X or less"
    if "or less" in low or "or lower" in low or "or below" in low:
        nums = re.findall(r"-?\d+", label)
        if nums:
            upper = int(nums[-1])
            return _normal_cdf(upper + 0.5, center, sigma)
        return 0.0

    # Tail: "X or more"
    if "or more" in low or "or higher" in low or "or above" in low:
        nums = re.findall(r"-?\d+", label)
        if nums:
            lower = int(nums[0])
            return 1.0 - _normal_cdf(lower - 0.5, center, sigma)
        return 0.0

    # Range: "44-45"
    range_match = re.match(r".*?(-?\d+)\s*[-–]\s*(\d+)", label)
    if range_match:
        lo_val = int(range_match.group(1))
        hi_val = int(range_match.group(2))
        return _normal_cdf(hi_val + 0.5, center, sigma) - _normal_cdf(lo_val - 0.5, center, sigma)

    # Single value: "8"
    single_match = re.search(r"(-?\d+)", label)
    if single_match:
        val = int(single_match.group(1))
        return _normal_cdf(val + 0.5, center, sigma) - _normal_cdf(val - 0.5, center, sigma)

    return 0.0


# ---------------------------------------------------------------------------
# Market comparison
# ---------------------------------------------------------------------------

def _compute_hedging(
    market: TemperatureMarket,
    center: float,
    sigma: float,
    budget: float = 10.0,
    min_model_prob: float = 0.03,
) -> HedgingStrategy | None:
    """Compare model probabilities with live market prices.

    Selects outcomes with model probability >= *min_model_prob* until
    ~90% cumulative coverage.  Returns comparison data (model prob,
    market price, edge) for each, sorted by model probability.

    Prices below 0.5¢ are treated as untradeable (set to 0).

    Returns None only if no outcomes meet the model probability threshold.
    """
    num_outcomes = len(market.outcomes)

    candidates: list[tuple[str, float, float, float]] = []
    for o in market.outcomes:
        model_p = _prob_for_outcome(o.label, center, sigma)
        if model_p >= min_model_prob:
            price = o.price if o.price > 0.005 else 0.0
            edge = model_p - price if price > 0 else 0.0
            candidates.append((o.label, price, model_p, edge))

    if not candidates:
        return None

    # Sort by model probability, take until 90% coverage
    candidates.sort(key=lambda x: x[2], reverse=True)
    selected: list[tuple[str, float, float, float]] = []
    cum_prob = 0.0
    for item in candidates:
        selected.append(item)
        cum_prob += item[2]
        if cum_prob >= 0.90:
            break

    bets = []
    model_prob_covered = 0.0
    for label, price, model_p, edge in selected:
        bets.append(HedgeBet(
            label=label,
            market_price=price,
            model_prob=model_p,
            edge=round(edge, 4),
            stake=0.0,
            shares=0.0,
            payout=0.0,
        ))
        model_prob_covered += model_p

    return HedgingStrategy(
        budget=0.0,
        bets=bets,
        total_staked=0.0,
        model_prob_covered=round(model_prob_covered, 4),
        expected_value=0.0,
        num_outcomes=num_outcomes,
        market_url=market.url,
        market_volume=market.volume,
    )


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
    budget: float = 10.0,
) -> BettingAnalysis:
    """Build a probability distribution over Polymarket temperature bins.

    The distribution is modeled as Normal(center, sigma) where:
      - center = aggregated tmax (bias-corrected, climate-anchored) + skewness nudge
      - sigma  = blended provider ensemble spread & historical variability

    Bins match Polymarket conventions:
      - US: 2°F bins aligned to even integers (42-43, 44-45, …)
      - Non-US: 1°C bins (7, 8, 9, …)

    If a live market is provided, computes a hedging strategy.
    """
    use_f = _is_us(country)
    unit = "°F" if use_f else "°C"

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
    bins = _bins_from_model(center, sigma, use_f, unit, step)

    # Mark the best bin (highest probability)
    if bins:
        best = max(bins, key=lambda b: b.prob)
        best.is_best = True

    # --- Hedging strategy ---
    hedging = None
    if market is not None and market.outcomes:
        hedging = _compute_hedging(market, center, sigma, budget)

    return BettingAnalysis(
        city=city,
        date=date,
        unit=unit,
        predicted=round(center, 1),
        sigma=round(sigma, 1),
        bins=bins,
        skewness=skew,
        confidence=agg.confidence,
        hedging=hedging,
    )


def _bins_from_model(
    center: float,
    sigma: float,
    use_f: bool,
    unit: str,
    step: int,
) -> list[BettingBin]:
    """Generate bins from our model."""
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

    lines.append("")

    # --- Table: model probabilities with bar chart ---
    max_label = max(len(b.label) for b in analysis.bins)
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

    best = next((b for b in analysis.bins if b.is_best), analysis.bins[0])
    sorted_bins = sorted(analysis.bins, key=lambda b: b.prob, reverse=True)
    h = analysis.hedging

    if h is not None and h.bets:
        # Market data available: show price table for context
        if h.market_url:
            lines.append(f"  Market : {h.market_url}")
            if h.market_volume:
                lines.append(f"  Volume : ${h.market_volume:,.0f}")
        if h.price_source == "live":
            lines.append(f"  Prices : LIVE (CLOB order book)")
        else:
            lines.append(f"  Prices : ⚠ SNAPSHOT (Gamma API – may be stale)")

        lines.append("")
        lines.append(f"  {'Range':<16} {'Model':>6} {'Price':>7}")
        lines.append(f"  {'─' * 32}")

        for bet in h.bets:
            model_str = f"{bet.model_prob * 100:>5.1f}%"
            if bet.market_price > 0:
                price_str = f"{bet.market_price * 100:>5.1f}¢"
            else:
                price_str = f"{'—':>7}"
            lines.append(f"  {bet.label:<16} {model_str} {price_str}")

        lines.append(f"  {'─' * 32}")
        lines.append("")

    # Recommendation — always based on model probability
    best_pct = best.prob * 100
    best_bet = None
    if h is not None:
        best_bet = next((b for b in h.bets if b.label == best.label), None)

    if best_bet and best_bet.market_price > 0:
        lines.append(f"  → BUY \"{best.label}\""
                     f" — our model gives {best_pct:.1f}% (market: {best_bet.market_price * 100:.0f}¢)")
    else:
        lines.append(f"  → BUY \"{best.label}\" — our model gives {best_pct:.1f}% probability")
        if not best_bet:
            lines.append(f"    If Polymarket price < {best_pct:.0f}¢, this is a VALUE BET")

    if len(sorted_bins) > 1:
        second = sorted_bins[1]
        sec_bet = None
        if h is not None:
            sec_bet = next((b for b in h.bets if b.label == second.label), None)
        if sec_bet and sec_bet.market_price > 0:
            lines.append(f"  → Also consider: \"{second.label}\""
                         f" ({second.prob * 100:.1f}%, market: {sec_bet.market_price * 100:.0f}¢)")
        else:
            lines.append(f"  → Also consider: \"{second.label}\" ({second.prob * 100:.1f}%)")

    # Risk assessment
    lines.append("")
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
