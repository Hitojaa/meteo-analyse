"""Polymarket temperature betting analysis.

Uses the multi-model forecast distribution to compute probability
for each temperature range and recommend optimal bets.

When a live Polymarket market is found, computes a value-betting
strategy that targets mispriced outcomes where our model probability
exceeds the market ask price.

Probability modeling uses Kernel Density Estimation (KDE) from the
actual provider values when enough data points are available, falling
back to Normal distribution otherwise. This captures asymmetric and
bimodal distributions that a simple Normal cannot represent.

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
    budget: float = 10.0
    hedging: HedgingStrategy | None = None
    max_picks: int = 3  # max outcomes in allocation


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """CDF of Normal(mu, sigma) evaluated at x."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def _normal_pdf(x: float, mu: float, sigma: float) -> float:
    """PDF of Normal(mu, sigma) evaluated at x."""
    if sigma <= 0:
        return float("inf") if x == mu else 0.0
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def _kde_pdf(x: float, samples: list[float], bandwidth: float) -> float:
    """Kernel Density Estimation using Gaussian kernels.

    Each sample contributes a Normal(sample, bandwidth) kernel.
    The PDF at x is the average of all kernel values.
    """
    if not samples:
        return 0.0
    return sum(_normal_pdf(x, s, bandwidth) for s in samples) / len(samples)


def _kde_cdf_numerical(
    x: float, samples: list[float], bandwidth: float,
    lo: float | None = None, n_steps: int = 200,
) -> float:
    """Numerically integrate KDE PDF from lo to x using the trapezoidal rule."""
    if lo is None:
        lo = min(samples) - 5 * bandwidth
    if x <= lo:
        return 0.0
    step = (x - lo) / n_steps
    total = 0.0
    prev_y = _kde_pdf(lo, samples, bandwidth)
    for i in range(1, n_steps + 1):
        xi = lo + i * step
        yi = _kde_pdf(xi, samples, bandwidth)
        total += (prev_y + yi) * 0.5 * step
        prev_y = yi
    return min(max(total, 0.0), 1.0)


def _silverman_bandwidth(samples: list[float]) -> float:
    """Silverman's rule of thumb for KDE bandwidth selection."""
    n = len(samples)
    if n < 2:
        return 1.0
    mean_val = sum(samples) / n
    std_val = (sum((x - mean_val) ** 2 for x in samples) / n) ** 0.5
    if std_val == 0:
        return 1.0
    # Silverman's rule: h = 0.9 * min(std, IQR/1.34) * n^(-1/5)
    sorted_s = sorted(samples)
    q1 = sorted_s[n // 4]
    q3 = sorted_s[3 * n // 4]
    iqr = q3 - q1
    spread = min(std_val, iqr / 1.34) if iqr > 0 else std_val
    return 0.9 * spread * n ** (-0.2)


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
    kde_samples: list[float] | None = None,
    kde_bw: float | None = None,
) -> float:
    """Compute our model's probability for a Polymarket outcome label.

    Uses KDE if samples are provided (>= 5 points), otherwise falls
    back to Normal(center, sigma).
    """
    import re
    low = label.lower()

    use_kde = kde_samples is not None and kde_bw is not None and len(kde_samples) >= 5

    def _cdf(x: float) -> float:
        if use_kde:
            return _kde_cdf_numerical(x, kde_samples, kde_bw)
        return _normal_cdf(x, center, sigma)

    # Tail: "X or less"
    if "or less" in low or "or lower" in low or "or below" in low:
        nums = re.findall(r"-?\d+", label)
        if nums:
            upper = int(nums[-1])
            return _cdf(upper + 0.5)
        return 0.0

    # Tail: "X or more"
    if "or more" in low or "or higher" in low or "or above" in low:
        nums = re.findall(r"-?\d+", label)
        if nums:
            lower = int(nums[0])
            return 1.0 - _cdf(lower - 0.5)
        return 0.0

    # Range: "44-45"
    range_match = re.match(r".*?(-?\d+)\s*[-–]\s*(\d+)", label)
    if range_match:
        lo_val = int(range_match.group(1))
        hi_val = int(range_match.group(2))
        return _cdf(hi_val + 0.5) - _cdf(lo_val - 0.5)

    # Single value: "8"
    single_match = re.search(r"(-?\d+)", label)
    if single_match:
        val = int(single_match.group(1))
        return _cdf(val + 0.5) - _cdf(val - 0.5)

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
    kde_samples: list[float] | None = None,
    kde_bw: float | None = None,
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
        model_p = _prob_for_outcome(
            o.label, center, sigma,
            kde_samples=kde_samples, kde_bw=kde_bw,
        )
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
    max_picks: int = 3,
) -> BettingAnalysis:
    """Build a probability distribution over Polymarket temperature bins.

    The distribution uses Kernel Density Estimation (KDE) from provider
    tmax values when enough data points are available (>= 5), falling
    back to Normal(center, sigma) otherwise.

    Sigma is computed from provider spread, historical variability, and
    weather instability, with auto-calibration from self-learning data.

    Bins match Polymarket conventions:
      - US: 2°F bins aligned to even integers (42-43, 44-45, …)
      - Non-US: 1°C bins (7, 8, 9, …)

    If a live market is provided, computes a hedging strategy.
    max_picks controls how many outcomes can be included in the allocation.
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

    # --- Apply instability factor (widens sigma in volatile weather) ---
    inst_factor = getattr(agg, "instability_factor", 1.0)
    sigma_c *= inst_factor

    # --- Convert to market unit ---
    if use_f:
        center = _c_to_f(center_c)
        sigma = sigma_c * 1.8
        step = 2
        # Convert provider values to °F for KDE
        kde_samples_raw = [_c_to_f(v) for v in tmax_c]
    else:
        center = center_c
        sigma = sigma_c
        step = 1
        kde_samples_raw = list(tmax_c)

    skew = agg.skewness_tmax
    center += skew * sigma * 0.10

    # --- KDE setup ---
    kde_samples: list[float] | None = None
    kde_bw: float | None = None
    if len(kde_samples_raw) >= 5:
        kde_samples = kde_samples_raw
        kde_bw = _silverman_bandwidth(kde_samples_raw)
        # Ensure bandwidth is reasonable
        kde_bw = max(kde_bw, 0.3 if not use_f else 0.5)

    # --- Generate bins ---
    bins = _bins_from_model(
        center, sigma, use_f, unit, step,
        kde_samples=kde_samples, kde_bw=kde_bw,
    )

    # Mark the best bin (highest probability)
    if bins:
        best = max(bins, key=lambda b: b.prob)
        best.is_best = True

    # --- Hedging strategy ---
    hedging = None
    if market is not None and market.outcomes:
        hedging = _compute_hedging(
            market, center, sigma, budget,
            kde_samples=kde_samples, kde_bw=kde_bw,
        )

    return BettingAnalysis(
        city=city,
        date=date,
        unit=unit,
        predicted=round(center, 1),
        sigma=round(sigma, 1),
        bins=bins,
        skewness=skew,
        confidence=agg.confidence,
        budget=budget,
        hedging=hedging,
        max_picks=max_picks,
    )


def _bins_from_model(
    center: float,
    sigma: float,
    use_f: bool,
    unit: str,
    step: int,
    kde_samples: list[float] | None = None,
    kde_bw: float | None = None,
) -> list[BettingBin]:
    """Generate bins from our model (KDE or Normal)."""
    use_kde = kde_samples is not None and kde_bw is not None and len(kde_samples) >= 5

    def _cdf(x: float) -> float:
        if use_kde:
            return _kde_cdf_numerical(x, kde_samples, kde_bw)
        return _normal_cdf(x, center, sigma)

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
            p = _cdf(v + 1.5) - _cdf(v - 0.5)
            label = f"{v}-{v + 1}{unit}"
        else:
            p = _cdf(v + 0.5) - _cdf(v - 0.5)
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
    p_low = _cdf(lower_bound - 0.5)
    bins.append(BettingBin(f"{lower_bound - 1}{unit} or less", p_low))

    for i in range(first_sig, last_sig + 1):
        bins.append(BettingBin(raw_bins[i][1], raw_bins[i][2]))

    last_v = raw_bins[last_sig][0]
    if use_f:
        upper_bound = last_v + 2
        p_high = 1.0 - _cdf(last_v + 1.5)
    else:
        upper_bound = last_v + 1
        p_high = 1.0 - _cdf(last_v + 0.5)
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

    # "Also consider" — pick the next-best tradeable outcome
    if len(sorted_bins) > 1:
        alt = None
        for cand in sorted_bins[1:]:
            if cand.label == best.label:
                continue
            if h is not None:
                cand_bet = next((b for b in h.bets if b.label == cand.label), None)
                if cand_bet and cand_bet.market_price > 0:
                    alt = (cand, cand_bet)
                    break
                # No market data for this candidate — skip if market exists
                if cand_bet and cand_bet.market_price == 0:
                    continue
            # No market at all — pick by model probability
            alt = (cand, None)
            break

        if alt:
            alt_bin, alt_bet = alt
            if alt_bet and alt_bet.market_price > 0:
                lines.append(f"  → Also consider: \"{alt_bin.label}\""
                             f" ({alt_bin.prob * 100:.1f}%, market: {alt_bet.market_price * 100:.0f}¢)")
            else:
                lines.append(f"  → Also consider: \"{alt_bin.label}\" ({alt_bin.prob * 100:.1f}%)")

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

    # --- N-outcome allocation strategy ---
    max_picks = getattr(analysis, "max_picks", 2)
    _format_allocation(lines, analysis, sorted_bins, h, max_picks, W)

    lines.append(f"  {'=' * W}")
    lines.append("")

    return "\n".join(lines)


def _format_allocation(
    lines: list[str],
    analysis: BettingAnalysis,
    sorted_bins: list[BettingBin],
    h: HedgingStrategy | None,
    max_picks: int,
    W: int,
) -> None:
    """Format the N-outcome dutch-book allocation section."""
    bankroll = analysis.budget
    MIN_PRICE = 0.05

    picks: list[tuple] = []  # (label, model_prob, market_price)
    if h is not None and h.bets:
        for cand in sorted_bins:
            if len(picks) >= max_picks:
                break
            cand_bet = next((b for b in h.bets if b.label == cand.label), None)
            if cand_bet and cand_bet.market_price >= MIN_PRICE:
                picks.append((cand.label, cand.prob, cand_bet.market_price))

    if len(picks) < 2:
        return

    # Smart reduction: if top 2 picks already have strong combined
    # probability (>=55%) and positive expected edge, only keep 2 trades
    # to concentrate the bankroll on the best opportunities.
    if len(picks) > 2:
        top2_prob = picks[0][1] + picks[1][1]
        top2_prices = picks[0][2] + picks[1][2]
        if top2_prob >= 0.55 and top2_prob > top2_prices:
            picks = picks[:2]

    sum_prices = sum(p[2] for p in picks)
    combined_prob = sum(p[1] for p in picks)

    lines.append("")
    lines.append(f"  {'─' * W}")
    lines.append(f"  ${bankroll:.0f} ALLOCATION ({len(picks)} outcomes)")
    lines.append(f"  {'─' * W}")

    # Dutch-book: allocate proportionally to price → equalizes payout
    # shares = bankroll / sum_prices (same for all outcomes)
    shares = bankroll / sum_prices
    profit_if_wins = shares - bankroll  # payout ($1/share) minus cost

    lines.append("")
    for label, p, c in picks:
        alloc = bankroll * c / sum_prices
        lines.append(
            f"  {label:<12}  ${alloc:>5.2f}  →  "
            f"{shares:.1f} shares @ {c * 100:.0f}¢"
            f"  (model: {p * 100:.0f}%)"
        )

    lines.append("")
    if sum_prices < 1.0:
        roi_win = (profit_if_wins / bankroll) * 100
        lines.append(f"  If any wins   →  ${shares:.2f}  "
                     f"(+${profit_if_wins:.2f}, ROI {roi_win:+.0f}%)")
    else:
        lines.append(f"  If any wins   →  ${shares:.2f}  "
                     f"(net {'+' if profit_if_wins >= 0 else ''}"
                     f"${profit_if_wins:.2f})")
    lines.append(f"  If none wins  →  -${bankroll:.2f}")

    lines.append("")
    ev_profit = combined_prob * shares - bankroll
    ev_roi = (ev_profit / bankroll) * 100
    lines.append(f"  Combined prob: {combined_prob * 100:.0f}%"
                 f"  |  Expected: {'+' if ev_profit >= 0 else ''}"
                 f"${ev_profit:.2f} (ROI {ev_roi:+.0f}%)")
