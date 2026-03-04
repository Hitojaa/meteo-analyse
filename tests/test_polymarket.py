"""Tests for the Polymarket betting analysis module."""

import math

import pytest

from models import AggregatedResult, DataQuality, ProviderResult
from polymarket import (
    BettingAnalysis,
    BettingBin,
    HedgeBet,
    HedgingStrategy,
    _compute_hedging,
    _is_us,
    _kde_cdf_numerical,
    _kde_pdf,
    _normal_cdf,
    _normal_pdf,
    _prob_for_outcome,
    _silverman_bandwidth,
    analyze,
    format_analysis,
)
from polymarket_api import MarketOutcome, TemperatureMarket


def _make_provider(tmax: float, name: str = "test") -> ProviderResult:
    return ProviderResult(
        provider_name=name,
        tmin_c=tmax - 10.0,
        tmax_c=tmax,
        date="2026-02-17",
        timezone="Europe/London",
        quality=DataQuality.DAILY_DIRECT,
    )


def _make_agg(
    tmax_c: float,
    skewness: float = 0.0,
    confidence: float = 70.0,
    hist_tmax_std: float | None = None,
    hist_tmax_mean: float | None = None,
) -> AggregatedResult:
    return AggregatedResult(
        tmin_c=tmax_c - 10.0,
        tmax_c=tmax_c,
        tmin_range=(tmax_c - 11.0, tmax_c - 9.0),
        tmax_range=(tmax_c - 1.0, tmax_c + 1.0),
        sources_used=5,
        skewness_tmin=0.0,
        skewness_tmax=skewness,
        confidence=confidence,
        hist_tmin_mean=None,
        hist_tmax_mean=hist_tmax_mean,
        hist_tmin_std=None,
        hist_tmax_std=hist_tmax_std,
        hist_sample_size=50 if hist_tmax_std else None,
    )


# ===========================================================================
# polymarket.py tests
# ===========================================================================

class TestIsUS:
    def test_united_states(self):
        assert _is_us("United States") is True

    def test_usa(self):
        assert _is_us("USA") is True

    def test_france(self):
        assert _is_us("France") is False

    def test_united_kingdom(self):
        assert _is_us("United Kingdom") is False

    def test_case_insensitive(self):
        assert _is_us("united states") is True
        assert _is_us("UNITED STATES") is True


class TestNormalCDF:
    def test_mean_gives_half(self):
        assert abs(_normal_cdf(0.0, 0.0, 1.0) - 0.5) < 0.001

    def test_large_positive(self):
        assert _normal_cdf(10.0, 0.0, 1.0) > 0.999

    def test_large_negative(self):
        assert _normal_cdf(-10.0, 0.0, 1.0) < 0.001

    def test_one_sigma(self):
        assert abs(_normal_cdf(1.0, 0.0, 1.0) - 0.8413) < 0.001

    def test_zero_sigma(self):
        assert _normal_cdf(5.0, 3.0, 0.0) == 1.0
        assert _normal_cdf(1.0, 3.0, 0.0) == 0.0


# ===========================================================================
# Analyze tests
# ===========================================================================

class TestAnalyze:
    def test_us_city_uses_fahrenheit(self):
        providers = [_make_provider(7.0 + i * 0.5) for i in range(5)]
        agg = _make_agg(8.0)
        result = analyze("New York", "2026-02-17", "United States", agg, providers)
        assert result.unit == "°F"

    def test_non_us_city_uses_celsius(self):
        providers = [_make_provider(7.0 + i * 0.5) for i in range(5)]
        agg = _make_agg(8.0)
        result = analyze("London", "2026-02-17", "United Kingdom", agg, providers)
        assert result.unit == "°C"

    def test_fahrenheit_bins_are_2f_wide(self):
        providers = [_make_provider(10.0 + i * 0.3) for i in range(8)]
        agg = _make_agg(11.0)
        result = analyze("NYC", "2026-02-17", "United States", agg, providers)
        middle_bins = [b for b in result.bins if "-" in b.label]
        for b in middle_bins:
            nums = b.label.replace("°F", "").split("-")
            assert int(nums[1]) - int(nums[0]) == 1

    def test_celsius_bins_are_1c_wide(self):
        providers = [_make_provider(7.0 + i * 0.3) for i in range(8)]
        agg = _make_agg(8.0)
        result = analyze("London", "2026-02-17", "United Kingdom", agg, providers)
        middle_bins = [b for b in result.bins if "or" not in b.label]
        for b in middle_bins:
            val = b.label.replace("°C", "")
            int(val)

    def test_probabilities_sum_to_one(self):
        providers = [_make_provider(10.0 + i * 0.5) for i in range(10)]
        agg = _make_agg(12.0, hist_tmax_std=2.5, hist_tmax_mean=11.0)
        result = analyze("Paris", "2026-02-17", "France", agg, providers)
        total = sum(b.prob for b in result.bins)
        assert abs(total - 1.0) < 0.01

    def test_best_bin_marked(self):
        providers = [_make_provider(10.0 + i * 0.2) for i in range(5)]
        agg = _make_agg(10.5)
        result = analyze("Berlin", "2026-02-17", "Germany", agg, providers)
        best_bins = [b for b in result.bins if b.is_best]
        assert len(best_bins) == 1

    def test_best_bin_has_highest_prob(self):
        providers = [_make_provider(10.0 + i * 0.2) for i in range(5)]
        agg = _make_agg(10.5)
        result = analyze("Berlin", "2026-02-17", "Germany", agg, providers)
        best = max(result.bins, key=lambda b: b.prob)
        assert best.is_best is True

    def test_predicted_near_center(self):
        providers = [_make_provider(15.0 + i * 0.1) for i in range(5)]
        agg = _make_agg(15.2)
        result = analyze("Madrid", "2026-02-17", "Spain", agg, providers)
        assert abs(result.predicted - 15.2) < 1.0

    def test_higher_historical_std_widens_distribution(self):
        providers = [_make_provider(10.0) for _ in range(5)]
        agg_narrow = _make_agg(10.0, hist_tmax_std=1.0, hist_tmax_mean=10.0)
        agg_wide = _make_agg(10.0, hist_tmax_std=5.0, hist_tmax_mean=10.0)
        result_narrow = analyze("A", "2026-02-17", "France", agg_narrow, providers)
        result_wide = analyze("A", "2026-02-17", "France", agg_wide, providers)
        assert result_wide.sigma >= result_narrow.sigma

    def test_skewness_shifts_center(self):
        providers = [_make_provider(10.0 + i * 0.2) for i in range(5)]
        agg_pos = _make_agg(10.5, skewness=1.5)
        agg_neg = _make_agg(10.5, skewness=-1.5)
        result_pos = analyze("A", "2026-02-17", "France", agg_pos, providers)
        result_neg = analyze("A", "2026-02-17", "France", agg_neg, providers)
        assert result_pos.predicted > result_neg.predicted

    def test_no_valid_data_raises(self):
        providers = [
            ProviderResult(
                provider_name="bad", tmin_c=5.0, tmax_c=None,
                date="2026-02-17", timezone="UTC",
            )
        ]
        agg = _make_agg(10.0)
        with pytest.raises(ValueError, match="No valid tmax"):
            analyze("X", "2026-02-17", "France", agg, providers)

    def test_single_provider_still_works(self):
        providers = [_make_provider(10.0)]
        agg = _make_agg(10.0)
        result = analyze("Solo", "2026-02-17", "France", agg, providers)
        assert len(result.bins) >= 3
        total = sum(b.prob for b in result.bins)
        assert abs(total - 1.0) < 0.01


# ===========================================================================
# Formatting tests
# ===========================================================================

class TestFormatAnalysis:
    def test_format_output(self):
        bins = [
            BettingBin("7°C or less", 0.05),
            BettingBin("8°C", 0.25),
            BettingBin("9°C", 0.40, is_best=True),
            BettingBin("10°C", 0.22),
            BettingBin("11°C or more", 0.08),
        ]
        analysis = BettingAnalysis(
            city="London", date="2026-02-17", unit="°C",
            predicted=9.1, sigma=1.2, bins=bins,
            skewness=0.1, confidence=72.0,
        )
        output = format_analysis(analysis)
        assert "POLYMARKET" in output
        assert "London" in output
        assert "BEST" in output
        assert "BUY" in output
        assert "9°C" in output

    def test_format_shows_risk(self):
        bins = [BettingBin("10°C", 1.0, is_best=True)]
        analysis = BettingAnalysis(
            city="Test", date="2026-01-01", unit="°C",
            predicted=10.0, sigma=1.0, bins=bins,
            skewness=0.0, confidence=80.0,
        )
        output = format_analysis(analysis)
        assert "LOW" in output

    def test_format_shows_skew_alert(self):
        bins = [BettingBin("10°C", 1.0, is_best=True)]
        analysis = BettingAnalysis(
            city="Test", date="2026-01-01", unit="°C",
            predicted=10.0, sigma=1.0, bins=bins,
            skewness=0.8, confidence=60.0,
        )
        output = format_analysis(analysis)
        assert "Skew alert" in output
        assert "higher" in output


# ===========================================================================
# _prob_for_outcome tests
# ===========================================================================

class TestProbForOutcome:
    def test_single_value(self):
        # center=10, sigma=2 → P(9.5 < X < 10.5) should be ~20%
        p = _prob_for_outcome("10°C", 10.0, 2.0)
        assert 0.15 < p < 0.25

    def test_range(self):
        # center=44, sigma=3 → P(43.5 < X < 45.5) should be meaningful
        p = _prob_for_outcome("44-45°F", 44.0, 3.0)
        assert 0.15 < p < 0.40

    def test_or_less_tail(self):
        # center=10, sigma=2 → P(X < 6.5) should be small
        p = _prob_for_outcome("6°C or less", 10.0, 2.0)
        assert p < 0.10

    def test_or_more_tail(self):
        # center=10, sigma=2 → P(X > 13.5) should be small
        p = _prob_for_outcome("14°C or more", 10.0, 2.0)
        assert p < 0.10

    def test_center_range_is_highest(self):
        center = 10.0
        sigma = 2.0
        p_center = _prob_for_outcome("10°C", center, sigma)
        p_off = _prob_for_outcome("14°C", center, sigma)
        assert p_center > p_off

    def test_unknown_label_returns_zero(self):
        assert _prob_for_outcome("nonsense", 10.0, 2.0) == 0.0


# ===========================================================================
# Hedging / dutching tests
# ===========================================================================

def _make_market(
    outcomes: list[tuple[str, float]],
    unit: str = "°C",
) -> TemperatureMarket:
    """Build a TemperatureMarket from (label, price) pairs."""
    return TemperatureMarket(
        title="Test temperature market",
        slug="test-market",
        url="https://polymarket.com/event/test-market",
        volume=50000.0,
        outcomes=[
            MarketOutcome(label=label, price=price, token_id=f"tok_{i}")
            for i, (label, price) in enumerate(outcomes)
        ],
        unit=unit,
        active=True,
    )


class TestComputeHedging:
    def test_returns_strategy_with_top_outcomes(self):
        """Returns a strategy with outcomes sorted by model probability."""
        market = _make_market([
            ("8°C or less", 0.02),
            ("9°C", 0.10),
            ("10°C", 0.20),
            ("11°C", 0.15),
            ("12°C or more", 0.05),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5)
        assert h is not None
        assert len(h.bets) >= 2
        # Sorted by model probability descending
        probs = [b.model_prob for b in h.bets]
        assert probs == sorted(probs, reverse=True)

    def test_no_stakes_allocated(self):
        """No Kelly sizing — all stakes are zero."""
        market = _make_market([
            ("9°C", 0.10),
            ("10°C", 0.20),
            ("11°C", 0.15),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5)
        assert h is not None
        for bet in h.bets:
            assert bet.stake == 0.0
        assert h.total_staked == 0.0

    def test_edge_computed_correctly(self):
        """Edge = model_prob - market_price for tradeable outcomes."""
        market = _make_market([
            ("10°C", 0.20),
            ("11°C", 0.15),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5)
        assert h is not None
        for bet in h.bets:
            if bet.market_price > 0:
                expected_edge = bet.model_prob - bet.market_price
                assert abs(bet.edge - expected_edge) < 0.01

    def test_includes_overpriced_outcomes(self):
        """Overpriced outcomes included when they have high model probability."""
        market = _make_market([
            ("9°C", 0.10),
            ("10°C", 0.50),  # overpriced
            ("11°C", 0.10),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5)
        assert h is not None
        labels = [b.label for b in h.bets]
        assert "10°C" in labels  # included despite being overpriced
        bet_10 = next(b for b in h.bets if b.label == "10°C")
        assert bet_10.edge < 0  # negative edge

    def test_untradeable_prices_set_to_zero(self):
        """Outcomes with price <= 0.5¢ get price and edge set to 0."""
        market = _make_market([
            ("10°C", 0.001),  # 0.1¢ — untradeable
            ("11°C", 0.15),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5)
        assert h is not None
        bet_10 = next((b for b in h.bets if b.label == "10°C"), None)
        assert bet_10 is not None
        assert bet_10.market_price == 0.0
        assert bet_10.edge == 0.0

    def test_returns_none_when_no_candidates(self):
        """Returns None when no outcomes have meaningful model probability."""
        market = _make_market([("9°C", 0.10), ("10°C", 0.20)])
        h = _compute_hedging(market, center=50.0, sigma=1.0)
        assert h is None

    def test_skips_low_model_prob_outcomes(self):
        """Outcomes far from center (low model prob) are excluded."""
        market = _make_market([
            ("1°C", 0.05),   # far from center=10
            ("10°C", 0.20),
            ("11°C", 0.15),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5)
        assert h is not None
        labels = [bet.label for bet in h.bets]
        assert "1°C" not in labels

    def test_coverage_reaches_90_percent(self):
        """Selected outcomes cover ~90% of model probability."""
        market = _make_market([
            ("8°C", 0.05), ("9°C", 0.10), ("10°C", 0.20),
            ("11°C", 0.15), ("12°C or more", 0.05),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5)
        assert h is not None
        assert h.model_prob_covered >= 0.85

    def test_market_url_and_volume_set(self):
        market = _make_market([
            ("10°C", 0.20),
            ("11°C", 0.15),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5)
        assert h is not None
        assert h.market_url == "https://polymarket.com/event/test-market"
        assert h.market_volume == 50000.0


class TestAnalyzeWithMarket:
    def test_hedging_attached_when_market_provided(self):
        providers = [_make_provider(10.0 + i * 0.3) for i in range(5)]
        agg = _make_agg(10.5)
        market = _make_market([
            ("9°C", 0.05),
            ("10°C", 0.30),
            ("11°C", 0.30),
            ("12°C or more", 0.05),
        ])
        result = analyze("Paris", "2026-02-17", "France", agg, providers, market=market)
        assert result.hedging is not None
        assert len(result.hedging.bets) >= 1

    def test_no_hedging_without_market(self):
        providers = [_make_provider(10.0 + i * 0.3) for i in range(5)]
        agg = _make_agg(10.5)
        result = analyze("Paris", "2026-02-17", "France", agg, providers)
        assert result.hedging is None

    def test_format_shows_recommendation_with_prices(self):
        providers = [_make_provider(10.0 + i * 0.3) for i in range(5)]
        agg = _make_agg(10.5)
        market = _make_market([
            ("9°C", 0.05),
            ("10°C", 0.30),
            ("11°C", 0.30),
            ("12°C or more", 0.05),
        ])
        result = analyze("Paris", "2026-02-17", "France", agg, providers, market=market)
        output = format_analysis(result)
        assert "RECOMMENDATION" in output
        assert "Model" in output
        assert "Price" in output


# ===========================================================================
# KDE tests
# ===========================================================================

class TestKDE:
    def test_kde_pdf_peaks_at_sample(self):
        samples = [10.0]
        bw = 1.0
        # PDF should peak at the sample value
        assert _kde_pdf(10.0, samples, bw) > _kde_pdf(12.0, samples, bw)

    def test_kde_pdf_multimodal(self):
        # Two clusters: should have peaks near both
        samples = [5.0, 5.1, 5.0, 15.0, 14.9, 15.1]
        bw = 0.5
        p_5 = _kde_pdf(5.0, samples, bw)
        p_15 = _kde_pdf(15.0, samples, bw)
        p_10 = _kde_pdf(10.0, samples, bw)
        assert p_5 > p_10
        assert p_15 > p_10

    def test_kde_cdf_bounds(self):
        samples = [10.0, 11.0, 12.0, 10.5, 11.5]
        bw = _silverman_bandwidth(samples)
        # CDF at very low should be ~0
        assert _kde_cdf_numerical(-100.0, samples, bw) < 0.01
        # CDF at very high should be ~1
        assert _kde_cdf_numerical(100.0, samples, bw) > 0.99

    def test_kde_cdf_monotonic(self):
        samples = [8.0, 9.0, 10.0, 11.0, 12.0]
        bw = _silverman_bandwidth(samples)
        prev = 0.0
        for x in range(5, 16):
            c = _kde_cdf_numerical(float(x), samples, bw)
            assert c >= prev
            prev = c

    def test_silverman_bandwidth_reasonable(self):
        samples = [10.0 + i * 0.5 for i in range(10)]
        bw = _silverman_bandwidth(samples)
        assert 0.1 < bw < 5.0

    def test_prob_for_outcome_with_kde(self):
        samples = [10.0, 10.5, 11.0, 10.2, 10.8]
        bw = _silverman_bandwidth(samples)
        # Probability near center should be higher than at edges
        p_center = _prob_for_outcome("10°C", 10.5, 1.0, kde_samples=samples, kde_bw=bw)
        p_edge = _prob_for_outcome("14°C", 10.5, 1.0, kde_samples=samples, kde_bw=bw)
        assert p_center > p_edge

    def test_kde_probs_sum_approx_one(self):
        """KDE-based bins should approximately sum to 1."""
        providers = [_make_provider(10.0 + i * 0.3) for i in range(10)]
        agg = _make_agg(11.0)
        result = analyze("Paris", "2026-02-17", "France", agg, providers)
        total = sum(b.prob for b in result.bins)
        assert abs(total - 1.0) < 0.05  # KDE integration may have small error


# ===========================================================================
# N-outcome allocation tests
# ===========================================================================

class TestNOutcomeAllocation:
    def test_max_picks_3_produces_up_to_3_outcomes(self):
        providers = [_make_provider(10.0 + i * 0.3) for i in range(8)]
        agg = _make_agg(10.5)
        market = _make_market([
            ("8°C or less", 0.05),
            ("9°C", 0.10),
            ("10°C", 0.25),
            ("11°C", 0.25),
            ("12°C", 0.10),
            ("13°C or more", 0.05),
        ])
        result = analyze("Paris", "2026-02-17", "France", agg, providers,
                        market=market, max_picks=3)
        output = format_analysis(result)
        assert "ALLOCATION" in output
        assert "outcomes" in output

    def test_smart_reduction_to_2_when_high_prob(self):
        """When top 2 picks have >=55% combined prob and positive edge, reduce to 2."""
        providers = [_make_provider(10.0 + i * 0.3) for i in range(8)]
        agg = _make_agg(10.5)
        # Top 2 bins (10°C, 11°C) have high model prob and prices below model
        market = _make_market([
            ("9°C", 0.08),
            ("10°C", 0.15),
            ("11°C", 0.15),
            ("12°C", 0.08),
        ])
        result = analyze("Paris", "2026-02-17", "France", agg, providers,
                        market=market, max_picks=3)
        output = format_analysis(result)
        assert "ALLOCATION" in output
        assert "2 outcomes" in output

    def test_max_picks_2_backwards_compatible(self):
        providers = [_make_provider(10.0 + i * 0.3) for i in range(5)]
        agg = _make_agg(10.5)
        market = _make_market([
            ("9°C", 0.10),
            ("10°C", 0.25),
            ("11°C", 0.25),
            ("12°C or more", 0.05),
        ])
        result = analyze("Paris", "2026-02-17", "France", agg, providers,
                        market=market, max_picks=2)
        output = format_analysis(result)
        assert "2 outcomes" in output


# ===========================================================================
# Instability factor tests
# ===========================================================================

class TestInstabilityFactor:
    def test_instability_widens_sigma(self):
        """Higher instability factor should produce wider distribution."""
        providers = [_make_provider(10.0) for _ in range(5)]
        agg_calm = _make_agg(10.0)
        agg_calm.instability_factor = 1.0
        agg_stormy = _make_agg(10.0)
        agg_stormy.instability_factor = 1.25

        result_calm = analyze("A", "2026-02-17", "France", agg_calm, providers)
        result_stormy = analyze("A", "2026-02-17", "France", agg_stormy, providers)
        assert result_stormy.sigma > result_calm.sigma
