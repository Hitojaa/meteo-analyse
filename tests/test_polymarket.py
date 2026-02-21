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
    _normal_cdf,
    _prob_for_outcome,
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
    def test_basic_value_bets_returns_strategy(self):
        """With underpriced outcomes, should find value bets."""
        market = _make_market([
            ("8°C or less", 0.02),
            ("9°C", 0.10),
            ("10°C", 0.20),
            ("11°C", 0.15),
            ("12°C or more", 0.05),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5, budget=10.0)
        assert h is not None
        assert len(h.bets) >= 2

    def test_budget_fully_allocated_to_value_bets(self):
        market = _make_market([
            ("9°C", 0.10),
            ("10°C", 0.20),
            ("11°C", 0.15),
            ("12°C or more", 0.05),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5, budget=10.0)
        assert h is not None
        # Budget is fully allocated across value bets (stake > 0)
        total_stake = sum(bet.stake for bet in h.bets if bet.stake > 0)
        assert abs(total_stake - 10.0) < 0.05
        # Non-value bets have zero stake
        for bet in h.bets:
            if bet.edge < 0.02:
                assert bet.stake == 0

    def test_only_staked_bets_have_positive_edge(self):
        """All bets with stake > 0 should have positive edge (model > price)."""
        market = _make_market([
            ("9°C", 0.10),
            ("10°C", 0.20),
            ("11°C", 0.50),  # overpriced → shown but no stake
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5, budget=10.0)
        assert h is not None
        for bet in h.bets:
            if bet.stake > 0:
                assert bet.edge > 0
                assert bet.model_prob > bet.market_price

    def test_positive_ev_on_underpriced_market(self):
        """When outcomes are underpriced, EV should be positive."""
        market = _make_market([
            ("9°C", 0.10),
            ("10°C", 0.15),
            ("11°C", 0.10),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5, budget=10.0)
        assert h is not None
        assert h.expected_value > 0

    def test_no_bets_when_all_overpriced(self):
        """When all outcomes are overpriced, should return None."""
        market = _make_market([
            ("9°C", 0.50),
            ("10°C", 0.80),
            ("11°C", 0.50),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5, budget=10.0)
        assert h is None

    def test_no_hedging_when_all_prices_zero(self):
        market = _make_market([
            ("9°C", 0.0),
            ("10°C", 0.0),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5, budget=10.0)
        assert h is None

    def test_skips_no_edge_outcomes(self):
        """Overpriced outcomes should be excluded."""
        market = _make_market([
            ("1°C", 0.05),   # far from center=10, no edge
            ("10°C", 0.20),  # model ~26%, price 20% → edge
            ("11°C", 0.15),  # model ~21%, price 15% → edge
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5, budget=10.0)
        assert h is not None
        labels = [bet.label for bet in h.bets]
        assert "1°C" not in labels

    def test_expected_value_computed(self):
        market = _make_market([
            ("9°C", 0.10),
            ("10°C", 0.20),
            ("11°C", 0.15),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5, budget=10.0)
        assert h is not None
        assert isinstance(h.expected_value, float)

    def test_market_url_and_volume_set(self):
        market = _make_market([
            ("10°C", 0.20),
            ("11°C", 0.15),
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5, budget=10.0)
        assert h is not None
        assert h.market_url == "https://polymarket.com/event/test-market"
        assert h.market_volume == 50000.0

    def test_kelly_allocates_more_to_bigger_edge(self):
        """Larger edge should get proportionally more stake."""
        market = _make_market([
            ("10°C", 0.10),  # model ~26%, big edge
            ("11°C", 0.18),  # model ~21%, smaller edge
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5, budget=10.0)
        assert h is not None
        staked = [b for b in h.bets if b.stake > 0]
        assert len(staked) == 2
        # First staked bet (sorted by model prob) should have higher stake
        # since 10°C has bigger edge
        staked.sort(key=lambda b: b.edge, reverse=True)
        assert staked[0].stake >= staked[1].stake

    def test_includes_overpriced_outcomes_for_context(self):
        """Overpriced outcomes within coverage should appear with stake=0."""
        market = _make_market([
            ("9°C", 0.10),   # model ~21%, edge +11%
            ("10°C", 0.50),  # model ~26%, edge -24% (overpriced)
            ("11°C", 0.10),  # model ~21%, edge +11%
        ])
        h = _compute_hedging(market, center=10.0, sigma=1.5, budget=10.0)
        assert h is not None
        labels = [b.label for b in h.bets]
        assert "10°C" in labels  # included for context
        bet_10 = next(b for b in h.bets if b.label == "10°C")
        assert bet_10.stake == 0  # but no money on it


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

    def test_format_shows_betting_strategy_section(self):
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
        assert "BETTING STRATEGY" in output
        assert "PROFIT" in output
        assert "★" in output
