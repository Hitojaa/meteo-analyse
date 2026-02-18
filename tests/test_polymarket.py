"""Tests for the Polymarket betting analysis module."""

import math

import pytest

from models import AggregatedResult, DataQuality, ProviderResult
from polymarket import (
    BettingAnalysis,
    BettingBin,
    VALUE_BET_THRESHOLD,
    _is_us,
    _normal_cdf,
    _prob_for_outcome,
    analyze,
    format_analysis,
    match_with_market,
)
from polymarket_api import (
    MarketOutcome,
    TemperatureMarket,
    normalize_outcome_key,
    _city_matches,
    _date_matches,
    _detect_unit,
)


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


def _make_market(outcomes_data: list[tuple[str, float]], unit: str = "°F") -> TemperatureMarket:
    """Helper to create a mock TemperatureMarket."""
    outcomes = [
        MarketOutcome(label=label, price=price, token_id=f"token_{i}")
        for i, (label, price) in enumerate(outcomes_data)
    ]
    return TemperatureMarket(
        title="Highest temperature in New York on February 17?",
        slug="highest-temperature-new-york-february-17",
        url="https://polymarket.com/event/highest-temperature-new-york-february-17",
        volume=72000.0,
        outcomes=outcomes,
        unit=unit,
        active=True,
    )


# ===========================================================================
# polymarket_api tests
# ===========================================================================

class TestNormalizeOutcomeKey:
    def test_range_with_unit(self):
        assert normalize_outcome_key("44-45°F") == "44-45"

    def test_range_without_unit(self):
        assert normalize_outcome_key("44-45") == "44-45"

    def test_single_celsius(self):
        assert normalize_outcome_key("8°C") == "8"

    def test_single_no_unit(self):
        assert normalize_outcome_key("8") == "8"

    def test_or_less(self):
        assert normalize_outcome_key("41°F or less") == "le41"

    def test_or_less_no_unit(self):
        assert normalize_outcome_key("41 or less") == "le41"

    def test_or_more(self):
        assert normalize_outcome_key("56°F or more") == "ge56"

    def test_ou_moins(self):
        assert normalize_outcome_key("41 ou moins") == "le41"

    def test_ou_plus(self):
        assert normalize_outcome_key("56 ou plus") == "ge56"

    def test_negative_temp(self):
        assert normalize_outcome_key("-5°C") == "-5"

    def test_negative_range(self):
        assert normalize_outcome_key("-2--1") == "-2--1"


class TestCityMatches:
    def test_exact_match(self):
        assert _city_matches("Highest temperature in New York on Feb 17?", "New York")

    def test_case_insensitive(self):
        assert _city_matches("Highest temperature in LONDON on Feb 17?", "london")

    def test_no_match(self):
        assert not _city_matches("Highest temperature in Paris on Feb 17?", "London")

    def test_alias_nyc(self):
        assert _city_matches("Highest temperature in NYC on Feb 17?", "new york")


class TestDateMatches:
    def test_full_date(self):
        assert _date_matches("Highest temperature in NYC on February 17?", "2026-02-17")

    def test_short_date(self):
        assert _date_matches("Highest temperature in NYC on Feb 17?", "2026-02-17")

    def test_no_match(self):
        assert not _date_matches("Highest temperature in NYC on Feb 18?", "2026-02-17")


class TestDetectUnit:
    def test_fahrenheit_in_title(self):
        assert _detect_unit("Temperature in Fahrenheit", "") == "°F"

    def test_celsius_in_description(self):
        assert _detect_unit("Temperature", "degrees Celsius") == "°C"

    def test_default_fahrenheit(self):
        assert _detect_unit("Temperature forecast", "") == "°F"


# ===========================================================================
# polymarket.py tests — existing
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


class TestProbForOutcome:
    def test_range_probability(self):
        # P(44-45) with center=45, sigma=2
        p = _prob_for_outcome("44-45", 45.0, 2.0, True)
        assert 0.1 < p < 0.5

    def test_single_value(self):
        # P(10) with center=10, sigma=1
        p = _prob_for_outcome("10", 10.0, 1.0, False)
        assert 0.2 < p < 0.5

    def test_or_less_tail(self):
        # P(≤35) with center=45, sigma=2 — should be very small
        p = _prob_for_outcome("35 or less", 45.0, 2.0, True)
        assert p < 0.01

    def test_or_more_tail(self):
        # P(≥55) with center=45, sigma=2 — should be very small
        p = _prob_for_outcome("55 or more", 45.0, 2.0, True)
        assert p < 0.01

    def test_center_bin_has_highest_prob(self):
        bins = ["42-43", "44-45", "46-47"]
        probs = [_prob_for_outcome(b, 45.0, 2.0, True) for b in bins]
        # 44-45 contains the center (45) so should be highest
        assert probs[1] == max(probs)


# ===========================================================================
# Market matching tests
# ===========================================================================

class TestMatchWithMarket:
    def test_exact_match_computes_edge(self):
        bins = [
            BettingBin("42-43°F", 0.15),
            BettingBin("44-45°F", 0.35),
            BettingBin("46-47°F", 0.30),
        ]
        market = _make_market([
            ("42-43", 0.20),
            ("44-45", 0.30),
            ("46-47", 0.35),
        ])
        match_with_market(bins, market)

        assert bins[0].market_price == 0.20
        assert bins[0].edge == pytest.approx(-0.05, abs=0.001)
        assert bins[1].market_price == 0.30
        assert bins[1].edge == pytest.approx(0.05, abs=0.001)
        assert bins[1].is_value_bet is True  # edge >= 5%

    def test_value_bet_threshold(self):
        bins = [BettingBin("44-45°F", 0.30)]
        market = _make_market([("44-45", 0.26)])  # edge = 4%, below 5%
        match_with_market(bins, market)
        assert bins[0].is_value_bet is False

        bins2 = [BettingBin("44-45°F", 0.30)]
        market2 = _make_market([("44-45", 0.24)])  # edge = 6%, above 5%
        match_with_market(bins2, market2)
        assert bins2[0].is_value_bet is True

    def test_no_match_leaves_none(self):
        bins = [BettingBin("44-45°F", 0.35)]
        market = _make_market([("50-51", 0.10)])
        match_with_market(bins, market)
        assert bins[0].market_price is None
        assert bins[0].edge is None

    def test_tail_bins_match(self):
        bins = [
            BettingBin("41°F or less", 0.05),
            BettingBin("56°F or more", 0.02),
        ]
        market = _make_market([
            ("41 or less", 0.04),
            ("56 or more", 0.01),
        ])
        match_with_market(bins, market)
        assert bins[0].market_price == 0.04
        assert bins[1].market_price == 0.01


# ===========================================================================
# Analyze with market
# ===========================================================================

class TestAnalyzeWithMarket:
    def test_market_found_flag(self):
        providers = [_make_provider(10.0 + i * 0.3) for i in range(5)]
        agg = _make_agg(11.0)
        market = _make_market([
            ("49 or less", 0.10),
            ("50-51", 0.30),
            ("52-53", 0.35),
            ("54-55", 0.20),
            ("56 or more", 0.05),
        ])
        result = analyze("NYC", "2026-02-17", "United States", agg, providers, market=market)
        assert result.market_found is True
        assert result.market_url is not None
        assert result.market_volume == 72000.0

    def test_bins_match_market_outcomes(self):
        providers = [_make_provider(10.0) for _ in range(3)]
        agg = _make_agg(10.0)
        market = _make_market([
            ("7", 0.10),
            ("8", 0.30),
            ("9", 0.35),
            ("10", 0.20),
            ("11", 0.05),
        ], unit="°C")
        result = analyze("London", "2026-02-17", "United Kingdom", agg, providers, market=market)
        # Should have exactly 5 bins matching the market
        assert len(result.bins) == 5

    def test_value_bets_collected(self):
        providers = [_make_provider(10.0) for _ in range(5)]
        agg = _make_agg(10.0)
        # Market underprices "10" — our model should see it as value
        market = _make_market([
            ("9", 0.50),
            ("10", 0.15),  # market says 15% but our model likely says more
            ("11", 0.35),
        ], unit="°C")
        result = analyze("Test", "2026-02-17", "France", agg, providers, market=market)
        # Check that value_bets list is populated if any edge >= threshold
        for vb in result.value_bets:
            assert vb.edge is not None
            assert vb.edge >= VALUE_BET_THRESHOLD

    def test_no_market_falls_back_to_model(self):
        providers = [_make_provider(10.0 + i * 0.2) for i in range(5)]
        agg = _make_agg(10.5)
        result = analyze("Berlin", "2026-02-17", "Germany", agg, providers, market=None)
        assert result.market_found is False
        assert result.market_url is None
        assert len(result.bins) >= 3


# ===========================================================================
# Analyze without market (existing tests)
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
    def test_format_no_market(self):
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
        assert "not found" in output
        assert "BEST" in output
        assert "BUY" in output
        assert "9°C" in output

    def test_format_with_market(self):
        bins = [
            BettingBin("44-45°F", 0.32, market_price=0.28, edge=0.04),
            BettingBin("46-47°F", 0.34, market_price=0.30, edge=0.04, is_best=True),
            BettingBin("48-49°F", 0.20, market_price=0.12, edge=0.08, is_value_bet=True),
        ]
        analysis = BettingAnalysis(
            city="New York", date="2026-02-17", unit="°F",
            predicted=46.2, sigma=2.1, bins=bins,
            skewness=0.0, confidence=75.0,
            market_found=True,
            market_url="https://polymarket.com/event/test",
            market_volume=72000.0,
            value_bets=[bins[2]],
        )
        output = format_analysis(analysis)
        assert "Live market: YES" in output
        assert "Model" in output
        assert "Market" in output
        assert "Edge" in output
        assert "VALUE" in output
        assert "$72,000" in output

    def test_format_market_no_value_bets(self):
        bins = [
            BettingBin("8°C", 0.30, market_price=0.31, edge=-0.01, is_best=True),
        ]
        analysis = BettingAnalysis(
            city="Paris", date="2026-02-17", unit="°C",
            predicted=8.0, sigma=1.0, bins=bins,
            skewness=0.0, confidence=70.0,
            market_found=True,
            market_url="https://polymarket.com/event/test",
            market_volume=50000.0,
            value_bets=[],
        )
        output = format_analysis(analysis)
        assert "No strong value bet" in output
        assert "efficient" in output

    def test_format_low_volume_warning(self):
        bins = [
            BettingBin("8°C", 0.40, market_price=0.20, edge=0.20, is_value_bet=True, is_best=True),
        ]
        analysis = BettingAnalysis(
            city="Small", date="2026-02-17", unit="°C",
            predicted=8.0, sigma=1.0, bins=bins,
            skewness=0.0, confidence=60.0,
            market_found=True,
            market_url="https://polymarket.com/event/test",
            market_volume=5000.0,
            value_bets=[bins[0]],
        )
        output = format_analysis(analysis)
        assert "WARNING" in output
        assert "Low volume" in output

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
