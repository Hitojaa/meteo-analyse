"""Tests for the Polymarket betting analysis module."""

import math

import pytest

from models import AggregatedResult, DataQuality, ProviderResult
from polymarket import (
    BettingAnalysis,
    BettingBin,
    _is_us,
    _normal_cdf,
    analyze,
    format_analysis,
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
        # P(X ≤ μ+σ) ≈ 0.8413
        assert abs(_normal_cdf(1.0, 0.0, 1.0) - 0.8413) < 0.001

    def test_zero_sigma(self):
        assert _normal_cdf(5.0, 3.0, 0.0) == 1.0
        assert _normal_cdf(1.0, 3.0, 0.0) == 0.0


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
        # Middle bins should contain "-" indicating range (e.g., "50-51°F")
        middle_bins = [b for b in result.bins if "-" in b.label]
        for b in middle_bins:
            nums = b.label.replace("°F", "").split("-")
            assert int(nums[1]) - int(nums[0]) == 1  # 2°F range: e.g. 50-51

    def test_celsius_bins_are_1c_wide(self):
        providers = [_make_provider(7.0 + i * 0.3) for i in range(8)]
        agg = _make_agg(8.0)
        result = analyze("London", "2026-02-17", "United Kingdom", agg, providers)
        # Middle bins should be single integers like "8°C"
        middle_bins = [b for b in result.bins if "or" not in b.label]
        for b in middle_bins:
            val = b.label.replace("°C", "")
            int(val)  # should not raise

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
        assert abs(result.predicted - 15.2) < 1.0  # in °C, close to agg tmax

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


class TestFormatAnalysis:
    def test_format_contains_key_elements(self):
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
        assert "2026-02-17" in output
        assert "BEST" in output
        assert "RECOMMENDATION" in output
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
