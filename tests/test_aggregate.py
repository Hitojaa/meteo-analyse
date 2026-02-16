"""Tests for the aggregation module."""

import math

import pytest

from aggregate import (
    _filter_outliers,
    _mad,
    _median,
    _skewness,
    _skewness_blend,
    _weighted_mean,
    _climate_anchoring,
    _confidence_score,
    aggregate,
)
from history import HistoricalStats
from models import DataQuality, ProviderResult


def _make_result(
    tmin: float | None, tmax: float | None, quality: DataQuality = DataQuality.DAILY_DIRECT
) -> ProviderResult:
    return ProviderResult(
        provider_name="test",
        tmin_c=tmin,
        tmax_c=tmax,
        date="2025-06-15",
        timezone="Europe/Paris",
        quality=quality,
    )


class TestMedianAndMad:
    def test_median_odd(self):
        assert _median([1, 3, 5]) == 3

    def test_median_even(self):
        assert _median([1, 3, 5, 7]) == 4.0

    def test_mad(self):
        # values: [1, 2, 3, 4, 100]
        # median = 3, deviations = [2, 1, 0, 1, 97], MAD = median([0,1,1,2,97]) = 1
        assert _mad([1, 2, 3, 4, 100]) == 1


class TestFilterOutliers:
    def test_no_outliers(self):
        vals = [10.0, 11.0, 10.5]
        weights = [1.0, 1.0, 1.0]
        fv, fw = _filter_outliers(vals, weights)
        assert fv == vals
        assert fw == weights

    def test_removes_outlier(self):
        # Median=10.5, MAD for [10, 10.5, 11, 50] -> median=10.75
        # deviations: [0.75, 0.25, 0.25, 39.25] -> MAD=0.5
        # threshold = 2.5 * 0.5 = 1.25 -> 50 is way out
        vals = [10.0, 10.5, 11.0, 50.0]
        weights = [1.0, 1.0, 1.0, 1.0]
        fv, fw = _filter_outliers(vals, weights)
        assert 50.0 not in fv
        assert len(fv) == 3

    def test_two_values_no_filtering(self):
        vals = [10.0, 50.0]
        weights = [1.0, 1.0]
        fv, fw = _filter_outliers(vals, weights)
        assert fv == vals


class TestWeightedMean:
    def test_equal_weights(self):
        assert _weighted_mean([10.0, 20.0], [1.0, 1.0]) == 15.0

    def test_unequal_weights(self):
        # (10*2 + 20*1) / 3 = 40/3 ≈ 13.33
        result = _weighted_mean([10.0, 20.0], [2.0, 1.0])
        assert abs(result - 13.333) < 0.01


class TestSkewness:
    def test_symmetric_distribution(self):
        # Perfectly symmetric values should have skewness ≈ 0
        skew = _skewness([10.0, 11.0, 12.0, 13.0, 14.0])
        assert abs(skew) < 0.01

    def test_right_skewed(self):
        # One high outlier → positive skewness
        skew = _skewness([10.0, 10.5, 11.0, 11.5, 20.0])
        assert skew > 0.5

    def test_left_skewed(self):
        # One low outlier → negative skewness
        skew = _skewness([1.0, 10.0, 10.5, 11.0, 11.5])
        assert skew < -0.5

    def test_too_few_values(self):
        assert _skewness([1.0, 2.0]) == 0.0
        assert _skewness([1.0]) == 0.0

    def test_all_identical(self):
        assert _skewness([5.0, 5.0, 5.0]) == 0.0


class TestSkewnessBlend:
    def test_symmetric_returns_weighted_mean(self):
        vals = [10.0, 11.0, 12.0, 13.0, 14.0]
        weights = [1.0] * 5
        result = _skewness_blend(vals, weights)
        expected = _weighted_mean(vals, weights)
        assert abs(result - expected) < 0.1

    def test_skewed_blends_toward_median(self):
        # Right-skewed: one high value
        vals = [10.0, 10.5, 11.0, 11.5, 25.0]
        weights = [1.0] * 5
        w_mean = _weighted_mean(vals, weights)
        med = _median(vals)
        result = _skewness_blend(vals, weights)
        # Result should be between median and mean, closer to median
        assert med <= result <= w_mean


class TestClimateAnchoring:
    def test_no_correction_within_normal(self):
        # Estimate within 1.5σ of historical → no change
        result = _climate_anchoring(12.0, [11.0, 12.0, 13.0], 11.0, 3.0)
        assert result == 12.0

    def test_correction_for_extreme_anomaly(self):
        # Estimate is 3σ above historical mean → should pull back
        result = _climate_anchoring(20.0, [18.0, 20.0, 22.0], 10.0, 3.0)
        assert result < 20.0  # Should be pulled toward 10.0
        assert result > 10.0  # But not too much

    def test_zero_std_no_correction(self):
        result = _climate_anchoring(15.0, [14.0, 15.0, 16.0], 10.0, 0.0)
        assert result == 15.0


class TestConfidenceScore:
    def test_high_confidence(self):
        # Many sources, low spread, consistent with history
        vals = [10.0, 10.2, 10.1, 10.3, 10.0, 10.1, 10.2, 10.0]
        score = _confidence_score(vals, 10.0, 2.0, 10.1)
        assert score >= 80

    def test_low_confidence_high_spread(self):
        # Few sources, high spread
        vals = [5.0, 15.0, 25.0]
        score = _confidence_score(vals, 10.0, 3.0, 15.0)
        assert score < 70

    def test_neutral_without_history(self):
        vals = [10.0, 11.0, 12.0]
        score = _confidence_score(vals, None, None, 11.0)
        assert 30 < score < 80


class TestAggregate:
    def test_basic_aggregation(self):
        results = [
            _make_result(5.0, 20.0),
            _make_result(6.0, 21.0),
            _make_result(5.5, 20.5),
        ]
        agg = aggregate(results)
        assert 5.0 <= agg.tmin_c <= 6.0
        assert 20.0 <= agg.tmax_c <= 21.0
        assert agg.sources_used == 3
        assert agg.warning is None

    def test_outlier_excluded(self):
        results = [
            _make_result(5.0, 20.0),
            _make_result(5.5, 20.5),
            _make_result(6.0, 21.0),
            _make_result(50.0, 80.0),  # outlier
        ]
        agg = aggregate(results)
        # The outlier should be excluded; result should be close to 5-6 / 20-21
        assert agg.tmin_c < 10.0
        assert agg.tmax_c < 25.0

    def test_single_source_warning(self):
        results = [_make_result(5.0, 20.0)]
        agg = aggregate(results)
        assert agg.tmin_c == 5.0
        assert agg.tmax_c == 20.0
        assert agg.warning is not None
        assert "1 source" in agg.warning

    def test_skips_none_values(self):
        results = [
            _make_result(5.0, 20.0),
            _make_result(None, None),
            _make_result(6.0, 21.0),
        ]
        agg = aggregate(results)
        assert agg.sources_used == 2

    def test_skips_nan_values(self):
        results = [
            _make_result(5.0, 20.0),
            _make_result(float("nan"), float("nan")),
            _make_result(6.0, 21.0),
        ]
        agg = aggregate(results)
        assert agg.sources_used == 2

    def test_no_valid_results_raises(self):
        results = [_make_result(None, None)]
        with pytest.raises(ValueError, match="No valid"):
            aggregate(results)

    def test_quality_weighting(self):
        # Hourly-computed result should have lower weight
        results = [
            _make_result(10.0, 25.0, DataQuality.DAILY_DIRECT),
            _make_result(10.0, 25.0, DataQuality.DAILY_DIRECT),
            _make_result(12.0, 27.0, DataQuality.COMPUTED_FROM_HOURLY),
        ]
        agg = aggregate(results)
        # Result should be closer to 10/25 than to 12/27
        assert agg.tmin_c < 11.0
        assert agg.tmax_c < 26.0

    def test_has_skewness_and_confidence(self):
        results = [
            _make_result(5.0, 20.0),
            _make_result(5.5, 20.5),
            _make_result(6.0, 21.0),
        ]
        agg = aggregate(results)
        assert isinstance(agg.skewness_tmin, float)
        assert isinstance(agg.skewness_tmax, float)
        assert isinstance(agg.confidence, float)
        assert 0 <= agg.confidence <= 100

    def test_with_historical_data(self):
        results = [
            _make_result(5.0, 20.0),
            _make_result(5.5, 20.5),
            _make_result(6.0, 21.0),
        ]
        hist = HistoricalStats(
            tmin_mean=5.0,
            tmin_std=2.0,
            tmax_mean=20.0,
            tmax_std=3.0,
            sample_size=50,
            years_covered=5,
        )
        agg = aggregate(results, historical=hist)
        assert agg.hist_tmin_mean == 5.0
        assert agg.hist_tmax_mean == 20.0
        assert agg.hist_tmin_std == 2.0
        assert agg.hist_tmax_std == 3.0
        assert agg.hist_sample_size == 50

    def test_historical_anchoring_corrects_extreme(self):
        # All providers predict way above historical norm
        results = [
            _make_result(20.0, 40.0),
            _make_result(21.0, 41.0),
            _make_result(22.0, 42.0),
        ]
        hist = HistoricalStats(
            tmin_mean=5.0,
            tmin_std=2.0,
            tmax_mean=20.0,
            tmax_std=3.0,
            sample_size=50,
            years_covered=5,
        )
        agg_with = aggregate(results, historical=hist)
        agg_without = aggregate(results)
        # With historical anchoring, extreme values should be pulled back slightly
        assert agg_with.tmin_c <= agg_without.tmin_c
        assert agg_with.tmax_c <= agg_without.tmax_c
