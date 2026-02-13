"""Tests for the aggregation module."""

import math

import pytest

from aggregate import _filter_outliers, _mad, _median, _weighted_mean, aggregate
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
