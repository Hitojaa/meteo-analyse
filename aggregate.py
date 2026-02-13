"""Robust aggregation of temperature forecasts from multiple providers."""

from __future__ import annotations

import math
from statistics import median

from models import AggregatedResult, ProviderResult


def _median(values: list[float]) -> float:
    return median(values)


def _mad(values: list[float]) -> float:
    """Median Absolute Deviation."""
    med = _median(values)
    return _median([abs(v - med) for v in values])


def _filter_outliers(
    values: list[float], weights: list[float], mad_threshold: float = 2.5
) -> tuple[list[float], list[float]]:
    """Remove outliers using MAD-based filtering.

    Values farther than mad_threshold * MAD from the median are discarded.
    If MAD is 0 (all values identical or only 2), no filtering is applied.
    """
    if len(values) <= 2:
        return values, weights

    med = _median(values)
    mad = _mad(values)

    if mad == 0:
        return values, weights

    filtered_vals: list[float] = []
    filtered_weights: list[float] = []
    for v, w in zip(values, weights):
        if abs(v - med) <= mad_threshold * mad:
            filtered_vals.append(v)
            filtered_weights.append(w)

    # Safeguard: if everything was filtered, keep all
    if not filtered_vals:
        return values, weights

    return filtered_vals, filtered_weights


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total_w = sum(weights)
    if total_w == 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total_w


def aggregate(results: list[ProviderResult]) -> AggregatedResult:
    """Compute a robust aggregated tmin/tmax from provider results.

    Steps:
      1. Filter out results with None/NaN values.
      2. If fewer than 2 valid sources, return best estimate with a warning.
      3. Compute median, MAD-filter outliers, then weighted mean.
    """
    valid = [
        r
        for r in results
        if r.tmin_c is not None
        and r.tmax_c is not None
        and not math.isnan(r.tmin_c)
        and not math.isnan(r.tmax_c)
    ]

    if not valid:
        raise ValueError("No valid provider results to aggregate")

    warning: str | None = None
    if len(valid) < 2:
        warning = (
            f"Only {len(valid)} source(s) available – "
            "result may be less reliable."
        )

    tmins = [r.tmin_c for r in valid]
    tmaxs = [r.tmax_c for r in valid]
    weights = [r.quality.weight for r in valid]

    # Filter outliers independently for tmin and tmax
    tmins_f, w_tmin = _filter_outliers(tmins, weights)
    tmaxs_f, w_tmax = _filter_outliers(tmaxs, weights)

    agg_tmin = _weighted_mean(tmins_f, w_tmin)
    agg_tmax = _weighted_mean(tmaxs_f, w_tmax)

    return AggregatedResult(
        tmin_c=round(agg_tmin, 1),
        tmax_c=round(agg_tmax, 1),
        tmin_range=(round(min(tmins), 1), round(max(tmins), 1)),
        tmax_range=(round(min(tmaxs), 1), round(max(tmaxs), 1)),
        sources_used=len(valid),
        warning=warning,
    )
