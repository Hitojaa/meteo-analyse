"""Robust aggregation of temperature forecasts from multiple providers.

Combines MAD-based outlier filtering with:
  - Provider bias correction (from self-learning verification)
  - Accuracy-based weighting (providers with lower MAE get more weight)
  - Skewness-aware blending (mean ↔ median)
  - Historical climatological anchoring (Bayesian shrinkage)
  - Confidence scoring
"""

from __future__ import annotations

import math
from statistics import median
from statistics import stdev as _stdev

from history import HistoricalStats
from models import AggregatedResult, ProviderResult
from verification import ProviderAccuracy


# Minimum verified forecasts before applying bias correction
MIN_VERIFICATIONS_FOR_BIAS = 3


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


def _skewness(values: list[float]) -> float:
    """Fisher-Pearson coefficient of skewness.

    Measures asymmetry of the distribution:
      > 0 → right-skewed (tail toward higher values, some providers predict much higher)
      < 0 → left-skewed  (tail toward lower values, some providers predict much lower)
      ≈ 0 → symmetric distribution
    """
    n = len(values)
    if n < 3:
        return 0.0
    m = sum(values) / n
    m2 = sum((x - m) ** 2 for x in values) / n
    if m2 == 0:
        return 0.0
    m3 = sum((x - m) ** 3 for x in values) / n
    return m3 / (m2**1.5)


def _skewness_blend(values: list[float], weights: list[float]) -> float:
    """Blend weighted mean and median based on skewness magnitude.

    When the distribution is symmetric (skew ≈ 0), returns the weighted mean.
    When skewed, progressively blends toward the median which is more robust
    to asymmetric outliers.

    The blend factor uses a sigmoid-like mapping: |skew| / (1 + |skew|)
      - skew = 0   → blend = 0.00 → 100% mean
      - skew = 0.5 → blend = 0.33 → 67% mean, 33% median
      - skew = 1.0 → blend = 0.50 → 50% mean, 50% median
      - skew = 2.0 → blend = 0.67 → 33% mean, 67% median
    """
    w_mean = _weighted_mean(values, weights)
    med = _median(values)
    skew = abs(_skewness(values))

    blend = skew / (1.0 + skew)
    return w_mean * (1.0 - blend) + med * blend


def _climate_anchoring(
    estimate: float,
    provider_values: list[float],
    hist_mean: float,
    hist_std: float,
) -> float:
    """Bayesian shrinkage toward climatological mean for extreme anomalies.

    Only activates when the forecast deviates > 1.5σ from the historical
    norm. The amount of shrinkage depends on:
      - How extreme the anomaly is (higher z → more correction)
      - How much providers disagree (higher spread → more correction)

    Maximum shrinkage is capped at 15% to avoid over-correcting.
    """
    if hist_std == 0:
        return estimate

    provider_spread = _stdev(provider_values) if len(provider_values) > 1 else 0.0

    z_anomaly = abs(estimate - hist_mean) / hist_std

    if z_anomaly <= 1.5:
        return estimate

    # Uncertainty ratio: high provider disagreement → more correction
    uncertainty = min(provider_spread / hist_std, 1.0) if hist_std > 0 else 0.0

    shrinkage = min(0.15, (z_anomaly - 1.5) * 0.05 * max(uncertainty, 0.3))

    return estimate * (1.0 - shrinkage) + hist_mean * shrinkage


def _accuracy_multiplier(
    provider_name: str,
    accuracy: dict[str, ProviderAccuracy] | None,
) -> float:
    """Compute a weight multiplier based on provider's verified accuracy.

    Returns a value between 0.7 (poor accuracy) and 1.3 (excellent accuracy).
    Providers with no accuracy data get a neutral 1.0.

    Based on combined MAE: lower MAE → higher weight.
      - MAE = 0°C → 1.3 (perfect)
      - MAE = 1°C → 1.0 (baseline)
      - MAE = 2°C → 0.83
      - MAE ≥ 3°C → 0.7 (floor)
    """
    if accuracy is None:
        return 1.0
    acc = accuracy.get(provider_name)
    if acc is None or acc.n < MIN_VERIFICATIONS_FOR_BIAS:
        return 1.0
    combined_mae = (acc.tmin_mae + acc.tmax_mae) / 2.0
    return max(0.7, min(1.3, 2.0 / (1.0 + combined_mae)))


def _confidence_score(
    values: list[float],
    hist_mean: float | None,
    hist_std: float | None,
    estimate: float,
    has_accuracy: bool = False,
) -> float:
    """Compute a 0–100 confidence score for the aggregated estimate.

    Based on four factors:
      1. Number of sources (more independent models → more reliable)
      2. Provider agreement (lower IQR → higher confidence)
      3. Historical consistency (closer to climatological norm → higher confidence)
      4. Self-learning data available (verified accuracy boosts confidence)
    """
    n = len(values)

    # Factor 1: Source count (diminishing returns, max at 8)
    source_score = min(n / 8.0, 1.0)

    # Factor 2: Provider agreement via IQR
    if n > 1:
        sorted_vals = sorted(values)
        q1_idx = max(0, n // 4)
        q3_idx = min(n - 1, 3 * n // 4)
        iqr = sorted_vals[q3_idx] - sorted_vals[q1_idx]
        spread_score = max(0.0, 1.0 - iqr / 10.0)
    else:
        spread_score = 0.3

    # Factor 3: Historical consistency
    if hist_mean is not None and hist_std is not None and hist_std > 0:
        z = abs(estimate - hist_mean) / hist_std
        hist_score = max(0.0, 1.0 - z / 4.0)
    else:
        hist_score = 0.5  # neutral if no history

    # Factor 4: Self-learning bonus
    learning_bonus = 0.08 if has_accuracy else 0.0

    score = source_score * 0.3 + spread_score * 0.40 + hist_score * 0.25 + learning_bonus
    return min(round(score * 100, 0), 99)  # Cap at 99 (never 100% certain)


def aggregate(
    results: list[ProviderResult],
    historical: HistoricalStats | None = None,
    provider_accuracy: dict[str, ProviderAccuracy] | None = None,
) -> AggregatedResult:
    """Compute a robust aggregated tmin/tmax from provider results.

    Steps:
      1. Filter out results with None/NaN values.
      2. Apply per-provider bias correction (if accuracy data available).
      3. Compute quality × accuracy weights.
      4. MAD-filter outliers.
      5. Skewness-aware blend between weighted mean and median.
      6. If historical data available, apply climate anchoring.
      7. Compute confidence score.
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

    # Raw values (for display in range)
    raw_tmins = [r.tmin_c for r in valid]
    raw_tmaxs = [r.tmax_c for r in valid]

    # Bias-corrected values (for aggregation)
    tmins = list(raw_tmins)
    tmaxs = list(raw_tmaxs)
    bias_corrected = 0

    if provider_accuracy:
        for i, r in enumerate(valid):
            acc = provider_accuracy.get(r.provider_name)
            if acc and acc.n >= MIN_VERIFICATIONS_FOR_BIAS:
                tmins[i] -= acc.tmin_bias
                tmaxs[i] -= acc.tmax_bias
                bias_corrected += 1

    # Weights: quality × accuracy multiplier
    weights = [
        r.quality.weight * _accuracy_multiplier(r.provider_name, provider_accuracy)
        for r in valid
    ]

    # Filter outliers independently for tmin and tmax
    tmins_f, w_tmin = _filter_outliers(tmins, weights)
    tmaxs_f, w_tmax = _filter_outliers(tmaxs, weights)

    # Skewness-aware estimation
    skew_tmin = _skewness(tmins_f)
    skew_tmax = _skewness(tmaxs_f)

    agg_tmin = _skewness_blend(tmins_f, w_tmin)
    agg_tmax = _skewness_blend(tmaxs_f, w_tmax)

    # Historical anchoring
    hist_tmin_mean = None
    hist_tmax_mean = None
    hist_tmin_std = None
    hist_tmax_std = None
    hist_sample_size = None

    if historical is not None:
        hist_tmin_mean = historical.tmin_mean
        hist_tmax_mean = historical.tmax_mean
        hist_tmin_std = historical.tmin_std
        hist_tmax_std = historical.tmax_std
        hist_sample_size = historical.sample_size
        agg_tmin = _climate_anchoring(
            agg_tmin, tmins_f, historical.tmin_mean, historical.tmin_std
        )
        agg_tmax = _climate_anchoring(
            agg_tmax, tmaxs_f, historical.tmax_mean, historical.tmax_std
        )

    # Confidence score (take the lower of tmin/tmax)
    has_accuracy = provider_accuracy is not None and bias_corrected > 0
    conf_tmin = _confidence_score(
        tmins_f, hist_tmin_mean, hist_tmin_std, agg_tmin, has_accuracy
    )
    conf_tmax = _confidence_score(
        tmaxs_f, hist_tmax_mean, hist_tmax_std, agg_tmax, has_accuracy
    )
    confidence = min(conf_tmin, conf_tmax)

    return AggregatedResult(
        tmin_c=round(agg_tmin, 1),
        tmax_c=round(agg_tmax, 1),
        tmin_range=(round(min(raw_tmins), 1), round(max(raw_tmins), 1)),
        tmax_range=(round(min(raw_tmaxs), 1), round(max(raw_tmaxs), 1)),
        sources_used=len(valid),
        warning=warning,
        skewness_tmin=round(skew_tmin, 2),
        skewness_tmax=round(skew_tmax, 2),
        confidence=confidence,
        hist_tmin_mean=hist_tmin_mean,
        hist_tmax_mean=hist_tmax_mean,
        hist_tmin_std=hist_tmin_std,
        hist_tmax_std=hist_tmax_std,
        hist_sample_size=hist_sample_size,
    )
