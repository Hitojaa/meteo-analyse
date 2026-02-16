"""Data models for meteo_avg."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DataQuality(str, Enum):
    DAILY_DIRECT = "daily_direct"
    COMPUTED_FROM_HOURLY = "computed_from_hourly"

    @property
    def weight(self) -> float:
        return {
            DataQuality.DAILY_DIRECT: 1.0,
            DataQuality.COMPUTED_FROM_HOURLY: 0.9,
        }[self]


@dataclass
class GeoLocation:
    name: str
    display_name: str
    lat: float
    lon: float
    country: str
    timezone: str


@dataclass
class ProviderResult:
    provider_name: str
    tmin_c: float | None
    tmax_c: float | None
    date: str  # YYYY-MM-DD
    timezone: str
    quality: DataQuality = DataQuality.DAILY_DIRECT


@dataclass
class ProviderError:
    provider_name: str
    error: str


@dataclass
class AggregatedResult:
    tmin_c: float
    tmax_c: float
    tmin_range: tuple[float, float]  # (lowest tmin, highest tmin)
    tmax_range: tuple[float, float]  # (lowest tmax, highest tmax)
    sources_used: int
    warning: str | None = None
    skewness_tmin: float = 0.0
    skewness_tmax: float = 0.0
    confidence: float = 0.0
    hist_tmin_mean: float | None = None
    hist_tmax_mean: float | None = None
    hist_tmin_std: float | None = None
    hist_tmax_std: float | None = None
    hist_sample_size: int | None = None


@dataclass
class ForecastReport:
    location: GeoLocation
    date: str
    timezone: str
    aggregated: AggregatedResult
    per_provider: list[ProviderResult] = field(default_factory=list)
    errors: list[ProviderError] = field(default_factory=list)
