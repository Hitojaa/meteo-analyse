# meteo_avg – Multi-source weather forecast aggregator

CLI tool that predicts daily min/max temperatures for a city by aggregating data from multiple weather APIs with robust outlier detection.

## Installation

```bash
pip install -r requirements.txt
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `WEATHERAPI_KEY` | No | API key from [weatherapi.com](https://www.weatherapi.com/) (free tier) |
| `OPENWEATHER_KEY` | No | API key from [openweathermap.org](https://openweathermap.org/api) (free tier) |

Open-Meteo requires no API key. If optional keys are missing, those providers are skipped.

## Usage

```bash
# Basic – today's forecast for Paris
python meteo_avg.py "Paris"

# Specific date
python meteo_avg.py "Lyon" --date 2025-06-15

# JSON output
python meteo_avg.py "Marseille" --json
```

### Example console output

```
============================================================
  Location : Paris, Île-de-France, France métropolitaine, France
  Date     : 2025-06-15
  Timezone : Europe/Paris
============================================================

  Aggregated forecast (°C):
    Tmin :    12.5 °C   (range: 12.1 – 13.5)
    Tmax :    24.6 °C   (range: 24.3 – 25.0)
    Sources used: 3

  Provider             Tmin (°C)  Tmax (°C) Quality
  --------------------------------------------------------------
  Open-Meteo                12.1       24.3 daily_direct
  WeatherAPI                13.5       25.0 daily_direct
  OpenWeatherMap            12.0       24.5 computed_from_hourly
```

### Example JSON output

```json
{
  "location": {
    "name": "Paris",
    "display_name": "Paris, Île-de-France, France métropolitaine, France",
    "lat": 48.8534951,
    "lon": 2.3483915,
    "country": "France",
    "timezone": "Europe/Paris"
  },
  "date": "2025-06-15",
  "timezone": "Europe/Paris",
  "aggregated": {
    "tmin_c": 12.5,
    "tmax_c": 24.6,
    "tmin_range": [12.1, 13.5],
    "tmax_range": [24.3, 25.0],
    "sources_used": 3,
    "warning": null
  },
  "per_provider": [
    {
      "provider_name": "Open-Meteo",
      "tmin_c": 12.1,
      "tmax_c": 24.3,
      "date": "2025-06-15",
      "timezone": "Europe/Paris",
      "quality": "daily_direct"
    }
  ],
  "errors": []
}
```

## Tests

```bash
pytest tests/ -v
```

## Architecture

```
meteo_avg.py          # CLI entrypoint
geocode.py            # Nominatim geocoding
aggregate.py          # MAD-based robust aggregation
cache.py              # Local JSON cache (~/.cache/meteo_avg/, TTL 30 min)
models.py             # Data models (dataclasses)
providers/
  open_meteo.py       # Open-Meteo (no key required)
  weatherapi.py       # WeatherAPI.com (key via WEATHERAPI_KEY)
  openweather.py      # OpenWeatherMap (key via OPENWEATHER_KEY)
tests/
  test_aggregate.py   # Aggregation + outlier tests
  test_providers.py   # Provider response parsing (mocked)
```

## Aggregation method

1. Filter out `None`/`NaN` values
2. Compute median for tmin and tmax independently
3. Compute MAD (Median Absolute Deviation) and exclude outliers beyond 2.5 × MAD
4. Compute weighted mean of remaining values (daily direct = 1.0, computed from hourly = 0.9)
5. If fewer than 2 valid sources, return best estimate with a warning

## Limitations

- **API divergence**: different providers may use different weather models, leading to varying forecasts.
- **Nominatim rate limits**: 1 request/second, no bulk use. The cache helps avoid repeated calls.
- **OpenWeatherMap free tier**: 5-day/3h forecast only – no direct daily min/max; computed from 3-hour intervals.
- **WeatherAPI free tier**: forecast limited to 3 days ahead.
- **Date range**: forecasts are limited to each provider's available range (typically 3–7 days).
