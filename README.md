# meteo-avg

**Intelligent multi-source weather forecast aggregation with self-learning accuracy.**

meteo-avg combines temperature forecasts from 13+ weather models (including AI-based models), applies robust statistical aggregation with skewness correction, and learns from its own prediction errors to improve over time.

## Features

- **13 NWP models** — ECMWF IFS, ECMWF AIFS (AI), NOAA GFS, GFS GraphCast (AI), DWD ICON, Meteo-France, Canadian GEM, JMA, UK Met Office, MET Norway, KNMI, DMI, + optional WeatherAPI & OpenWeatherMap
- **Skewness-aware aggregation** — blends weighted mean and median based on Fisher-Pearson skewness coefficient; asymmetric distributions lean toward the more robust median
- **Historical climatology** — fetches 5-year temperature normals from Open-Meteo Archive API for Bayesian anchoring and anomaly detection
- **Self-learning** — stores past predictions and verifies them against observed temperatures; builds a per-location accuracy profile that improves with every run
- **Per-provider bias correction** — subtracts systematic forecast errors (e.g. if ECMWF consistently predicts 0.5°C too cold in Paris, it corrects for that)
- **Accuracy-based weighting** — providers with lower MAE get more influence on the final result (0.7x–1.3x multiplier)
- **Confidence scoring** — 0–99% score based on source count, provider agreement, historical consistency, and self-learning data
- **Automatic fallback** — if extra models cause API errors, falls back to core models transparently
- **Smart caching** — 30-min forecast cache, 30-day historical cache, persistent accuracy database

## How it gets smarter over time

```
Run 1:  Fetches forecasts → saves prediction → displays result
Run 2:  Fetches forecasts → saves prediction → displays result
...
Run N (>5 days later):
        Verifies past predictions against actual observed temperatures
        → computes per-provider bias and MAE
        → applies bias correction to current forecast
        → weights accurate providers higher
        → displays improved result with accuracy stats
```

The verification system uses a rolling window of the last 30 forecasts per provider per location. Data is stored in `~/.cache/meteo_avg/`.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.12+. Only dependency is `httpx` (and `tzdata` for Windows).

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `WEATHERAPI_KEY` | No | API key from [weatherapi.com](https://www.weatherapi.com/) (free tier) |
| `OPENWEATHER_KEY` | No | API key from [openweathermap.org](https://openweathermap.org/api) (free tier) |

Open-Meteo (the primary source with 13 models) requires no API key. If optional keys are missing, those providers are simply skipped.

## Usage

```bash
# Basic – today's forecast
python meteo_avg.py "Paris"

# Specific date
python meteo_avg.py "London" --date 2026-02-20

# JSON output (for scripting / integration)
python meteo_avg.py "Tokyo" --json

# Fast mode (skip historical data fetch)
python meteo_avg.py "Berlin" --no-history
```

### Example output

```
============================================================================
  Location : Paris, Ile-de-France, France metropolitaine, France
  Date     : 2026-02-16
  Timezone : Europe/Paris
============================================================================

  Aggregated forecast:
    Tmin :   -1.2 °C /   29.8 °F   (range: -4.5 – 1.0 °C)
    Tmax :    8.3 °C /   46.9 °F   (range: 5.1 – 10.2 °C)
    Sources used: 13
    Confidence: 82%
    Skewness: Tmin -0.42 (low bias)  |  Tmax symmetric

  Historical context (5-year climatology ±5 days, n=55):
    Tmin avg:  -0.8 °C (σ 2.9°C)  |  Tmax avg:   7.5 °C (σ 2.4°C)
    Anomaly : Tmin -0.4°C vs normal  |  Tmax +0.8°C vs normal

  Self-learning (8 verified forecast(s) for this location):
    Best providers: ECMWF AIFS 0.25° (AI) (MAE 0.7°C), ECMWF IFS 0.25° (MAE 0.9°C)
    Bias corrections applied: 4 provider(s) adjusted

  Provider                       Tmin           Tmax    MAE Quality
  ---------------------------------------------------------------------------
  Open-Meteo (Best Match)    -1.0°C/30.2°F   8.1°C/46.6°F  0.8° daily_direct
  ECMWF IFS 0.25°            -1.2°C/29.8°F   8.5°C/47.3°F  0.9° daily_direct
  ECMWF AIFS 0.25° (AI)      -1.1°C/30.0°F   8.2°C/46.8°F  0.7° daily_direct
  NOAA GFS                   -2.1°C/28.2°F   9.0°C/48.2°F  1.2° daily_direct
  GFS GraphCast (AI)         -1.5°C/29.3°F   8.8°C/47.8°F    -  daily_direct
  DWD ICON                   -1.8°C/28.8°F   7.9°C/46.2°F  1.0° daily_direct
  ...
```

## Aggregation pipeline

The aggregation applies multiple layers of statistical intelligence:

```
Raw provider forecasts (13+ models)
  │
  ├─ 1. Filter None/NaN values
  ├─ 2. Bias correction (subtract systematic errors per provider)
  ├─ 3. Accuracy-based weighting (quality × accuracy multiplier)
  ├─ 4. MAD outlier filtering (remove values > 2.5×MAD from median)
  ├─ 5. Skewness-aware blending (mean ↔ median based on asymmetry)
  ├─ 6. Historical climate anchoring (Bayesian shrinkage for extremes)
  └─ 7. Confidence scoring
          │
          ▼
  Final aggregated Tmin / Tmax with confidence %
```

### Step details

| Step | Method | Purpose |
|------|--------|---------|
| **Outlier filtering** | Median Absolute Deviation (MAD), threshold 2.5× | Remove wildly incorrect models |
| **Skewness correction** | Fisher-Pearson coefficient, sigmoid blend | When providers disagree asymmetrically, lean toward median |
| **Climate anchoring** | Bayesian shrinkage, max 15% | Pull back extreme anomalies (>1.5σ from 5-year norm) |
| **Bias correction** | Per-provider mean error subtraction | Remove systematic warm/cold biases |
| **Accuracy weighting** | Inverse MAE, range 0.7x–1.3x | Give reliable providers more influence |
| **Confidence** | Multi-factor (sources, IQR, history, learning) | Quantify forecast reliability |

## Architecture

```
meteo_avg.py              CLI entrypoint & orchestration
aggregate.py              Statistical aggregation pipeline
history.py                Historical climatology (Open-Meteo Archive API)
verification.py           Self-learning forecast verification system
geocode.py                City → coordinates (Nominatim + Open-Meteo fallback)
cache.py                  Local JSON cache (~/.cache/meteo_avg/)
models.py                 Data models (dataclasses)
providers/
  open_meteo.py           Open-Meteo multi-model (13 NWP models, no key)
  weatherapi.py           WeatherAPI.com (optional, key required)
  openweather.py          OpenWeatherMap (optional, key required)
tests/
  test_aggregate.py       47 tests: aggregation, skewness, bias, confidence
  test_providers.py       Provider response parsing (mocked HTTP)
```

### Cache structure

```
~/.cache/meteo_avg/
├── {city}_{date}.json          Forecast cache (TTL: 30 min)
├── history/
│   └── hist_{lat}_{lon}_{mm}_{dd}.json   Climate normals (TTL: 30 days)
├── forecasts/
│   └── {lat}_{lon}_{date}.json           Past predictions (for verification)
└── accuracy/
    └── {lat}_{lon}.json                  Provider accuracy stats (persistent)
```

## Weather models

### Core models (via Open-Meteo, no API key)

| Model | Institution | Type |
|-------|------------|------|
| ECMWF IFS 0.25° | European Centre (EU) | Traditional NWP |
| ECMWF AIFS 0.25° | European Centre (EU) | AI-based |
| NOAA GFS | NOAA (US) | Traditional NWP |
| GFS GraphCast | Google / NOAA | AI-based |
| DWD ICON | Deutscher Wetterdienst (DE) | Traditional NWP |
| Meteo-France | Meteo-France (FR) | Traditional NWP |
| Canadian GEM | CMC (CA) | Traditional NWP |
| JMA | Japan Met Agency (JP) | Traditional NWP |
| UK Met Office | UKMO (GB) | Traditional NWP |
| MET Norway | MET Norway (NO) | Traditional NWP |
| KNMI | KNMI (NL) | Regional NWP |
| DMI | DMI (DK) | Regional NWP |

### Optional providers (API key required)

| Provider | Endpoint | Quality |
|----------|----------|---------|
| WeatherAPI.com | Daily forecast (3 days) | daily_direct (weight 1.0) |
| OpenWeatherMap | 5-day/3h forecast | computed_from_hourly (weight 0.9) |

## Tests

```bash
# Run all 47 tests
pytest tests/ -v
```

## Limitations

- **Forecast range**: depends on each model (typically 3–16 days ahead)
- **Historical data lag**: Open-Meteo Archive API has ~5 day delay, so self-learning verification starts after 5+ days
- **Nominatim rate limits**: 1 request/second; the cache mitigates repeated calls
- **OpenWeatherMap free tier**: 5-day/3h only, no direct daily min/max
- **WeatherAPI free tier**: limited to 3 days ahead
- **Self-learning cold start**: the first few runs have no accuracy data; the system improves after 3+ verified forecasts per location

## License

MIT
