"""Tests for provider response parsing (mocked HTTP)."""

from unittest.mock import MagicMock, patch

import pytest

from models import DataQuality


class TestOpenMeteo:
    """Test Open-Meteo multi-model provider with mocked responses."""

    SAMPLE_RESPONSE = {
        "daily": {
            "time": ["2025-06-15"],
            # best_match (no suffix)
            "temperature_2m_max": [24.3],
            "temperature_2m_min": [12.1],
            # ecmwf model
            "temperature_2m_max_ecmwf_ifs025": [23.8],
            "temperature_2m_min_ecmwf_ifs025": [11.5],
            # gfs model
            "temperature_2m_max_gfs_seamless": [25.0],
            "temperature_2m_min_gfs_seamless": [12.8],
        }
    }

    @patch("providers.open_meteo.httpx.Client")
    def test_parse_multi_model(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.SAMPLE_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        from providers.open_meteo import fetch

        results = fetch(48.85, 2.35, "2025-06-15", "Europe/Paris")

        assert isinstance(results, list)
        assert len(results) >= 3  # best_match + ecmwf + gfs at minimum

        names = {r.provider_name for r in results}
        assert "Open-Meteo (Best Match)" in names
        assert "ECMWF IFS 0.25°" in names
        assert "NOAA GFS" in names

        best = next(r for r in results if r.provider_name == "Open-Meteo (Best Match)")
        assert best.tmin_c == 12.1
        assert best.tmax_c == 24.3
        assert best.quality == DataQuality.DAILY_DIRECT

    @patch("providers.open_meteo.httpx.Client")
    def test_skips_models_without_data(self, mock_client_cls):
        """Models not present in response should be silently skipped."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "daily": {
                "time": ["2025-06-15"],
                "temperature_2m_max": [24.3],
                "temperature_2m_min": [12.1],
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        from providers.open_meteo import fetch

        results = fetch(48.85, 2.35, "2025-06-15", "Europe/Paris")

        # Only best_match has data
        assert len(results) == 1
        assert results[0].provider_name == "Open-Meteo (Best Match)"


class TestWeatherAPI:
    """Test WeatherAPI provider with mocked responses."""

    SAMPLE_RESPONSE = {
        "forecast": {
            "forecastday": [
                {
                    "date": "2025-06-15",
                    "day": {
                        "maxtemp_c": 25.0,
                        "mintemp_c": 13.5,
                    },
                }
            ]
        }
    }

    @patch.dict("os.environ", {"WEATHERAPI_KEY": "test-key"})
    @patch("providers.weatherapi.httpx.Client")
    def test_parse_response(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.SAMPLE_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        from providers.weatherapi import fetch

        result = fetch(48.85, 2.35, "2025-06-15", "Europe/Paris")

        assert result.provider_name == "WeatherAPI"
        assert result.tmin_c == 13.5
        assert result.tmax_c == 25.0

    def test_missing_key_raises(self):
        from providers.weatherapi import fetch

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="WEATHERAPI_KEY"):
                fetch(48.85, 2.35, "2025-06-15", "Europe/Paris")


class TestOpenWeather:
    """Test OpenWeatherMap provider with mocked responses."""

    SAMPLE_RESPONSE = {
        "list": [
            # 2025-06-15 entries (UTC timestamps mapped to Europe/Paris)
            {"dt": 1750003200, "main": {"temp": 14.0}},  # 2025-06-15 12:00 UTC
            {"dt": 1750014000, "main": {"temp": 22.5}},  # 2025-06-15 15:00 UTC
            {"dt": 1750024800, "main": {"temp": 18.0}},  # 2025-06-15 18:00 UTC
            {"dt": 1750035600, "main": {"temp": 13.0}},  # 2025-06-15 21:00 UTC
            # Next day entry (should be excluded)
            {"dt": 1750089600, "main": {"temp": 30.0}},  # 2025-06-16 12:00 UTC
        ]
    }

    @patch.dict("os.environ", {"OPENWEATHER_KEY": "test-key"})
    @patch("providers.openweather.httpx.Client")
    def test_parse_response(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.SAMPLE_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        from providers.openweather import fetch

        result = fetch(48.85, 2.35, "2025-06-15", "Europe/Paris")

        assert result.provider_name == "OpenWeatherMap"
        assert result.quality == DataQuality.COMPUTED_FROM_HOURLY
        # Should only include entries for 2025-06-15 in Europe/Paris
        assert result.tmin_c <= 14.0
        assert result.tmax_c >= 22.5

    def test_missing_key_raises(self):
        from providers.openweather import fetch

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENWEATHER_KEY"):
                fetch(48.85, 2.35, "2025-06-15", "Europe/Paris")
