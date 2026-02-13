"""Tests for provider response parsing (mocked HTTP)."""

from unittest.mock import MagicMock, patch

import pytest

from models import DataQuality


class TestOpenMeteo:
    """Test Open-Meteo provider with mocked responses."""

    SAMPLE_RESPONSE = {
        "daily": {
            "time": ["2025-06-15"],
            "temperature_2m_max": [24.3],
            "temperature_2m_min": [12.1],
        }
    }

    @patch("providers.open_meteo.httpx.Client")
    def test_parse_response(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.SAMPLE_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        from providers.open_meteo import fetch

        result = fetch(48.85, 2.35, "2025-06-15", "Europe/Paris")

        assert result.provider_name == "Open-Meteo"
        assert result.tmin_c == 12.1
        assert result.tmax_c == 24.3
        assert result.quality == DataQuality.DAILY_DIRECT


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
