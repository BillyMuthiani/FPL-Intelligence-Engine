"""Tests for configuration module."""

from pathlib import Path

import pytest

from fpl_engine.config import Settings, get_settings, reload_settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = Settings()
        assert settings.environment == "development"
        assert settings.debug is False
        assert settings.fpl_base_url == "https://fantasy.premierleague.com/api"
        assert settings.fpl_timeout_seconds == 30.0
        assert settings.fpl_max_retries == 3
        assert settings.random_seed == 42
        assert settings.log_level == "INFO"
        assert settings.log_format == "json"

    def test_environment_property(self) -> None:
        """Test environment property helpers."""
        dev_settings = Settings(environment="development")
        assert dev_settings.is_development is True
        assert dev_settings.is_production is False

        prod_settings = Settings(environment="production")
        assert prod_settings.is_production is True
        assert prod_settings.is_development is False

    def test_data_dirs_are_paths(self) -> None:
        """Test that data directories are Path objects."""
        settings = Settings()
        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.raw_data_dir, Path)
        assert isinstance(settings.processed_data_dir, Path)

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable overrides."""
        monkeypatch.setenv("FPL_BASE_URL", "https://custom.api.url")
        monkeypatch.setenv("ENVIRONMENT", "testing")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        # Need to reload to pick up env vars
        reload_settings()
        settings = get_settings()

        assert settings.fpl_base_url == "https://custom.api.url"
        assert settings.environment == "testing"
        assert settings.debug is True
        assert settings.log_level == "DEBUG"

    def test_timeout_validation(self) -> None:
        """Test timeout validation."""
        with pytest.raises(ValueError):
            Settings(fpl_timeout_seconds=0)

        with pytest.raises(ValueError):
            Settings(fpl_timeout_seconds=-1)

        # Valid
        settings = Settings(fpl_timeout_seconds=10.0)
        assert settings.fpl_timeout_seconds == 10.0

    def test_max_retries_validation(self) -> None:
        """Test max retries validation."""
        with pytest.raises(ValueError):
            Settings(fpl_max_retries=-1)

        settings = Settings(fpl_max_retries=5)
        assert settings.fpl_max_retries == 5

    def test_caching(self) -> None:
        """Test that get_settings returns cached instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reload_clears_cache(self) -> None:
        """Test that reload_settings clears cache."""
        s1 = get_settings()
        s2 = reload_settings()
        assert s1 is not s2


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_valid_log_levels(self) -> None:
        """Test valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level

    def test_invalid_log_level(self) -> None:
        """Test invalid log level - pydantic should coerce or validate."""
        # Pydantic will accept any string for log_level
        settings = Settings(log_level="INVALID")
        assert settings.log_level == "INVALID"

    def test_backoff_factor_validation(self) -> None:
        """Test retry backoff factor validation."""
        with pytest.raises(ValueError):
            Settings(fpl_retry_backoff_factor=-1)

        settings = Settings(fpl_retry_backoff_factor=1.0)
        assert settings.fpl_retry_backoff_factor == 1.0
