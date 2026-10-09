"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parents[3] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: str = Field(default="development", description="Runtime environment")
    debug: bool = Field(default=False, description="Enable debug mode")

    # FPL API
    fpl_base_url: str = Field(
        default="https://fantasy.premierleague.com/api",
        description="Base URL for official FPL API",
    )
    fpl_timeout_seconds: float = Field(
        default=30.0,
        description="HTTP timeout for FPL API requests",
        gt=0,
    )
    fpl_max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for failed requests",
        ge=0,
    )
    fpl_retry_backoff_factor: float = Field(
        default=0.5,
        description="Backoff factor for retry delays",
        ge=0,
    )

    # Optional external providers
    understat_base_url: str | None = Field(
        default=None,
        description="Base URL for Understat API (optional)",
    )

    # Randomness for reproducibility
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducible simulations",
    )

    # Data storage
    data_dir: Path = Field(
        default=Path(__file__).parents[3] / "data",
        description="Base directory for data storage",
    )
    raw_data_dir: Path = Field(
        default=Path(__file__).parents[3] / "data/raw",
        description="Directory for raw data files",
    )
    processed_data_dir: Path = Field(
        default=Path(__file__).parents[3] / "data/processed",
        description="Directory for processed data files",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    log_format: str = Field(
        default="json",
        description="Log format: 'json' or 'console'",
    )

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Force reload of settings (clears cache)."""
    get_settings.cache_clear()
    return get_settings()
