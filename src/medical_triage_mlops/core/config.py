from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Medical Triage API"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    model_path: str = "models/model.joblib"
    onnx_model_path: str = "models/model.onnx"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
