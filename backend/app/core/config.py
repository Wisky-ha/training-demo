"""Central application configuration.

All runtime settings are defined here so API, persistence, and future services
can use the same environment-backed configuration object.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-backed settings for the backend service.

    Environment variables use the ``MODEL_PLATFORM_`` prefix, for example
    ``MODEL_PLATFORM_DATABASE_URL``.  A local ``.env`` file is optional.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="MODEL_PLATFORM_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "模型训练可视化平台"
    app_version: str = "0.1.0"
    environment: Literal["development", "testing", "production"] = "development"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    # An absolute default keeps the SQLite file location independent of the
    # directory from which uvicorn is started. It can be replaced with any
    # SQLAlchemy-compatible database URL through the environment.
    database_url: str = Field(
        default_factory=lambda: (
            f"sqlite:///{(PROJECT_ROOT / 'data' / 'platform.db').as_posix()}"
        )
    )
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    model_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "models")
    script_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "scripts")
    upload_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "uploads")

    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    def ensure_directories(self) -> None:
        """Create configured storage directories when the service starts."""

        for directory in (
            self.data_dir,
            self.model_dir,
            self.script_dir,
            self.upload_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""

    return Settings()


# Convenient import for modules that do not need to override configuration.
settings = get_settings()
