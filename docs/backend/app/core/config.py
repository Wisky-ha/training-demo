"""Central application settings.

Settings are read from environment variables and an optional ``.env`` file.
Environment variables use the ``APP_`` prefix by default; ``DATABASE_URL`` is
also accepted for convenience when running the backend locally.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve the checked-in configuration locations from this module rather than
# from the process working directory.  The working-directory ``.env`` entry is
# retained below so local overrides still work when the package is installed.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_PROJECT_DIR = _BACKEND_DIR.parent


class Settings(BaseSettings):
    """Runtime configuration shared by the API and infrastructure modules."""

    app_name: str = Field(
        default="模型训练可视化平台",
        validation_alias=AliasChoices("APP_NAME", "APP_APP_NAME"),
    )
    environment: Literal["development", "testing", "production"] = Field(
        default="development",
        validation_alias=AliasChoices("APP_ENVIRONMENT", "APP_APP_ENVIRONMENT"),
    )
    debug: bool = Field(
        default=False,
        validation_alias=AliasChoices("APP_DEBUG", "APP_APP_DEBUG"),
    )
    database_url: str = Field(
        default="sqlite:///./data/app.db",
        validation_alias=AliasChoices("APP_DATABASE_URL", "DATABASE_URL"),
    )
    model_storage_dir: Path = Field(
        default=Path("data/models"),
        validation_alias=AliasChoices("APP_MODEL_STORAGE_DIR", "APP_APP_MODEL_STORAGE_DIR"),
    )
    upload_storage_dir: Path = Field(
        default=Path("data/uploads"),
        validation_alias=AliasChoices("APP_UPLOAD_STORAGE_DIR", "APP_APP_UPLOAD_STORAGE_DIR"),
    )
    # Scripts may use their own root; otherwise they share the artifact root
    # selected by ``storage_root`` (or the backwards-compatible model root).
    script_storage_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "APP_SCRIPT_STORAGE_DIR",
            "APP_APP_SCRIPT_STORAGE_DIR",
            "script_dir",
        ),
    )
    # A dedicated root can be configured for all persisted artifacts.  When it
    # is omitted, the existing model directory remains the compatible default.
    storage_root: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "APP_STORAGE_ROOT",
            "APP_FILE_STORAGE_ROOT",
            "APP_APP_STORAGE_ROOT",
            "APP_APP_FILE_STORAGE_ROOT",
            "file_storage_root",
            "file_storage_dir",
            "artifact_storage_dir",
        ),
    )
    max_script_size_bytes: int = Field(
        default=5 * 1024 * 1024,
        gt=0,
        validation_alias=AliasChoices(
            "APP_MAX_SCRIPT_SIZE_BYTES",
            "APP_APP_MAX_SCRIPT_SIZE_BYTES",
        ),
    )
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173"],
        validation_alias=AliasChoices("APP_ALLOWED_ORIGINS", "APP_APP_ALLOWED_ORIGINS"),
    )

    model_config = SettingsConfigDict(
        # backend/.env is the fallback, while a project-root or current
        # working-directory .env can provide the normal local override.
        env_file=(
            str(_BACKEND_DIR / ".env"),
            str(_PROJECT_DIR / ".env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    @property
    def database_path(self) -> Path | str:
        """Return the SQLite filename represented by ``database_url``.

        The foundation intentionally uses Python's built-in sqlite3 module,
        while retaining a URL-shaped setting so the storage implementation can
        be replaced later without changing application configuration.
        """

        if self.database_url == "sqlite:///:memory:":
            return ":memory:"
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("database_url must be a sqlite:/// URL")
        raw_path = self.database_url[len(prefix) :]
        if not raw_path:
            raise ValueError("database_url must include a SQLite database path")
        return Path(raw_path).expanduser()

    @property
    def file_storage_root(self) -> Path:
        """Return the root used by the artifact file storage service.

        ``model_storage_dir`` was part of the original configuration, so it is
        deliberately retained as the fallback for backwards compatibility.
        """

        return self.storage_root or self.model_storage_dir

    @property
    def script_file_storage_root(self) -> Path:
        """Return the root used for uploaded Python script source files."""

        return self.script_storage_dir or self.file_storage_root

    def ensure_storage_directories(self) -> None:
        """Create configured application storage directories when needed."""

        self.model_storage_dir.mkdir(parents=True, exist_ok=True)
        self.upload_storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_storage_root.mkdir(parents=True, exist_ok=True)
        self.script_file_storage_root.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the process-wide, cached settings instance."""

    return Settings()
