"""Runtime configuration from environment variables (prefix ``BV_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BV_", env_file=(REPO_ROOT / ".env", ".env"), extra="ignore")

    data_dir: Path = Field(default=REPO_ROOT / "data", description="uploads, artifacts, cache and the SQLite database")
    models_dir: Path = Field(default=REPO_ROOT / "models" / "weights")
    database_url: str | None = Field(default=None, description="SQLAlchemy URL; defaults to SQLite in data_dir")

    max_upload_mb: int = 1024
    max_duration_s: float = 1800.0
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")

    max_concurrent_jobs: int = 1
    max_queued_jobs: int = 4

    api_key: str | None = Field(default=None, description="if set, every /api request needs header X-API-Key")
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    retention_days: int = 0  # 0 = keep sessions until deleted by the user
    enable_demo: bool = True
    log_level: str = "INFO"

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'badmintonvision.db').as_posix()}"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def stroke_models_dir(self) -> Path:
        return self.data_dir / "models" / "strokes"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.uploads_dir,
            self.sessions_dir,
            self.cache_dir,
            self.tmp_dir,
            self.stroke_models_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
