"""FastAPI application entrypoint.

Run locally from the repository root with:

    uvicorn backend.app.main:app --reload
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import Settings, get_settings
from .db.session import initialize_database


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application with explicit settings when needed in tests."""

    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        active_settings.ensure_storage_directories()
        initialize_database(settings=active_settings)
        yield

    application = FastAPI(
        title=active_settings.app_name,
        version="0.1.0",
        debug=active_settings.debug,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health", tags=["system"])
    @application.get("/api/health", include_in_schema=False)
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "environment": active_settings.environment,
            "app_name": active_settings.app_name,
        }

    return application


app = create_app()
