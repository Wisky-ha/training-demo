"""FastAPI application entrypoint.

Run locally from the repository root with:

    python -m uvicorn --app-dir docs backend.app.main:app --reload

Alternatively, run ``python -m uvicorn backend.app.main:app --reload``
from the ``docs`` directory.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import Settings, get_settings
from .db.session import (
    create_database_engine,
    create_session_factory,
    get_session,
    initialize_database,
)
from .datasets.router import router as dataset_router
from .preprocessing.router import router as preprocessing_router
from .scripts.router import router as script_router
from .training_jobs.router import router as training_job_router
from .model_router import alerts_router, router as model_router
from .mcp_router import router as mcp_router
from .services.model_baseline import ModelBaselineService


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application with explicit settings when needed in tests."""

    active_settings = settings or get_settings()
    # Keep the engine and session factory on the app instance so a test or a
    # deployment with explicit settings never accidentally writes to the
    # process-wide default database.
    engine = create_database_engine(active_settings)
    session_factory = create_session_factory(engine)

    training_job_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="training-job")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        active_settings.ensure_storage_directories()
        initialize_database(engine=engine)
        with session_factory() as session:
            ModelBaselineService(session).initialize_baselines()
        try:
            yield
        finally:
            training_job_executor.shutdown(wait=True, cancel_futures=True)
            engine.dispose()

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
    application.state.settings = active_settings
    application.state.session_factory = session_factory
    application.state.training_job_executor = training_job_executor

    def app_session():
        session = session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Routes depend on the standard database dependency, while this override
    # makes create_app(Settings(...)) isolated and keeps existing test/deployment
    # overrides working as expected.
    application.dependency_overrides[get_session] = app_session
    application.include_router(dataset_router)
    application.include_router(script_router)
    application.include_router(preprocessing_router)
    application.include_router(training_job_router)
    application.include_router(model_router)
    application.include_router(alerts_router)
    application.include_router(mcp_router)

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
