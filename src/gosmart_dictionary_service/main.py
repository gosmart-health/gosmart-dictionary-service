"""FastAPI application entry point for GoSmart DS (Dictionary Service)."""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from gosmart_dictionary_service.api.v1.router import api_v1_router
from gosmart_dictionary_service.core.config import get_settings
from gosmart_dictionary_service.core.logging import bind_log_context, clear_log_context, get_logger, setup_logging
from gosmart_dictionary_service.db.neo4j import neo4j_client
from gosmart_dictionary_service.db.schema_loader import SchemaInitializationError

logger = get_logger("gosmart_ds.server")


def logo_print() -> None:
    logo = """
 ██████╗  ██████╗ ███████╗███╗   ███╗ █████╗ ██████╗ ████████╗
██╔════╝ ██╔═══██╗██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝
██║  ███╗██║   ██║███████╗██╔████╔██║███████║██████╔╝   ██║
██║   ██║██║   ██║╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║
╚██████╔╝╚██████╔╝███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║
 ╚═════╝  ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝

██████╗ ███████╗
██╔══██╗██╔════╝
██║  ██║███████╗
██║  ██║╚════██║
██████╔╝███████║
╚═════╝ ╚══════╝

Copyright (c) 2026, GoSmart Health, Apache 2.0 License
https://gosmart.health
"""
    print(logo)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan for managing structured logging, DB drivers, and background services."""
    logo_print()
    setup_logging()
    settings = get_settings()
    logger.info(
        "server_starting",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        dev_mode=settings.DEV_MODE,
        log_level=settings.LOG_LEVEL,
        log_format=settings.LOG_FORMAT,
        log_rotation_days=settings.LOG_ROTATION_DAYS,
    )
    logger.info("db_configured", db_connection=settings.DB_CONNECTION_URI, db_database=settings.DB_DATABASE)

    # Initialize Neo4j Driver & verify connectivity
    try:
        await neo4j_client.connect()
        connected = await neo4j_client.verify_connectivity()
        if connected:
            logger.info(
                "neo4j_connected",
                db_connection=settings.DB_CONNECTION_URI,
                db_database=settings.DB_DATABASE,
            )
        elif not settings.DEV_MODE:
            raise ConnectionError(f"Unable to verify Neo4j connectivity at {settings.DB_CONNECTION_URI}")
        else:
            logger.warning(
                "neo4j_init_notice",
                db_connection=settings.DB_CONNECTION_URI,
                dev_mode=True,
            )
    except SchemaInitializationError as e:
        logger.critical("schema_initialization_fatal_error", error=str(e), exc_info=True)
        raise
    except Exception as e:
        if not settings.DEV_MODE:
            logger.critical("neo4j_init_fatal_error", error=str(e), exc_info=True)
            raise
        logger.warning("neo4j_init_notice", error=str(e), dev_mode=True)

    # NOTE: Background Pykka actors (AuditActor, IngestionActor, EmbeddingActor),
    # schema initialization, admin bootstrap, and MCP session manager will be wired here
    # as their respective modules are migrated.
    yield

    # Clean shutdown
    logger.info("server_shutting_down", app_name=settings.APP_NAME)
    await neo4j_client.close()
    logger.info("server_shutdown_complete")


def create_app() -> FastAPI:
    """FastAPI Application factory."""
    settings = get_settings()

    fastapi_app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "FastAPI Backend for GoSmart DS — Clinical Terminology Dictionary and "
            "Ontology Graph Service with Neo4j and Genkit"
        ),
        lifespan=lifespan,
    )

    # Structured Request Tracing Middleware
    @fastapi_app.middleware("http")
    async def structlog_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        client = request.client
        bind_log_context(
            request_id=request_id,
            http_method=request.method,
            path=request.url.path,
            client_ip=client.host if client is not None else "unknown",
        )
        start_time = time.perf_counter()
        logger.info("http_request_received")

        try:
            response: Response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info("http_request_finished", status_code=response.status_code, duration_ms=duration_ms)
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error("http_request_failed", error=str(exc), duration_ms=duration_ms, exc_info=True)
            raise
        finally:
            clear_log_context()

    # CORS
    # noinspection PyTypeChecker
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    fastapi_app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    @fastapi_app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs_url": "/docs",
            "api_v1": settings.API_V1_PREFIX,
            "db_connection": settings.DB_CONNECTION_URI,
        }

    return fastapi_app


app = create_app()


def run() -> None:
    """Entrypoint to run the server directly."""
    settings = get_settings()
    pkg_dir = str(Path(__file__).resolve().parent)
    src_dir = str(Path(__file__).resolve().parent.parent)
    uvicorn.run(
        "gosmart_dictionary_service.main:app",
        app_dir=src_dir,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        reload_dirs=[pkg_dir] if settings.RELOAD else None,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    run()
