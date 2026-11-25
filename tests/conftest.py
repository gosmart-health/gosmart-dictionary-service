"""Shared fixtures for the GoSmart DS test suite."""

import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from logging.handlers import TimedRotatingFileHandler
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gosmart_dictionary_service.core.config import Settings, get_settings
from gosmart_dictionary_service.main import app as main_app
from gosmart_dictionary_service.main import create_app


def make_test_settings(**overrides: Any) -> Settings:
    """Construct isolated Settings without loading .env files."""
    kwargs: dict[str, Any] = {"_env_file": None, **overrides}
    return Settings(**kwargs)


def _close_rotating_file_handlers() -> None:
    """Close and detach any TimedRotatingFileHandler left on the root logger."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, TimedRotatingFileHandler):
            handler.close()
            root.removeHandler(handler)


@pytest.fixture
def isolated_app_lifespan() -> Iterator[None]:
    """Disable production startup side effects for mocked unit API tests."""

    @asynccontextmanager
    async def no_lifespan(_app: FastAPI):
        yield

    original_lifespan = main_app.router.lifespan_context
    main_app.router.lifespan_context = no_lifespan
    try:
        yield
    finally:
        main_app.router.lifespan_context = original_lifespan


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """An in-process HTTP client with mocked Neo4j and file logging disabled."""
    test_settings = make_test_settings(
        DEV_MODE=True,
        LOG_TO_FILE=False,
        LOG_FORMAT="json",
    )
    get_settings.cache_clear()
    monkeypatch.setattr("gosmart_dictionary_service.core.config.get_settings", lambda: test_settings)
    monkeypatch.setattr("gosmart_dictionary_service.core.logging.get_settings", lambda: test_settings)
    monkeypatch.setattr("gosmart_dictionary_service.main.get_settings", lambda: test_settings)
    monkeypatch.setattr("gosmart_dictionary_service.db.neo4j.Neo4jClient.connect", AsyncMock())
    monkeypatch.setattr(
        "gosmart_dictionary_service.db.neo4j.Neo4jClient.verify_connectivity", AsyncMock(return_value=True)
    )
    monkeypatch.setattr("gosmart_dictionary_service.db.neo4j.Neo4jClient.close", AsyncMock())

    fastapi_app = create_app()
    fastapi_app.dependency_overrides[get_settings] = lambda: test_settings
    try:
        with TestClient(fastapi_app) as test_client:
            yield test_client
    finally:
        fastapi_app.dependency_overrides.clear()
        _close_rotating_file_handlers()
        get_settings.cache_clear()
