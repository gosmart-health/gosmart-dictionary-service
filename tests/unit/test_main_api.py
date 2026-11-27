"""Tests for the FastAPI routes, middleware, and lifespan in ``gosmart_dictionary_service.main``."""

import json
import logging
import tempfile
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from gosmart_dictionary_service import __version__
from gosmart_dictionary_service.core.config import get_settings
from gosmart_dictionary_service.db.schema_loader import SchemaInitializationError
from gosmart_dictionary_service.main import app, create_app
from tests.conftest import make_test_settings


def _cleanup_file_handlers() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, TimedRotatingFileHandler):
            handler.close()
            root.removeHandler(handler)


def test_root_returns_service_metadata(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "GoSmart DS"
    assert payload["version"] == __version__
    assert payload["docs_url"] == "/docs"
    assert payload["api_v1"] == "/api/v1"
    assert "db_connection" in payload


def test_health_endpoint_healthy(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "HEALTHY"
    assert payload["version"] == __version__
    assert payload["services"]["neo4j"]["status"] == "HEALTHY"


def test_health_endpoint_degraded_when_neo4j_offline(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "gosmart_dictionary_service.db.neo4j.Neo4jClient.verify_connectivity",
        AsyncMock(return_value=False),
    )
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "DEGRADED"
    assert payload["services"]["neo4j"]["status"] == "UNHEALTHY"


def test_request_id_middleware_echoes_and_generates_header(client: TestClient) -> None:
    res_generated = client.get("/")
    assert res_generated.status_code == 200
    assert res_generated.headers.get("X-Request-ID")

    res_propagated = client.get("/", headers={"X-Request-ID": "trace-ds-001"})
    assert res_propagated.status_code == 200
    assert res_propagated.headers.get("X-Request-ID") == "trace-ds-001"


def test_structlog_middleware_writes_request_events_to_log_file(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "gosmart_ds_middleware.log"
        settings = make_test_settings(
            DEV_MODE=True,
            LOG_DIR=tmpdir,
            LOG_FILE="gosmart_ds_middleware.log",
            LOG_TO_FILE=True,
            LOG_FORMAT="json",
        )
        get_settings.cache_clear()
        monkeypatch.setattr("gosmart_dictionary_service.core.config.get_settings", lambda: settings)
        monkeypatch.setattr("gosmart_dictionary_service.core.logging.get_settings", lambda: settings)
        monkeypatch.setattr("gosmart_dictionary_service.main.get_settings", lambda: settings)
        monkeypatch.setattr("gosmart_dictionary_service.db.neo4j.Neo4jClient.connect", AsyncMock())
        monkeypatch.setattr(
            "gosmart_dictionary_service.db.neo4j.Neo4jClient.verify_connectivity", AsyncMock(return_value=True)
        )
        monkeypatch.setattr("gosmart_dictionary_service.db.neo4j.Neo4jClient.close", AsyncMock())
        monkeypatch.setattr("gosmart_dictionary_service.main.apply_schema_async", AsyncMock(return_value=0))
        monkeypatch.setattr(
            "gosmart_dictionary_service.main.system_service.bootstrap_system",
            AsyncMock(return_value={"success": True}),
        )

        test_app = create_app()
        try:
            with TestClient(test_app) as test_client:
                resp = test_client.get("/", headers={"X-Request-ID": "REQ-MW-42"})
                assert resp.status_code == 200

            for handler in logging.getLogger().handlers:
                handler.flush()

            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            events = {rec["event"]: rec for rec in records}
            assert "server_starting" in events
            assert "http_request_received" in events
            assert "http_request_finished" in events
            assert "server_shutdown_complete" in events

            finished = events["http_request_finished"]
            assert finished["request_id"] == "REQ-MW-42"
            assert finished["http_method"] == "GET"
            assert finished["path"] == "/"
            assert finished["status_code"] == 200
            assert isinstance(finished["duration_ms"], (int, float))
        finally:
            _cleanup_file_handlers()
            get_settings.cache_clear()


def test_lifespan_applies_schema_and_bootstraps_system_when_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_test_settings(DEV_MODE=True, LOG_TO_FILE=False)
    get_settings.cache_clear()
    apply_schema_mock = AsyncMock(return_value=12)
    bootstrap_mock = AsyncMock(return_value={"success": True})

    monkeypatch.setattr("gosmart_dictionary_service.core.config.get_settings", lambda: settings)
    monkeypatch.setattr("gosmart_dictionary_service.core.logging.get_settings", lambda: settings)
    monkeypatch.setattr("gosmart_dictionary_service.main.get_settings", lambda: settings)
    monkeypatch.setattr("gosmart_dictionary_service.db.neo4j.Neo4jClient.connect", AsyncMock())
    monkeypatch.setattr(
        "gosmart_dictionary_service.db.neo4j.Neo4jClient.verify_connectivity", AsyncMock(return_value=True)
    )
    monkeypatch.setattr("gosmart_dictionary_service.db.neo4j.Neo4jClient.close", AsyncMock())
    monkeypatch.setattr("gosmart_dictionary_service.main.apply_schema_async", apply_schema_mock)
    monkeypatch.setattr("gosmart_dictionary_service.main.system_service.bootstrap_system", bootstrap_mock)

    test_app = create_app()
    try:
        with TestClient(test_app):
            pass
        apply_schema_mock.assert_awaited_once()
        bootstrap_mock.assert_awaited_once_with(
            acted_by="startup",
            settings=settings,
            ensure_schema=False,
            seed_default_llm_configs=True,
        )
    finally:
        get_settings.cache_clear()


def test_lifespan_fails_closed_on_schema_initialization_error(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_test_settings(DEV_MODE=True, LOG_TO_FILE=False)
    get_settings.cache_clear()

    monkeypatch.setattr("gosmart_dictionary_service.core.config.get_settings", lambda: settings)
    monkeypatch.setattr("gosmart_dictionary_service.core.logging.get_settings", lambda: settings)
    monkeypatch.setattr("gosmart_dictionary_service.main.get_settings", lambda: settings)
    monkeypatch.setattr("gosmart_dictionary_service.db.neo4j.Neo4jClient.connect", AsyncMock())
    monkeypatch.setattr(
        "gosmart_dictionary_service.db.neo4j.Neo4jClient.verify_connectivity", AsyncMock(return_value=True)
    )
    monkeypatch.setattr("gosmart_dictionary_service.db.neo4j.Neo4jClient.close", AsyncMock())
    monkeypatch.setattr(
        "gosmart_dictionary_service.main.apply_schema_async",
        AsyncMock(side_effect=SchemaInitializationError("DDL failure", script_file="dicom_schema.cypher")),
    )

    test_app = create_app()
    try:
        with pytest.raises(SchemaInitializationError):
            with TestClient(test_app):
                pass
    finally:
        get_settings.cache_clear()


def test_lifespan_fails_when_neo4j_offline_in_production_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_test_settings(DEV_MODE=False, LOG_TO_FILE=False)
    get_settings.cache_clear()
    monkeypatch.setattr("gosmart_dictionary_service.core.config.get_settings", lambda: settings)
    monkeypatch.setattr("gosmart_dictionary_service.core.logging.get_settings", lambda: settings)
    monkeypatch.setattr("gosmart_dictionary_service.main.get_settings", lambda: settings)
    monkeypatch.setattr("gosmart_dictionary_service.db.neo4j.Neo4jClient.connect", AsyncMock())
    monkeypatch.setattr(
        "gosmart_dictionary_service.db.neo4j.Neo4jClient.verify_connectivity", AsyncMock(return_value=False)
    )
    monkeypatch.setattr("gosmart_dictionary_service.db.neo4j.Neo4jClient.close", AsyncMock())

    test_app = create_app()
    try:
        with pytest.raises(ConnectionError):
            with TestClient(test_app):
                pass
    finally:
        get_settings.cache_clear()


def test_openapi_metadata_matches_package_version(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    payload = response.json()
    assert payload["info"]["title"] == "GoSmart DS"
    assert payload["info"]["version"] == __version__


def test_documentation_routes_are_enabled(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert app.openapi_url == "/openapi.json"
