"""Unit tests for GoSmart DS structured logging and trimmed configuration."""

import json
import logging
import tempfile
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pytest
import structlog

from gosmart_dictionary_service.core.logging import (
    Neo4jAlreadyExistsLogFilter,
    bind_log_context,
    clear_log_context,
    get_logger,
    setup_logging,
    unbind_log_context,
)
from tests.conftest import make_test_settings


def _cleanup_file_handlers() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, TimedRotatingFileHandler):
            handler.close()
            root.removeHandler(handler)


def test_settings_defaults_and_trimmed_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GoSmart DS defaults and confirm DICOM/FHIR/NLU/cloud fields are removed."""
    for key in (
        "GOSMART_DS_APP_NAME",
        "GOSMART_DS_PORT",
        "GOSMART_DS_LOG_FILE",
        "GOSMART_DS_DB_CONNECTION_URI",
        "NEO4J_URI",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = make_test_settings()
    assert settings.APP_NAME == "GoSmart DS"
    assert settings.PORT == 8081
    assert settings.LOG_FILE == "gosmart_ds.log"
    assert settings.API_KEY_PREFIX == "gs_dict_"
    assert settings.DB_CONNECTION_URI == "bolt://localhost:7688"

    for removed_field in (
        "DICOM_AE_TITLE",
        "DICOM_SCP_PORT",
        "DICOM_STAGING_DIR",
        "FHIR_BASE_URL",
        "NLU_MODEL_PATH",
        "STORAGE_ROOT_URI",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AWS_ACCESS_KEY_ID",
        "AZURE_STORAGE_CONNECTION_STRING",
    ):
        assert not hasattr(settings, removed_field)


def test_logging_setup_and_7_day_rotation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            settings = make_test_settings(
                LOG_DIR=tmpdir,
                LOG_FILE="gosmart_ds_test.log",
                LOG_ROTATION_DAYS=7,
                LOG_BACKUP_COUNT=8,
                LOG_TO_FILE=True,
                LOG_FORMAT="json",
            )
            setup_logging(settings)

            root = logging.getLogger()
            file_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
            assert len(file_handlers) == 1
            fh = file_handlers[0]

            assert fh.when == "D"
            assert fh.interval in (7, 7 * 86400)
            assert fh.backupCount == 8
            assert Path(fh.baseFilename).name == "gosmart_ds_test.log"
        finally:
            _cleanup_file_handlers()


def test_logging_console_format_configures_console_renderer() -> None:
    settings = make_test_settings(
        LOG_TO_FILE=False,
        LOG_FORMAT="console",
    )
    setup_logging(settings)

    root = logging.getLogger()
    assert len(root.handlers) == 1
    formatter = root.handlers[0].formatter
    assert isinstance(formatter, structlog.stdlib.ProcessorFormatter)
    assert any(isinstance(p, structlog.dev.ConsoleRenderer) for p in formatter.processors)


def test_structlog_json_values_and_context() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            log_path = Path(tmpdir) / "gosmart_ds_json.log"
            settings = make_test_settings(
                LOG_DIR=tmpdir,
                LOG_FILE="gosmart_ds_json.log",
                LOG_TO_FILE=True,
                LOG_FORMAT="json",
            )
            setup_logging(settings)

            logger = get_logger("gosmart_ds.test")
            bind_log_context(request_id="REQ-12345", caller="graphite", transient="drop-me")
            unbind_log_context("transient")

            metadata_payload = {
                "vocabulary": "SNOMED",
                "code": "59282003",
                "tags": ["pulmonary", "embolism"],
                "score": 0.98,
            }
            logger.info(
                "concept_resolution_event",
                status="SUCCESS",
                concept_metadata=metadata_payload,
                execution_time_ms=12.4,
            )

            clear_log_context()

            for handler in logging.getLogger().handlers:
                handler.flush()

            assert log_path.exists()
            lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert len(lines) >= 1

            last_log = json.loads(lines[-1])
            assert last_log["event"] == "concept_resolution_event"
            assert last_log["logger"] == "gosmart_ds.test"
            assert last_log["request_id"] == "REQ-12345"
            assert last_log["caller"] == "graphite"
            assert "transient" not in last_log
            assert last_log["status"] == "SUCCESS"
            assert last_log["execution_time_ms"] == 12.4
            assert last_log["concept_metadata"] == metadata_payload
            assert "timestamp" in last_log
            assert last_log["level"] == "info"
        finally:
            clear_log_context()
            _cleanup_file_handlers()


def test_neo4j_already_exists_notification_filter() -> None:
    """Verify that verbose Neo4j schema 'already exists' notifications are suppressed."""
    filter_instance = Neo4jAlreadyExistsLogFilter()

    record_already_exists = logging.LogRecord(
        name="neo4j.notifications",
        level=logging.INFO,
        pathname="result.py",
        lineno=346,
        msg=(
            "Received notification from DBMS server: <GqlStatusObject gql_status='00NA0', "
            'status_description="note: successful completion - index or constraint already exists.">'
        ),
        args=(),
        exc_info=None,
    )
    assert filter_instance.filter(record_already_exists) is False

    record_normal_warning = logging.LogRecord(
        name="neo4j.notifications",
        level=logging.WARNING,
        pathname="result.py",
        lineno=346,
        msg=(
            "Received notification from DBMS server: <GqlStatusObject gql_status='01000', "
            'status_description="warning: query plan sub-optimal">'
        ),
        args=(),
        exc_info=None,
    )
    assert filter_instance.filter(record_normal_warning) is True
