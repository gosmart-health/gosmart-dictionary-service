"""Shared fixtures for the dictionary service test suite."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from gosmart_dictionary_service.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """An in-process HTTP client bound to the FastAPI app (no network, no server)."""
    with TestClient(app) as test_client:
        yield test_client
