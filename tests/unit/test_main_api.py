"""Smoke tests for the FastAPI routes declared in ``gosmart_dictionary_service.main``."""

from fastapi.testclient import TestClient

from gosmart_dictionary_service import __version__
from gosmart_dictionary_service.main import app


def test_root_returns_hello_world(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_say_hello_interpolates_name(client: TestClient) -> None:
    response = client.get("/hello/User")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello User"}


def test_openapi_metadata_matches_package_version(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    payload = response.json()
    assert payload["info"]["title"] == "Gosmart.Health Dictionary Service"
    assert payload["info"]["version"] == __version__


def test_documentation_routes_are_enabled(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert app.openapi_url == "/openapi.json"
