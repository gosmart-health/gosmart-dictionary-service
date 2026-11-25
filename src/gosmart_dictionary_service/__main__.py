"""Allow ``python -m gosmart_dictionary_service`` to start the API server."""

from gosmart_dictionary_service.main import run


def main() -> None:
    """Serve the FastAPI app with uvicorn using configured settings."""
    run()


if __name__ == "__main__":
    main()
