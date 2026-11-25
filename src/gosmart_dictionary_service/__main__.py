"""Allow ``python -m gosmart_dictionary_service`` to start the API server."""

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def main() -> None:
    """Serve the FastAPI app with uvicorn (reload enabled for local development)."""
    uvicorn.run(
        "gosmart_dictionary_service.main:app",
        host=HOST,
        port=PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()
