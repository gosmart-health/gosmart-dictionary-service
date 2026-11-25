---
title: Getting Started
description: Install, run, and test the Gosmart.Health Dictionary Service (GoSmart DS) locally.
---

# Getting Started

## Requirements

- Python 3.14 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
# Runtime + dev + docs dependencies
uv sync --all-groups
```

## Configuration & Logging

Copy `.env.example` to `.env` to configure `GOSMART_DS_*` settings locally. Key defaults:

- `GOSMART_DS_PORT=8081` (FastAPI HTTP port)
- `GOSMART_DS_DB_CONNECTION_URI=bolt://localhost:7688` (dedicated local Neo4j instance)
- `GOSMART_DS_LOG_LEVEL=INFO`, `GOSMART_DS_LOG_FORMAT=json` (`json` or `console`)
- `GOSMART_DS_LOG_DIR=./logs`, `GOSMART_DS_LOG_FILE=gosmart_ds.log` (7-day rotating JSON logs)

## Running the API server

```bash
# Auto-reload on http://127.0.0.1:8081
uv run python -m gosmart_dictionary_service

# ...or directly through uvicorn
uv run uvicorn gosmart_dictionary_service.main:app --port 8081 --reload
```

Interactive API docs are served at <http://127.0.0.1:8081/docs>, and health checks at <http://127.0.0.1:8081/api/v1/health>.

## Project layout

```
gosmart-dictionary-service/
├── pyproject.toml                     # Metadata, dependencies, tool config
├── .env.example                       # Sample GOSMART_DS_* configuration
├── src/
│   └── gosmart_dictionary_service/    # Importable package (src layout)
│       ├── __init__.py                # Package docstring + __version__
│       ├── __main__.py                # `python -m gosmart_dictionary_service`
│       ├── main.py                    # FastAPI app factory, lifespan & request logging
│       ├── core/                      # Settings (`GOSMART_DS_*`) & structlog logging
│       ├── db/                        # Neo4j client & Cypher schema loader
│       └── api/                       # Versioned REST endpoints (`/api/v1`)
├── tests/
│   ├── conftest.py                    # Shared fixtures (TestClient)
│   └── unit/                          # Unit tests
└── docs/
    └── zensical/                      # This documentation site
```

!!! note "Package naming"
    The distribution name is `gosmart-dictionary-service`; the importable
    package name is `gosmart_dictionary_service` because Python module names
    cannot contain hyphens.

## Testing and linting

```bash
uv run pytest            # run the test suite
uv run ruff check .      # lint
uv run ruff format .     # format
```

## Documentation site

Run from `docs/zensical/`:

```bash
uv run --group docs zensical serve                    # live preview
uv run --group docs zensical build --strict --clean   # production build
```
