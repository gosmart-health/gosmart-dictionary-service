---
title: Getting Started
description: Install, run, and test the Gosmart.Health Dictionary Service locally.
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

## Running the API server

```bash
# Auto-reload on http://127.0.0.1:8000
uv run python -m gosmart_dictionary_service

# ...or directly through uvicorn
uv run uvicorn gosmart_dictionary_service.main:app --reload
```

Interactive API docs are served at <http://127.0.0.1:8000/docs>.

## Project layout

```
gosmart-dictionary-service/
├── pyproject.toml                     # Metadata, dependencies, tool config
├── src/
│   └── gosmart_dictionary_service/    # Importable package (src layout)
│       ├── __init__.py                # Package docstring + __version__
│       ├── __main__.py                # `python -m gosmart_dictionary_service`
│       └── main.py                    # FastAPI app
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
