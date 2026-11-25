# Gosmart.Health Dictionary Service (GoSmart DS)

Terminology dictionary and ontology graph service for the Gosmart.Health
platform: a FastAPI API backed by a Neo4j knowledge graph, with Genkit-powered
LLM flows for terminology resolution and enrichment.

## Stack

| Concern     | Module                                                           |
|-------------|------------------------------------------------------------------|
| API         | [`fastapi`](https://fastapi.tiangolo.com/) + `uvicorn[standard]` |
| Config      | [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) (`GOSMART_DS_*`) |
| Logging     | [`structlog`](https://www.structlog.org/) + 7-day rotating JSON file logs |
| LLM flows   | [`genkit`](https://python.api.genkit.dev/)                       |
| Graph store | [`neo4j`](https://neo4j.com/docs/python-manual/current/)         |
| Docs site   | [`zensical`](https://zensical.org/) (`docs` dependency group)    |

## Requirements

- Python 3.14 (pinned in [`.python-version`](./.python-version))
- [uv](https://docs.astral.sh/uv/)

## Project layout

```
gosmart-dictionary-service/
├── pyproject.toml                     # Project metadata, dependencies, tool config
├── .env.example                       # Sample GOSMART_DS_* configuration
├── src/
│   └── gosmart_dictionary_service/    # Importable package (src layout)
│       ├── __init__.py                # Package docstring + __version__
│       ├── __main__.py                # `python -m gosmart_dictionary_service`
│       ├── main.py                    # FastAPI app factory, lifespan & request logging middleware
│       ├── core/                      # Settings (`GOSMART_DS_*`) & structlog logging setup
│       ├── db/                        # Neo4j async/sync driver & Cypher schema loader
│       ├── api/                       # Versioned REST endpoints (`/api/v1`)
│       ├── model/                     # Pydantic domain models
│       └── service/                   # Core business services
├── tests/
│   ├── conftest.py                    # Shared fixtures (TestClient)
│   └── unit/                          # Unit tests
└── docs/
    └── zensical/                      # Documentation site source
```

> The distribution name is `gosmart-dictionary-service`; the importable package
> name is `gosmart_dictionary_service` (Python module names cannot contain
> hyphens).

## Configuration & Logging

Configuration is loaded from `.env` (and environment variables prefixed with `GOSMART_DS_`). Copy `.env.example` to `.env` to customize locally:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GOSMART_DS_HOST` | `0.0.0.0` | FastAPI host binding |
| `GOSMART_DS_PORT` | `8081` | FastAPI HTTP port |
| `GOSMART_DS_RELOAD` | `true` | Uvicorn auto-reload |
| `GOSMART_DS_DEV_MODE` | `false` | Allow startup if Neo4j is unreachable |
| `GOSMART_DS_DB_CONNECTION_URI` | `bolt://localhost:7688` | Dedicated Neo4j Bolt URI |
| `GOSMART_DS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `GOSMART_DS_LOG_FORMAT` | `json` | Console renderer: `json` or `console` |
| `GOSMART_DS_LOG_DIR` | `./logs` | Rotating log file output directory |
| `GOSMART_DS_LOG_FILE` | `gosmart_ds.log` | Base filename for JSON rotating logs |
| `GOSMART_DS_LOG_ROTATION_DAYS` | `7` | Rotation interval in days |
| `GOSMART_DS_LOG_BACKUP_COUNT` | `8` | Number of rotated backups retained |
| `GOSMART_DS_LOG_TO_FILE` | `true` | Enable rotating JSON file logging |

## Getting started

```bash
# Install runtime + dev + docs dependencies
uv sync --all-groups

# Start the API server (default port 8081)
uv run python -m gosmart_dictionary_service

# ...or directly through uvicorn
uv run uvicorn gosmart_dictionary_service.main:app --port 8081 --reload
```

Interactive API docs: <http://127.0.0.1:8081/docs>.
Health endpoint: <http://127.0.0.1:8081/api/v1/health>.
The template requests in [`test_main.http`](./test_main.http) can be run from
the IDE HTTP client against a running server.

## Development

```bash
uv run pytest            # run the test suite
uv run ruff check .      # lint
uv run ruff format .     # format
```

## Documentation site

Docs tooling lives in the `docs` dependency group (kept out of runtime deps).
Run from `docs/zensical/`:

```bash
uv run --group docs zensical serve                    # live preview
uv run --group docs zensical build --strict --clean   # production build into ./site/
```

See [`docs/zensical/README.md`](./docs/zensical/README.md).

## License

Copyright (c) 2026, GoSmart Health — Apache-2.0
License. <https://gosmart.health>
