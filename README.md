# Gosmart.Health Dictionary Service

Terminology dictionary and ontology graph service for the Gosmart.Health
platform: a FastAPI API
backed by a Neo4j knowledge graph, with Genkit-powered LLM flows for
terminology resolution and enrichment.

## Stack

| Concern     | Module                                                           |
|-------------|------------------------------------------------------------------|
| API         | [`fastapi`](https://fastapi.tiangolo.com/) + `uvicorn[standard]` |
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
├── src/
│   └── gosmart_dictionary_service/    # Importable package (src layout)
│       ├── __init__.py                # Package docstring + __version__
│       ├── __main__.py                # `python -m gosmart_dictionary_service`
│       └── main.py                    # FastAPI app
├── tests/
│   ├── conftest.py                    # Shared fixtures (TestClient)
│   └── unit/                          # Unit tests
└── docs/
    └── zensical/                      # Documentation site source
```

> The distribution name is `gosmart-dictionary-service`; the importable package
> name is `gosmart_dictionary_service` (Python module names cannot contain
> hyphens).

## Getting started

```bash
# Install runtime + dev + docs dependencies
uv sync --all-groups

# Start the API server with auto-reload on http://127.0.0.1:8000
uv run python -m gosmart_dictionary_service

# ...or directly through uvicorn
uv run uvicorn gosmart_dictionary_service.main:app --reload
```

Interactive API docs: <http://127.0.0.1:8000/docs>.
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
