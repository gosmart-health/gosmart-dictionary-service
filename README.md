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
| LLM flows   | [`genkit`](https://python.api.genkit.dev/) (`genkit-ollama`, `genkit-google-genai`) |
| Graph store | [`neo4j`](https://neo4j.com/docs/python-manual/current/)         |
| Concurrency | [`pykka`](https://pykka.readthedocs.io/) (Actor Model)           |
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
│       ├── __main__.py                # CLI (`gosmart-ds serve | bootstrap | ingest-dicom`)
│       ├── main.py                    # FastAPI app factory, lifespan & request logging middleware
│       ├── core/                      # Settings (`GOSMART_DS_*`), security & structlog setup
│       ├── db/                        # Neo4j async/sync driver & Cypher schema loader/DDL
│       ├── api/                       # Versioned REST endpoints (`/api/v1`)
│       ├── actor/                     # Pykka ThreadingActors (DicomIngestionActor, EmbeddingActor)
│       ├── ingestor/                  # Batched Neo4j dictionary ingesters (DicomIngesterService)
│       ├── model/                     # Pydantic domain models (System, Group, LLM_Config, Embedding, DICOM)
│       └── service/                   # Core business services (SystemService, LLMConfigService, EmbeddingService)
├── tests/
│   ├── conftest.py                    # Shared fixtures (TestClient)
│   ├── unit/                          # Unit tests
│   └── integration/                   # Live Neo4j integration tests (auto-skip when offline)
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
| `GOSMART_DS_VECTOR_EMBED_PROVIDER` | `ollama` | Default embedding provider (`ollama`, `vertexai`, `googleai`) |
| `GOSMART_DS_VECTOR_EMBED_SERVER_URL` | `http://localhost:11434` | Default Ollama/custom embedding server URL |
| `GOSMART_DS_VECTOR_EMBED_MODEL` | `embeddinggemma-2:270m` | Default vector embedding model name |
| `GOSMART_DS_VECTOR_EMBED_DIMENSIONS` | `768` | Output vector dimensionality for Neo4j vector index |
| `GOSMART_DS_VECTOR_EMBED_TAG` | `embedding-v1` | Default `LLM_Config.tag` seeded for embeddings |
| `GOSMART_DS_INFERENCE_PROVIDER` | `ollama` | Default inference provider (`ollama`, `vertexai`, `googleai`) |
| `GOSMART_DS_INFERENCE_SERVER_URL` | `http://localhost:11434` | Default Ollama/custom inference server URL |
| `GOSMART_DS_INFERENCE_MODEL` | `gemma4:e2b` | Default inference model name |
| `GOSMART_DS_INFERENCE_TAG` | `inference-v1` | Default `LLM_Config.tag` seeded for inference |
| `GOSMART_DS_VERTEX_AI_API_KEY` | *(unset)* | Optional Vertex AI / Google AI API key (referenced via `${GOSMART_DS_VERTEX_AI_API_KEY}`) |
| `GOSMART_DS_VERTEX_AI_LOCATION` | `us-central1` | Default Google Cloud / Vertex AI region |
| `GOSMART_DS_VERTEX_AI_PROJECT_ID` | *(unset)* | Optional Google Cloud Project ID for Vertex AI |
| `GOSMART_DS_LLM_CONFIG_ENCRYPTION_KEY` | *(unset)* | Optional Fernet passphrase/key to encrypt literal LLM API keys at rest |
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
# ...or explicitly via CLI
uv run gosmart-ds serve

# Bootstrap Neo4j schema + :System + 'GoSmart Health' :Group + default :LLM_Config nodes
uv run gosmart-ds bootstrap

# Validate DICOM seed files (dry run — zero DB writes)
uv run gosmart-ds ingest-dicom --source-dir ./data-source/dicom --dry-run

# Ingest DICOM standard dictionary into Neo4j (and optionally compute 768-dim embeddings)
uv run gosmart-ds ingest-dicom --source-dir ./data-source/dicom --bootstrap
uv run gosmart-ds ingest-dicom --source-dir ./data-source/dicom --bootstrap --compute-embeddings
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
