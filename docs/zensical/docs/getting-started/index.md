---
title: Getting Started
description: Install, run, ingest, and test the Gosmart.Health Dictionary Service (GoSmart DS) locally.
---

# Getting Started

## Requirements

- Python 3.14 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
# Install runtime + dev + docs dependencies
uv sync --all-groups
```

## Configuration & Logging

Configuration is loaded from `.env` (and environment variables prefixed with
`GOSMART_DS_*`). Copy `.env.example` to `.env` to customize locally:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GOSMART_DS_HOST` | `0.0.0.0` | FastAPI host binding |
| `GOSMART_DS_PORT` | `8081` | FastAPI HTTP port |
| `GOSMART_DS_RELOAD` | `true` | Uvicorn auto-reload |
| `GOSMART_DS_DEV_MODE` | `false` | Allow startup if Neo4j is unreachable; enables local verification link fallback when `GOSMART_DS_BREVO_API_KEY` is unset |
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
| `GOSMART_DS_NLU_MODEL_PATH` | `./nlu_models/intent_models.zip` | Path to the trained `gosmart-clinical-nlu` spaCy model bundle |
| `GOSMART_DS_NLU_LLM_EXPANSION_ENABLED` | `true` | Enable LLM alternate search phrase synthesis (`gemma4:e2b`) during NLU parsing |
| `GOSMART_DS_NLU_MAX_ALTERNATE_PHRASES` | `3` | Maximum alternate clinical/DICOM search phrases synthesized per query |
| `GOSMART_DS_NLU_SCHEMA_CACHE_TTL_SEC` | `60` | TTL (seconds) for caching active Neo4j loaded coding schemas during NLU schema checks |
| `GOSMART_DS_VERTEX_AI_API_KEY` | *(unset)* | Optional Vertex AI / Google AI API key (referenced via `${GOSMART_DS_VERTEX_AI_API_KEY}`) |
| `GOSMART_DS_VERTEX_AI_LOCATION` | `us-central1` | Default Google Cloud / Vertex AI region |
| `GOSMART_DS_VERTEX_AI_PROJECT_ID` | *(unset)* | Optional Google Cloud Project ID for Vertex AI |
| `GOSMART_DS_LLM_CONFIG_ENCRYPTION_KEY` | *(unset)* | Optional Fernet passphrase/key to encrypt literal LLM API keys at rest |
| `GOSMART_DS_JWT_SECRET_KEY` | *(dev default)* | Secret key used to sign JWT access tokens and browser session cookies (set a strong 32+ char value in production) |
| `GOSMART_DS_API_KEY_PREFIX` | `gs_dict_` | Prefix used for generated `X-API-Key` credentials |
| `GOSMART_DS_SESSION_COOKIE_NAME` | `gosmart_ds_session` | `HttpOnly` session cookie name for the HTML web portal |
| `GOSMART_DS_SESSION_COOKIE_SECURE` | `false` | Set `true` behind HTTPS / Cloudflare so session cookies require TLS |
| `GOSMART_DS_BREVO_API_KEY` | *(unset)* | Brevo transactional email API key (`api-key` header) for account verification emails; in `DEV_MODE=true` when unset, verification links are logged and displayed locally |
| `GOSMART_DS_BREVO_API_URL` | `https://api.brevo.com/v3/smtp/email` | Brevo transactional email API endpoint URL |
| `GOSMART_DS_BREVO_SENDER_EMAIL` | `no-reply@gosmart.health` | Verified sender email address for Brevo transactional emails |
| `GOSMART_DS_BREVO_SENDER_NAME` | `GoSmart Health Dictionary Service` | Sender display name for Brevo transactional emails |
| `GOSMART_DS_PUBLIC_BASE_URL` | `http://127.0.0.1:8081` | Public origin URL used to construct `/verify-email?token=...` links |
| `GOSMART_DS_EMAIL_VERIFICATION_EXPIRE_HOURS` | `24` | Expiration window (hours) for registration verification links |
| `GOSMART_DS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `GOSMART_DS_LOG_FORMAT` | `json` | Console renderer: `json` or `console` |
| `GOSMART_DS_LOG_DIR` | `./logs` | Rotating log file output directory |
| `GOSMART_DS_LOG_FILE` | `gosmart_ds.log` | Base filename for JSON rotating logs |
| `GOSMART_DS_LOG_ROTATION_DAYS` | `7` | Rotation interval in days |
| `GOSMART_DS_LOG_BACKUP_COUNT` | `8` | Number of rotated backups retained |
| `GOSMART_DS_LOG_TO_FILE` | `true` | Enable rotating JSON file logging |

## Running the API server

```bash
# Start the API server (default port 8081)
uv run python -m gosmart_dictionary_service
# ...or explicitly via CLI
uv run gosmart-ds serve
```

### Web Portal & Endpoints

- **Web Portal (Multi-Page HTML)**:
  - Home: <http://127.0.0.1:8081/> (serves HTML to browsers; returns JSON service metadata when `Accept: application/json` or default `*/*` is requested)
  - Account Registration: <http://127.0.0.1:8081/register>
  - Email Verification: <http://127.0.0.1:8081/verify-email>
  - Log In: <http://127.0.0.1:8081/login>
  - API Key Management: <http://127.0.0.1:8081/keys>
  - Grounded Dictionary AI Chat: <http://127.0.0.1:8081/chat> (unified Clinical + DICOM NLU grounding with scope selector, related-concept toggle, and collapsible tabular FHIR citations)
  - Ontology & Dictionary Explorer: <http://127.0.0.1:8081/explore> (entity drill-down inspector with FHIR coding macros, categorized relationship tables, and Graph-UI-ready neighborhood JSON)
- **REST API & Docs**:
  - Interactive API docs: <http://127.0.0.1:8081/docs>
  - Health endpoint: <http://127.0.0.1:8081/api/v1/health>
  - Auth & API Keys: `/api/v1/auth/register`, `/api/v1/auth/verify-email`, `/api/v1/auth/login`, `/api/v1/auth/me`, `/api/v1/auth/keys`
  - AI Chat: `POST /api/v1/chat` (supports `scope` = `auto` | `clinical` | `dicom`, `include_related`, NLU routing metadata, warnings, and drill-down citations; authenticated via session cookie, JWT Bearer token, or `X-API-Key`)

The template requests in `test_main.http` can be run from the IDE HTTP client
against a running server.

## Bootstrapping & Ingestion

```bash
# Bootstrap Neo4j schema + :System + 'GoSmart Health' :Group + default :LLM_Config nodes
uv run gosmart-ds bootstrap

# Validate DICOM seed files (dry run — zero DB writes)
uv run gosmart-ds ingest-dicom --source-dir ./data-source/dicom --dry-run

# Ingest DICOM standard dictionary into Neo4j (and optionally compute 768-dim embeddings)
uv run gosmart-ds ingest-dicom --source-dir ./data-source/dicom --bootstrap
uv run gosmart-ds ingest-dicom --source-dir ./data-source/dicom --bootstrap --compute-embeddings

# Validate OHDSI Athena clinical terms CSVs (dry run — zero DB writes)
uv run gosmart-ds ingest-clinical-terms --source-dir ./data-source/clinical-terms --profile full --dry-run

# Ingest full clinical graph (SNOMED, LOINC, ICD10CM, UCUM; ~1.50M concepts — recommended 32 GB Neo4j host)
uv run gosmart-ds ingest-clinical-terms --source-dir ./data-source/clinical-terms --profile full --bootstrap

# Ingest reduced demo-core clinical graph (~455k valid concepts across Condition/Measurement/Procedure/Anatomy/Unit — fits 8 GB Linux VM)
uv run gosmart-ds ingest-clinical-terms --source-dir ./data-source/clinical-terms --profile demo-core --bootstrap
```

### Training the spaCy NLU Model (`gosmart-clinical-nlu`)

GoSmart DS registers the `find_coding_schema` intent plugin via the
`gosmart_clinical_nlu.intents` entry point
(`src/gosmart_dictionary_service/nlu/find_coding_schema.py`) with training data
under `training/find_coding_schema/`:

```bash
# List discovered built-in and external intents
uv run gosmart-clinical-nlu list-intents

# Train the intent bundle into nlu_models/intent_models.zip
./devop-scripts/nlu-train.sh

# Inspect and smoke-test the trained bundle
uv run gosmart-clinical-nlu inspect --model nlu_models/intent_models.zip
uv run gosmart-clinical-nlu parse --model nlu_models/intent_models.zip "Look up DICOM (0008, 0010)"
```

### Unified Search & Clinical NLU Service

`SearchService` (`src/gosmart_dictionary_service/service/search_service.py`)
pairs with `ClinicalNLUService`
(`src/gosmart_dictionary_service/service/nlu_service.py`) to run every query
through a 5-stage grounded discovery pipeline:

1. **Trained spaCy NLU Extraction**: Loads `nlu_models/intent_models.zip` via `gosmart-clinical-nlu` to extract coding schemas, clinical codes, DICOM tags, and DICOM UIDs, and dynamically alerts the caller via `SearchResponse.warnings` when a mentioned schema is not loaded in the active Neo4j database (e.g., demo subsets or unsupported schemas like `RxNorm`).
2. **Deterministic Exact Hits (`score = 1.0`)**: Resolved codes/tags remain at the top of `hits` with `MatchSource.EXACT_CODE` (unknown codes in loaded schemas emit a warning and are skipped so raw numbers are never hallucinated).
3. **Reverse-Lookup Query Rewriting**: Replaces extracted code spans in the query string with their human-readable Neo4j `Concept.display` / `DicomElement.name` (`nlu.rewritten_query`).
4. **LLM Alternate Phrase Synthesis (`gemma4:e2b`)**: Uses the active `inference-v1` `LLM_Config` via Genkit (`expand_with_llm=True` by default, with graceful fallback if offline) to synthesize medical/DICOM synonyms (`nlu.synthesized_phrases`).
5. **Discounted Hybrid Aggregation**: Executes Lucene full-text + HNSW/embedding vector search across both Clinical Concepts and DICOM entities for the rewritten query and synthesized phrases, aggregating discovered matches beneath deterministic hits at discounted probabilities (`<= 0.85`).

```python
from gosmart_dictionary_service.model.search import ResponseFormat, SearchEntityType
from gosmart_dictionary_service.service.search_service import search_service

# Unified natural-language search (NLU + LLM expansion + Lucene + [:IS_A] convergence + embedding rerank)
res = await search_service.search("acute pneumonia", entity_types=[SearchEntityType.CONCEPT])

# Code lookup + automatic reverse-lookup & cross-domain (Clinical + DICOM) discovery
disc = await search_service.search("LOINC 87279-6")

# Forward + reverse + 2-hop shared-axis related concept aggregation
rel = await search_service.search("Find something related to LOINC 87279-6")
compact_mcp_payload = rel.to_wire(ResponseFormat.COMPACT)
```

Interactive Jupyter playgrounds:

- `notebooks/dicom_graph_playground.ipynb`
- `notebooks/clinical_terms_playground.ipynb`

## Project layout

```
gosmart-dictionary-service/
├── pyproject.toml                     # Project metadata, dependencies, NLU entry points, tool config
├── .env.example                       # Sample GOSMART_DS_* configuration
├── training/                          # spaCy NLU training corpus (`config.json`, `find_coding_schema/`)
├── nlu_models/                        # Trained `gosmart-clinical-nlu` spaCy bundle (`intent_models.zip`)
├── devop-scripts/                     # Operational scripts (`nlu-train.sh`)
├── notebooks/                         # Interactive Jupyter playgrounds (DICOM & Clinical Terms)
├── src/
│   └── gosmart_dictionary_service/    # Importable package (src layout)
│       ├── __init__.py                # Package docstring + __version__
│       ├── __main__.py                # CLI (`gosmart-ds serve | bootstrap | ingest-dicom | ingest-clinical-terms`)
│       ├── main.py                    # FastAPI app factory, lifespan, static assets & web mount
│       ├── core/                      # Settings (`GOSMART_DS_*`), security, auth deps & structlog
│       ├── db/                        # Neo4j async/sync driver & Cypher schema loader/DDL
│       ├── api/                       # Versioned REST endpoints (`/api/v1/health`, `/auth`, `/chat`)
│       ├── web/                       # Multi-page HTML portal (Jinja2 templates + static CSS/JS)
│       ├── actor/                     # Pykka ThreadingActors (DicomIngestionActor, ClinicalTermsIngestionActor, EmbeddingActor)
│       ├── ingestor/                  # Batched Neo4j dictionary ingesters (DicomIngesterService, ClinicalTermsIngesterService)
│       ├── model/                     # Pydantic domain models (System, Auth, LLM_Config, Embedding, Search, DICOM, Clinical)
│       ├── nlu/                       # External `gosmart-clinical-nlu` intent specs (`find_coding_schema`)
│       └── service/                   # Core business services (System, User, Email, Chat, LLMConfig, Embedding, Search, ClinicalNLU)
├── tests/
│   ├── conftest.py                    # Shared fixtures (TestClient)
│   ├── unit/                          # Unit tests
│   └── integration/                   # Live Neo4j integration tests (auto-skip when offline)
└── docs/
    └── zensical/                      # Documentation site source
```

!!! note "Package naming"
    The distribution name is `gosmart-dictionary-service`; the importable
    package name is `gosmart_dictionary_service` because Python module names
    cannot contain hyphens.

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

See `docs/zensical/README.md` for site build and deployment details.
