---
title: Home
description: Gosmart.Health Dictionary Service — a terminology dictionary and ontology graph service with a FastAPI API and multi-page HTML web portal backed by a Neo4j knowledge graph, Brevo registration, and Genkit-powered LLM flows.
---

# Gosmart.Health Dictionary Service

Terminology dictionary and ontology graph service for the Gosmart.Health
platform: a FastAPI API and multi-page HTML web portal backed by a **Neo4j**
knowledge graph, Brevo email registration, and **Genkit**-powered LLM flows for
terminology resolution and enrichment.

## Why

- **Single source of truth** for terminologies shared across the Gosmart.Health
  platform (GRAPHITE, the FHIR plugin, and downstream clients).
- **Graph-native queries** — hierarchical and associative term relationships are
  traversals, not joins.
- **AI-assisted mapping** — Genkit flows turn free text into candidate coded
  concepts with structured, validated output.

## Stack

| Concern     | Module                                                           |
| ----------- | ---------------------------------------------------------------- |
| API & Web   | [FastAPI](https://fastapi.tiangolo.com/) + `uvicorn[standard]` + `jinja2` |
| Auth & Mail | `argon2-cffi`, `pyjwt`, `httpx` + [Brevo API](https://developers.brevo.com/) |
| Config      | [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) (`GOSMART_DS_*`) |
| Logging     | [`structlog`](https://www.structlog.org/) + 7-day rotating JSON file logs |
| LLM flows   | [Genkit](https://python.api.genkit.dev/) (`genkit-ollama`, `genkit-google-genai`) |
| NLU         | `gosmart-clinical-nlu` (spaCy `textcat` + `entity_ruler` intent plugin + Genkit query expansion) |
| Graph store | [Neo4j Python driver](https://neo4j.com/docs/python-manual/current/) |
| Concurrency | [`pykka`](https://pykka.readthedocs.io/) (Actor Model) |
| Docs site   | [Zensical](https://zensical.org/) (`docs` dependency group) |

## Next

See [Getting Started](getting-started/index.md) to install and run the service, or
read [Architecture](architecture.md) for the full system design.
