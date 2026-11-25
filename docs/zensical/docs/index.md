---
title: Home
description: Gosmart.Health Dictionary Service — a terminology dictionary API backed by a Neo4j knowledge graph and Genkit LLM flows.
---

# Gosmart.Health Dictionary Service

The Dictionary Service exposes clinical terminology (codes, value sets, and
their relationships) over a FastAPI HTTP API. Terms live in a **Neo4j**
knowledge graph, and **Genkit** flows provide LLM-assisted terminology
resolution and enrichment.

## Why

- **Single source of truth** for terminologies shared across the Gosmart.Health
  platform (GRAPHITE, the FHIR plugin, and downstream clients).
- **Graph-native queries** — hierarchical and associative term relationships are
  traversals, not joins.
- **AI-assisted mapping** — Genkit flows turn free text into candidate coded
  concepts with structured, validated output.

## Stack

| Concern     | Module                                                            |
| ----------- | ----------------------------------------------------------------- |
| API         | [FastAPI](https://fastapi.tiangolo.com/) + `uvicorn[standard]`     |
| LLM flows   | [Genkit](https://python.api.genkit.dev/)                          |
| Graph store | [Neo4j Python driver](https://neo4j.com/docs/python-manual/current/) |
| Docs site   | [Zensical](https://zensical.org/)                                 |

## Next

See [Getting Started](getting-started/index.md) to install and run the service.
