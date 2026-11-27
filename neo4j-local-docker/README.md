# Neo4j Docker Test Server Setup

## What is This?

This directory contains the Docker Compose file to run the standalone Neo4j Graph Database for GoSmart DS development, clinical terminology ingestion, and testing. Using the Community Edition, you can launch and use it without AuraDB Free node/relationship caps.

## Getting Started

- Download and set up Docker Desktop (or Docker Engine on Linux).
- Database persistence is automatically managed via a Docker named volume (`neo4j_data_ds`), eliminating host permission issues across macOS, Linux, and Windows.

## Memory Sizing Presets

`docker-compose.yaml` defaults are tuned for a **32 GB RAM host** ingesting the full OHDSI Athena clinical graph (`--profile full`: ~1.50M `:Concept` nodes, ~5.1M relationships, plus Lucene full-text indexes):

| Deployment Profile | Host RAM | Environment Overrides |
| :--- | :--- | :--- |
| **Full (`--profile full`)** *(default)* | 32 GB (e.g., Windows/WSL2 or Linux) | `NEO4J_HEAP_INITIAL=4G NEO4J_HEAP_MAX=8G NEO4J_PAGECACHE=8G NEO4J_TX_MAX=4G` |
| **Valid (`--profile valid`)** | 16 GB | `NEO4J_HEAP_INITIAL=2G NEO4J_HEAP_MAX=4G NEO4J_PAGECACHE=4G NEO4J_TX_MAX=2G` |
| **Demo (`--profile demo-core`)** | **8 GB Linux VM** | `NEO4J_HEAP_INITIAL=1G NEO4J_HEAP_MAX=3G NEO4J_PAGECACHE=2G NEO4J_TX_MAX=1G` |

> **Windows / WSL2 Note:** When running the 32 GB full preset under Docker Desktop on Windows, set `memory=20GB` (or higher) in `%USERPROFILE%\.wslconfig` so WSL2 does not cap the Neo4j container below 16 GB.

## Authentication is Disabled

- To facilitate local development and testing, authentication is disabled in `docker-compose.yaml` via `NEO4J_AUTH=none`.
- **Do not use `NEO4J_AUTH=none` in production!**

## Ports Exposed

- `7475` (mapped to container `7474`): Neo4j Browser HTTP UI
- `7688` (mapped to container `7687`): Bolt protocol

Ensure inbound firewall rules allow `7688` (and optionally `7475`) when connecting from another machine on your LAN.

## Starting

```shell
# Full 32 GB host defaults
docker compose up -d

# 8 GB Linux demo VM override
NEO4J_HEAP_INITIAL=1G NEO4J_HEAP_MAX=3G NEO4J_PAGECACHE=2G NEO4J_TX_MAX=1G docker compose up -d
```

## Cloud Server Alternative

Neo4j AuraDB Free supports up to **200,000 nodes**, **400,000 relationships**, and **1 GB storage** (with 72-hour write inactivity auto-pause). That fits the DICOM dictionary alone (~15k nodes), but will not fit the full SNOMED/LOINC/ICD-10-CM clinical graph; use self-hosted Neo4j Community Edition or AuraDB Professional for clinical ontologies.

## References

- [Neo4j Docker Hub](https://hub.docker.com/_/neo4j/)
- [Neo4j Aura](https://neo4j.com/videos/getting-started-with-aura-free-tier/)
