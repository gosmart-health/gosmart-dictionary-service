"""CLI and module entrypoint for GoSmart DS (``gosmart-ds`` / ``python -m gosmart_dictionary_service``)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

from gosmart_dictionary_service.actor.clinical_terms_ingestion_actor import ClinicalTermsIngestionActor
from gosmart_dictionary_service.actor.dicom_ingestion_actor import DicomIngestionActor
from gosmart_dictionary_service.actor.embedding_actor import EmbeddingActor
from gosmart_dictionary_service.core.logging import setup_logging
from gosmart_dictionary_service.db.neo4j import neo4j_client
from gosmart_dictionary_service.main import run
from gosmart_dictionary_service.model.actor_message import ClinicalTermsIngestMessage, DicomIngestMessage
from gosmart_dictionary_service.model.clinical.constants import ClinicalIngestionProfile
from gosmart_dictionary_service.model.llm_config import DEFAULT_EMBEDDING_CONFIG_TAG
from gosmart_dictionary_service.model.result import STATUS_OK
from gosmart_dictionary_service.service.mcp_service import build_mcp_transport_security, run_mcp_server
from gosmart_dictionary_service.service.system_service import system_service


def _parse_csv_list(raw: Optional[str]) -> Optional[List[str]]:
    if not raw or not raw.strip():
        return None
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items if items else None


def _parse_int_csv_list(raw: Optional[str]) -> Optional[List[int]]:
    str_items = _parse_csv_list(raw)
    if str_items is None:
        return None
    return [int(item) for item in str_items]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gosmart-ds",
        description="GoSmart Health Dictionary Service CLI (defaults to launching the FastAPI server).",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve (default when no subcommand given)
    subparsers.add_parser("serve", help="Start the FastAPI HTTP server with uvicorn")

    # mcp
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Start the GoSmart DS Model Context Protocol (MCP) server (stdio, streamable-http, or sse)",
    )
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    mcp_parser.add_argument(
        "--host",
        default=None,
        help="Host binding when --transport is 'streamable-http' or 'sse' (default: 127.0.0.1)",
    )
    mcp_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port binding when --transport is 'streamable-http' or 'sse' (default: 8000)",
    )

    # bootstrap
    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Apply Neo4j DDL schema and bootstrap :System, 'GoSmart Health' :Group, and default :LLM_Config nodes",
    )
    bootstrap_parser.add_argument("--acted-by", default="cli", help="Audit actor identifier (default: cli)")
    bootstrap_parser.add_argument("--test-id", default=None, help="Optional test_id for isolated test runs")
    bootstrap_parser.add_argument(
        "--no-llm-configs",
        action="store_true",
        help="Skip seeding default embedding-v1 and inference-v1 LLM_Config nodes",
    )

    # ingest-dicom
    ingest_parser = subparsers.add_parser(
        "ingest-dicom",
        help="Ingest Innolitics DICOM standard JSON files into Neo4j via Pykka actors",
    )
    ingest_parser.add_argument(
        "--source-dir",
        default="./data-source/dicom",
        help="Path to directory containing DICOM standard JSON files (default: ./data-source/dicom)",
    )
    ingest_parser.add_argument("--batch-size", type=int, default=1000, help="Neo4j UNWIND batch size (default: 1000)")
    ingest_parser.add_argument("--acted-by", default="cli_dicom_ingestor", help="Audit actor identifier")
    ingest_parser.add_argument("--test-id", default=None, help="Optional test_id for isolated test runs")
    ingest_parser.add_argument("--bootstrap", action="store_true", help="Bootstrap System/Group/LLM_Config first")
    ingest_parser.add_argument("--no-schema", action="store_true", help="Skip applying Neo4j DDL schema first")
    ingest_parser.add_argument("--no-references", action="store_true", help="Skip ingesting references.json nodes")
    ingest_parser.add_argument(
        "--no-reference-edges",
        action="store_true",
        help="Skip ingesting :REFERENCES_STANDARD edges",
    )
    ingest_parser.add_argument(
        "--compute-embeddings",
        action="store_true",
        help="Compute and persist :Embedding nodes for DICOM name and description fields",
    )
    ingest_parser.add_argument(
        "--embedding-tag",
        default=DEFAULT_EMBEDDING_CONFIG_TAG,
        help=f"LLM_Config tag for embeddings (default: {DEFAULT_EMBEDDING_CONFIG_TAG})",
    )
    ingest_parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Texts per embedding provider batch (default: 64)",
    )
    ingest_parser.add_argument(
        "--no-skip-existing-embeddings",
        action="store_true",
        help="Recompute embeddings even when existing :Embedding text is unchanged",
    )
    ingest_parser.add_argument(
        "--raw-embedding-text",
        action="store_true",
        help="Embed raw name/description text without DICOM entity context prefix",
    )
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate seed files and report counts without writing to Neo4j or calling LLMs",
    )

    # ingest-clinical-terms
    clinical_parser = subparsers.add_parser(
        "ingest-clinical-terms",
        help="Stream and ingest OHDSI Athena clinical terminology CSVs into Neo4j via Pykka actors",
    )
    clinical_parser.add_argument(
        "--source-dir",
        default="./data-source/clinical-terms",
        help="Path to directory containing OHDSI Athena CSV files (default: ./data-source/clinical-terms)",
    )
    clinical_parser.add_argument(
        "--profile",
        choices=[p.value for p in ClinicalIngestionProfile],
        default=ClinicalIngestionProfile.FULL.value,
        help="Preset scope: 'full' (~1.50M), 'valid' (~765k), or 'demo-core' (~455k for 8 GB Linux demo)",
    )
    clinical_parser.add_argument(
        "--vocabularies",
        default=None,
        help="Comma-separated vocabulary_id allowlist (default: SNOMED,LOINC,ICD10CM,UCUM)",
    )
    clinical_parser.add_argument(
        "--all-vocabularies",
        action="store_true",
        help="Ingest all vocabularies in the Athena export (including OSM/OMOP metadata)",
    )
    clinical_parser.add_argument(
        "--domains",
        default=None,
        help="Comma-separated domain_id filter (e.g. 'Condition,Measurement,Procedure,Spec Anatomic Site,Unit')",
    )
    dep_group = clinical_parser.add_mutually_exclusive_group()
    dep_group.add_argument(
        "--exclude-deprecated",
        action="store_true",
        help="Exclude deprecated/upgraded ('D'/'U') concepts",
    )
    dep_group.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include deprecated/upgraded ('D'/'U') concepts (overrides 'valid'/'demo-core' profile defaults)",
    )
    clinical_parser.add_argument("--no-synonyms", action="store_true", help="Skip attaching CONCEPT_SYNONYM.csv")
    clinical_parser.add_argument(
        "--languages",
        default=None,
        help="Comma-separated Athena language_concept_ids for synonyms (default: 4180186 for English)",
    )
    clinical_parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Include synonyms across all languages",
    )
    clinical_parser.add_argument(
        "--max-synonyms",
        type=int,
        default=25,
        help="Maximum deduplicated synonyms stored per Concept node (default: 25)",
    )
    clinical_parser.add_argument(
        "--no-relationships",
        action="store_true",
        help="Skip ingesting CONCEPT_RELATIONSHIP.csv ontology edges",
    )
    clinical_parser.add_argument(
        "--no-attribute-edges",
        action="store_true",
        help="Skip :HAS_ATTRIBUTE defining relationship edges (keep :IS_A, :MAPS_TO, :REPLACED_BY)",
    )
    clinical_parser.add_argument(
        "--include-rf2-metadata-edges",
        action="store_true",
        help="Include SNOMED RF2 administrative edges ('Has status', 'Has Module')",
    )
    clinical_parser.add_argument(
        "--no-replaced-by-edges",
        action="store_true",
        help="Skip historical :REPLACED_BY replacement edges",
    )
    clinical_parser.add_argument(
        "--max-concepts",
        type=int,
        default=None,
        help="Optional cap on ingested Concept nodes (useful for dev/smoke testing)",
    )
    clinical_parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Neo4j UNWIND MERGE batch size (default: 5000)",
    )
    clinical_parser.add_argument("--acted-by", default="cli_clinical_terms_ingestor", help="Audit actor identifier")
    clinical_parser.add_argument("--test-id", default=None, help="Optional test_id for isolated test runs")
    clinical_parser.add_argument("--bootstrap", action="store_true", help="Bootstrap System/Group/LLM_Config first")
    clinical_parser.add_argument("--no-schema", action="store_true", help="Skip applying Neo4j DDL schema first")
    clinical_parser.add_argument(
        "--compute-embeddings",
        action="store_true",
        help="Compute and persist :Embedding nodes for ingested :Concept nodes (opt-in; slow on full dataset)",
    )
    clinical_parser.add_argument(
        "--embed-all-concepts",
        action="store_true",
        help="Embed non-standard and deprecated concepts too (default restricts to valid Standard/Classification)",
    )
    clinical_parser.add_argument(
        "--embed-synonyms",
        action="store_true",
        help="Also generate separate :Embedding nodes for concept synonyms",
    )
    clinical_parser.add_argument(
        "--embedding-tag",
        default=DEFAULT_EMBEDDING_CONFIG_TAG,
        help=f"LLM_Config tag for embeddings (default: {DEFAULT_EMBEDDING_CONFIG_TAG})",
    )
    clinical_parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Texts per embedding provider batch (default: 64)",
    )
    clinical_parser.add_argument(
        "--no-skip-existing-embeddings",
        action="store_true",
        help="Recompute embeddings even when existing :Embedding text is unchanged",
    )
    clinical_parser.add_argument(
        "--raw-embedding-text",
        action="store_true",
        help="Embed raw concept display text without vocabulary/domain/code context prefix",
    )
    clinical_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stream and validate Athena CSVs and report counts without writing to Neo4j or calling LLMs",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for GoSmart DS."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command in (None, "serve"):
        run()
        return 0

    if args.command == "mcp":
        log_stream = sys.stderr if args.transport == "stdio" else sys.stdout
        setup_logging(stream=log_stream)
        run_kwargs: Dict[str, Any] = {}
        if args.transport in ("streamable-http", "sse"):
            run_kwargs["transport_security"] = build_mcp_transport_security()
            if args.host:
                run_kwargs["host"] = args.host
            if args.port is not None:
                run_kwargs["port"] = args.port
        try:
            run_mcp_server(transport=args.transport, **run_kwargs)
            return 0
        finally:
            try:
                asyncio.run(neo4j_client.close())
            except Exception:
                if neo4j_client._sync_driver is not None:
                    neo4j_client._sync_driver.close()
                    neo4j_client._sync_driver = None

    setup_logging()

    if args.command == "bootstrap":
        try:
            result = system_service.bootstrap_system_sync(
                acted_by=args.acted_by,
                test_id=args.test_id,
                ensure_schema=True,
                seed_default_llm_configs=not args.no_llm_configs,
            )
            print(json.dumps(result, indent=2, default=str))
            return 0
        finally:
            if neo4j_client._sync_driver is not None:
                neo4j_client._sync_driver.close()
                neo4j_client._sync_driver = None

    if args.command == "ingest-dicom":
        embedding_actor = EmbeddingActor.start() if args.compute_embeddings else None
        ingestion_actor = DicomIngestionActor.start(embedding_actor_ref=embedding_actor)
        try:
            msg = DicomIngestMessage(
                source_dir=args.source_dir,
                batch_size=args.batch_size,
                acted_by=args.acted_by,
                test_id=args.test_id,
                ensure_schema=not args.no_schema,
                bootstrap_system=args.bootstrap,
                include_references=not args.no_references,
                include_reference_edges=not args.no_reference_edges,
                compute_embeddings=args.compute_embeddings,
                embedding_tag=args.embedding_tag,
                embedding_batch_size=args.embedding_batch_size,
                skip_existing_embeddings=not args.no_skip_existing_embeddings,
                use_contextual_embedding_text=not args.raw_embedding_text,
                dry_run=args.dry_run,
            )
            res = ingestion_actor.ask(msg)
            if res.status != STATUS_OK or res.value is None:
                print(json.dumps(res.model_dump(), indent=2, default=str), file=sys.stderr)
                return 1
            print(res.value.model_dump_json(indent=2))
            return 0
        finally:
            ingestion_actor.stop()
            if embedding_actor is not None:
                embedding_actor.stop()
            if neo4j_client._sync_driver is not None:
                neo4j_client._sync_driver.close()
                neo4j_client._sync_driver = None

    if args.command == "ingest-clinical-terms":
        include_deprecated: Optional[bool] = None
        if args.exclude_deprecated:
            include_deprecated = False
        elif args.include_deprecated:
            include_deprecated = True

        embedding_actor = EmbeddingActor.start() if args.compute_embeddings else None
        ingestion_actor = ClinicalTermsIngestionActor.start(embedding_actor_ref=embedding_actor)
        try:
            msg = ClinicalTermsIngestMessage(
                source_dir=args.source_dir,
                profile=ClinicalIngestionProfile(args.profile),
                vocabularies=_parse_csv_list(args.vocabularies),
                all_vocabularies=args.all_vocabularies,
                domains=_parse_csv_list(args.domains),
                include_deprecated=include_deprecated,
                include_synonyms=not args.no_synonyms,
                synonym_language_ids=_parse_int_csv_list(args.languages),
                all_synonym_languages=args.all_languages,
                max_synonyms_per_concept=args.max_synonyms,
                include_relationships=not args.no_relationships,
                include_attribute_edges=not args.no_attribute_edges,
                include_rf2_metadata_edges=args.include_rf2_metadata_edges,
                include_replaced_by_edges=not args.no_replaced_by_edges,
                max_concepts=args.max_concepts,
                batch_size=args.batch_size,
                acted_by=args.acted_by,
                test_id=args.test_id,
                ensure_schema=not args.no_schema,
                bootstrap_system=args.bootstrap,
                compute_embeddings=args.compute_embeddings,
                embed_standard_only=not args.embed_all_concepts,
                embed_synonyms=args.embed_synonyms,
                embedding_tag=args.embedding_tag,
                embedding_batch_size=args.embedding_batch_size,
                skip_existing_embeddings=not args.no_skip_existing_embeddings,
                use_contextual_embedding_text=not args.raw_embedding_text,
                dry_run=args.dry_run,
            )
            res = ingestion_actor.ask(msg)
            if res.status != STATUS_OK or res.value is None:
                print(json.dumps(res.model_dump(), indent=2, default=str), file=sys.stderr)
                return 1
            print(res.value.model_dump_json(indent=2))
            return 0
        finally:
            ingestion_actor.stop()
            if embedding_actor is not None:
                embedding_actor.stop()
            if neo4j_client._sync_driver is not None:
                neo4j_client._sync_driver.close()
                neo4j_client._sync_driver = None

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
