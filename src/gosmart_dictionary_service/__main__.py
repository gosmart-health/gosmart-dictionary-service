"""CLI and module entrypoint for GoSmart DS (``gosmart-ds`` / ``python -m gosmart_dictionary_service``)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Optional

from gosmart_dictionary_service.actor.dicom_ingestion_actor import DicomIngestionActor
from gosmart_dictionary_service.actor.embedding_actor import EmbeddingActor
from gosmart_dictionary_service.core.logging import setup_logging
from gosmart_dictionary_service.db.neo4j import neo4j_client
from gosmart_dictionary_service.main import run
from gosmart_dictionary_service.model.actor_message import DicomIngestMessage
from gosmart_dictionary_service.model.llm_config import DEFAULT_EMBEDDING_CONFIG_TAG
from gosmart_dictionary_service.model.result import STATUS_OK
from gosmart_dictionary_service.service.system_service import system_service


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gosmart-ds",
        description="GoSmart Health Dictionary Service CLI (defaults to launching the FastAPI server).",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve (default when no subcommand given)
    subparsers.add_parser("serve", help="Start the FastAPI HTTP server with uvicorn")

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

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for GoSmart DS."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command in (None, "serve"):
        run()
        return 0

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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
