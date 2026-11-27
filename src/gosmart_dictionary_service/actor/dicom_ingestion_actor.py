"""Pykka ThreadingActor orchestrating DICOM dictionary ingestion and optional embedding dispatch."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pykka
from pydantic import ValidationError

from gosmart_dictionary_service.core.logging import get_logger
from gosmart_dictionary_service.ingestor.dicom_ingestor import (
    DicomIngesterService,
    DicomIngestionOptions,
    DicomIngestionReport,
    dicom_ingester_service,
)
from gosmart_dictionary_service.model.actor_message import (
    DicomCleanupMessage,
    DicomIngestMessage,
    EmbeddingBatchMessage,
    EmbeddingItem,
)
from gosmart_dictionary_service.model.result import (
    STATUS_BAD_REQUEST,
    STATUS_INTERNAL_ERROR,
    STATUS_OK,
    Result,
)
from gosmart_dictionary_service.service.embedding_service import EmbeddingBatchReport

logger = get_logger("gosmart_ds.actor.dicom_ingestion")


class DicomIngestionActor(pykka.ThreadingActor):
    """Background worker actor running DICOM JSON parsing, Neo4j ingestion, and embedding dispatch."""

    def __init__(
        self,
        ingester: Optional[DicomIngesterService] = None,
        embedding_actor_ref: Optional[pykka.ActorRef] = None,
    ) -> None:
        super().__init__()
        self._ingester = ingester or dicom_ingester_service
        self._embedding_actor_ref = embedding_actor_ref

    def _dispatch_embeddings_to_actor(
        self,
        items: List[EmbeddingItem],
        options: DicomIngestionOptions,
    ) -> EmbeddingBatchReport:
        assert self._embedding_actor_ref is not None
        msg = EmbeddingBatchMessage(
            items=items,
            tag=options.embedding_tag,
            group_code=options.embedding_group_code,
            group_name=options.embedding_group_name,
            batch_size=options.embedding_batch_size,
            skip_existing=options.skip_existing_embeddings,
            test_id=options.test_id,
            acted_by=options.acted_by,
        )
        res: Result[EmbeddingBatchReport] = self._embedding_actor_ref.ask(msg)
        if res.status != STATUS_OK or res.value is None:
            raise RuntimeError(f"EmbeddingActor failed (status={res.status}): {res.message}")
        return res.value

    def on_receive(self, message: Any) -> Result[Any]:
        # Handle cleanup requests
        if isinstance(message, DicomCleanupMessage) or (
            isinstance(message, dict) and message.get("type") == "CLEANUP_TEST_DATA"
        ):
            try:
                cleanup_req = (
                    message if isinstance(message, DicomCleanupMessage) else DicomCleanupMessage.model_validate(message)
                )
            except ValidationError as exc:
                return Result(
                    status=STATUS_BAD_REQUEST,
                    message=f"Invalid DicomCleanupMessage: {exc}",
                    value_type="dict",
                    value=None,
                )
            try:
                payload: Dict[str, Any] = self._ingester.cleanup_test_data_sync(cleanup_req.test_id)
                return Result(
                    status=STATUS_OK,
                    message=payload.get("message", "Test cleanup completed."),
                    value_type="dict",
                    value=payload,
                )
            except Exception as exc:
                logger.error("dicom_cleanup_actor_failed", error=str(exc), exc_info=True)
                return Result(
                    status=STATUS_INTERNAL_ERROR,
                    message=f"DICOM test cleanup failed: {exc}",
                    value_type="dict",
                    value=None,
                )

        # Handle DICOM ingestion requests
        try:
            if isinstance(message, DicomIngestMessage):
                ingest_req = message
            elif isinstance(message, dict):
                ingest_req = DicomIngestMessage.model_validate(message)
            else:
                return Result(
                    status=STATUS_BAD_REQUEST,
                    message=f"Unsupported message type for DicomIngestionActor: {type(message).__name__}",
                    value_type="DicomIngestionReport",
                    value=None,
                )
        except ValidationError as exc:
            logger.warning("dicom_ingestion_actor_invalid_message", error=str(exc))
            return Result(
                status=STATUS_BAD_REQUEST,
                message=f"Invalid DicomIngestMessage: {exc}",
                value_type="DicomIngestionReport",
                value=None,
            )

        try:
            options = DicomIngestionOptions(
                source_dir=ingest_req.source_dir,
                batch_size=ingest_req.batch_size,
                acted_by=ingest_req.acted_by,
                test_id=ingest_req.test_id,
                ensure_schema=ingest_req.ensure_schema,
                bootstrap_system=ingest_req.bootstrap_system,
                include_references=ingest_req.include_references,
                include_reference_edges=ingest_req.include_reference_edges,
                dedupe_reference_edges_per_owner=ingest_req.dedupe_reference_edges_per_owner,
                compute_embeddings=ingest_req.compute_embeddings,
                embedding_tag=ingest_req.embedding_tag,
                embedding_group_code=ingest_req.embedding_group_code,
                embedding_group_name=ingest_req.embedding_group_name,
                embedding_batch_size=ingest_req.embedding_batch_size,
                skip_existing_embeddings=ingest_req.skip_existing_embeddings,
                use_contextual_embedding_text=ingest_req.use_contextual_embedding_text,
                dry_run=ingest_req.dry_run,
            )
            callback = self._dispatch_embeddings_to_actor if self._embedding_actor_ref is not None else None
            report: DicomIngestionReport = self._ingester.ingest_sync(options, embed_callback=callback)
            return Result(
                status=STATUS_OK,
                message=(
                    f"DICOM ingestion completed (dry_run={report.dry_run}, "
                    f"nodes={sum(report.node_counts.values())}, "
                    f"relations={sum(report.relation_counts.values())})."
                ),
                value_type="DicomIngestionReport",
                value=report,
            )
        except Exception as exc:
            logger.error("dicom_ingestion_actor_failed", error=str(exc), exc_info=True)
            return Result(
                status=STATUS_INTERNAL_ERROR,
                message=f"DICOM ingestion failed: {exc}",
                value_type="DicomIngestionReport",
                value=None,
            )


__all__ = ["DicomIngestionActor"]
