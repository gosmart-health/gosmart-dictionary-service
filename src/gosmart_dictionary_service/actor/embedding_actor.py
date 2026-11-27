"""Pykka ThreadingActor for background vector embedding computation and Neo4j registration."""

from __future__ import annotations

from typing import Any, Optional

import pykka
from pydantic import ValidationError

from gosmart_dictionary_service.core.logging import get_logger
from gosmart_dictionary_service.model.actor_message import EmbeddingBatchMessage
from gosmart_dictionary_service.model.result import (
    STATUS_BAD_REQUEST,
    STATUS_INTERNAL_ERROR,
    STATUS_OK,
    Result,
)
from gosmart_dictionary_service.service.embedding_service import (
    EmbeddingBatchReport,
    EmbeddingService,
    embedding_service,
)

logger = get_logger("gosmart_ds.actor.embedding")


class EmbeddingActor(pykka.ThreadingActor):
    """Background worker actor computing L2-normalized embeddings and persisting them in Neo4j."""

    def __init__(self, service: Optional[EmbeddingService] = None) -> None:
        super().__init__()
        self._service = service or embedding_service

    def on_receive(self, message: Any) -> Result[EmbeddingBatchReport]:
        try:
            if isinstance(message, EmbeddingBatchMessage):
                req = message
            elif isinstance(message, dict):
                req = EmbeddingBatchMessage.model_validate(message)
            else:
                return Result(
                    status=STATUS_BAD_REQUEST,
                    message=f"Unsupported message type for EmbeddingActor: {type(message).__name__}",
                    value_type="EmbeddingBatchReport",
                    value=None,
                )
        except ValidationError as exc:
            logger.warning("embedding_actor_invalid_message", error=str(exc))
            return Result(
                status=STATUS_BAD_REQUEST,
                message=f"Invalid EmbeddingBatchMessage: {exc}",
                value_type="EmbeddingBatchReport",
                value=None,
            )

        try:
            report = self._service.embed_items_sync(
                req.items,
                tag=req.tag,
                group_id=req.group_id,
                group_code=req.group_code,
                group_name=req.group_name,
                batch_size=req.batch_size,
                skip_existing=req.skip_existing,
                test_id=req.test_id,
                acted_by=req.acted_by,
            )
            return Result(
                status=STATUS_OK,
                message=(
                    f"Embedded {report.embedded_count}/{report.total_items} items "
                    f"(skipped={report.skipped_existing_count}, persisted={report.persisted_count})."
                ),
                value_type="EmbeddingBatchReport",
                value=report,
            )
        except Exception as exc:
            logger.error("embedding_actor_execution_failed", error=str(exc), exc_info=True)
            return Result(
                status=STATUS_INTERNAL_ERROR,
                message=f"Embedding batch failed: {exc}",
                value_type="EmbeddingBatchReport",
                value=None,
            )


__all__ = ["EmbeddingActor"]
