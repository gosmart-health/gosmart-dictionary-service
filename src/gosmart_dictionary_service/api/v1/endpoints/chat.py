"""REST API endpoint for the grounded Dictionary AI Chat assistant."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from gosmart_dictionary_service.core.auth_dependency import get_current_user
from gosmart_dictionary_service.service.chat_service import (
    ChatProviderError,
    ChatRequest,
    ChatResponse,
    chat_service,
)

router = APIRouter(prefix="/chat", tags=["AI Chat"])


@router.post("", response_model=ChatResponse)
async def post_chat_message(
    payload: ChatRequest,
    _current_user: Dict[str, Any] = Depends(get_current_user),
) -> ChatResponse:
    """Answer a terminology/DICOM question using Genkit grounded with Neo4j dictionary retrieval."""
    try:
        return await chat_service.chat(payload)
    except ChatProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
