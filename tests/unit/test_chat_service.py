"""Unit tests for the grounded Dictionary AI Chat service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from gosmart_dictionary_service.model.llm_config import LLM_Config
from gosmart_dictionary_service.service.chat_service import (
    ChatCitation,
    ChatMessage,
    ChatProviderError,
    ChatRequest,
    DictionaryChatService,
    build_lucene_query,
    extract_dicom_lookup_hints,
)
from tests.conftest import make_test_settings


class _MockNeo4jChatClient:
    """Scripted Neo4j client returning DICOM exact/fulltext query hits."""

    def __init__(self, responses: List[List[Dict[str, Any]]]) -> None:
        self._responses = list(responses)
        self.queries: List[str] = []

    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self.queries.append(query)
        if not self._responses:
            return []
        return self._responses.pop(0)


class _StubLLMConfigService:
    def __init__(self, cfg: Optional[LLM_Config]) -> None:
        self._cfg = cfg

    async def get_llm_config_by_tag(self, *_args: Any, **_kwargs: Any) -> Optional[LLM_Config]:
        return self._cfg


def test_dicom_lookup_hints_and_lucene_sanitization() -> None:
    tags, keywords = extract_dicom_lookup_hints("What is (0018,0050) SliceThickness and 00180082?")
    assert tags == ["00180050", "00180082"]
    assert "SliceThickness" in keywords

    lucene = build_lucene_query("Find (0018,0050): slice thickness [mm] & inversion-time?!")
    assert "slice" in lucene.lower()
    assert "thickness" in lucene.lower()
    assert "inversion" in lucene.lower()
    for special in ("(", ")", ":", "[", "]", "?", "!"):
        assert special not in lucene


@pytest.mark.anyio
async def test_chat_service_grounds_prompt_with_neo4j_dicom_citations() -> None:
    mock_db = _MockNeo4jChatClient(
        [
            # Stage 1: Exact DICOM lookup
            [
                {
                    "tag": "(0018,0050)",
                    "tag_hex": "00180050",
                    "name": "Slice Thickness",
                    "keyword": "SliceThickness",
                    "vr": "DS",
                    "vm": "1",
                    "retired": False,
                    "description": "Nominal slice thickness, in mm.",
                }
            ],
            # Stage 2a: Fulltext elements (includes duplicate that must be deduplicated)
            [
                {
                    "tag": "(0018,0050)",
                    "tag_hex": "00180050",
                    "name": "Slice Thickness",
                    "keyword": "SliceThickness",
                    "vr": "DS",
                    "vm": "1",
                    "retired": False,
                    "description": "Nominal slice thickness, in mm.",
                    "score": 4.2,
                },
                {
                    "tag": "(0018,0088)",
                    "tag_hex": "00180088",
                    "name": "Spacing Between Slices",
                    "keyword": "SpacingBetweenSlices",
                    "vr": "DS",
                    "vm": "1",
                    "retired": False,
                    "description": "Spacing between slices, in mm.",
                    "score": 2.9,
                },
            ],
            # Stage 2b: Fulltext structures
            [
                {
                    "label": "DicomModule",
                    "identifier": "image-plane",
                    "name": "Image Plane",
                    "description": "Specifies spatial orientation and slice thickness.",
                    "score": 2.1,
                }
            ],
        ]
    )
    inference_cfg = LLM_Config(
        tag="inference-v1",
        provider="ollama",
        model="gemma4:e2b",
        url="http://localhost:11434",
    )
    captured_prompt: Dict[str, str] = {}

    def _fake_generate(system_prompt: str, prompt: str, cfg: LLM_Config) -> str:
        captured_prompt["system"] = system_prompt
        captured_prompt["prompt"] = prompt
        return f"SliceThickness is (0018,0050) with VR=DS (answered by {cfg.model})."

    svc = DictionaryChatService(
        client=mock_db,  # type: ignore[arg-type]
        llm_config_service=_StubLLMConfigService(inference_cfg),  # type: ignore[arg-type]
        settings=make_test_settings(),
        generate_fn=_fake_generate,
    )

    resp = await svc.chat(
        ChatRequest(
            message="What is DICOM (0018,0050) SliceThickness?",
            history=[ChatMessage(role="user", content="Hi"), ChatMessage(role="assistant", content="Hello!")],
            include_context=True,
            context_limit=5,
        )
    )

    assert resp.provider == "ollama"
    assert resp.model == "gemma4:e2b"
    assert "(0018,0050)" in resp.reply
    # Deduplicated: (0018,0050), (0018,0088), and Image Plane module
    assert len(resp.citations) == 3
    assert resp.citations[0].identifier == "(0018,0050)"
    assert resp.citations[0].source == "exact"
    assert "SliceThickness" in captured_prompt["prompt"]
    assert "### Recent Conversation" in captured_prompt["prompt"]


@pytest.mark.anyio
async def test_chat_service_degrades_gracefully_when_context_disabled_or_provider_fails() -> None:
    svc_no_ctx = DictionaryChatService(
        llm_config_service=_StubLLMConfigService(None),  # type: ignore[arg-type]
        settings=make_test_settings(INFERENCE_PROVIDER="ollama", INFERENCE_MODEL="gemma4:e2b"),
        search_fn=lambda _q, _lim: [
            ChatCitation(entity_type="DicomElement", identifier="(0010,0010)", name="Patient's Name")
        ],
        generate_fn=lambda _sys, _prompt, _cfg: "General answer without context.",
    )
    resp = await svc_no_ctx.chat(ChatRequest(message="Explain DICOM VR.", include_context=False))
    assert resp.citations == []
    assert resp.reply == "General answer without context."

    def _failing_gen(_sys: str, _prompt: str, _cfg: LLM_Config) -> str:
        raise ChatProviderError("Ollama connection refused")

    svc_fail = DictionaryChatService(
        llm_config_service=_StubLLMConfigService(None),  # type: ignore[arg-type]
        settings=make_test_settings(),
        search_fn=lambda _q, _lim: [],
        generate_fn=_failing_gen,
    )
    with pytest.raises(ChatProviderError, match="Ollama connection refused"):
        await svc_fail.chat(ChatRequest(message="Hello"))
