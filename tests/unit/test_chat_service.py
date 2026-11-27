"""Unit tests for the grounded Dictionary AI Chat service backed by SearchService."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pytest

from gosmart_dictionary_service.model.llm_config import LLM_Config
from gosmart_dictionary_service.model.search import (
    ExtractedCodeEntity,
    FHIRCoding,
    MatchSource,
    NLUIntent,
    NLUParseResult,
    RelatedConceptHit,
    RelationDirection,
    SearchEntityType,
    SearchHit,
    SearchResponse,
)
from gosmart_dictionary_service.service.chat_service import (
    ChatCitation,
    ChatMessage,
    ChatProviderError,
    ChatRequest,
    DictionaryChatService,
    build_lucene_query,
    extract_dicom_lookup_hints,
    search_hit_to_citation,
)
from tests.conftest import make_test_settings


class _StubLLMConfigService:
    def __init__(self, cfg: Optional[LLM_Config]) -> None:
        self._cfg = cfg

    async def get_llm_config_by_tag(self, *_args: Any, **_kwargs: Any) -> Optional[LLM_Config]:
        return self._cfg


class _StubSearchService:
    """In-memory SearchService stub recording call parameters and returning a scripted SearchResponse."""

    def __init__(self, response: SearchResponse) -> None:
        self._response = response
        self.calls: List[Dict[str, Any]] = []

    async def search(
        self,
        query: str,
        *,
        entity_types: Optional[Sequence[Any]] = None,
        include_related: bool = False,
        related_limit: int = 8,
        embedding_tag: str = "embedding-v1",
        limit: int = 8,
        test_id: Optional[str] = None,
        **_kwargs: Any,
    ) -> SearchResponse:
        self.calls.append(
            {
                "query": query,
                "entity_types": list(entity_types) if entity_types is not None else None,
                "include_related": include_related,
                "related_limit": related_limit,
                "embedding_tag": embedding_tag,
                "limit": limit,
                "test_id": test_id,
            }
        )
        return self._response


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


def test_search_hit_to_citation_maps_clinical_dicom_and_related_drilldowns() -> None:
    clinical_hit = SearchHit(
        id="uuid-loinc-87279-6",
        entity_type=SearchEntityType.CONCEPT,
        score=1.0,
        match_sources=[MatchSource.EXACT_CODE],
        coding=FHIRCoding(system="http://loinc.org", code="87279-6", display="CT Chest for screening"),
        macro="http://loinc.org|87279-6|CT Chest for screening",
        properties={
            "vocabulary": "LOINC",
            "domain": "Measurement",
            "standard": "S",
            "deprecated": False,
            "synonyms": ["Screening chest CT", "Low-dose chest CT"],
        },
        related=[
            RelatedConceptHit(
                id="uuid-snomed-169069000",
                entity_type=SearchEntityType.CONCEPT,
                direction=RelationDirection.FORWARD,
                relation_reason=MatchSource.MAPS_TO,
                relationship_id="Maps to",
                score=0.96,
                coding=FHIRCoding(
                    system="http://snomed.info/sct",
                    code="169069000",
                    display="Computed tomography of chest",
                ),
                macro="http://snomed.info/sct|169069000|Computed tomography of chest",
                properties={"vocabulary": "SNOMED", "domain": "Procedure"},
            )
        ],
    )
    c_cit = search_hit_to_citation(clinical_hit)
    assert c_cit.entity_type == "Concept"
    assert c_cit.identifier == "LOINC:87279-6"
    assert c_cit.vocabulary == "LOINC"
    assert c_cit.domain == "Measurement"
    assert c_cit.standard_concept == "S"
    assert c_cit.source == "exact"
    assert c_cit.macro == "http://loinc.org|87279-6|CT Chest for screening"
    assert c_cit.drilldown_url is not None and "id=uuid-loinc-87279-6" in c_cit.drilldown_url
    assert c_cit.snippet is not None and "Screening chest CT" in c_cit.snippet
    assert len(c_cit.related) == 1
    assert c_cit.related[0].identifier == "SNOMED:169069000"
    assert "id=uuid-snomed-169069000" in c_cit.related[0].drilldown_url

    dicom_hit = SearchHit(
        id="uuid-dicom-00180050",
        entity_type=SearchEntityType.DICOM_ELEMENT,
        score=0.81,
        match_sources=[MatchSource.FULLTEXT, MatchSource.VECTOR, MatchSource.HYBRID],
        coding=FHIRCoding(
            system="http://gosmart.health/terminology/dicom-tag",
            code="00180050",
            display="Slice Thickness",
        ),
        macro="http://gosmart.health/terminology/dicom-tag|00180050|Slice Thickness",
        properties={
            "tag": "(0018,0050)",
            "tag_hex": "00180050",
            "keyword": "SliceThickness",
            "vr": "DS",
            "vm": "1",
            "retired": False,
            "description": "Nominal slice thickness, in mm.",
        },
    )
    d_cit = search_hit_to_citation(dicom_hit)
    assert d_cit.identifier == "(0018,0050)"
    assert d_cit.keyword == "SliceThickness"
    assert d_cit.vr == "DS"
    assert d_cit.vm == "1"
    assert d_cit.source == "hybrid"
    assert d_cit.drilldown_url is not None and "id=uuid-dicom-00180050" in d_cit.drilldown_url


@pytest.mark.anyio
async def test_chat_service_delegates_to_unified_search_service_and_propagates_nlu_and_warnings() -> None:
    scripted_search = SearchResponse(
        query="LOINC 87279-6 and RxNorm 12345",
        nlu=NLUParseResult(
            raw_query="LOINC 87279-6 and RxNorm 12345",
            normalized_query="LOINC 87279-6 and RxNorm 12345",
            rewritten_query="CT Chest for screening",
            synthesized_phrases=["thoracic CT screening"],
            intent=NLUIntent.SEARCH,
            extracted_codes=[
                ExtractedCodeEntity(
                    vocabulary_id="LOINC",
                    code="87279-6",
                    raw_span="LOINC 87279-6",
                    resolved=True,
                    resolved_display="CT Chest for screening",
                    is_schema_loaded=True,
                )
            ],
            unsupported_vocabularies=["RXNORM"],
            provider="gosmart-nlu",
        ),
        total_hits=2,
        hits=[
            SearchHit(
                id="uuid-loinc-87279-6",
                entity_type=SearchEntityType.CONCEPT,
                score=1.0,
                match_sources=[MatchSource.EXACT_CODE],
                coding=FHIRCoding(system="http://loinc.org", code="87279-6", display="CT Chest for screening"),
                macro="http://loinc.org|87279-6|CT Chest for screening",
                properties={"vocabulary": "LOINC", "domain": "Measurement", "standard": "S"},
            ),
            SearchHit(
                id="uuid-dicom-00180050",
                entity_type=SearchEntityType.DICOM_ELEMENT,
                score=0.77,
                match_sources=[MatchSource.HYBRID],
                coding=FHIRCoding(
                    system="http://gosmart.health/terminology/dicom-tag",
                    code="00180050",
                    display="Slice Thickness",
                ),
                macro="http://gosmart.health/terminology/dicom-tag|00180050|Slice Thickness",
                properties={"tag": "(0018,0050)", "keyword": "SliceThickness", "vr": "DS", "vm": "1"},
            ),
        ],
        warnings=["Coding schema 'RXNORM' is not loaded in the active Neo4j database (loaded: DICOM, LOINC, SNOMED)."],
        duration_ms=14.8,
    )
    stub_search = _StubSearchService(scripted_search)
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
        return f"LOINC 87279-6 is CT Chest for screening (answered by {cfg.model})."

    svc = DictionaryChatService(
        llm_config_service=_StubLLMConfigService(inference_cfg),  # type: ignore[arg-type]
        search_service=stub_search,  # type: ignore[arg-type]
        settings=make_test_settings(),
        generate_fn=_fake_generate,
    )

    resp = await svc.chat(
        ChatRequest(
            message="LOINC 87279-6 and RxNorm 12345",
            history=[ChatMessage(role="user", content="Hi"), ChatMessage(role="assistant", content="Hello!")],
            include_context=True,
            context_limit=6,
            scope="clinical",
            include_related=True,
            test_id="unit_chat_test",
        )
    )

    assert len(stub_search.calls) == 1
    assert stub_search.calls[0]["entity_types"] == ["clinical"]
    assert stub_search.calls[0]["include_related"] is True
    assert stub_search.calls[0]["limit"] == 6
    assert stub_search.calls[0]["test_id"] == "unit_chat_test"

    assert resp.provider == "ollama"
    assert resp.model == "gemma4:e2b"
    assert resp.nlu is not None
    assert resp.nlu.intent == "search"
    assert resp.nlu.rewritten_query == "CT Chest for screening"
    assert resp.nlu.extracted_codes == ["LOINC:87279-6"]
    assert resp.search_duration_ms == 14.8
    assert len(resp.warnings) == 1
    assert "RXNORM" in resp.warnings[0]
    assert len(resp.citations) == 2
    assert resp.citations[0].identifier == "LOINC:87279-6"
    assert resp.citations[1].identifier == "(0018,0050)"

    # Verify NLU metadata, warnings, clinical macros, and history were injected into prompt
    assert "### NLU Routing Metadata" in captured_prompt["prompt"]
    assert "### Dictionary Warnings & Guardrails" in captured_prompt["prompt"]
    assert "RXNORM" in captured_prompt["prompt"]
    assert "http://loinc.org|87279-6|CT Chest for screening" in captured_prompt["prompt"]
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
    assert resp.nlu is None
    assert resp.reply == "General answer without context."

    # Custom search_fn automatically gets a fallback drilldown_url populated when include_context=True
    resp_with_custom = await svc_no_ctx.chat(ChatRequest(message="Explain (0010,0010)", include_context=True))
    assert len(resp_with_custom.citations) == 1
    assert resp_with_custom.citations[0].drilldown_url is not None
    assert resp_with_custom.citations[0].drilldown_url.startswith("/explore?")

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
