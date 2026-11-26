"""Unit tests for SearchService (mocked Neo4j and EmbeddingService)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

import pytest

from gosmart_dictionary_service.model.llm_config import LLM_Config
from gosmart_dictionary_service.model.search import (
    MatchSource,
    NLUIntent,
    RelationDirection,
    ResponseFormat,
    SearchEntityType,
    normalize_entity_types,
)
from gosmart_dictionary_service.service.search_service import (
    SearchService,
    build_lucene_query,
    cosine_similarity,
)


class _FakeNeo4jClient:
    """In-memory query dispatcher for unit-testing SearchService without a live database."""

    def __init__(self, responses: Dict[str, List[Dict[str, Any]]]) -> None:
        self.responses = responses
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params = dict(parameters or {})
        self.calls.append((query, params))
        for marker, rows in self.responses.items():
            if marker in query:
                return list(rows)
        return []


class _FakeLLMConfigService:
    def __init__(self, config: Optional[LLM_Config]) -> None:
        self._config = config

    async def get_llm_config_by_tag(self, tag: str, *, group_code: str = "") -> Optional[LLM_Config]:
        return self._config


class _FakeEmbeddingService:
    def __init__(
        self,
        *,
        vectors_by_substring: Optional[Dict[str, List[float]]] = None,
        fail: bool = False,
    ) -> None:
        self._vectors_by_substring = vectors_by_substring or {}
        self._fail = fail
        self.calls: List[List[str]] = []

    async def compute_vectors(self, texts: Sequence[str], config: LLM_Config) -> List[List[float]]:
        self.calls.append(list(texts))
        if self._fail:
            raise RuntimeError("Ollama connection refused")
        out: List[List[float]] = []
        for text in texts:
            matched: Optional[List[float]] = None
            for needle, vec in self._vectors_by_substring.items():
                if needle.lower() in text.lower():
                    matched = vec
                    break
            out.append(matched if matched is not None else [0.1, 0.9, 0.0])
        return out


def _make_dummy_llm_config() -> LLM_Config:
    return LLM_Config(
        tag="embedding-v1",
        provider="ollama",
        model="embeddinggemma-2:270m",
        url="http://localhost:11434",
        dimensions=3,
    )


def test_lucene_escaping_and_cosine_helpers() -> None:
    assert build_lucene_query("mg/dL") == r"mg\/dL"
    assert build_lucene_query("(0018,0050)") == r"\(0018,0050\)"
    assert build_lucene_query("acute pneumonia?") == r"acute pneumonia\?"
    assert build_lucene_query("   ") == ""

    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([], []) == 0.0


def test_normalize_entity_types_aliases() -> None:
    clinical = normalize_entity_types(["clinical"])
    assert clinical == [SearchEntityType.CONCEPT]

    dicom = normalize_entity_types(["dicom"])
    assert dicom is not None
    assert SearchEntityType.DICOM_ELEMENT in dicom
    assert SearchEntityType.DICOM_CIOD in dicom

    explicit = normalize_entity_types([SearchEntityType.CONCEPT, "DicomElement", "Concept"])
    assert explicit == [SearchEntityType.CONCEPT, SearchEntityType.DICOM_ELEMENT]


def test_search_clinical_parent_convergence_and_embedding_rerank() -> None:
    fake_client = _FakeNeo4jClient(
        {
            "concept_fulltext_idx": [
                {
                    "id": "cid-aspiration",
                    "vocabulary_id": "SNOMED",
                    "code": "35031000119100",
                    "display": "Acute aspiration pneumonia",
                    "system_uri": "http://snomed.info/sct",
                    "coding_macro": "http://snomed.info/sct|35031000119100|Acute aspiration pneumonia",
                    "domain_id": "Condition",
                    "concept_class_id": "Disorder",
                    "standard_concept": "S",
                    "invalid_reason": None,
                    "is_deprecated": False,
                    "synonyms": ["Acute aspiration pneumonia (disorder)"],
                    "lexical_score": 16.0,
                },
                {
                    "id": "cid-cocci",
                    "vocabulary_id": "SNOMED",
                    "code": "184431000119108",
                    "display": "Acute pneumonia caused by coccidioidomycosis",
                    "system_uri": "http://snomed.info/sct",
                    "coding_macro": (
                        "http://snomed.info/sct|184431000119108|Acute pneumonia caused by coccidioidomycosis"
                    ),
                    "domain_id": "Condition",
                    "concept_class_id": "Disorder",
                    "standard_concept": "S",
                    "invalid_reason": None,
                    "is_deprecated": False,
                    "synonyms": [],
                    "lexical_score": 15.0,
                },
            ],
            "count(DISTINCT sid) AS convergence": [
                {
                    "id": "cid-pneumonia",
                    "vocabulary_id": "SNOMED",
                    "code": "233604007",
                    "display": "Pneumonia",
                    "system_uri": "http://snomed.info/sct",
                    "coding_macro": "http://snomed.info/sct|233604007|Pneumonia",
                    "domain_id": "Condition",
                    "concept_class_id": "Disorder",
                    "standard_concept": "S",
                    "invalid_reason": None,
                    "is_deprecated": False,
                    "synonyms": ["Pneumonia (disorder)"],
                    "convergence": 2,
                }
            ],
            "collect(DISTINCT r.relationship_id": [
                {"id": "cid-pneumonia", "attrs": ["Has finding site: Structure of parenchyma of lung"]},
                {"id": "cid-aspiration", "attrs": []},
                {"id": "cid-cocci", "attrs": []},
            ],
        }
    )
    fake_emb = _FakeEmbeddingService(
        vectors_by_substring={
            "233604007": [0.99, 0.14, 0.0],  # Canonical parent semantically closest!
            "35031000119100": [0.50, 0.86, 0.0],
            "184431000119108": [0.30, 0.95, 0.0],
            "acute pneumonia": [1.0, 0.0, 0.0],
        }
    )
    svc = SearchService(client=fake_client, embedding_service=fake_emb)  # type: ignore[arg-type]
    svc.llm_config_service = _FakeLLMConfigService(_make_dummy_llm_config())  # type: ignore[assignment]

    resp = asyncio.run(svc.search("acute pneumonia", entity_types=[SearchEntityType.CONCEPT], limit=5))

    assert resp.nlu.intent == NLUIntent.SEARCH
    assert resp.warnings == []
    assert resp.total_hits == 3
    assert resp.hits[0].coding.code == "233604007"
    assert MatchSource.IS_A_PARENT in resp.hits[0].match_sources
    assert MatchSource.HYBRID in resp.hits[0].match_sources
    assert resp.hits[0].vector_score is not None and resp.hits[0].vector_score > 0.95

    # Verify FHIR and Compact projections
    fhir_wire = resp.to_wire(ResponseFormat.FHIR)
    assert fhir_wire["hits"][0]["coding"]["system"] == "http://snomed.info/sct"
    compact_wire = resp.to_wire(ResponseFormat.COMPACT)
    assert compact_wire["hits"][0]["macro"] == "http://snomed.info/sct|233604007|Pneumonia"
    assert "nlu" not in compact_wire


def test_search_graceful_fallback_when_embedding_provider_offline() -> None:
    fake_client = _FakeNeo4jClient(
        {
            "concept_fulltext_idx": [
                {
                    "id": "cid-1",
                    "vocabulary_id": "SNOMED",
                    "code": "233604007",
                    "display": "Pneumonia",
                    "system_uri": "http://snomed.info/sct",
                    "coding_macro": "http://snomed.info/sct|233604007|Pneumonia",
                    "domain_id": "Condition",
                    "concept_class_id": "Disorder",
                    "standard_concept": "S",
                    "invalid_reason": None,
                    "is_deprecated": False,
                    "synonyms": [],
                    "lexical_score": 14.5,
                }
            ]
        }
    )
    fake_emb = _FakeEmbeddingService(fail=True)
    svc = SearchService(client=fake_client, embedding_service=fake_emb)  # type: ignore[arg-type]
    svc.llm_config_service = _FakeLLMConfigService(_make_dummy_llm_config())  # type: ignore[assignment]

    resp = asyncio.run(svc.search("pneumonia", entity_types=[SearchEntityType.CONCEPT], rerank_with_embeddings=True))

    assert resp.total_hits == 1
    assert resp.hits[0].coding.code == "233604007"
    assert resp.hits[0].vector_score is None
    assert len(resp.warnings) == 1
    assert "falling back" in resp.warnings[0].lower()


def test_search_related_lookup_aggregates_forward_reverse_and_shared_axes() -> None:
    fake_client = _FakeNeo4jClient(
        {
            "OPTIONAL MATCH (c)-[:BELONGS_TO_SYSTEM]->(cs:CodeSystem)": [
                {
                    "id": "cid-loinc-seed",
                    "vocabulary_id": "LOINC",
                    "code": "87279-6",
                    "display": "CT Chest for screening",
                    "system_uri": "http://loinc.org",
                    "system_version": "LOINC 2.82",
                    "coding_macro": "http://loinc.org|87279-6|CT Chest for screening",
                    "domain_id": "Procedure",
                    "concept_class_id": "Clinical Observation",
                    "standard_concept": "S",
                    "invalid_reason": None,
                    "is_deprecated": False,
                    "synonyms": [],
                }
            ],
            "'forward' AS direction": [
                {
                    "id": "cid-axis-ct",
                    "direction": "forward",
                    "rel_type": "HAS_ATTRIBUTE",
                    "relationship_id": "Has method",
                    "vocabulary_id": "LOINC",
                    "code": "LP6207-7",
                    "display": "Computed tomography",
                    "system_uri": "http://loinc.org",
                    "domain_id": "Observation",
                    "standard_concept": None,
                    "invalid_reason": None,
                }
            ],
            "'reverse' AS direction": [
                {
                    "id": "cid-panel",
                    "direction": "reverse",
                    "rel_type": "HAS_ATTRIBUTE",
                    "relationship_id": "Panel contains",
                    "vocabulary_id": "LOINC",
                    "code": "PANEL-1",
                    "display": "Chest CT Screening Panel",
                    "system_uri": "http://loinc.org",
                    "domain_id": "Procedure",
                    "standard_concept": "S",
                    "invalid_reason": None,
                }
            ],
            "count(DISTINCT axis) AS shared_count": [
                {
                    "id": "cid-sibling-ct",
                    "shared_count": 3,
                    "shared_axes": ["Computed tomography", "Chest", "Doc"],
                    "vocabulary_id": "LOINC",
                    "code": "24627-2",
                    "display": "CT Chest WO contrast",
                    "system_uri": "http://loinc.org",
                    "domain_id": "Procedure",
                    "standard_concept": "S",
                    "invalid_reason": None,
                }
            ],
        }
    )
    svc = SearchService(client=fake_client, embedding_service=_FakeEmbeddingService())  # type: ignore[arg-type]

    resp = svc.search_sync(
        "Find something related to LOINC 87279-6",
        rerank_with_embeddings=False,
    )

    assert resp.nlu.intent == NLUIntent.RELATED_LOOKUP
    assert resp.total_hits == 1
    seed = resp.hits[0]
    assert seed.coding.code == "87279-6"
    assert len(seed.related) == 3

    reasons = {r.relation_reason for r in seed.related}
    assert MatchSource.HAS_ATTRIBUTE in reasons
    assert MatchSource.USED_AS_ATTRIBUTE_BY in reasons
    assert MatchSource.SHARED_ATTRIBUTE in reasons

    shared_hit = next(r for r in seed.related if r.relation_reason == MatchSource.SHARED_ATTRIBUTE)
    assert shared_hit.coding.code == "24627-2"
    assert shared_hit.direction == RelationDirection.BOTH
    assert "Computed tomography" in shared_hit.shared_axes
