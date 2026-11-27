"""Unit tests for ClinicalNLUService (trained spaCy bundle, schema checks, reverse-lookup, and LLM synthesis)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

import pytest

from gosmart_dictionary_service.core.config import Settings
from gosmart_dictionary_service.model.llm_config import LLM_Config
from gosmart_dictionary_service.model.search import NLUIntent
from gosmart_dictionary_service.service.nlu_service import (
    ClinicalNLUService,
    NLUParserProtocol,
    parse_synthesized_phrases,
)


class _MockNeo4jClient:
    """In-memory Neo4j fake supporting schema inspection and exact reverse-lookups."""

    def __init__(
        self,
        *,
        code_systems: Optional[List[str]] = None,
        has_dicom: bool = False,
        concepts: Optional[Dict[tuple[str, str], Dict[str, Any]]] = None,
        dicom_elements: Optional[Dict[str, Dict[str, Any]]] = None,
        dicom_sops: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.code_systems = code_systems
        self.has_dicom = has_dicom
        self.concepts = concepts or {}
        self.dicom_elements = dicom_elements or {}
        self.dicom_sops = dicom_sops or {}
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        del database
        params = dict(parameters or {})
        self.calls.append((query, params))

        if "collect(DISTINCT cs.vocabulary_id) AS code_systems" in query:
            if self.code_systems is None and not self.has_dicom:
                return []
            return [{"code_systems": list(self.code_systems or []), "has_dicom": self.has_dicom}]

        if "MATCH (c:Concept {vocabulary_id: $vocab, code: $code})" in query:
            key = (str(params.get("vocab")), str(params.get("code")))
            hit = self.concepts.get(key)
            return [hit] if hit is not None else []

        if "MATCH (e:DicomElement)" in query:
            tag_hex = str(params.get("tag_hex") or "")
            raw = str(params.get("raw") or "").lower()
            if tag_hex in self.dicom_elements:
                return [self.dicom_elements[tag_hex]]
            for elem in self.dicom_elements.values():
                if str(elem.get("keyword") or "").lower() == raw:
                    return [elem]
            return []

        if "MATCH (s:DicomSOPClass {sop_class_uid: $uid})" in query:
            uid = str(params.get("uid") or "")
            sop = self.dicom_sops.get(uid)
            return [sop] if sop is not None else []

        return []


def test_clinical_nlu_implements_protocol_and_loads_trained_bundle() -> None:
    client = _MockNeo4jClient(
        code_systems=["LOINC", "SNOMED", "ICD10CM", "UCUM"],
        has_dicom=True,
        concepts={
            ("LOINC", "87279-6"): {
                "vocabulary_id": "LOINC",
                "code": "87279-6",
                "display": "CT Chest for screening",
                "domain_id": "Procedure",
                "definition": "Chest CT screening study",
            }
        },
    )
    svc = ClinicalNLUService(
        client=client,  # type: ignore[arg-type]
        synthesize_fn=lambda seed, ctx, cfg, n: ["thoracic CT screening", "low dose chest computed tomography"],
    )
    assert isinstance(svc, NLUParserProtocol)

    res = svc.parse("What is LOINC 87279-6?")
    assert res.provider == "gosmart-nlu"
    assert res.intent == NLUIntent.EXACT_LOOKUP
    assert res.nlu_confidence is not None and res.nlu_confidence > 0.85
    assert len(res.extracted_codes) == 1
    assert res.extracted_codes[0].vocabulary_id == "LOINC"
    assert res.extracted_codes[0].code == "87279-6"
    assert res.extracted_codes[0].resolved is True
    assert res.extracted_codes[0].resolved_display == "CT Chest for screening"
    assert res.rewritten_query == "CT Chest for screening"
    assert res.synthesized_phrases == ["thoracic CT screening", "low dose chest computed tomography"]
    assert res.warnings == []


def test_multi_code_reverse_lookup_and_right_to_left_query_rewrite() -> None:
    client = _MockNeo4jClient(
        code_systems=["LOINC", "ICD10CM", "UCUM"],
        has_dicom=True,
        concepts={
            ("LOINC", "87279-6"): {
                "vocabulary_id": "LOINC",
                "code": "87279-6",
                "display": "CT Chest for screening",
                "domain_id": "Procedure",
                "definition": None,
            },
            ("ICD10CM", "E11.9"): {
                "vocabulary_id": "ICD10CM",
                "code": "E11.9",
                "display": "Type 2 diabetes mellitus without complications",
                "domain_id": "Condition",
                "definition": None,
            },
            ("UCUM", "mg/dL"): {
                "vocabulary_id": "UCUM",
                "code": "mg/dL",
                "display": "Milligram per deciliter",
                "domain_id": "Unit",
                "definition": None,
            },
        },
        dicom_elements={
            "00180050": {
                "tag": "(0018,0050)",
                "tag_hex": "00180050",
                "keyword": "SliceThickness",
                "name": "Slice Thickness",
                "vr": "DS",
                "description": "Nominal reconstructed slice thickness, in mm.",
            }
        },
    )
    captured_context: List[str] = []

    def _fake_synth(seed: str, ctx: Sequence[str], cfg: LLM_Config, n: int) -> Sequence[str]:
        del cfg, n
        captured_context.extend(ctx)
        return [f"{seed} imaging protocol"]

    svc = ClinicalNLUService(client=client, synthesize_fn=_fake_synth)  # type: ignore[arg-type]

    # 1) Multi-code clinical + DICOM in one utterance
    res = svc.parse("Compare LOINC 87279-6 and DICOM (0018,0050)")
    assert res.intent == NLUIntent.SEARCH
    assert [(e.vocabulary_id, e.code) for e in res.extracted_codes] == [
        ("LOINC", "87279-6"),
        ("DICOM_TAG", "00180050"),
    ]
    assert res.rewritten_query == "CT Chest for screening and Slice Thickness"
    assert len(captured_context) == 2
    assert any("SliceThickness" in c for c in captured_context)

    # 2) ICD-10 -> ICD10CM fallback + UCUM case-sensitive fallback
    res_icd = svc.parse("ICD-10 E11.9", expand_with_llm=False)
    assert res_icd.extracted_codes[0].vocabulary_id == "ICD10CM"
    assert res_icd.extracted_codes[0].resolved is True
    assert res_icd.rewritten_query == "Type 2 diabetes mellitus without complications"

    res_ucum = svc.parse("UCUM mg/dL", expand_with_llm=False)
    assert res_ucum.extracted_codes[0].vocabulary_id == "UCUM"
    assert res_ucum.extracted_codes[0].code == "mg/dL"
    assert res_ucum.rewritten_query == "Milligram per deciliter"


def test_dynamic_schema_alerts_for_demo_subset_and_unsupported_vocabularies() -> None:
    # Simulate a demo Neo4j database where ONLY DICOM and LOINC are loaded
    demo_client = _MockNeo4jClient(
        code_systems=["LOINC"],
        has_dicom=True,
        concepts={},
    )
    synth_calls: List[str] = []
    svc = ClinicalNLUService(
        client=demo_client,  # type: ignore[arg-type]
        synthesize_fn=lambda seed, ctx, cfg, n: synth_calls.append(seed) or ["chest CT scan"],  # type: ignore[func-returns-value]
    )

    # 1) Explicit code in SNOMED (unloaded in this demo DB)
    res_snomed = svc.parse("What is SNOMED 233604007?")
    assert res_snomed.unsupported_vocabularies == ["SNOMED"]
    assert res_snomed.extracted_codes[0].is_schema_loaded is False
    assert res_snomed.extracted_codes[0].resolved is False
    assert res_snomed.rewritten_query == ""
    assert synth_calls == []  # Unloaded code stripped -> empty seed -> LLM not invoked on raw number
    assert any("SNOMED" in w and "DICOM, LOINC" in w for w in res_snomed.warnings)

    # 2) Completely unsupported schema (RxNorm)
    res_rx = svc.parse("Look up RxNorm 1191")
    assert res_rx.unsupported_vocabularies == ["RxNorm"]
    assert any("RxNorm" in w for w in res_rx.warnings)

    # 3) Free-text mention of unloaded schema ("Find SNOMED codes for chest CT"):
    #    Alerts user that SNOMED is not loaded, clears inferred_vocabularies so LOINC/DICOM can still match!
    res_free = svc.parse("Find SNOMED codes for chest CT", expand_with_llm=False)
    assert res_free.unsupported_vocabularies == ["SNOMED"]
    assert res_free.inferred_vocabularies == []
    assert any("SNOMED" in w for w in res_free.warnings)


def test_unknown_code_in_loaded_schema_is_skipped_and_never_sent_to_llm() -> None:
    client = _MockNeo4jClient(
        code_systems=["LOINC"],
        has_dicom=True,
        concepts={
            ("LOINC", "87279-6"): {
                "vocabulary_id": "LOINC",
                "code": "87279-6",
                "display": "CT Chest for screening",
                "domain_id": "Procedure",
                "definition": None,
            }
        },
    )
    seeds_seen: List[str] = []

    def _record_synth(seed: str, ctx: Sequence[str], cfg: LLM_Config, n: int) -> Sequence[str]:
        del ctx, cfg, n
        seeds_seen.append(seed)
        return ["thoracic CT screening"]

    svc = ClinicalNLUService(client=client, synthesize_fn=_record_synth)  # type: ignore[arg-type]

    # Pure unknown code -> skipped completely, LLM NOT called
    res_unknown = svc.parse("What is LOINC 999999-9?")
    assert res_unknown.extracted_codes[0].resolved is False
    assert res_unknown.rewritten_query == ""
    assert res_unknown.synthesized_phrases == []
    assert seeds_seen == []
    assert any("999999-9" in w and "not found" in w for w in res_unknown.warnings)

    # Mixed known + unknown code -> unknown code stripped, known display preserved
    res_mixed = svc.parse("Compare LOINC 87279-6 and LOINC 999999-9")
    assert res_mixed.rewritten_query == "CT Chest for screening"
    assert seeds_seen == ["CT Chest for screening"]
    assert res_mixed.synthesized_phrases == ["thoracic CT screening"]
    assert any("999999-9" in w for w in res_mixed.warnings)


def test_parse_synthesized_phrases_and_offline_llm_fallback() -> None:
    # JSON array inside markdown fence + <think> block + duplicate filtering
    raw = (
        "<think>Thinking about synonyms...</think>\n"
        "```json\n"
        '["CT Chest for screening", "Thoracic computed tomography", "Low-dose chest CT", "Lung cancer CT screen"]\n'
        "```"
    )
    parsed = parse_synthesized_phrases(
        raw,
        exclude=["CT Chest for screening"],
        max_phrases=3,
    )
    assert parsed == [
        "Thoracic computed tomography",
        "Low-dose chest CT",
        "Lung cancer CT screen",
    ]

    # Bullet-list fallback
    bullets = "Here are alternates:\n1. Slice thickness attribute\n- Reconstructed image thickness\n"
    assert parse_synthesized_phrases(bullets, max_phrases=3) == [
        "Slice thickness attribute",
        "Reconstructed image thickness",
    ]

    # Offline LLM provider graceful warning fallback
    def _failing_synth(seed: str, ctx: Sequence[str], cfg: LLM_Config, n: int) -> Sequence[str]:
        raise ConnectionError("Ollama gemma4:e2b unreachable")

    settings = Settings(NLU_LLM_EXPANSION_ENABLED=True, NLU_MAX_ALTERNATE_PHRASES=3)
    svc = ClinicalNLUService(
        client=_MockNeo4jClient(),  # type: ignore[arg-type]
        settings=settings,
        synthesize_fn=_failing_synth,
    )
    res = asyncio.run(svc.parse_async("acute pneumonia"))
    assert res.synthesized_phrases == []
    assert len(res.warnings) == 1
    assert "ConnectionError" in res.warnings[0]


@pytest.mark.parametrize(
    ("utterance", "expected_intent"),
    [
        ("Find something related to LOINC 87279-6 with contrast", NLUIntent.RELATED_LOOKUP),
        ("Map ICD-10-CM E11.9 to SNOMED", NLUIntent.CROSSWALK),
        ("Show subtypes of SNOMED 233604007", NLUIntent.EXPAND_SUBTYPES),
    ],
)
def test_ontology_intents_skip_default_llm_expansion(utterance: str, expected_intent: NLUIntent) -> None:
    called = False

    def _spy(seed: str, ctx: Sequence[str], cfg: LLM_Config, n: int) -> Sequence[str]:
        nonlocal called
        called = True
        return ["should not run"]

    svc = ClinicalNLUService(client=_MockNeo4jClient(), synthesize_fn=_spy)  # type: ignore[arg-type]
    res = svc.parse(utterance)
    assert res.intent == expected_intent
    assert called is False
