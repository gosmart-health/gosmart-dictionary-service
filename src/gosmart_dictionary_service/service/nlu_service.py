"""Clinical NLU query-parsing protocol, deterministic stub, and trained service implementation.

This module provides:
- ``NLUParserProtocol``: Uniform sync/async parser interface consumed by ``SearchService``.
- ``StubNLUService``: Lightweight regex/heuristic parser with zero external dependencies.
- ``ClinicalNLUService``: Full NLU pipeline backed by the trained ``gosmart-clinical-nlu``
  spaCy model bundle (``nlu_models/intent_models.zip``), dynamic Neo4j schema availability
  alerts, reverse-lookup query rewriting, and Genkit inference LLM phrase synthesis
  (defaulting to Ollama ``gemma4:e2b`` via ``inference-v1``).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
import time
from collections.abc import Awaitable, Callable, Collection, Coroutine, Sequence
from pathlib import Path
from typing import Any, Dict, List, LiteralString, Optional, Protocol, Set, Tuple, TypeVar, Union, runtime_checkable

from genkit import Genkit
from genkit_google_genai import GoogleAI, VertexAI
from genkit_ollama import Ollama
from gosmart_clinical_nlu import IntentModels, load_intent_models, register_intent_spec

from gosmart_dictionary_service.core.config import PROJECT_ROOT, Settings, get_settings
from gosmart_dictionary_service.core.logging import get_logger
from gosmart_dictionary_service.db.neo4j import Neo4jClient, neo4j_client
from gosmart_dictionary_service.model.llm_config import (
    DEFAULT_INFERENCE_CONFIG_TAG,
    LLM_Config,
    build_default_inference_llm_config,
)
from gosmart_dictionary_service.model.search import (
    DICOM_ENTITY_TYPES,
    ExtractedCodeEntity,
    NLUIntent,
    NLUParseResult,
    SearchEntityType,
)
from gosmart_dictionary_service.nlu.find_coding_schema import (
    FIND_CODING_SCHEMA_SPEC,
    normalize_coding_code,
    normalize_coding_schema,
    normalize_dicom_tag,
    normalize_dicom_uid,
)
from gosmart_dictionary_service.service.llm_config_service import LLMConfigService
from gosmart_dictionary_service.service.system_service import DEFAULT_GROUP_CODE, normalize_test_id

logger = get_logger("gosmart_ds.nlu")

_T = TypeVar("_T")

SynthesizePhrasesFn = Callable[
    [str, Sequence[str], LLM_Config, int],
    Union[Sequence[str], Awaitable[Sequence[str]]],
]

# Canonical vocabulary alias normalization (used by StubNLUService for backward compatibility)
_VOCAB_ALIASES: Dict[str, str] = {
    "snomed": "SNOMED",
    "snomed-ct": "SNOMED",
    "snomedct": "SNOMED",
    "sct": "SNOMED",
    "loinc": "LOINC",
    "icd10cm": "ICD10CM",
    "icd-10-cm": "ICD10CM",
    "icd10-cm": "ICD10CM",
    "icd10": "ICD10CM",
    "icd-10": "ICD10CM",
    "ucum": "UCUM",
    "icd11": "ICD11",
    "icd-11": "ICD11",
    "rxnorm": "RxNorm",
    "cpt4": "CPT4",
    "cpt": "CPT4",
    "hcpcs": "HCPCS",
    "dcm": "DCM",
}

# Explicit vocabulary + code patterns (e.g. "LOINC 87279-6", "SNOMED:233604007", "ICD-10-CM E11.9")
_EXPLICIT_VOCAB_CODE_RE = re.compile(
    r"\b(?P<vocab>SNOMED(?:[- ]?CT)?|SCT|LOINC|ICD[- ]?10(?:[- ]?(?:CM|PCS))?|ICD[- ]?11|ICD|"
    r"UCUM|RXNORM|CPT[- ]?4?|HCPCS|NDC|ATC|MESH|RADLEX|DICOM|DCM)"
    r"[\s:/#-]+(?:(?:code|tag|element|attribute|concept|term)[\s:/#-]+)?"
    r"(?P<code>[A-Za-z0-9][A-Za-z0-9._/\[\]%+-]{0,31})(?=$|[\s?,:;!.)\]])",
    re.IGNORECASE,
)

# Standalone recognizable clinical codes without explicit vocabulary prefix
_BARE_LOINC_RE = re.compile(r"^\s*(?P<code>(?:LP)?\d{1,7}-\d)\s*$", re.IGNORECASE)
_BARE_ICD10CM_RE = re.compile(r"^\s*(?P<code>[A-TV-Z]\d{2}(?:\.[A-Za-z0-9]{1,4})?)\s*$", re.IGNORECASE)
_BARE_SNOMED_RE = re.compile(r"^\s*(?P<code>\d{6,18})\s*$")

# DICOM tag "(0018,0050)", "0018,0050", "00180050" and DICOM UID "1.2.840.10008.*"
_DICOM_TAG_PAREN_RE = re.compile(r"\(?\b(?P<g>[0-9A-Fa-f]{4}),\s*(?P<e>[0-9A-Fa-f]{4})\b\)?")
_BARE_DICOM_HEX_RE = re.compile(r"^\s*(?P<hex>[0-9A-Fa-f]{8})\s*$")
_DICOM_UID_RE = re.compile(r"\b(?P<uid>1\.2\.840\.10008(?:\.\d+)+)\b")

# Intent cues
_RELATED_CUE_RE = re.compile(
    r"\b(?:related\s+to|similar\s+to|associated\s+with|connected\s+to|neighborhood\s+(?:of|for)|around)\b",
    re.IGNORECASE,
)
_CROSSWALK_CUE_RE = re.compile(
    r"\b(?:map(?:ped|ping)?|crosswalk|translate|convert|equivalent(?:\s+in)?)\b",
    re.IGNORECASE,
)
_CROSSWALK_TARGET_RE = re.compile(
    r"\b(?:to|into|in)\s+(?P<vocab>SNOMED(?:[- ]?CT)?|LOINC|ICD[- ]?10(?:[- ]?CM)?|ICD[- ]?11|"
    r"UCUM|RXNORM|CPT[- ]?4?|HCPCS|NDC|ATC|MESH|RADLEX)\b",
    re.IGNORECASE,
)
_SUBTYPES_CUE_RE = re.compile(
    r"\b(?:subtypes?\s+of|children\s+of|descendants?\s+of|expand(?:\s+descendants)?\s+(?:of)?|under)\b",
    re.IGNORECASE,
)

# Entity/vocabulary hint tokens in free-text queries
_DICOM_HINT_RE = re.compile(
    r"\b(?:dicom|ciod|sop\s*class|vr|vm|tag|de-?identification|confidentiality|ps3\.\d+)\b",
    re.IGNORECASE,
)
_VOCAB_MENTION_RE = re.compile(
    r"\b(SNOMED(?:[- ]?CT)?|LOINC|ICD[- ]?10(?:[- ]?(?:CM|PCS))?|ICD[- ]?11|UCUM|"
    r"RXNORM|CPT[- ]?4?|HCPCS|NDC|ATC|MESH|RADLEX)\b",
    re.IGNORECASE,
)

# Scaffold filler phrases stripped when computing modifier/search text
_FILLER_PREFIX_RE = re.compile(
    r"^(?:please\s+)?(?:find|search(?:\s+for)?|look\s*up|lookup|get|show(?:\s+me)?|what\s+(?:is|are)|"
    r"tell\s+me\s+about|explain|define|describe|compare|fetch|retrieve|resolve|check)\s+"
    r"(?:the\s+)?(?:meaning\s+of\s+|definition\s+of\s+)?(?:something|anything|concepts?|terms?|codes?|tags?)?\s*",
    re.IGNORECASE,
)

# Adjacent schema/code connector checks for pairing spaCy entities
_PRECEDING_SCHEMA_CONNECTOR_RE = re.compile(
    r"^[\s:/#-]*(?:codes?|concepts?|terms?|tags?|elements?|attributes?|uids?|ids?|sop\s+class)?[\s:/#-]*$",
    re.IGNORECASE,
)
_FOLLOWING_SCHEMA_CONNECTOR_RE = re.compile(
    r"^\s+(?:in|from|under|of|for)\s+(?:the\s+)?$",
    re.IGNORECASE,
)

# English words following a vocabulary mention that are NOT codes
_NON_CODE_WORDS: Set[str] = {
    "code",
    "codes",
    "concept",
    "concepts",
    "term",
    "terms",
    "tag",
    "tags",
    "element",
    "elements",
    "attribute",
    "attributes",
    "uid",
    "uids",
    "for",
    "to",
    "in",
    "of",
    "from",
    "and",
    "or",
    "with",
    "without",
    "about",
    "on",
    "by",
    "is",
    "are",
    "related",
    "search",
    "lookup",
    "mapping",
    "mappings",
    "procedures",
    "conditions",
    "radiology",
    "playbook",
    "tests",
    "values",
    "units",
    "standard",
    "dictionary",
}

_TRAILING_ORPHAN_WORDS_RE = re.compile(
    r"(?:\s+(?:and|or|vs\.?|versus|with|for|in|of|from|to|on|about|under|between))+\s*$",
    re.IGNORECASE,
)
_LEADING_ORPHAN_WORDS_RE = re.compile(
    r"^(?:(?:and|or|vs\.?|versus|for|in|of|from|to|on|about|between)\s+)+",
    re.IGNORECASE,
)
_DUPLICATE_CONJUNCTION_RE = re.compile(r"\b(and|or)\s+(?:and|or)\b", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_JSON_ARRAY_SUBSTRING_RE = re.compile(r"\[.*\]", re.DOTALL)
_LEADING_BULLET_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s*")

# Neo4j queries used by ClinicalNLUService for dynamic schema checks & code reverse-lookups
_LOADED_SCHEMAS_CYPHER: LiteralString = """
OPTIONAL MATCH (cs:CodeSystem)
WHERE (($test_id IS NULL AND cs.test_id IS NULL) OR ($test_id IS NOT NULL AND cs.test_id = $test_id))
WITH collect(DISTINCT cs.vocabulary_id) AS code_systems
RETURN code_systems,
       EXISTS {
           MATCH (e:DicomElement)
           WHERE (($test_id IS NULL AND e.test_id IS NULL) OR ($test_id IS NOT NULL AND e.test_id = $test_id))
       } AS has_dicom
"""

_REVERSE_LOOKUP_CONCEPT_CYPHER: LiteralString = """
MATCH (c:Concept {vocabulary_id: $vocab, code: $code})
WHERE (($test_id IS NULL AND c.test_id IS NULL) OR ($test_id IS NOT NULL AND c.test_id = $test_id))
RETURN c.vocabulary_id AS vocabulary_id,
       c.code AS code,
       c.display AS display,
       c.domain_id AS domain_id,
       c.definition AS definition
LIMIT 1
"""

_REVERSE_LOOKUP_DICOM_ELEMENT_CYPHER: LiteralString = """
MATCH (e:DicomElement)
WHERE (($test_id IS NULL AND e.test_id IS NULL) OR ($test_id IS NOT NULL AND e.test_id = $test_id))
  AND (
      e.tag_hex = $tag_hex
      OR e.tag = $tag_formatted
      OR toLower(e.keyword) = toLower($raw)
  )
RETURN e.tag AS tag,
       e.tag_hex AS tag_hex,
       e.keyword AS keyword,
       e.name AS name,
       e.vr AS vr,
       e.description AS description
LIMIT 1
"""

_REVERSE_LOOKUP_DICOM_SOP_CYPHER: LiteralString = """
MATCH (s:DicomSOPClass {sop_class_uid: $uid})
WHERE (($test_id IS NULL AND s.test_id IS NULL) OR ($test_id IS NOT NULL AND s.test_id = $test_id))
RETURN s.sop_class_uid AS code,
       s.name AS name,
       s.ciod_name AS ciod_name
LIMIT 1
"""


def _normalize_vocab(raw_vocab: str) -> str:
    compact = re.sub(r"\s+", "", raw_vocab.strip().lower())
    return _VOCAB_ALIASES.get(compact, raw_vocab.strip().upper())


def _run_coro_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Execute an async coroutine safely from synchronous code (including inside running event loops)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: List[_T] = []
    error_box: List[BaseException] = []

    def _runner() -> None:
        try:
            result_box.append(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - thread bridge error relay
            error_box.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error_box:
        raise error_box[0]
    return result_box[0]


def _spans_overlap(span: Tuple[int, int], existing: Sequence[Tuple[int, int]]) -> bool:
    start, end = span
    return any(start < ex_end and end > ex_start for ex_start, ex_end in existing)


def _infer_vocab_from_bare_code(raw_code: str) -> Optional[Tuple[str, str]]:
    """Infer ``(vocabulary_id, normalized_code)`` from an unambiguous standalone code token."""
    stripped = raw_code.strip()
    if _DICOM_UID_RE.fullmatch(stripped):
        return ("DICOM_UID", normalize_dicom_uid(stripped))
    paren_match = _DICOM_TAG_PAREN_RE.fullmatch(stripped)
    if paren_match:
        return ("DICOM_TAG", f"{paren_match.group('g')}{paren_match.group('e')}".upper())
    hex_match = _BARE_DICOM_HEX_RE.match(stripped)
    if hex_match:
        candidate_hex = hex_match.group("hex").upper()
        if re.search(r"[A-F]", candidate_hex) or candidate_hex.startswith("00"):
            return ("DICOM_TAG", candidate_hex)
    loinc_match = _BARE_LOINC_RE.match(stripped)
    if loinc_match:
        return ("LOINC", loinc_match.group("code").upper())
    icd_match = _BARE_ICD10CM_RE.match(stripped)
    if icd_match:
        return ("ICD10CM", icd_match.group("code").upper())
    snomed_match = _BARE_SNOMED_RE.match(stripped)
    if snomed_match:
        return ("SNOMED", snomed_match.group("code"))
    return None


def _is_schema_loaded(schema: str, loaded_schemas: Set[str]) -> bool:
    """Return True if ``schema`` is satisfied by the active Neo4j database's ``loaded_schemas``."""
    upper_loaded = {item.upper() for item in loaded_schemas}
    key = schema.strip().upper()
    if key in {"DICOM", "DICOM_TAG", "DICOM_UID", "DCM"}:
        return "DICOM" in upper_loaded or "DCM" in upper_loaded
    if key == "ICD10":
        return bool({"ICD10", "ICD10CM", "ICD10PCS"} & upper_loaded)
    return key in upper_loaded


def _candidate_vocab_ids(vocab: str) -> Tuple[str, ...]:
    """Return ordered Neo4j ``vocabulary_id`` candidates for an extracted schema identifier."""
    upper = vocab.strip().upper()
    if upper == "ICD10":
        return ("ICD10", "ICD10CM", "ICD10PCS")
    return (vocab.strip(),)


def parse_synthesized_phrases(
    raw_text: str,
    *,
    exclude: Sequence[str] = (),
    max_phrases: int = 3,
) -> List[str]:
    """Parse LLM-generated alternate search phrases from JSON array or line-oriented text."""
    if max_phrases <= 0:
        return []
    cleaned = _THINK_BLOCK_RE.sub("", raw_text or "").strip()
    cleaned = _CODE_FENCE_RE.sub("", cleaned).strip()
    if not cleaned:
        return []

    raw_items: List[str] = []
    parsed_json: Any = None
    try:
        parsed_json = json.loads(cleaned)
    except Exception:
        arr_match = _JSON_ARRAY_SUBSTRING_RE.search(cleaned)
        if arr_match:
            try:
                parsed_json = json.loads(arr_match.group(0))
            except Exception:
                parsed_json = None

    if isinstance(parsed_json, list):
        raw_items = [str(item) for item in parsed_json if isinstance(item, (str, int, float))]
    elif isinstance(parsed_json, dict):
        for key in ("phrases", "alternates", "queries", "items"):
            val = parsed_json.get(key)
            if isinstance(val, list):
                raw_items = [str(item) for item in val if isinstance(item, (str, int, float))]
                break
    else:
        for line in cleaned.splitlines():
            stripped = _LEADING_BULLET_RE.sub("", line.strip()).strip().strip("\"',")
            lower = stripped.lower()
            if (
                not stripped
                or stripped in {"[", "]", "{", "}"}
                or lower.startswith(("here are", "sure", "alternate", "json"))
            ):
                continue
            raw_items.append(stripped)

    seen_lower = {" ".join(s.strip().lower().split()) for s in exclude if s and s.strip()}
    phrases: List[str] = []
    for item in raw_items:
        norm = " ".join(item.strip().strip("\"'").split())
        if not norm or len(norm) < 2 or len(norm) > 120:
            continue
        key = norm.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        phrases.append(norm)
        if len(phrases) >= max_phrases:
            break
    return phrases


@runtime_checkable
class NLUParserProtocol(Protocol):
    """Protocol implemented by both ``StubNLUService`` and ``ClinicalNLUService``."""

    def parse(self, query: str) -> NLUParseResult:
        """Parse a raw user query synchronously into intent, entities, and search hints."""
        ...

    async def parse_async(self, query: str) -> NLUParseResult:
        """Parse a raw user query asynchronously."""
        ...


class StubNLUService:
    """Deterministic regex/heuristic NLU stub with zero model/DB/LLM dependencies."""

    provider_name: str = "stub"

    def parse(
        self,
        query: str,
        *,
        test_id: Optional[str] = None,
        expand_with_llm: Optional[bool] = None,
    ) -> NLUParseResult:
        """Parse a user query into intent, extracted codes, and inferred entity/vocabulary hints."""
        del test_id, expand_with_llm
        raw = query or ""
        cleaned = " ".join(raw.strip().split())
        if not cleaned:
            return NLUParseResult(
                raw_query=raw,
                normalized_query="",
                intent=NLUIntent.SEARCH,
                provider=self.provider_name,
            )

        extracted: List[ExtractedCodeEntity] = []
        matched_spans: List[Tuple[int, int]] = []
        inferred_entity_types: List[SearchEntityType] = []
        inferred_vocabs: List[str] = []

        # 1) DICOM SOP Class UID (most specific pattern)
        for match in _DICOM_UID_RE.finditer(cleaned):
            extracted.append(
                ExtractedCodeEntity(
                    vocabulary_id="DICOM_UID",
                    code=match.group("uid"),
                    raw_span=match.group(0),
                    confidence=0.99,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )
            matched_spans.append(match.span())
            if SearchEntityType.DICOM_SOP_CLASS not in inferred_entity_types:
                inferred_entity_types.append(SearchEntityType.DICOM_SOP_CLASS)

        # 2) DICOM Tag "(0018,0050)" / "0018,0050" or bare 8-char hex
        for match in _DICOM_TAG_PAREN_RE.finditer(cleaned):
            tag_hex = f"{match.group('g')}{match.group('e')}".upper()
            extracted.append(
                ExtractedCodeEntity(
                    vocabulary_id="DICOM_TAG",
                    code=tag_hex,
                    raw_span=match.group(0),
                    confidence=0.99,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )
            matched_spans.append(match.span())
            if SearchEntityType.DICOM_ELEMENT not in inferred_entity_types:
                inferred_entity_types.append(SearchEntityType.DICOM_ELEMENT)

        if not extracted:
            bare_hex = _BARE_DICOM_HEX_RE.match(cleaned)
            if bare_hex:
                candidate_hex = bare_hex.group("hex").upper()
                if re.search(r"[A-F]", candidate_hex) or candidate_hex.startswith("00"):
                    extracted.append(
                        ExtractedCodeEntity(
                            vocabulary_id="DICOM_TAG",
                            code=candidate_hex,
                            raw_span=cleaned,
                            confidence=0.95,
                            start_char=0,
                            end_char=len(cleaned),
                        )
                    )
                    matched_spans.append((0, len(cleaned)))
                    inferred_entity_types.append(SearchEntityType.DICOM_ELEMENT)

        # 3) Explicit "VOCAB CODE" patterns (e.g. "LOINC 87279-6", "SNOMED 233604007", "ICD10CM E11.9")
        for match in _EXPLICIT_VOCAB_CODE_RE.finditer(cleaned):
            if _spans_overlap(match.span(), matched_spans):
                continue
            vocab = _normalize_vocab(match.group("vocab"))
            code = match.group("code").strip()
            if code.lower() in _NON_CODE_WORDS:
                continue
            if vocab == "DICOM":
                vocab = "DICOM_UID" if code.startswith("1.2.840.10008.") else "DICOM_TAG"
            extracted.append(
                ExtractedCodeEntity(
                    vocabulary_id=vocab,
                    code=code.upper() if vocab == "ICD10CM" else code,
                    raw_span=match.group(0),
                    confidence=0.98,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )
            matched_spans.append(match.span())
            if vocab in {"DICOM_TAG", "DICOM_UID"}:
                et = SearchEntityType.DICOM_SOP_CLASS if vocab == "DICOM_UID" else SearchEntityType.DICOM_ELEMENT
                if et not in inferred_entity_types:
                    inferred_entity_types.append(et)
            else:
                if SearchEntityType.CONCEPT not in inferred_entity_types:
                    inferred_entity_types.append(SearchEntityType.CONCEPT)
                if vocab not in inferred_vocabs:
                    inferred_vocabs.append(vocab)

        # 4) Bare clinical code shapes (only when the entire query is just a code)
        if not extracted:
            bare_loinc = _BARE_LOINC_RE.match(cleaned)
            bare_icd10 = _BARE_ICD10CM_RE.match(cleaned)
            bare_snomed = _BARE_SNOMED_RE.match(cleaned)
            if bare_loinc:
                extracted.append(
                    ExtractedCodeEntity(
                        vocabulary_id="LOINC",
                        code=bare_loinc.group("code").upper(),
                        raw_span=cleaned,
                        confidence=0.95,
                        start_char=0,
                        end_char=len(cleaned),
                    )
                )
                matched_spans.append((0, len(cleaned)))
                inferred_entity_types.append(SearchEntityType.CONCEPT)
                inferred_vocabs.append("LOINC")
            elif bare_icd10:
                extracted.append(
                    ExtractedCodeEntity(
                        vocabulary_id="ICD10CM",
                        code=bare_icd10.group("code").upper(),
                        raw_span=cleaned,
                        confidence=0.92,
                        start_char=0,
                        end_char=len(cleaned),
                    )
                )
                matched_spans.append((0, len(cleaned)))
                inferred_entity_types.append(SearchEntityType.CONCEPT)
                inferred_vocabs.append("ICD10CM")
            elif bare_snomed:
                extracted.append(
                    ExtractedCodeEntity(
                        vocabulary_id="SNOMED",
                        code=bare_snomed.group("code"),
                        raw_span=cleaned,
                        confidence=0.90,
                        start_char=0,
                        end_char=len(cleaned),
                    )
                )
                matched_spans.append((0, len(cleaned)))
                inferred_entity_types.append(SearchEntityType.CONCEPT)
                inferred_vocabs.append("SNOMED")

        # 5) Determine target vocabularies for crosswalks and general vocabulary/entity hints
        target_vocabs: List[str] = []
        for match in _CROSSWALK_TARGET_RE.finditer(cleaned):
            v = _normalize_vocab(match.group("vocab"))
            if v not in target_vocabs:
                target_vocabs.append(v)

        if not inferred_vocabs:
            for match in _VOCAB_MENTION_RE.finditer(cleaned):
                v = _normalize_vocab(match.group(1))
                if v not in inferred_vocabs:
                    inferred_vocabs.append(v)

        if inferred_vocabs and SearchEntityType.CONCEPT not in inferred_entity_types:
            inferred_entity_types.append(SearchEntityType.CONCEPT)

        if _DICOM_HINT_RE.search(cleaned):
            if not any(et in DICOM_ENTITY_TYPES for et in inferred_entity_types):
                inferred_entity_types.append(SearchEntityType.DICOM_ELEMENT)

        # 6) Intent classification
        has_related_cue = bool(_RELATED_CUE_RE.search(cleaned))
        has_crosswalk_cue = bool(_CROSSWALK_CUE_RE.search(cleaned))
        has_subtypes_cue = bool(_SUBTYPES_CUE_RE.search(cleaned))

        modifier_text: Optional[str] = None
        if extracted:
            if has_crosswalk_cue:
                intent = NLUIntent.CROSSWALK
            elif has_subtypes_cue:
                intent = NLUIntent.EXPAND_SUBTYPES
            elif has_related_cue:
                intent = NLUIntent.RELATED_LOOKUP
                modifier_text = self._extract_modifier_text(cleaned, matched_spans)
            else:
                remainder = self._extract_modifier_text(cleaned, matched_spans)
                if not remainder:
                    intent = NLUIntent.EXACT_LOOKUP
                else:
                    intent = NLUIntent.RELATED_LOOKUP
                    modifier_text = remainder
        else:
            intent = NLUIntent.SEARCH

        normalized_query = _FILLER_PREFIX_RE.sub("", cleaned).strip() or cleaned
        return NLUParseResult(
            raw_query=raw,
            normalized_query=normalized_query,
            intent=intent,
            extracted_codes=extracted,
            inferred_entity_types=inferred_entity_types,
            inferred_vocabularies=inferred_vocabs,
            inferred_domains=[],
            target_vocabularies=target_vocabs,
            modifier_text=modifier_text,
            provider=self.provider_name,
        )

    async def parse_async(
        self,
        query: str,
        *,
        test_id: Optional[str] = None,
        expand_with_llm: Optional[bool] = None,
    ) -> NLUParseResult:
        """Asynchronous wrapper around ``parse``."""
        return self.parse(query, test_id=test_id, expand_with_llm=expand_with_llm)

    @staticmethod
    def _extract_modifier_text(cleaned: str, spans: List[Tuple[int, int]]) -> Optional[str]:
        """Remove extracted code spans and routing scaffold words to isolate trailing search modifiers."""
        pieces: List[str] = []
        cursor = 0
        for start, end in sorted(spans):
            if start > cursor:
                pieces.append(cleaned[cursor:start])
            cursor = max(cursor, end)
        if cursor < len(cleaned):
            pieces.append(cleaned[cursor:])
        residual = " ".join(pieces)
        residual = _FILLER_PREFIX_RE.sub("", residual)
        residual = _RELATED_CUE_RE.sub("", residual)
        residual = _CROSSWALK_CUE_RE.sub("", residual)
        residual = _SUBTYPES_CUE_RE.sub("", residual)
        residual = re.sub(r"^[\s?,:;.-]+|[\s?,:;.-]+$", "", residual)
        residual = " ".join(residual.split())
        if not residual or residual.lower() in {"for", "of", "the", "a", "an", "to", "in", "on", "about"}:
            return None
        return residual


class ClinicalNLUService:
    """Full Clinical & DICOM NLU service backed by ``gosmart-clinical-nlu``, Neo4j, and Genkit LLM."""

    provider_name: str = "gosmart-nlu"

    SYNTHESIS_SYSTEM_PROMPT: str = (
        "You are a clinical terminology and DICOM data dictionary search query expander.\n"
        "Given a clinical or medical imaging search query (and optional resolved dictionary context), "
        "generate concise alternate search phrases using standard medical synonyms, clinical terminology "
        "phrasing (LOINC, SNOMED CT, ICD-10-CM), and DICOM attribute terminology.\n"
        "Rules:\n"
        '1. Return ONLY a JSON array of strings, for example: ["phrase 1", "phrase 2", "phrase 3"]\n'
        "2. Each phrase must be 2 to 8 words long.\n"
        "3. Do NOT invent or output numeric clinical codes or hexadecimal DICOM tags.\n"
        "4. Do NOT include explanations, markdown prose, or numbering outside the JSON array."
    )

    _model_cache_lock = threading.Lock()
    _model_cache: Dict[Path, IntentModels] = {}

    def __init__(
        self,
        *,
        client: Optional[Neo4jClient] = None,
        llm_config_service: Optional[LLMConfigService] = None,
        settings: Optional[Settings] = None,
        model_path: Optional[str | Path] = None,
        loaded_schemas: Optional[Collection[str]] = None,
        synthesize_fn: Optional[SynthesizePhrasesFn] = None,
    ) -> None:
        self._settings = settings
        self.client = client or neo4j_client
        self.llm_config_service = llm_config_service or LLMConfigService(client=self.client)
        self._explicit_model_path = Path(model_path) if model_path is not None else None
        self._static_loaded_schemas: Optional[Set[str]] = (
            {str(s).strip() for s in loaded_schemas if str(s).strip()} if loaded_schemas is not None else None
        )
        self._synthesize_fn = synthesize_fn
        self._schema_cache: Dict[Optional[str], Tuple[float, Optional[Set[str]]]] = {}
        self._client_lock = threading.Lock()
        self._genkit_clients: Dict[Tuple[Any, ...], Genkit] = {}

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

    def parse(
        self,
        query: str,
        *,
        test_id: Optional[str] = None,
        expand_with_llm: Optional[bool] = None,
    ) -> NLUParseResult:
        """Synchronously execute the full NLU parse, Neo4j code reverse-lookup, and LLM expansion."""
        return _run_coro_sync(self.parse_async(query, test_id=test_id, expand_with_llm=expand_with_llm))

    async def parse_async(
        self,
        query: str,
        *,
        test_id: Optional[str] = None,
        expand_with_llm: Optional[bool] = None,
    ) -> NLUParseResult:
        """Execute the full NLU phase asynchronously."""
        raw = query or ""
        cleaned = " ".join(raw.strip().split())
        if not cleaned:
            return NLUParseResult(
                raw_query=raw,
                normalized_query="",
                intent=NLUIntent.SEARCH,
                provider=self.provider_name,
            )

        norm_test_id = normalize_test_id(test_id)
        warnings: List[str] = []

        # 1) Run trained spaCy NLU bundle + supplementary pattern extraction
        extracted, mentioned_schemas, nlu_confidence = self._extract_entities_and_schemas(cleaned, warnings)
        matched_spans: List[Tuple[int, int]] = [
            (ent.start_char, ent.end_char)
            for ent in extracted
            if ent.start_char is not None and ent.end_char is not None
        ]

        # 2) Crosswalk targets, general hints, and intent classification
        target_vocabs: List[str] = []
        for match in _CROSSWALK_TARGET_RE.finditer(cleaned):
            v = normalize_coding_schema(match.group("vocab"))
            if v not in target_vocabs:
                target_vocabs.append(v)

        inferred_vocabs: List[str] = []
        inferred_entity_types: List[SearchEntityType] = []
        for ent in extracted:
            if ent.vocabulary_id == "DICOM_UID":
                if SearchEntityType.DICOM_SOP_CLASS not in inferred_entity_types:
                    inferred_entity_types.append(SearchEntityType.DICOM_SOP_CLASS)
            elif ent.vocabulary_id == "DICOM_TAG":
                if SearchEntityType.DICOM_ELEMENT not in inferred_entity_types:
                    inferred_entity_types.append(SearchEntityType.DICOM_ELEMENT)
            else:
                if SearchEntityType.CONCEPT not in inferred_entity_types:
                    inferred_entity_types.append(SearchEntityType.CONCEPT)
                if ent.vocabulary_id not in inferred_vocabs:
                    inferred_vocabs.append(ent.vocabulary_id)

        if not inferred_vocabs:
            for schema in mentioned_schemas:
                if schema not in {"DICOM", "DCM"} and schema not in target_vocabs and schema not in inferred_vocabs:
                    inferred_vocabs.append(schema)

        if _DICOM_HINT_RE.search(cleaned) or "DICOM" in mentioned_schemas:
            if not any(et in DICOM_ENTITY_TYPES for et in inferred_entity_types):
                inferred_entity_types.append(SearchEntityType.DICOM_ELEMENT)

        has_related_cue = bool(_RELATED_CUE_RE.search(cleaned))
        has_crosswalk_cue = bool(_CROSSWALK_CUE_RE.search(cleaned))
        has_subtypes_cue = bool(_SUBTYPES_CUE_RE.search(cleaned))

        modifier_text: Optional[str] = None
        if extracted:
            if has_crosswalk_cue:
                intent = NLUIntent.CROSSWALK
            elif has_subtypes_cue:
                intent = NLUIntent.EXPAND_SUBTYPES
            elif has_related_cue:
                intent = NLUIntent.RELATED_LOOKUP
                modifier_text = StubNLUService._extract_modifier_text(cleaned, matched_spans)
            else:
                remainder = StubNLUService._extract_modifier_text(cleaned, matched_spans)
                intent = NLUIntent.EXACT_LOOKUP if not remainder else NLUIntent.SEARCH
        else:
            intent = NLUIntent.SEARCH

        # 3) Dynamic database schema availability check & user alerts
        loaded_schemas = await self.get_loaded_schemas(test_id=norm_test_id)
        unsupported_vocabularies: List[str] = []
        if loaded_schemas is not None:
            available_list_str = ", ".join(sorted(loaded_schemas))
            checked_schemas: Set[str] = set()

            for ent in extracted:
                schema_label = "DICOM" if ent.vocabulary_id in {"DICOM_TAG", "DICOM_UID"} else ent.vocabulary_id
                ent.is_schema_loaded = _is_schema_loaded(ent.vocabulary_id, loaded_schemas)
                if not ent.is_schema_loaded:
                    ent.resolved = False
                    if schema_label not in unsupported_vocabularies:
                        unsupported_vocabularies.append(schema_label)
                    if schema_label.upper() not in checked_schemas:
                        checked_schemas.add(schema_label.upper())
                        warnings.append(
                            f"Coding schema '{schema_label}' is not loaded in the active dictionary database "
                            f"(available schemas: {available_list_str})."
                        )

            for schema in (*mentioned_schemas, *target_vocabs):
                if schema.upper() in checked_schemas:
                    continue
                if not _is_schema_loaded(schema, loaded_schemas):
                    checked_schemas.add(schema.upper())
                    if schema not in unsupported_vocabularies:
                        unsupported_vocabularies.append(schema)
                    warnings.append(
                        f"Coding schema '{schema}' is not loaded in the active dictionary database "
                        f"(available schemas: {available_list_str})."
                    )

            # Filter inferred_vocabularies to loaded schemas so free-text queries (e.g. "Find SNOMED codes
            # for chest CT" on a DICOM+LOINC demo DB) still discover relevant concepts across loaded schemas
            inferred_vocabs = [
                ("ICD10CM" if v == "ICD10" and "ICD10CM" in {s.upper() for s in loaded_schemas} else v)
                for v in inferred_vocabs
                if _is_schema_loaded(v, loaded_schemas)
            ]

        if inferred_vocabs and SearchEntityType.CONCEPT not in inferred_entity_types:
            inferred_entity_types.append(SearchEntityType.CONCEPT)

        # 4) Reverse-lookup found codes in Neo4j & rewrite query string
        context_notes: List[str] = []
        for ent in extracted:
            if ent.is_schema_loaded is False:
                continue
            resolved_ok, display_text, context_line = await self._reverse_lookup_entity(ent, test_id=norm_test_id)
            ent.resolved = resolved_ok
            ent.resolved_display = display_text
            if resolved_ok and context_line:
                context_notes.append(context_line)
            elif not resolved_ok:
                label = ent.raw_span or f"{ent.vocabulary_id} {ent.code}"
                warnings.append(f"Code '{label}' was not found in the active dictionary database and was skipped.")

        normalized_query = _FILLER_PREFIX_RE.sub("", cleaned).strip() or cleaned
        if extracted:
            rewritten_query: Optional[str] = self._rewrite_query_with_resolutions(cleaned, extracted)
        else:
            rewritten_query = normalized_query

        # 5) Synthesize alternate clinical/DICOM search phrases via inference LLM (gemma4:e2b)
        should_expand = (
            expand_with_llm
            if expand_with_llm is not None
            else (self.settings.NLU_LLM_EXPANSION_ENABLED and intent in {NLUIntent.SEARCH, NLUIntent.EXACT_LOOKUP})
        )
        synthesized_phrases: List[str] = []
        seed_phrase = (rewritten_query or "").strip()
        if should_expand and seed_phrase and self.settings.NLU_MAX_ALTERNATE_PHRASES > 0:
            synthesized_phrases = await self._synthesize_phrases(
                seed_query=seed_phrase,
                raw_query=raw,
                normalized_query=normalized_query,
                context_notes=context_notes,
                max_phrases=self.settings.NLU_MAX_ALTERNATE_PHRASES,
                warnings=warnings,
            )

        return NLUParseResult(
            raw_query=raw,
            normalized_query=normalized_query,
            rewritten_query=rewritten_query,
            synthesized_phrases=synthesized_phrases,
            intent=intent,
            nlu_confidence=round(nlu_confidence, 4) if nlu_confidence is not None else None,
            extracted_codes=extracted,
            inferred_entity_types=inferred_entity_types,
            inferred_vocabularies=inferred_vocabs,
            inferred_domains=[],
            target_vocabularies=target_vocabs,
            unsupported_vocabularies=unsupported_vocabularies,
            modifier_text=modifier_text,
            warnings=warnings,
            provider=self.provider_name,
        )

    # -------------------------------------------------------------------------
    # Trained spaCy Bundle Loading & Multi-Entity Extraction
    # -------------------------------------------------------------------------

    def _resolve_model_bundle_path(self) -> Optional[Path]:
        if self._explicit_model_path is not None:
            candidate = self._explicit_model_path.expanduser()
            return candidate.resolve() if candidate.exists() else None

        raw_cfg = self.settings.NLU_MODEL_PATH.strip()
        if not raw_cfg:
            return None
        direct = Path(raw_cfg).expanduser()
        if direct.exists():
            return direct.resolve()
        repo_rel = (PROJECT_ROOT / raw_cfg).resolve()
        if repo_rel.exists():
            return repo_rel
        return None

    def _get_loaded_models(self) -> Optional[IntentModels]:
        bundle_path = self._resolve_model_bundle_path()
        if bundle_path is None:
            return None
        with self._model_cache_lock:
            cached = self._model_cache.get(bundle_path)
            if cached is not None:
                return cached
            try:
                register_intent_spec(FIND_CODING_SCHEMA_SPEC)
                loaded = load_intent_models(bundle_path)
                self._model_cache[bundle_path] = loaded
                return loaded
            except Exception as exc:
                logger.warning("nlu_model_bundle_load_failed", path=str(bundle_path), error=str(exc))
                return None

    def _extract_entities_and_schemas(
        self,
        cleaned: str,
        warnings: List[str],
    ) -> Tuple[List[ExtractedCodeEntity], List[str], Optional[float]]:
        extracted: List[ExtractedCodeEntity] = []
        occupied_spans: List[Tuple[int, int]] = []
        mentioned_schemas: List[str] = []
        nlu_confidence: Optional[float] = None

        models = self._get_loaded_models()
        loaded_model = models.get("find_coding_schema") if models else None
        if loaded_model is not None:
            doc = loaded_model.nlp(cleaned)
            cat_score = doc.cats.get(FIND_CODING_SCHEMA_SPEC.label)
            if cat_score is not None:
                nlu_confidence = float(cat_score)
            ent_conf = round(max(nlu_confidence or 0.95, 0.85), 4)

            ents = list(doc.ents)
            paired_schema_indices: Set[int] = set()

            for idx, ent in enumerate(ents):
                if ent.label_ in {"DICOM_UID", "DICOM_TAG"}:
                    vocab_id = ent.label_
                    norm_val = (
                        normalize_dicom_uid(ent.text) if ent.label_ == "DICOM_UID" else normalize_dicom_tag(ent.text)
                    )
                    span_start, span_end = ent.start_char, ent.end_char
                    if idx > 0 and (idx - 1) not in paired_schema_indices and ents[idx - 1].label_ == "CODING_SCHEMA":
                        prev = ents[idx - 1]
                        between = cleaned[prev.end_char : ent.start_char]
                        if normalize_coding_schema(prev.text) in {"DICOM", "DCM"} and (
                            _PRECEDING_SCHEMA_CONNECTOR_RE.match(between) is not None
                        ):
                            paired_schema_indices.add(idx - 1)
                            span_start = prev.start_char
                    elif (
                        idx + 1 < len(ents)
                        and (idx + 1) not in paired_schema_indices
                        and ents[idx + 1].label_ == "CODING_SCHEMA"
                    ):
                        nxt = ents[idx + 1]
                        between = cleaned[ent.end_char : nxt.start_char]
                        if normalize_coding_schema(nxt.text) in {"DICOM", "DCM"} and (
                            _FOLLOWING_SCHEMA_CONNECTOR_RE.match(between) is not None
                        ):
                            paired_schema_indices.add(idx + 1)
                            span_end = nxt.end_char

                    extracted.append(
                        ExtractedCodeEntity(
                            vocabulary_id=vocab_id,
                            code=norm_val,
                            raw_span=cleaned[span_start:span_end],
                            confidence=ent_conf,
                            start_char=span_start,
                            end_char=span_end,
                        )
                    )
                    occupied_spans.append((span_start, span_end))

                elif ent.label_ == "CODING_CODE":
                    raw_code = ent.text.strip()
                    if raw_code.lower() in _NON_CODE_WORDS:
                        continue

                    paired_vocab: Optional[str] = None
                    span_start, span_end = ent.start_char, ent.end_char

                    # Prefer immediately preceding CODING_SCHEMA ("LOINC 87279-6", "SNOMED code 233604007")
                    if idx > 0 and (idx - 1) not in paired_schema_indices and ents[idx - 1].label_ == "CODING_SCHEMA":
                        prev = ents[idx - 1]
                        between = cleaned[prev.end_char : ent.start_char]
                        if _PRECEDING_SCHEMA_CONNECTOR_RE.match(between) is not None:
                            paired_schema_indices.add(idx - 1)
                            paired_vocab = normalize_coding_schema(prev.text)
                            span_start = prev.start_char

                    # Otherwise check immediately following CODING_SCHEMA ("24627-2 in LOINC")
                    if (
                        paired_vocab is None
                        and idx + 1 < len(ents)
                        and (idx + 1) not in paired_schema_indices
                        and ents[idx + 1].label_ == "CODING_SCHEMA"
                    ):
                        nxt = ents[idx + 1]
                        between = cleaned[ent.end_char : nxt.start_char]
                        if _FOLLOWING_SCHEMA_CONNECTOR_RE.match(between) is not None:
                            paired_schema_indices.add(idx + 1)
                            paired_vocab = normalize_coding_schema(nxt.text)
                            span_end = nxt.end_char

                    if paired_vocab is not None:
                        if paired_vocab == "DICOM":
                            vocab_id = "DICOM_UID" if raw_code.startswith("1.2.840.10008.") else "DICOM_TAG"
                            code_val = (
                                normalize_dicom_uid(raw_code)
                                if vocab_id == "DICOM_UID"
                                else normalize_dicom_tag(raw_code)
                            )
                        elif paired_vocab == "UCUM":
                            vocab_id = "UCUM"
                            code_val = raw_code
                        else:
                            vocab_id = paired_vocab
                            code_val = normalize_coding_code(raw_code)
                    else:
                        inferred_bare = _infer_vocab_from_bare_code(raw_code)
                        if inferred_bare is None:
                            continue
                        vocab_id, code_val = inferred_bare

                    extracted.append(
                        ExtractedCodeEntity(
                            vocabulary_id=vocab_id,
                            code=code_val,
                            raw_span=cleaned[span_start:span_end],
                            confidence=ent_conf,
                            start_char=span_start,
                            end_char=span_end,
                        )
                    )
                    occupied_spans.append((span_start, span_end))

            # Collect unpaired CODING_SCHEMA spans (we filter any consumed by supplementary regex below)
            unpaired_schema_spans: List[Tuple[int, int, str]] = [
                (ent.start_char, ent.end_char, normalize_coding_schema(ent.text))
                for idx, ent in enumerate(ents)
                if ent.label_ == "CODING_SCHEMA" and idx not in paired_schema_indices
            ]
        else:
            unpaired_schema_spans = []
            warnings.append("Trained NLU bundle unavailable; using deterministic pattern fallback.")

        # Supplementary non-overlapping regex pass for unspaced tokens (e.g. "SNOMED:233604007")
        # and custom synthetic test codes (e.g. "LOINC T-87279-6").
        for match in _DICOM_UID_RE.finditer(cleaned):
            if _spans_overlap(match.span(), occupied_spans):
                continue
            extracted.append(
                ExtractedCodeEntity(
                    vocabulary_id="DICOM_UID",
                    code=match.group("uid"),
                    raw_span=match.group(0),
                    confidence=0.99,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )
            occupied_spans.append(match.span())

        for match in _DICOM_TAG_PAREN_RE.finditer(cleaned):
            if _spans_overlap(match.span(), occupied_spans):
                continue
            extracted.append(
                ExtractedCodeEntity(
                    vocabulary_id="DICOM_TAG",
                    code=f"{match.group('g')}{match.group('e')}".upper(),
                    raw_span=match.group(0),
                    confidence=0.99,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )
            occupied_spans.append(match.span())

        for match in _EXPLICIT_VOCAB_CODE_RE.finditer(cleaned):
            if _spans_overlap(match.span(), occupied_spans):
                continue
            raw_code = match.group("code").strip()
            if raw_code.lower() in _NON_CODE_WORDS:
                continue
            vocab = normalize_coding_schema(match.group("vocab"))
            if vocab == "DICOM":
                vocab = "DICOM_UID" if raw_code.startswith("1.2.840.10008.") else "DICOM_TAG"
                code_val = normalize_dicom_uid(raw_code) if vocab == "DICOM_UID" else normalize_dicom_tag(raw_code)
            elif vocab == "UCUM":
                code_val = raw_code
            else:
                code_val = normalize_coding_code(raw_code)

            extracted.append(
                ExtractedCodeEntity(
                    vocabulary_id=vocab,
                    code=code_val,
                    raw_span=match.group(0),
                    confidence=0.96,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )
            occupied_spans.append(match.span())

        if not extracted:
            inferred_bare = _infer_vocab_from_bare_code(cleaned)
            if inferred_bare is not None:
                vocab_id, code_val = inferred_bare
                extracted.append(
                    ExtractedCodeEntity(
                        vocabulary_id=vocab_id,
                        code=code_val,
                        raw_span=cleaned,
                        confidence=0.92,
                        start_char=0,
                        end_char=len(cleaned),
                    )
                )
                occupied_spans.append((0, len(cleaned)))

        # Finalize standalone schema mentions that were not consumed by any code span
        for s_start, s_end, schema_name in unpaired_schema_spans:
            if not _spans_overlap((s_start, s_end), occupied_spans) and schema_name not in mentioned_schemas:
                mentioned_schemas.append(schema_name)

        for match in _VOCAB_MENTION_RE.finditer(cleaned):
            if _spans_overlap(match.span(), occupied_spans):
                continue
            schema_name = normalize_coding_schema(match.group(1))
            if schema_name not in mentioned_schemas:
                mentioned_schemas.append(schema_name)

        extracted.sort(key=lambda e: e.start_char if e.start_char is not None else 0)
        return extracted, mentioned_schemas, nlu_confidence

    # -------------------------------------------------------------------------
    # Dynamic Neo4j Schema Availability & Reverse Lookup
    # -------------------------------------------------------------------------

    async def get_loaded_schemas(self, *, test_id: Optional[str] = None) -> Optional[Set[str]]:
        """Inspect active Neo4j ``CodeSystem`` and DICOM nodes (with TTL caching) to list loaded schemas."""
        if self._static_loaded_schemas is not None:
            return set(self._static_loaded_schemas)

        ttl = self.settings.NLU_SCHEMA_CACHE_TTL_SEC
        now = time.monotonic()
        cached = self._schema_cache.get(test_id)
        if cached is not None and ttl > 0 and (now - cached[0]) < ttl:
            return set(cached[1]) if cached[1] is not None else None

        try:
            rows = await self.client.execute_query(_LOADED_SCHEMAS_CYPHER, {"test_id": test_id})
        except Exception as exc:
            logger.debug("nlu_loaded_schemas_query_skipped", error=str(exc))
            return None

        if not rows:
            return None

        row = rows[0]
        raw_systems = row.get("code_systems") or []
        has_dicom = bool(row.get("has_dicom"))
        schemas: Set[str] = {str(item).strip() for item in raw_systems if item and str(item).strip()}
        if has_dicom:
            schemas.add("DICOM")

        # Treat completely empty unseeded/mocked responses as unknown rather than alerting on every schema
        resolved: Optional[Set[str]] = schemas if schemas else None
        self._schema_cache[test_id] = (now, resolved)
        return set(resolved) if resolved is not None else None

    async def _reverse_lookup_entity(
        self,
        entity: ExtractedCodeEntity,
        *,
        test_id: Optional[str],
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Look up ``entity`` in Neo4j and return ``(resolved, display_text, llm_context_line)``."""
        try:
            if entity.vocabulary_id == "DICOM_UID":
                rows = await self.client.execute_query(
                    _REVERSE_LOOKUP_DICOM_SOP_CYPHER,
                    {"uid": entity.code, "test_id": test_id},
                )
                if not rows:
                    return (False, None, None)
                row = rows[0]
                name = str(row.get("name") or "").strip()
                ciod = str(row.get("ciod_name") or "").strip()
                ctx = f"DICOM SOP Class {entity.code}: {name}" + (f" (CIOD: {ciod})" if ciod else "")
                return (bool(name), name or None, ctx)

            if entity.vocabulary_id == "DICOM_TAG":
                hex_only = re.sub(r"[^0-9A-Fa-f]", "", entity.code).upper()
                tag_hex = hex_only if len(hex_only) == 8 else entity.code.upper()
                tag_fmt = f"({tag_hex[:4]},{tag_hex[4:]})" if len(tag_hex) == 8 else entity.code
                rows = await self.client.execute_query(
                    _REVERSE_LOOKUP_DICOM_ELEMENT_CYPHER,
                    {
                        "tag_hex": tag_hex,
                        "tag_formatted": tag_fmt,
                        "raw": entity.code,
                        "test_id": test_id,
                    },
                )
                if not rows:
                    return (False, None, None)
                row = rows[0]
                name = str(row.get("name") or row.get("keyword") or "").strip()
                kw = str(row.get("keyword") or "").strip()
                vr = str(row.get("vr") or "").strip()
                desc = " ".join(str(row.get("description") or "").split())[:180]
                ctx = f"DICOM {row.get('tag') or tag_fmt} Keyword={kw} VR={vr}: {name}"
                if desc:
                    ctx += f" — {desc}"
                return (bool(name), name or None, ctx)

            for candidate_vocab in _candidate_vocab_ids(entity.vocabulary_id):
                code_candidates = [entity.code]
                # Preserve case-sensitive fallback (e.g. UCUM 'mg/dL') if raw_span ends with a mixed-case token
                if entity.raw_span:
                    tail_token = entity.raw_span.split()[-1].lstrip(":/#-")
                    if tail_token and tail_token not in code_candidates:
                        code_candidates.append(tail_token)

                for code_val in code_candidates:
                    rows = await self.client.execute_query(
                        _REVERSE_LOOKUP_CONCEPT_CYPHER,
                        {"vocab": candidate_vocab, "code": code_val, "test_id": test_id},
                    )
                    if rows:
                        row = rows[0]
                        matched_vocab = str(row.get("vocabulary_id") or candidate_vocab)
                        matched_code = str(row.get("code") or code_val)
                        display = str(row.get("display") or "").strip()
                        domain = str(row.get("domain_id") or "").strip()
                        defn = " ".join(str(row.get("definition") or "").split())[:180]
                        entity.vocabulary_id = matched_vocab
                        entity.code = matched_code
                        ctx = f"{matched_vocab} {matched_code}"
                        if domain:
                            ctx += f" ({domain})"
                        ctx += f": {display}"
                        if defn:
                            ctx += f" — {defn}"
                        return (bool(display), display or None, ctx)
        except Exception as exc:
            logger.warning(
                "nlu_reverse_lookup_failed",
                vocabulary=entity.vocabulary_id,
                code=entity.code,
                error=str(exc),
            )
        return (False, None, None)

    @staticmethod
    def _rewrite_query_with_resolutions(cleaned: str, extracted: Sequence[ExtractedCodeEntity]) -> str:
        """Replace resolved code spans with ``resolved_display`` and strip unresolved/unsupported spans."""
        rewritten = cleaned
        spanned = [e for e in extracted if e.start_char is not None and e.end_char is not None]
        for ent in sorted(spanned, key=lambda item: int(item.start_char or 0), reverse=True):
            start = int(ent.start_char or 0)
            end = int(ent.end_char or 0)
            replacement = ent.resolved_display if (ent.resolved and ent.resolved_display) else ""
            rewritten = f"{rewritten[:start]}{replacement}{rewritten[end:]}"

        rewritten = _FILLER_PREFIX_RE.sub("", rewritten)
        rewritten = _DUPLICATE_CONJUNCTION_RE.sub(r"\1", rewritten)
        rewritten = re.sub(r"^[\s?,:;.\-!]+|[\s?,:;.\-!]+$", "", rewritten)
        rewritten = _TRAILING_ORPHAN_WORDS_RE.sub("", rewritten)
        rewritten = _LEADING_ORPHAN_WORDS_RE.sub("", rewritten)
        rewritten = re.sub(r"^[\s?,:;.\-!]+|[\s?,:;.\-!]+$", "", rewritten)
        rewritten = " ".join(rewritten.split())

        if rewritten.lower() in {
            "the",
            "a",
            "an",
            "code",
            "codes",
            "tag",
            "tags",
            "concept",
            "concepts",
            "term",
            "terms",
            "dicom",
        }:
            return ""
        return rewritten

    # -------------------------------------------------------------------------
    # Genkit LLM Alternate Phrase Synthesis (gemma4:e2b via inference-v1)
    # -------------------------------------------------------------------------

    async def _resolve_inference_config(self) -> LLM_Config:
        target_tag = self.settings.INFERENCE_TAG or DEFAULT_INFERENCE_CONFIG_TAG
        try:
            cfg = await self.llm_config_service.get_llm_config_by_tag(
                target_tag,
                group_code=DEFAULT_GROUP_CODE,
            )
            if cfg is not None and (cfg.tag == target_tag or cfg.dimensions is None):
                return cfg
        except Exception as exc:
            logger.debug("nlu_inference_config_lookup_fallback", tag=target_tag, error=str(exc))

        return build_default_inference_llm_config(DEFAULT_GROUP_CODE, settings=self.settings)

    def _get_or_create_genkit_client(self, config: LLM_Config) -> Tuple[Genkit, str]:
        provider = config.provider.strip().lower()
        resolved_key = config.resolve_api_key()
        project_id = config.project_id or self.settings.VERTEX_AI_PROJECT_ID
        location = config.location or self.settings.VERTEX_AI_LOCATION
        fingerprint = config.client_fingerprint(resolved_api_key=resolved_key, fallback_project_id=project_id)

        with self._client_lock:
            ai = self._genkit_clients.get(fingerprint)
            if ai is None:
                if provider == "ollama":
                    plugin: Any = Ollama(base_url=config.url or None)
                elif provider in {"googleai", "gemini"}:
                    plugin = GoogleAI(api_key=resolved_key or None)
                elif provider == "vertexai":
                    plugin = VertexAI(
                        api_key=resolved_key or None,
                        project=project_id,
                        location=location,
                    )
                else:
                    raise ValueError(f"Unsupported NLU inference provider '{provider}'.")
                ai = Genkit(plugins=[plugin])
                self._genkit_clients[fingerprint] = ai

        if provider == "ollama":
            model_ref = config.model if config.model.startswith("ollama/") else f"ollama/{config.model}"
        elif provider in {"googleai", "gemini"}:
            model_ref = f"googleai/{config.model.removeprefix('googleai/')}"
        elif provider == "vertexai":
            model_ref = f"vertexai/{config.model.removeprefix('vertexai/')}"
        else:
            raise ValueError(f"Unsupported NLU inference provider '{provider}'.")

        return ai, model_ref

    async def _synthesize_phrases(
        self,
        *,
        seed_query: str,
        raw_query: str,
        normalized_query: str,
        context_notes: Sequence[str],
        max_phrases: int,
        warnings: List[str],
    ) -> List[str]:
        config = await self._resolve_inference_config()
        exclude_set = [raw_query, normalized_query, seed_query]

        try:
            if self._synthesize_fn is not None:
                custom = self._synthesize_fn(seed_query, context_notes, config, max_phrases)
                items = await custom if inspect.isawaitable(custom) else list(custom)
                return parse_synthesized_phrases(
                    json.dumps(list(items)),
                    exclude=exclude_set,
                    max_phrases=max_phrases,
                )

            prompt_lines = [f"Search query: {seed_query}"]
            if context_notes:
                prompt_lines.append("Resolved dictionary context:\n" + "\n".join(f"- {c}" for c in context_notes))
            prompt_lines.append(
                f"Generate up to {max_phrases} distinct alternate clinical and DICOM search phrases as a JSON array:"
            )
            prompt = "\n\n".join(prompt_lines)

            ai, model_ref = self._get_or_create_genkit_client(config)
            gen_cfg: Dict[str, Any] = {"temperature": 0.1}
            if config.extra_config:
                gen_cfg.update(config.extra_config)

            response = await ai.generate(
                model=model_ref,
                system=self.SYNTHESIS_SYSTEM_PROMPT,
                prompt=prompt,
                config=gen_cfg,
            )
            text = (getattr(response, "text", None) or "").strip()
            return parse_synthesized_phrases(text, exclude=exclude_set, max_phrases=max_phrases)
        except Exception as exc:
            logger.warning(
                "nlu_phrase_synthesis_fallback",
                provider=config.provider,
                model=config.model,
                error=str(exc),
            )
            msg = (
                f"LLM phrase synthesis unavailable ({config.provider}/{config.model}: {exc.__class__.__name__}); "
                "continuing with deterministic/lexical search."
            )
            if msg not in warnings:
                warnings.append(msg)
            return []


stub_nlu_service = StubNLUService()
clinical_nlu_service = ClinicalNLUService()

__all__ = [
    "ClinicalNLUService",
    "NLUParserProtocol",
    "StubNLUService",
    "SynthesizePhrasesFn",
    "clinical_nlu_service",
    "parse_synthesized_phrases",
    "stub_nlu_service",
]
