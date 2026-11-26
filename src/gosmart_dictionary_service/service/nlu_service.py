"""NLU query-parsing protocol and deterministic stub implementation.

This module defines ``NLUParserProtocol`` so ``SearchService`` can run immediately with
``StubNLUService`` (regex/heuristic code and intent detection) and later swap in the
trained clinical NLU migrated from ``gosmart-graphite`` without modifying callers.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

from gosmart_dictionary_service.model.search import (
    DICOM_ENTITY_TYPES,
    ExtractedCodeEntity,
    NLUIntent,
    NLUParseResult,
    SearchEntityType,
)

# Canonical vocabulary alias normalization
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
    r"\b(?P<vocab>SNOMED(?:[- ]?CT)?|SCT|LOINC|ICD[- ]?10(?:[- ]?CM)?|ICD[- ]?11|UCUM|RXNORM|CPT4?|HCPCS|DCM)"
    r"[\s:/#-]+(?P<code>[A-Za-z0-9][A-Za-z0-9._/%-]{0,31})\b",
    re.IGNORECASE,
)

# Standalone recognizable clinical codes without explicit vocabulary prefix
_BARE_LOINC_RE = re.compile(r"^\s*(?P<code>\d{1,7}-\d)\s*$")
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
    r"\b(?:to|into|in)\s+(?P<vocab>SNOMED(?:[- ]?CT)?|LOINC|ICD[- ]?10(?:[- ]?CM)?|ICD[- ]?11|UCUM)\b",
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
    r"\b(SNOMED(?:[- ]?CT)?|LOINC|ICD[- ]?10(?:[- ]?CM)?|UCUM)\b",
    re.IGNORECASE,
)

# Scaffold filler phrases stripped when computing modifier/search text
_FILLER_PREFIX_RE = re.compile(
    r"^(?:please\s+)?(?:find|search(?:\s+for)?|look\s*up|lookup|get|show(?:\s+me)?|what\s+is)\s+"
    r"(?:something|anything|concepts?|terms?|codes?)?\s*",
    re.IGNORECASE,
)


def _normalize_vocab(raw_vocab: str) -> str:
    compact = re.sub(r"\s+", "", raw_vocab.strip().lower())
    return _VOCAB_ALIASES.get(compact, raw_vocab.strip().upper())


@runtime_checkable
class NLUParserProtocol(Protocol):
    """Protocol implemented by both ``StubNLUService`` and the future migrated ``gosmart-graphite`` NLU."""

    def parse(self, query: str) -> NLUParseResult:
        """Parse a raw user query synchronously into intent, entities, and search hints."""
        ...

    async def parse_async(self, query: str) -> NLUParseResult:
        """Parse a raw user query asynchronously."""
        ...


class StubNLUService:
    """Deterministic regex/heuristic NLU stub pending migration of the trained ``gosmart-graphite`` NLU."""

    provider_name: str = "stub"

    def parse(self, query: str) -> NLUParseResult:
        """Parse a user query into intent, extracted codes, and inferred entity/vocabulary hints."""
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
                )
            )
            matched_spans.append(match.span())
            if SearchEntityType.DICOM_ELEMENT not in inferred_entity_types:
                inferred_entity_types.append(SearchEntityType.DICOM_ELEMENT)

        if not extracted:
            bare_hex = _BARE_DICOM_HEX_RE.match(cleaned)
            # Treat as DICOM hex if it contains A-F or starts with a standard DICOM group prefix (00xx)
            if bare_hex:
                candidate_hex = bare_hex.group("hex").upper()
                if re.search(r"[A-F]", candidate_hex) or candidate_hex.startswith("00"):
                    extracted.append(
                        ExtractedCodeEntity(
                            vocabulary_id="DICOM_TAG",
                            code=candidate_hex,
                            raw_span=cleaned,
                            confidence=0.95,
                        )
                    )
                    matched_spans.append((0, len(cleaned)))
                    inferred_entity_types.append(SearchEntityType.DICOM_ELEMENT)

        # 3) Explicit "VOCAB CODE" patterns (e.g. "LOINC 87279-6", "SNOMED 233604007", "ICD10CM E11.9")
        for match in _EXPLICIT_VOCAB_CODE_RE.finditer(cleaned):
            vocab = _normalize_vocab(match.group("vocab"))
            code = match.group("code").strip()
            # Guard against English filler words after a vocabulary name (e.g. "SNOMED codes", "LOINC search")
            if code.lower() in {
                "code",
                "codes",
                "concept",
                "concepts",
                "term",
                "terms",
                "for",
                "to",
                "in",
                "of",
                "from",
                "and",
                "or",
                "related",
                "search",
                "lookup",
                "mapping",
                "mappings",
                "procedures",
                "conditions",
                "radiology",
                "playbook",
            }:
                continue
            extracted.append(
                ExtractedCodeEntity(
                    vocabulary_id=vocab,
                    code=code.upper() if vocab == "ICD10CM" else code,
                    raw_span=match.group(0),
                    confidence=0.98,
                )
            )
            matched_spans.append(match.span())
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
                        code=bare_loinc.group("code"),
                        raw_span=cleaned,
                        confidence=0.95,
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
                # If the query is essentially just the code (or "lookup <code>"), treat as EXACT_LOOKUP
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

    async def parse_async(self, query: str) -> NLUParseResult:
        """Asynchronous wrapper around ``parse`` (keeps protocol compatible with async ML/LLM NLU)."""
        return self.parse(query)

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
        residual = re.sub(r"^[\s,:;.-]+|[\s,:;.-]+$", "", residual)
        residual = " ".join(residual.split())
        # Ignore bare stop-word leftovers like "for" or "of"
        if not residual or residual.lower() in {"for", "of", "the", "a", "an", "to", "in", "on", "about"}:
            return None
        return residual


stub_nlu_service = StubNLUService()

__all__ = [
    "NLUParserProtocol",
    "StubNLUService",
    "stub_nlu_service",
]
