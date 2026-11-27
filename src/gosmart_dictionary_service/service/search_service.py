"""Unified Clinical Terminology & DICOM Dictionary Search and Ontology Traversal Service."""

from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from collections.abc import Coroutine, Sequence
from typing import Any, Dict, List, LiteralString, Optional, Set, Tuple, TypeVar

from gosmart_dictionary_service.core.config import Settings, get_settings
from gosmart_dictionary_service.core.logging import get_logger
from gosmart_dictionary_service.db.neo4j import Neo4jClient, neo4j_client
from gosmart_dictionary_service.model.llm_config import DEFAULT_EMBEDDING_CONFIG_TAG
from gosmart_dictionary_service.model.search import (
    DEFAULT_UNIFIED_ENTITY_TYPES,
    DICOM_ENTITY_TYPES,
    FHIRCoding,
    MatchSource,
    NLUIntent,
    NLUParseResult,
    RelatedConceptHit,
    RelationDirection,
    SearchEntityType,
    SearchHit,
    SearchResponse,
    SubsumptionOutcome,
    normalize_entity_types,
)
from gosmart_dictionary_service.service.embedding_service import EmbeddingService
from gosmart_dictionary_service.service.llm_config_service import LLMConfigService
from gosmart_dictionary_service.service.nlu_service import (
    ClinicalNLUService,
    NLUParserProtocol,
    StubNLUService,
)
from gosmart_dictionary_service.service.system_service import DEFAULT_GROUP_CODE, normalize_test_id

logger = get_logger("gosmart_ds.search")

_T = TypeVar("_T")

# Lucene reserved special characters requiring escaping
_LUCENE_SPECIAL_RE = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')

# Maximum hub degree for 2-hop shared-attribute traversal (skips generic hubs like 'Point in time')
_MAX_SHARED_AXIS_DEGREE = 5000

# Probability discount & cap constants for hybrid discovery beneath deterministic code matches
_REWRITTEN_DISCOUNT_WITH_EXACT = 0.78
_REWRITTEN_CAP_WITH_EXACT = 0.85
_SYNTHESIZED_PHRASE_DISCOUNT = 0.68
_SYNTHESIZED_PHRASE_CAP = 0.78
_CONSENSUS_BOOST_PER_EXTRA_PHRASE = 0.03

# Priority weights for ordering aggregated related concepts
_RELATION_BASE_SCORE: Dict[MatchSource, float] = {
    MatchSource.REPLACED_BY: 1.00,
    MatchSource.MAPS_TO: 0.96,
    MatchSource.MAPPED_FROM: 0.92,
    MatchSource.IS_A_PARENT: 0.88,
    MatchSource.IS_A_CHILD: 0.84,
    MatchSource.HAS_ATTRIBUTE: 0.80,
    MatchSource.REPLACES: 0.76,
    MatchSource.SHARED_ATTRIBUTE: 0.74,
    MatchSource.USED_AS_ATTRIBUTE_BY: 0.68,
    MatchSource.SIBLING: 0.65,
    MatchSource.DICOM_RELATION: 0.82,
}

# -----------------------------------------------------------------------------
# Cypher Queries
# -----------------------------------------------------------------------------

_LOOKUP_CONCEPT_CYPHER: LiteralString = """
MATCH (c:Concept {vocabulary_id: $vocab, code: $code})
WHERE (($test_id IS NULL AND c.test_id IS NULL) OR ($test_id IS NOT NULL AND c.test_id = $test_id))
OPTIONAL MATCH (c)-[:BELONGS_TO_SYSTEM]->(cs:CodeSystem)
RETURN c.id AS id,
       c.vocabulary_id AS vocabulary_id,
       c.code AS code,
       c.display AS display,
       coalesce(c.system_uri, cs.system_uri, '') AS system_uri,
       cs.version AS system_version,
       c.coding_macro AS coding_macro,
       c.domain_id AS domain_id,
       c.concept_class_id AS concept_class_id,
       c.standard_concept AS standard_concept,
       c.invalid_reason AS invalid_reason,
       coalesce(c.is_deprecated, c.invalid_reason IS NOT NULL) AS is_deprecated,
       c.athena_concept_id AS athena_concept_id,
       coalesce(c.synonyms, []) AS synonyms,
       c.definition AS definition
LIMIT 1
"""

_LOOKUP_DICOM_ELEMENT_CYPHER: LiteralString = """
MATCH (e:DicomElement)
WHERE (($test_id IS NULL AND e.test_id IS NULL) OR ($test_id IS NOT NULL AND e.test_id = $test_id))
  AND (
      e.tag_hex = $tag_hex
      OR e.tag = $tag_formatted
      OR toLower(e.keyword) = toLower($raw)
  )
RETURN e.id AS id,
       'DicomElement' AS entity_type,
       e.tag AS tag,
       e.tag_hex AS tag_hex,
       e.keyword AS keyword,
       e.name AS name,
       e.vr AS vr,
       e.vm AS vm,
       coalesce(e.is_retired, false) AS is_retired,
       e.description AS description,
       coalesce(e.system_uri, 'http://gosmart.health/terminology/dicom-tag') AS system_uri,
       e.coding_macro AS coding_macro
LIMIT 1
"""

_LOOKUP_DICOM_SOP_CYPHER: LiteralString = """
MATCH (s:DicomSOPClass {sop_class_uid: $uid})
WHERE (($test_id IS NULL AND s.test_id IS NULL) OR ($test_id IS NOT NULL AND s.test_id = $test_id))
RETURN s.id AS id,
       'DicomSOPClass' AS entity_type,
       s.sop_class_uid AS code,
       s.name AS name,
       s.ciod_name AS ciod_name,
       coalesce(s.system_uri, 'http://gosmart.health/terminology/dicom-sop-class') AS system_uri
LIMIT 1
"""

_DICOM_ELEMENT_RELATED_CYPHER: LiteralString = """
MATCH (e:DicomElement {id: $node_id})
CALL (e) {
    MATCH (m:DicomModule)-[u:USES_ELEMENT]->(e)
    WHERE (($test_id IS NULL AND m.test_id IS NULL) OR ($test_id IS NOT NULL AND m.test_id = $test_id))
    RETURN m.id AS id,
           'DicomModule' AS entity_type,
           'reverse' AS direction,
           'dicom_relation' AS relation_reason,
           'USES_ELEMENT' AS relationship_id,
           coalesce(m.module_id, m.id) AS code,
           m.name AS display,
           coalesce(m.system_uri, 'http://gosmart.health/terminology/dicom-module') AS system_uri,
           {requirement_type: u.requirement_type, path: u.path} AS extra_props
    LIMIT $limit
    UNION ALL
    MATCH (e)-[:HAS_CONFIDENTIALITY_PROFILE]->(cp:DicomConfidentialityProfile)
    WHERE (($test_id IS NULL AND cp.test_id IS NULL) OR ($test_id IS NOT NULL AND cp.test_id = $test_id))
    RETURN cp.id AS id,
           'DicomConfidentialityProfile' AS entity_type,
           'forward' AS direction,
           'dicom_relation' AS relation_reason,
           'HAS_CONFIDENTIALITY_PROFILE' AS relationship_id,
           coalesce(cp.attribute_id, cp.tag, cp.id) AS code,
           cp.name AS display,
           coalesce(cp.system_uri, 'http://gosmart.health/terminology/dicom-confidentiality') AS system_uri,
           {basic_profile: cp.basic_profile, tag: cp.tag} AS extra_props
    LIMIT $limit
}
RETURN id, entity_type, direction, relation_reason, relationship_id, code, display, system_uri, extra_props
LIMIT $limit
"""

_CONCEPT_FULLTEXT_CYPHER: LiteralString = """
CALL db.index.fulltext.queryNodes('concept_fulltext_idx', $lucene_query)
YIELD node AS c, score
WHERE (($test_id IS NULL AND c.test_id IS NULL) OR ($test_id IS NOT NULL AND c.test_id = $test_id))
  AND ($vocabs IS NULL OR c.vocabulary_id IN $vocabs)
  AND ($domains IS NULL OR c.domain_id IN $domains)
  AND (NOT $valid_only OR c.invalid_reason IS NULL)
  AND (NOT $standard_only OR c.standard_concept = 'S')
RETURN c.id AS id,
       c.vocabulary_id AS vocabulary_id,
       c.code AS code,
       c.display AS display,
       coalesce(c.system_uri, '') AS system_uri,
       c.coding_macro AS coding_macro,
       c.domain_id AS domain_id,
       c.concept_class_id AS concept_class_id,
       c.standard_concept AS standard_concept,
       c.invalid_reason AS invalid_reason,
       coalesce(c.is_deprecated, c.invalid_reason IS NOT NULL) AS is_deprecated,
       c.athena_concept_id AS athena_concept_id,
       coalesce(c.synonyms, [])[0..6] AS synonyms,
       score AS lexical_score
ORDER BY lexical_score DESC
LIMIT $candidate_limit
"""

_CONCEPT_PARENT_CONVERGENCE_CYPHER: LiteralString = """
UNWIND $seed_ids AS sid
MATCH (child:Concept {id: sid})-[:IS_A]->(parent:Concept)
WHERE (($test_id IS NULL AND parent.test_id IS NULL) OR ($test_id IS NOT NULL AND parent.test_id = $test_id))
  AND ($vocabs IS NULL OR parent.vocabulary_id IN $vocabs)
  AND ($domains IS NULL OR parent.domain_id IN $domains)
  AND (NOT $valid_only OR parent.invalid_reason IS NULL)
  AND (NOT $standard_only OR parent.standard_concept = 'S')
WITH parent, count(DISTINCT sid) AS convergence
RETURN parent.id AS id,
       parent.vocabulary_id AS vocabulary_id,
       parent.code AS code,
       parent.display AS display,
       coalesce(parent.system_uri, '') AS system_uri,
       parent.coding_macro AS coding_macro,
       parent.domain_id AS domain_id,
       parent.concept_class_id AS concept_class_id,
       parent.standard_concept AS standard_concept,
       parent.invalid_reason AS invalid_reason,
       coalesce(parent.is_deprecated, parent.invalid_reason IS NOT NULL) AS is_deprecated,
       parent.athena_concept_id AS athena_concept_id,
       coalesce(parent.synonyms, [])[0..6] AS synonyms,
       convergence
ORDER BY convergence DESC
LIMIT $parent_limit
"""

_HYDRATE_CONCEPT_ATTRS_CYPHER: LiteralString = """
UNWIND $concept_ids AS cid
MATCH (c:Concept {id: cid})
OPTIONAL MATCH (c)-[r:HAS_ATTRIBUTE]->(a:Concept)
WHERE (($test_id IS NULL AND a.test_id IS NULL) OR ($test_id IS NOT NULL AND a.test_id = $test_id))
WITH cid, collect(DISTINCT r.relationship_id + ': ' + a.display)[0..5] AS attrs
RETURN cid AS id, attrs
"""

_RELATED_FORWARD_CYPHER: LiteralString = """
MATCH (c:Concept {vocabulary_id: $vocab, code: $code})
WHERE (($test_id IS NULL AND c.test_id IS NULL) OR ($test_id IS NOT NULL AND c.test_id = $test_id))
MATCH (c)-[r:IS_A|MAPS_TO|REPLACED_BY|HAS_ATTRIBUTE]->(dst:Concept)
WHERE (($test_id IS NULL AND dst.test_id IS NULL) OR ($test_id IS NOT NULL AND dst.test_id = $test_id))
  AND (NOT $valid_only OR dst.invalid_reason IS NULL)
RETURN dst.id AS id,
       'forward' AS direction,
       type(r) AS rel_type,
       coalesce(r.relationship_id, type(r)) AS relationship_id,
       r.equivalence AS equivalence,
       dst.vocabulary_id AS vocabulary_id,
       dst.code AS code,
       dst.display AS display,
       coalesce(dst.system_uri, '') AS system_uri,
       dst.coding_macro AS coding_macro,
       dst.domain_id AS domain_id,
       dst.concept_class_id AS concept_class_id,
       dst.standard_concept AS standard_concept,
       dst.invalid_reason AS invalid_reason
LIMIT $limit
"""

_RELATED_REVERSE_CYPHER: LiteralString = """
MATCH (c:Concept {vocabulary_id: $vocab, code: $code})
WHERE (($test_id IS NULL AND c.test_id IS NULL) OR ($test_id IS NOT NULL AND c.test_id = $test_id))
MATCH (src:Concept)-[r:IS_A|MAPS_TO|REPLACED_BY|HAS_ATTRIBUTE]->(c)
WHERE (($test_id IS NULL AND src.test_id IS NULL) OR ($test_id IS NOT NULL AND src.test_id = $test_id))
  AND (NOT $valid_only OR src.invalid_reason IS NULL)
RETURN src.id AS id,
       'reverse' AS direction,
       type(r) AS rel_type,
       coalesce(r.relationship_id, type(r)) AS relationship_id,
       r.equivalence AS equivalence,
       src.vocabulary_id AS vocabulary_id,
       src.code AS code,
       src.display AS display,
       coalesce(src.system_uri, '') AS system_uri,
       src.coding_macro AS coding_macro,
       src.domain_id AS domain_id,
       src.concept_class_id AS concept_class_id,
       src.standard_concept AS standard_concept,
       src.invalid_reason AS invalid_reason
LIMIT $limit
"""

_RELATED_SHARED_ATTRS_CYPHER: LiteralString = """
MATCH (c:Concept {vocabulary_id: $vocab, code: $code})
WHERE (($test_id IS NULL AND c.test_id IS NULL) OR ($test_id IS NOT NULL AND c.test_id = $test_id))
MATCH (c)-[:HAS_ATTRIBUTE]->(axis:Concept)
WHERE (($test_id IS NULL AND axis.test_id IS NULL) OR ($test_id IS NOT NULL AND axis.test_id = $test_id))
  AND COUNT { (axis)<-[:HAS_ATTRIBUTE]-() } <= $max_axis_degree
WITH c, axis
MATCH (other:Concept)-[:HAS_ATTRIBUTE]->(axis)
WHERE other.id <> c.id
  AND (($test_id IS NULL AND other.test_id IS NULL) OR ($test_id IS NOT NULL AND other.test_id = $test_id))
  AND (NOT $valid_only OR other.invalid_reason IS NULL)
  AND other.vocabulary_id = c.vocabulary_id
WITH other,
     count(DISTINCT axis) AS shared_count,
     collect(DISTINCT axis.display)[0..5] AS shared_axes
WHERE shared_count >= $min_shared_axes
RETURN other.id AS id,
       shared_count,
       shared_axes,
       other.vocabulary_id AS vocabulary_id,
       other.code AS code,
       other.display AS display,
       coalesce(other.system_uri, '') AS system_uri,
       other.coding_macro AS coding_macro,
       other.domain_id AS domain_id,
       other.concept_class_id AS concept_class_id,
       other.standard_concept AS standard_concept,
       other.invalid_reason AS invalid_reason
ORDER BY shared_count DESC, other.display ASC
LIMIT $limit
"""

_EXPAND_DESCENDANTS_CYPHER_TEMPLATE = """
MATCH p = (desc:Concept)-[:IS_A*1..{max_depth}]->(root:Concept {{vocabulary_id: $vocab, code: $code}})
WHERE (($test_id IS NULL AND root.test_id IS NULL) OR ($test_id IS NOT NULL AND root.test_id = $test_id))
  AND (($test_id IS NULL AND desc.test_id IS NULL) OR ($test_id IS NOT NULL AND desc.test_id = $test_id))
  AND (NOT $valid_only OR desc.invalid_reason IS NULL)
WITH desc, min(length(p)) AS depth
RETURN desc.id AS id,
       depth,
       desc.vocabulary_id AS vocabulary_id,
       desc.code AS code,
       desc.display AS display,
       coalesce(desc.system_uri, '') AS system_uri,
       desc.coding_macro AS coding_macro,
       desc.domain_id AS domain_id,
       desc.concept_class_id AS concept_class_id,
       desc.standard_concept AS standard_concept,
       desc.invalid_reason AS invalid_reason,
       coalesce(desc.is_deprecated, desc.invalid_reason IS NOT NULL) AS is_deprecated
ORDER BY depth ASC, desc.display ASC
LIMIT $limit
"""

_CROSSWALK_CYPHER: LiteralString = """
MATCH (src:Concept {vocabulary_id: $vocab, code: $code})
WHERE (($test_id IS NULL AND src.test_id IS NULL) OR ($test_id IS NOT NULL AND src.test_id = $test_id))
CALL (src) {
    MATCH (src)-[r:MAPS_TO]->(dst:Concept)
    WHERE (($test_id IS NULL AND dst.test_id IS NULL) OR ($test_id IS NOT NULL AND dst.test_id = $test_id))
      AND ($target_vocabs IS NULL OR dst.vocabulary_id IN $target_vocabs)
    RETURN dst AS target,
           'forward' AS direction,
           'maps_to' AS reason,
           coalesce(r.relationship_id, 'Maps to') AS relationship_id,
           r.equivalence AS equivalence,
           1 AS hops
    UNION ALL
    MATCH (dst:Concept)-[r:MAPS_TO]->(src)
    WHERE (($test_id IS NULL AND dst.test_id IS NULL) OR ($test_id IS NOT NULL AND dst.test_id = $test_id))
      AND ($target_vocabs IS NULL OR dst.vocabulary_id IN $target_vocabs)
    RETURN dst AS target,
           'reverse' AS direction,
           'mapped_from' AS reason,
           coalesce(r.relationship_id, 'Maps to') AS relationship_id,
           r.equivalence AS equivalence,
           1 AS hops
    UNION ALL
    MATCH (src)-[:MAPS_TO]->(mid:Concept)<-[r2:MAPS_TO]-(dst:Concept)
    WHERE $max_hops >= 2
      AND dst.id <> src.id
      AND (($test_id IS NULL AND mid.test_id IS NULL) OR ($test_id IS NOT NULL AND mid.test_id = $test_id))
      AND (($test_id IS NULL AND dst.test_id IS NULL) OR ($test_id IS NOT NULL AND dst.test_id = $test_id))
      AND ($target_vocabs IS NULL OR dst.vocabulary_id IN $target_vocabs)
    RETURN dst AS target,
           'both' AS direction,
           'maps_to' AS reason,
           '2-hop Maps to via ' + mid.vocabulary_id + ':' + mid.code AS relationship_id,
           r2.equivalence AS equivalence,
           2 AS hops
}
WITH target, direction, reason, relationship_id, equivalence, min(hops) AS hops
RETURN target.id AS id,
       direction,
       reason,
       relationship_id,
       equivalence,
       hops,
       target.vocabulary_id AS vocabulary_id,
       target.code AS code,
       target.display AS display,
       coalesce(target.system_uri, '') AS system_uri,
       target.coding_macro AS coding_macro,
       target.domain_id AS domain_id,
       target.concept_class_id AS concept_class_id,
       target.standard_concept AS standard_concept,
       target.invalid_reason AS invalid_reason
ORDER BY hops ASC, target.vocabulary_id ASC, target.code ASC
LIMIT $limit
"""

_SUBSUMES_CYPHER_TEMPLATE = """
MATCH (a:Concept {{vocabulary_id: $a_vocab, code: $a_code}})
WHERE (($test_id IS NULL AND a.test_id IS NULL) OR ($test_id IS NOT NULL AND a.test_id = $test_id))
MATCH (b:Concept {{vocabulary_id: $b_vocab, code: $b_code}})
WHERE (($test_id IS NULL AND b.test_id IS NULL) OR ($test_id IS NOT NULL AND b.test_id = $test_id))
RETURN a.id = b.id AS is_equivalent,
       EXISTS {{ (b)-[:IS_A*1..{max_depth}]->(a) }} AS a_subsumes_b,
       EXISTS {{ (a)-[:IS_A*1..{max_depth}]->(b) }} AS b_subsumes_a
LIMIT 1
"""

_DICOM_ELEMENT_FULLTEXT_CYPHER: LiteralString = """
CALL db.index.fulltext.queryNodes('dicom_element_fulltext_idx', $lucene_query)
YIELD node AS e, score
WHERE (($test_id IS NULL AND e.test_id IS NULL) OR ($test_id IS NOT NULL AND e.test_id = $test_id))
  AND (NOT $valid_only OR coalesce(e.is_retired, false) = false)
RETURN e.id AS id,
       'DicomElement' AS entity_type,
       e.tag AS tag,
       e.tag_hex AS tag_hex,
       e.keyword AS keyword,
       e.name AS name,
       e.vr AS vr,
       e.vm AS vm,
       coalesce(e.is_retired, false) AS is_retired,
       e.description AS description,
       coalesce(e.system_uri, 'http://gosmart.health/terminology/dicom-tag') AS system_uri,
       e.coding_macro AS coding_macro,
       score AS lexical_score
ORDER BY lexical_score DESC
LIMIT $limit
"""

_DICOM_STRUCTURE_FULLTEXT_CYPHER: LiteralString = """
CALL db.index.fulltext.queryNodes('dicom_structure_fulltext_idx', $lucene_query)
YIELD node AS n, score
WHERE (($test_id IS NULL AND n.test_id IS NULL) OR ($test_id IS NOT NULL AND n.test_id = $test_id))
  AND any(lbl IN labels(n) WHERE lbl IN $labels)
RETURN n.id AS id,
       head([lbl IN labels(n) WHERE lbl IN $labels]) AS entity_type,
       coalesce(n.sop_class_uid, n.ciod_id, n.module_id, n.macro_id, n.id) AS code,
       n.name AS name,
       n.description AS description,
       n.ciod_name AS ciod_name,
       coalesce(n.system_uri, '') AS system_uri,
       score AS lexical_score
ORDER BY lexical_score DESC
LIMIT $limit
"""

_DICOM_VECTOR_SEARCH_CYPHER: LiteralString = """
CALL db.index.vector.queryNodes('embedding_vector_idx', $fetch_k, $query_vec)
YIELD node AS emb, score
WHERE (($test_id IS NULL AND emb.test_id IS NULL) OR ($test_id IS NOT NULL AND emb.test_id = $test_id))
  AND emb.modelTag = $model_tag
MATCH (emb)-[:EMBEDS]->(n)
WHERE (($test_id IS NULL AND n.test_id IS NULL) OR ($test_id IS NOT NULL AND n.test_id = $test_id))
  AND any(lbl IN labels(n) WHERE lbl IN $labels)
  AND (NOT $valid_only OR coalesce(n.is_retired, false) = false)
WITH n, max(score) AS vector_score
RETURN n.id AS id,
       head([lbl IN labels(n) WHERE lbl IN $labels]) AS entity_type,
       n.tag AS tag,
       n.tag_hex AS tag_hex,
       n.keyword AS keyword,
       coalesce(n.sop_class_uid, n.ciod_id, n.module_id, n.macro_id, n.attribute_id, n.tag_hex, n.id) AS code,
       n.name AS name,
       n.vr AS vr,
       n.vm AS vm,
       coalesce(n.is_retired, false) AS is_retired,
       n.description AS description,
       n.ciod_name AS ciod_name,
       n.basic_profile AS basic_profile,
       coalesce(n.system_uri, '') AS system_uri,
       n.coding_macro AS coding_macro,
       vector_score
ORDER BY vector_score DESC
LIMIT $limit
"""


# -----------------------------------------------------------------------------
# Pure Helpers
# -----------------------------------------------------------------------------


def escape_lucene_term(term: str) -> str:
    """Escape Lucene reserved characters in a single token."""
    return _LUCENE_SPECIAL_RE.sub(r"\\\1", term)


def build_lucene_query(text: str) -> str:
    """Sanitize a free-text query string into a safe Lucene token query."""
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if not tokens:
        return ""
    escaped = [escape_lucene_term(t) for t in tokens]
    return " ".join(escaped)


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric vectors (handles non-unit vectors safely)."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / denom))


def _normalize_dicom_tag(raw: str) -> Tuple[str, str]:
    """Normalize DICOM tag input like '(0018,0050)' or '00180050' into (tag_hex, formatted_tag)."""
    hex_only = re.sub(r"[^0-9A-Fa-f]", "", raw).upper()
    if len(hex_only) == 8:
        return hex_only, f"({hex_only[:4]},{hex_only[4:]})"
    return raw.strip().upper(), raw.strip()


def _vocab_lookup_candidates(vocabulary_id: str) -> Tuple[str, ...]:
    trimmed = vocabulary_id.strip()
    if trimmed.upper() == "ICD10":
        return ("ICD10", "ICD10CM", "ICD10PCS")
    return (trimmed,)


def _build_concept_hit(
    row: Dict[str, Any],
    *,
    score: float,
    match_sources: List[MatchSource],
    lexical_score: Optional[float] = None,
    vector_score: Optional[float] = None,
    related: Optional[List[RelatedConceptHit]] = None,
) -> SearchHit:
    system_uri = str(row.get("system_uri") or "")
    code = str(row.get("code") or "")
    display = str(row.get("display") or "")
    macro = str(row.get("coding_macro") or f"{system_uri}|{code}|{display}")
    props: Dict[str, Any] = {
        "vocabulary": row.get("vocabulary_id"),
        "domain": row.get("domain_id"),
        "concept_class": row.get("concept_class_id"),
        "standard": row.get("standard_concept"),
        "deprecated": bool(row.get("is_deprecated") or row.get("invalid_reason") is not None),
    }
    if row.get("invalid_reason"):
        props["invalid_reason"] = row["invalid_reason"]
    if row.get("athena_concept_id") is not None:
        props["athena_concept_id"] = row["athena_concept_id"]
    if row.get("synonyms"):
        props["synonyms"] = list(row["synonyms"])
    if row.get("definition"):
        props["definition"] = row["definition"]
    if row.get("depth") is not None:
        props["depth"] = row["depth"]
    return SearchHit(
        id=str(row["id"]),
        entity_type=SearchEntityType.CONCEPT,
        score=score,
        lexical_score=lexical_score,
        vector_score=vector_score,
        match_sources=match_sources,
        coding=FHIRCoding(
            system=system_uri,
            code=code,
            display=display,
            version=row.get("system_version"),
        ),
        macro=macro,
        properties=props,
        related=related or [],
    )


def _build_dicom_hit(
    row: Dict[str, Any],
    *,
    score: float,
    match_sources: List[MatchSource],
    lexical_score: Optional[float] = None,
    vector_score: Optional[float] = None,
    related: Optional[List[RelatedConceptHit]] = None,
) -> SearchHit:
    entity_type = SearchEntityType(str(row.get("entity_type") or SearchEntityType.DICOM_ELEMENT.value))
    system_uri = str(row.get("system_uri") or "http://gosmart.health/terminology/dicom-tag")
    display = str(row.get("name") or "")
    if entity_type == SearchEntityType.DICOM_ELEMENT:
        code = str(row.get("tag_hex") or row.get("code") or "")
        macro = str(row.get("coding_macro") or f"{system_uri}|{code}|{display}")
        props: Dict[str, Any] = {
            "tag": row.get("tag"),
            "tag_hex": row.get("tag_hex"),
            "keyword": row.get("keyword"),
            "vr": row.get("vr"),
            "vm": row.get("vm"),
            "retired": bool(row.get("is_retired", False)),
        }
        if row.get("description"):
            props["description"] = row["description"]
    else:
        code = str(row.get("code") or "")
        macro = f"{system_uri}|{code}|{display}"
        props = {}
        if row.get("description"):
            props["description"] = row["description"]
        if row.get("ciod_name"):
            props["ciod_name"] = row["ciod_name"]
        if row.get("basic_profile"):
            props["basic_profile"] = row["basic_profile"]

    return SearchHit(
        id=str(row["id"]),
        entity_type=entity_type,
        score=score,
        lexical_score=lexical_score,
        vector_score=vector_score,
        match_sources=match_sources,
        coding=FHIRCoding(system=system_uri, code=code, display=display),
        macro=macro,
        properties={k: v for k, v in props.items() if v is not None},
        related=related or [],
    )


def _map_rel_type_to_reason(rel_type: str, direction: RelationDirection) -> MatchSource:
    if rel_type == "IS_A":
        return MatchSource.IS_A_PARENT if direction == RelationDirection.FORWARD else MatchSource.IS_A_CHILD
    if rel_type == "MAPS_TO":
        return MatchSource.MAPS_TO if direction == RelationDirection.FORWARD else MatchSource.MAPPED_FROM
    if rel_type == "REPLACED_BY":
        return MatchSource.REPLACED_BY if direction == RelationDirection.FORWARD else MatchSource.REPLACES
    if rel_type == "HAS_ATTRIBUTE":
        return MatchSource.HAS_ATTRIBUTE if direction == RelationDirection.FORWARD else MatchSource.USED_AS_ATTRIBUTE_BY
    return MatchSource.SHARED_ATTRIBUTE


def _build_candidate_embedding_text(row: Dict[str, Any], attrs: Sequence[str]) -> str:
    vocab = row.get("vocabulary_id") or ""
    domain = row.get("domain_id") or ""
    code = row.get("code") or ""
    display = row.get("display") or ""
    parts = [f"{vocab} {domain} ({code}): {display}".strip()]
    syns = row.get("synonyms") or []
    if syns:
        parts.append("Synonyms: " + "; ".join(str(s) for s in syns[:4]))
    if attrs:
        parts.append("Attributes: " + "; ".join(attrs[:4]))
    return " | ".join(parts)


def _run_coro_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Execute an async coroutine safely from synchronous code (including inside running Jupyter loops)."""
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


# -----------------------------------------------------------------------------
# SearchService
# -----------------------------------------------------------------------------


class SearchService:
    """Unified Clinical Terminology & DICOM Dictionary search and ontology traversal service."""

    def __init__(
        self,
        *,
        client: Optional[Neo4jClient] = None,
        embedding_service: Optional[EmbeddingService] = None,
        nlu_service: Optional[NLUParserProtocol] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or neo4j_client
        self.llm_config_service = LLMConfigService(client=self.client)
        self.embedding_service = embedding_service or EmbeddingService(
            client=self.client,
            llm_config_service=self.llm_config_service,
            settings=self.settings,
        )
        self._owns_default_nlu = nlu_service is None
        self.nlu_service: NLUParserProtocol = nlu_service or ClinicalNLUService(
            client=self.client,
            llm_config_service=self.llm_config_service,
            settings=self.settings,
        )

    # -------------------------------------------------------------------------
    # High-Level Unified Entrypoint
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        entity_types: Optional[Sequence[SearchEntityType | str]] = None,
        vocabularies: Optional[Sequence[str]] = None,
        domains: Optional[Sequence[str]] = None,
        valid_only: bool = True,
        standard_only: bool = False,
        include_related: bool = False,
        related_direction: RelationDirection = RelationDirection.BOTH,
        related_limit: int = 12,
        include_shared_attributes: bool = True,
        rerank_with_embeddings: bool = True,
        expand_with_llm: Optional[bool] = None,
        embedding_tag: str = DEFAULT_EMBEDDING_CONFIG_TAG,
        limit: int = 10,
        test_id: Optional[str] = None,
    ) -> SearchResponse:
        """Execute a unified, NLU-guided search across Clinical Concepts and/or DICOM Dictionary entities."""
        t0 = time.perf_counter()
        norm_test_id = normalize_test_id(test_id)
        warnings: List[str] = []

        if isinstance(self.nlu_service, (ClinicalNLUService, StubNLUService)):
            if isinstance(self.nlu_service, ClinicalNLUService) and self._owns_default_nlu:
                self.nlu_service.llm_config_service = self.llm_config_service
            nlu = await self.nlu_service.parse_async(
                query,
                test_id=norm_test_id,
                expand_with_llm=expand_with_llm,
            )
        else:
            nlu = await self.nlu_service.parse_async(query)

        for nlu_warning in nlu.warnings:
            if nlu_warning not in warnings:
                warnings.append(nlu_warning)

        explicit_entity_types = normalize_entity_types(entity_types)
        target_entity_types = self._resolve_entity_types(explicit_entity_types, nlu)

        explicit_vocabs: Optional[List[str]] = (
            [v.strip() for v in vocabularies if v and v.strip()] if vocabularies else None
        )
        resolved_vocabs: Optional[List[str]] = (
            explicit_vocabs
            if explicit_vocabs is not None
            else (list(nlu.inferred_vocabularies) if nlu.inferred_vocabularies else None)
        )
        resolved_domains: Optional[List[str]] = (
            [d.strip() for d in domains if d and d.strip()]
            if domains
            else (list(nlu.inferred_domains) if nlu.inferred_domains else None)
        )

        if not nlu.normalized_query and not nlu.extracted_codes:
            return SearchResponse(
                query=query,
                nlu=nlu,
                total_hits=0,
                hits=[],
                warnings=warnings,
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        # 1) Pure ontology graph intents (RELATED_LOOKUP, CROSSWALK, EXPAND_SUBTYPES)
        if nlu.extracted_codes and nlu.intent in {
            NLUIntent.RELATED_LOOKUP,
            NLUIntent.CROSSWALK,
            NLUIntent.EXPAND_SUBTYPES,
        }:
            hits = await self._handle_code_intent(
                nlu=nlu,
                include_related=include_related,
                related_direction=related_direction,
                related_limit=related_limit,
                include_shared_attributes=include_shared_attributes,
                valid_only=valid_only,
                rerank_with_embeddings=rerank_with_embeddings,
                embedding_tag=embedding_tag,
                limit=limit,
                test_id=norm_test_id,
                warnings=warnings,
            )
            duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return SearchResponse(
                query=query,
                nlu=nlu,
                total_hits=len(hits),
                hits=hits,
                warnings=warnings,
                duration_ms=duration_ms,
            )

        # 2) Unified Discovery Pipeline (SEARCH & EXACT_LOOKUP):
        #    a) Resolve deterministic exact codes at probability 1.0
        #    b) Run hybrid Lucene + Vector search across Clinical + DICOM for the reverse-looked-up
        #       rewritten query and LLM-synthesized alternate phrases at discounted probabilities
        deterministic_by_id: Dict[str, SearchHit] = {}
        for entity in nlu.extracted_codes:
            if entity.is_schema_loaded is False or entity.resolved is False:
                continue
            if entity.vocabulary_id in {"DICOM_TAG", "DICOM_UID"}:
                det_hit = await self.lookup_dicom(
                    entity.code,
                    include_related=include_related,
                    related_limit=related_limit,
                    test_id=norm_test_id,
                )
            else:
                det_hit = await self.lookup_code(
                    entity.vocabulary_id,
                    entity.code,
                    include_related=include_related,
                    related_direction=related_direction,
                    include_shared_attributes=include_shared_attributes,
                    related_limit=related_limit,
                    test_id=norm_test_id,
                )

            if det_hit is not None:
                entity.resolved = True
                if not entity.resolved_display:
                    entity.resolved_display = det_hit.coding.display
                deterministic_by_id.setdefault(det_hit.id, det_hit)
            elif entity.resolved is None:
                entity.resolved = False
                label = entity.raw_span or f"{entity.vocabulary_id} {entity.code}"
                msg = f"Code '{label}' was not found in the active dictionary database and was skipped."
                if msg not in warnings:
                    warnings.append(msg)

        has_deterministic = bool(deterministic_by_id)
        weighted_phrases = self._build_weighted_discovery_phrases(
            query=query,
            nlu=nlu,
            has_deterministic=has_deterministic,
        )

        # When codes were extracted, search BOTH Clinical Concepts and DICOM entities by default
        # so queries like "LOINC 87279-6" discover related terminology & DICOM attributes together.
        if explicit_entity_types is not None:
            expansion_entity_types = explicit_entity_types
        elif nlu.extracted_codes:
            expansion_entity_types = sorted(DEFAULT_UNIFIED_ENTITY_TYPES, key=lambda et: et.value)
        else:
            expansion_entity_types = target_entity_types

        expansion_vocabs = explicit_vocabs if nlu.extracted_codes else resolved_vocabs

        discovered_hits = await self._execute_multi_phrase_discovery(
            weighted_phrases=weighted_phrases,
            deterministic_ids=set(deterministic_by_id.keys()),
            has_deterministic=has_deterministic,
            entity_types=expansion_entity_types,
            vocabularies=expansion_vocabs,
            domains=resolved_domains,
            valid_only=valid_only,
            standard_only=standard_only,
            rerank_with_embeddings=rerank_with_embeddings,
            embedding_tag=embedding_tag,
            limit=limit,
            test_id=norm_test_id,
            warnings=warnings,
        )

        if include_related and discovered_hits:
            await self._attach_related_to_hits(
                discovered_hits,
                direction=related_direction,
                include_shared_attributes=include_shared_attributes,
                valid_only=valid_only,
                related_limit=related_limit,
                test_id=norm_test_id,
            )

        hits = (list(deterministic_by_id.values()) + discovered_hits)[:limit]
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return SearchResponse(
            query=query,
            nlu=nlu,
            total_hits=len(hits),
            hits=hits,
            warnings=warnings,
            duration_ms=duration_ms,
        )

    def search_sync(self, query: str, **kwargs: Any) -> SearchResponse:
        """Synchronous convenience wrapper around ``search``."""
        return _run_coro_sync(self.search(query, **kwargs))

    # -------------------------------------------------------------------------
    # Granular Lookup & Ontology Graph Primitives
    # -------------------------------------------------------------------------

    async def lookup_code(
        self,
        vocabulary_id: str,
        code: str,
        *,
        resolve_deprecated: bool = True,
        include_related: bool = False,
        related_direction: RelationDirection = RelationDirection.BOTH,
        include_shared_attributes: bool = True,
        related_limit: int = 12,
        test_id: Optional[str] = None,
    ) -> Optional[SearchHit]:
        """Exact O(1) indexed lookup of a clinical ``(:Concept {vocabulary_id, code})``."""
        norm_test_id = normalize_test_id(test_id)
        rows: List[Dict[str, Any]] = []
        for candidate_vocab in _vocab_lookup_candidates(vocabulary_id):
            rows = await self.client.execute_query(
                _LOOKUP_CONCEPT_CYPHER,
                {"vocab": candidate_vocab, "code": code.strip(), "test_id": norm_test_id},
            )
            if rows:
                break
        if not rows:
            return None

        row = rows[0]
        related: List[RelatedConceptHit] = []
        if include_related:
            related = await self.find_related_concepts(
                row["vocabulary_id"],
                row["code"],
                direction=related_direction,
                include_shared_attributes=include_shared_attributes,
                limit=related_limit,
                test_id=norm_test_id,
            )
        elif resolve_deprecated and row.get("invalid_reason") is not None:
            forward_rels = await self.find_related_concepts(
                row["vocabulary_id"],
                row["code"],
                direction=RelationDirection.FORWARD,
                include_shared_attributes=False,
                limit=6,
                test_id=norm_test_id,
            )
            related = [r for r in forward_rels if r.relation_reason in {MatchSource.REPLACED_BY, MatchSource.MAPS_TO}]

        return _build_concept_hit(
            row,
            score=1.0,
            match_sources=[MatchSource.EXACT_CODE],
            related=related,
        )

    def lookup_code_sync(self, vocabulary_id: str, code: str, **kwargs: Any) -> Optional[SearchHit]:
        """Synchronous convenience wrapper around ``lookup_code``."""
        return _run_coro_sync(self.lookup_code(vocabulary_id, code, **kwargs))

    async def lookup_dicom(
        self,
        tag_or_keyword: str,
        *,
        include_related: bool = False,
        related_limit: int = 12,
        test_id: Optional[str] = None,
    ) -> Optional[SearchHit]:
        """Direct lookup of a DICOM Data Element (tag/hex/keyword) or SOP Class UID."""
        raw = tag_or_keyword.strip()
        if not raw:
            return None
        norm_test_id = normalize_test_id(test_id)

        if raw.startswith("1.2.840.10008."):
            rows = await self.client.execute_query(
                _LOOKUP_DICOM_SOP_CYPHER,
                {"uid": raw, "test_id": norm_test_id},
            )
            if not rows:
                return None
            return _build_dicom_hit(rows[0], score=1.0, match_sources=[MatchSource.EXACT_CODE])

        tag_hex, tag_formatted = _normalize_dicom_tag(raw)
        rows = await self.client.execute_query(
            _LOOKUP_DICOM_ELEMENT_CYPHER,
            {
                "tag_hex": tag_hex,
                "tag_formatted": tag_formatted,
                "raw": raw,
                "test_id": norm_test_id,
            },
        )
        if not rows:
            return None

        row = rows[0]
        related: List[RelatedConceptHit] = []
        if include_related:
            rel_rows = await self.client.execute_query(
                _DICOM_ELEMENT_RELATED_CYPHER,
                {"node_id": row["id"], "limit": related_limit, "test_id": norm_test_id},
            )
            for rr in rel_rows:
                sys_uri = str(rr.get("system_uri") or "")
                c_code = str(rr.get("code") or "")
                disp = str(rr.get("display") or "")
                related.append(
                    RelatedConceptHit(
                        id=str(rr["id"]),
                        entity_type=SearchEntityType(str(rr["entity_type"])),
                        direction=RelationDirection(str(rr["direction"])),
                        relation_reason=MatchSource.DICOM_RELATION,
                        relationship_id=str(rr.get("relationship_id") or ""),
                        score=_RELATION_BASE_SCORE[MatchSource.DICOM_RELATION],
                        coding=FHIRCoding(system=sys_uri, code=c_code, display=disp),
                        macro=f"{sys_uri}|{c_code}|{disp}",
                        properties=dict(rr.get("extra_props") or {}),
                    )
                )

        return _build_dicom_hit(
            row,
            score=1.0,
            match_sources=[MatchSource.EXACT_CODE],
            related=related,
        )

    def lookup_dicom_sync(self, tag_or_keyword: str, **kwargs: Any) -> Optional[SearchHit]:
        """Synchronous convenience wrapper around ``lookup_dicom``."""
        return _run_coro_sync(self.lookup_dicom(tag_or_keyword, **kwargs))

    async def find_related_concepts(
        self,
        vocabulary_id: str,
        code: str,
        *,
        direction: RelationDirection = RelationDirection.BOTH,
        include_shared_attributes: bool = True,
        valid_only: bool = True,
        min_shared_axes: int = 2,
        limit: int = 20,
        test_id: Optional[str] = None,
    ) -> List[RelatedConceptHit]:
        """Aggregate forward, reverse, and 2-hop shared-attribute related concepts for a seed code."""
        norm_test_id = normalize_test_id(test_id)
        per_branch_limit = max(limit * 2, 24)
        base_params: Dict[str, Any] = {
            "vocab": vocabulary_id.strip(),
            "code": code.strip(),
            "valid_only": valid_only,
            "limit": per_branch_limit,
            "test_id": norm_test_id,
        }

        deduped: Dict[str, RelatedConceptHit] = {}

        def _register(hit: RelatedConceptHit) -> None:
            existing = deduped.get(hit.id)
            if existing is None or hit.score > existing.score:
                deduped[hit.id] = hit

        # 1) Forward edges: (c)-[:IS_A|MAPS_TO|REPLACED_BY|HAS_ATTRIBUTE]->(dst)
        if direction in {RelationDirection.FORWARD, RelationDirection.BOTH}:
            fwd_rows = await self.client.execute_query(_RELATED_FORWARD_CYPHER, base_params)
            for row in fwd_rows:
                reason = _map_rel_type_to_reason(str(row["rel_type"]), RelationDirection.FORWARD)
                _register(self._row_to_related_hit(row, RelationDirection.FORWARD, reason))

        # 2) Reverse edges: (src)-[:IS_A|MAPS_TO|REPLACED_BY|HAS_ATTRIBUTE]->(c)
        if direction in {RelationDirection.REVERSE, RelationDirection.BOTH}:
            rev_rows = await self.client.execute_query(_RELATED_REVERSE_CYPHER, base_params)
            for row in rev_rows:
                reason = _map_rel_type_to_reason(str(row["rel_type"]), RelationDirection.REVERSE)
                _register(self._row_to_related_hit(row, RelationDirection.REVERSE, reason))

        # 3) 2-hop shared-attribute discovery (Forward -> Reverse across specific non-hub axes)
        if include_shared_attributes:
            shared_params = {
                **base_params,
                "max_axis_degree": _MAX_SHARED_AXIS_DEGREE,
                "min_shared_axes": max(1, min_shared_axes),
            }
            shared_rows = await self.client.execute_query(_RELATED_SHARED_ATTRS_CYPHER, shared_params)
            if not shared_rows and min_shared_axes > 1:
                shared_params["min_shared_axes"] = 1
                shared_rows = await self.client.execute_query(_RELATED_SHARED_ATTRS_CYPHER, shared_params)

            for row in shared_rows:
                shared_count = int(row.get("shared_count") or 1)
                axes = [str(a) for a in (row.get("shared_axes") or []) if a]
                score = _RELATION_BASE_SCORE[MatchSource.SHARED_ATTRIBUTE] + min(shared_count, 5) * 0.04
                _register(
                    self._row_to_related_hit(
                        row,
                        RelationDirection.BOTH,
                        MatchSource.SHARED_ATTRIBUTE,
                        score_override=score,
                        relationship_id=f"shared_attributes({shared_count})",
                        shared_axes=axes,
                    )
                )

        ranked = sorted(
            deduped.values(),
            key=lambda item: (-item.score, item.coding.display.lower()),
        )
        return ranked[:limit]

    def find_related_concepts_sync(
        self,
        vocabulary_id: str,
        code: str,
        **kwargs: Any,
    ) -> List[RelatedConceptHit]:
        """Synchronous convenience wrapper around ``find_related_concepts``."""
        return _run_coro_sync(self.find_related_concepts(vocabulary_id, code, **kwargs))

    async def expand_descendants(
        self,
        vocabulary_id: str,
        code: str,
        *,
        max_depth: int = 2,
        valid_only: bool = True,
        limit: int = 100,
        test_id: Optional[str] = None,
    ) -> List[SearchHit]:
        """Expand valid ``[:IS_A*1..max_depth]`` subtypes beneath a clinical root concept (FHIR ``$expand``)."""
        bounded_depth = max(1, min(int(max_depth), 8))
        norm_test_id = normalize_test_id(test_id)
        query = _EXPAND_DESCENDANTS_CYPHER_TEMPLATE.format(max_depth=bounded_depth)
        rows = await self.client.execute_query(
            query,
            {
                "vocab": vocabulary_id.strip(),
                "code": code.strip(),
                "valid_only": valid_only,
                "limit": limit,
                "test_id": norm_test_id,
            },
        )
        hits: List[SearchHit] = []
        for row in rows:
            depth = int(row.get("depth") or 1)
            score = max(0.1, round(1.0 - (depth - 1) * 0.1, 4))
            hits.append(
                _build_concept_hit(
                    row,
                    score=score,
                    match_sources=[MatchSource.IS_A_CHILD],
                )
            )
        return hits

    def expand_descendants_sync(self, vocabulary_id: str, code: str, **kwargs: Any) -> List[SearchHit]:
        """Synchronous convenience wrapper around ``expand_descendants``."""
        return _run_coro_sync(self.expand_descendants(vocabulary_id, code, **kwargs))

    async def crosswalk(
        self,
        vocabulary_id: str,
        code: str,
        *,
        target_vocabularies: Optional[Sequence[str]] = None,
        max_hops: int = 2,
        limit: int = 25,
        test_id: Optional[str] = None,
    ) -> List[RelatedConceptHit]:
        """Translate a concept across code systems via direct and 2-hop ``[:MAPS_TO]`` paths."""
        norm_test_id = normalize_test_id(test_id)
        target_vocabs = [v.strip() for v in target_vocabularies if v and v.strip()] if target_vocabularies else None
        rows = await self.client.execute_query(
            _CROSSWALK_CYPHER,
            {
                "vocab": vocabulary_id.strip(),
                "code": code.strip(),
                "target_vocabs": target_vocabs,
                "max_hops": max(1, min(int(max_hops), 2)),
                "limit": limit,
                "test_id": norm_test_id,
            },
        )
        deduped: Dict[str, RelatedConceptHit] = {}
        for row in rows:
            hops = int(row.get("hops") or 1)
            direction = RelationDirection(str(row.get("direction") or "forward"))
            reason = MatchSource(str(row.get("reason") or MatchSource.MAPS_TO.value))
            score = 0.98 if hops == 1 else 0.80
            hit = self._row_to_related_hit(row, direction, reason, score_override=score)
            existing = deduped.get(hit.id)
            if existing is None or hit.score > existing.score:
                deduped[hit.id] = hit
        return sorted(deduped.values(), key=lambda item: (-item.score, item.coding.code))[:limit]

    def crosswalk_sync(self, vocabulary_id: str, code: str, **kwargs: Any) -> List[RelatedConceptHit]:
        """Synchronous convenience wrapper around ``crosswalk``."""
        return _run_coro_sync(self.crosswalk(vocabulary_id, code, **kwargs))

    async def subsumes(
        self,
        ancestor_vocabulary: str,
        ancestor_code: str,
        descendant_vocabulary: str,
        descendant_code: str,
        *,
        max_depth: int = 10,
        test_id: Optional[str] = None,
    ) -> SubsumptionOutcome:
        """Evaluate FHIR Terminology Service ``$subsumes`` between two clinical concepts."""
        bounded_depth = max(1, min(int(max_depth), 15))
        norm_test_id = normalize_test_id(test_id)
        query = _SUBSUMES_CYPHER_TEMPLATE.format(max_depth=bounded_depth)
        rows = await self.client.execute_query(
            query,
            {
                "a_vocab": ancestor_vocabulary.strip(),
                "a_code": ancestor_code.strip(),
                "b_vocab": descendant_vocabulary.strip(),
                "b_code": descendant_code.strip(),
                "test_id": norm_test_id,
            },
        )
        if not rows:
            return "not-subsumed"
        row = rows[0]
        if row.get("is_equivalent"):
            return "equivalent"
        if row.get("a_subsumes_b"):
            return "subsumes"
        if row.get("b_subsumes_a"):
            return "subsumed-by"
        return "not-subsumed"

    def subsumes_sync(
        self,
        ancestor_vocabulary: str,
        ancestor_code: str,
        descendant_vocabulary: str,
        descendant_code: str,
        **kwargs: Any,
    ) -> SubsumptionOutcome:
        """Synchronous convenience wrapper around ``subsumes``."""
        return _run_coro_sync(
            self.subsumes(
                ancestor_vocabulary,
                ancestor_code,
                descendant_vocabulary,
                descendant_code,
                **kwargs,
            )
        )

    # -------------------------------------------------------------------------
    # Internal Routing & Hybrid Search Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_entity_types(
        explicit: Optional[List[SearchEntityType]],
        nlu: NLUParseResult,
    ) -> List[SearchEntityType]:
        if explicit is not None:
            return explicit
        if nlu.inferred_entity_types:
            return list(nlu.inferred_entity_types)
        return sorted(DEFAULT_UNIFIED_ENTITY_TYPES, key=lambda et: et.value)

    @staticmethod
    def _build_weighted_discovery_phrases(
        *,
        query: str,
        nlu: NLUParseResult,
        has_deterministic: bool,
    ) -> List[Tuple[str, float, float, Tuple[MatchSource, ...]]]:
        """Build ``(phrase_text, weight, cap, extra_sources)`` tuples for multi-phrase discovery."""
        if nlu.rewritten_query is not None:
            primary = nlu.rewritten_query.strip()
        elif nlu.extracted_codes and all(e.resolved is False for e in nlu.extracted_codes):
            # Guardrail: never send unresolved raw code numbers into lexical/vector discovery
            primary = ""
        elif nlu.extracted_codes:
            rewritten = ClinicalNLUService._rewrite_query_with_resolutions(
                " ".join(query.strip().split()),
                nlu.extracted_codes,
            )
            primary = rewritten.strip()
        else:
            primary = (nlu.normalized_query or query).strip()

        phrases: List[Tuple[str, float, float, Tuple[MatchSource, ...]]] = []
        seen_lower: Set[str] = set()

        if primary:
            t2_weight = _REWRITTEN_DISCOUNT_WITH_EXACT if has_deterministic else 1.0
            t2_cap = _REWRITTEN_CAP_WITH_EXACT if has_deterministic else 1.0
            phrases.append((primary, t2_weight, t2_cap, ()))
            seen_lower.add(primary.lower())

        for alt in nlu.synthesized_phrases:
            clean_alt = " ".join(alt.strip().split())
            if not clean_alt or clean_alt.lower() in seen_lower:
                continue
            seen_lower.add(clean_alt.lower())
            phrases.append(
                (
                    clean_alt,
                    _SYNTHESIZED_PHRASE_DISCOUNT,
                    _SYNTHESIZED_PHRASE_CAP,
                    (MatchSource.LLM_EXPANSION,),
                )
            )
        return phrases

    async def _execute_multi_phrase_discovery(
        self,
        *,
        weighted_phrases: Sequence[Tuple[str, float, float, Tuple[MatchSource, ...]]],
        deterministic_ids: Set[str],
        has_deterministic: bool,
        entity_types: Sequence[SearchEntityType],
        vocabularies: Optional[List[str]],
        domains: Optional[List[str]],
        valid_only: bool,
        standard_only: bool,
        rerank_with_embeddings: bool,
        embedding_tag: str,
        limit: int,
        test_id: Optional[str],
        warnings: List[str],
    ) -> List[SearchHit]:
        """Run Clinical + DICOM hybrid search across weighted phrases and aggregate with discounted probabilities."""
        if not weighted_phrases:
            return []

        clinical_requested = SearchEntityType.CONCEPT in entity_types
        dicom_labels = [et for et in entity_types if et in DICOM_ENTITY_TYPES]
        if not clinical_requested and not dicom_labels:
            return []

        # Pre-batch phrase embeddings when running multiple phrases so we avoid N separate HTTP calls
        phrase_vec_map: Dict[str, List[float]] = {}
        candidate_vec_cache: Optional[Dict[str, List[float]]] = None
        if rerank_with_embeddings and len(weighted_phrases) > 1:
            phrase_texts = [p[0] for p in weighted_phrases]
            vecs = await self._safe_embed_texts(phrase_texts, embedding_tag=embedding_tag, warnings=warnings)
            if vecs is not None and len(vecs) == len(phrase_texts):
                phrase_vec_map = {txt: vec for txt, vec in zip(phrase_texts, vecs)}
                candidate_vec_cache = {}

        per_phrase_limit = max(limit * 2, 15)
        best_hit_by_id: Dict[str, SearchHit] = {}
        max_weighted_by_id: Dict[str, float] = {}
        max_cap_by_id: Dict[str, float] = {}
        phrase_hits_count: Dict[str, int] = {}
        merged_sources_by_id: Dict[str, Set[MatchSource]] = {}
        best_lex_by_id: Dict[str, float] = {}
        best_vec_by_id: Dict[str, float] = {}

        for phrase_text, weight, cap, extra_sources in weighted_phrases:
            q_vec = phrase_vec_map.get(phrase_text)
            phrase_results: List[SearchHit] = []

            if clinical_requested:
                c_hits = await self._search_clinical(
                    query_text=phrase_text,
                    vocabularies=vocabularies,
                    domains=domains,
                    valid_only=valid_only,
                    standard_only=standard_only,
                    rerank_with_embeddings=rerank_with_embeddings,
                    embedding_tag=embedding_tag,
                    limit=per_phrase_limit,
                    test_id=test_id,
                    warnings=warnings,
                    precomputed_query_vec=q_vec,
                    candidate_vec_cache=candidate_vec_cache,
                )
                phrase_results.extend(c_hits)

            if dicom_labels:
                d_hits = await self._search_dicom(
                    query_text=phrase_text,
                    entity_types=dicom_labels,
                    valid_only=valid_only,
                    rerank_with_embeddings=rerank_with_embeddings,
                    embedding_tag=embedding_tag,
                    limit=per_phrase_limit,
                    test_id=test_id,
                    warnings=warnings,
                    precomputed_query_vec=q_vec,
                )
                phrase_results.extend(d_hits)

            seen_in_this_phrase: Set[str] = set()
            for hit in phrase_results:
                if hit.id in deterministic_ids:
                    continue

                discounted = min(cap, round(hit.score * weight, 4))
                if hit.id not in seen_in_this_phrase:
                    seen_in_this_phrase.add(hit.id)
                    phrase_hits_count[hit.id] = phrase_hits_count.get(hit.id, 0) + 1

                src_set = merged_sources_by_id.setdefault(hit.id, set())
                src_set.update(hit.match_sources)
                src_set.update(extra_sources)

                if hit.lexical_score is not None:
                    best_lex_by_id[hit.id] = max(best_lex_by_id.get(hit.id, 0.0), hit.lexical_score)
                if hit.vector_score is not None:
                    prev_vec = best_vec_by_id.get(hit.id)
                    best_vec_by_id[hit.id] = hit.vector_score if prev_vec is None else max(prev_vec, hit.vector_score)

                max_cap_by_id[hit.id] = max(max_cap_by_id.get(hit.id, 0.0), cap)
                if hit.id not in max_weighted_by_id or discounted > max_weighted_by_id[hit.id]:
                    max_weighted_by_id[hit.id] = discounted
                    best_hit_by_id[hit.id] = hit

        aggregated: List[SearchHit] = []
        for hid, base_hit in best_hit_by_id.items():
            overall_cap = _REWRITTEN_CAP_WITH_EXACT if has_deterministic else max_cap_by_id[hid]
            consensus_boost = _CONSENSUS_BOOST_PER_EXTRA_PHRASE * max(0, phrase_hits_count.get(hid, 1) - 1)
            final_score = round(min(overall_cap, max_weighted_by_id[hid] + consensus_boost), 4)
            ordered_sources = sorted(merged_sources_by_id[hid], key=lambda s: s.value)
            aggregated.append(
                base_hit.model_copy(
                    update={
                        "score": final_score,
                        "match_sources": ordered_sources,
                        "lexical_score": best_lex_by_id.get(hid),
                        "vector_score": best_vec_by_id.get(hid),
                    }
                )
            )

        aggregated.sort(key=lambda h: (-h.score, h.coding.display.lower()))
        return aggregated[:limit]

    async def _handle_code_intent(
        self,
        *,
        nlu: NLUParseResult,
        include_related: bool,
        related_direction: RelationDirection,
        related_limit: int,
        include_shared_attributes: bool,
        valid_only: bool,
        rerank_with_embeddings: bool,
        embedding_tag: str,
        limit: int,
        test_id: Optional[str],
        warnings: List[str],
    ) -> List[SearchHit]:
        hits: List[SearchHit] = []
        for entity in nlu.extracted_codes:
            if entity.is_schema_loaded is False:
                continue
            if entity.vocabulary_id in {"DICOM_TAG", "DICOM_UID"}:
                want_related = include_related or nlu.intent == NLUIntent.RELATED_LOOKUP
                hit = await self.lookup_dicom(
                    entity.code,
                    include_related=want_related,
                    related_limit=related_limit,
                    test_id=test_id,
                )
                if hit is not None:
                    hits.append(hit)
                continue

            if nlu.intent == NLUIntent.EXACT_LOOKUP:
                hit = await self.lookup_code(
                    entity.vocabulary_id,
                    entity.code,
                    include_related=include_related,
                    related_direction=related_direction,
                    include_shared_attributes=include_shared_attributes,
                    related_limit=related_limit,
                    test_id=test_id,
                )
                if hit is not None:
                    hits.append(hit)
            elif nlu.intent == NLUIntent.RELATED_LOOKUP:
                seed = await self.lookup_code(
                    entity.vocabulary_id,
                    entity.code,
                    resolve_deprecated=False,
                    include_related=False,
                    test_id=test_id,
                )
                if seed is None:
                    continue
                resolved_vocab = str(seed.properties.get("vocabulary") or entity.vocabulary_id)
                related = await self.find_related_concepts(
                    resolved_vocab,
                    seed.coding.code,
                    direction=related_direction,
                    include_shared_attributes=include_shared_attributes,
                    valid_only=valid_only,
                    limit=max(related_limit, limit),
                    test_id=test_id,
                )
                if nlu.modifier_text and related and rerank_with_embeddings:
                    related = await self._rerank_related_with_embeddings(
                        modifier_text=nlu.modifier_text,
                        related=related,
                        embedding_tag=embedding_tag,
                        warnings=warnings,
                    )
                seed.related = related[:related_limit]
                hits.append(seed)
            elif nlu.intent == NLUIntent.CROSSWALK:
                seed = await self.lookup_code(
                    entity.vocabulary_id,
                    entity.code,
                    resolve_deprecated=False,
                    include_related=False,
                    test_id=test_id,
                )
                if seed is None:
                    continue
                resolved_vocab = str(seed.properties.get("vocabulary") or entity.vocabulary_id)
                mapped = await self.crosswalk(
                    resolved_vocab,
                    seed.coding.code,
                    target_vocabularies=nlu.target_vocabularies or None,
                    limit=max(related_limit, limit),
                    test_id=test_id,
                )
                seed.related = mapped[:related_limit]
                hits.append(seed)
            elif nlu.intent == NLUIntent.EXPAND_SUBTYPES:
                seed = await self.lookup_code(
                    entity.vocabulary_id,
                    entity.code,
                    resolve_deprecated=False,
                    include_related=False,
                    test_id=test_id,
                )
                resolved_vocab = (
                    str(seed.properties.get("vocabulary") or entity.vocabulary_id)
                    if seed is not None
                    else entity.vocabulary_id
                )
                resolved_code = seed.coding.code if seed is not None else entity.code
                if seed is not None:
                    hits.append(seed)
                descendants = await self.expand_descendants(
                    resolved_vocab,
                    resolved_code,
                    max_depth=2,
                    valid_only=valid_only,
                    limit=max(limit - len(hits), 1),
                    test_id=test_id,
                )
                hits.extend(descendants)

        return hits[:limit]

    async def _search_clinical(
        self,
        *,
        query_text: str,
        vocabularies: Optional[List[str]],
        domains: Optional[List[str]],
        valid_only: bool,
        standard_only: bool,
        rerank_with_embeddings: bool,
        embedding_tag: str,
        limit: int,
        test_id: Optional[str],
        warnings: List[str],
        precomputed_query_vec: Optional[List[float]] = None,
        candidate_vec_cache: Optional[Dict[str, List[float]]] = None,
    ) -> List[SearchHit]:
        lucene_query = build_lucene_query(query_text)
        if not lucene_query:
            return []

        candidate_limit = max(limit * 5, 40)
        lexical_rows = await self.client.execute_query(
            _CONCEPT_FULLTEXT_CYPHER,
            {
                "lucene_query": lucene_query,
                "vocabs": vocabularies,
                "domains": domains,
                "valid_only": valid_only,
                "standard_only": standard_only,
                "candidate_limit": candidate_limit,
                "test_id": test_id,
            },
        )
        if not lexical_rows:
            return []

        # Collect candidates and boost canonical 1-hop [:IS_A] parents of top Lucene hits
        candidates: Dict[str, Dict[str, Any]] = {}
        lexical_scores: Dict[str, float] = {}
        sources: Dict[str, Set[MatchSource]] = {}
        convergence_counts: Dict[str, int] = {}

        max_lex = max(float(r.get("lexical_score") or 0.0) for r in lexical_rows) or 1.0
        for row in lexical_rows:
            cid = str(row["id"])
            candidates[cid] = row
            lexical_scores[cid] = float(row.get("lexical_score") or 0.0)
            sources[cid] = {MatchSource.FULLTEXT}
            convergence_counts[cid] = 0

        top_seed_ids = [str(r["id"]) for r in lexical_rows[:15]]
        parent_rows = await self.client.execute_query(
            _CONCEPT_PARENT_CONVERGENCE_CYPHER,
            {
                "seed_ids": top_seed_ids,
                "vocabs": vocabularies,
                "domains": domains,
                "valid_only": valid_only,
                "standard_only": standard_only,
                "parent_limit": min(limit, 10),
                "test_id": test_id,
            },
        )
        for prow in parent_rows:
            pid = str(prow["id"])
            conv = int(prow.get("convergence") or 0)
            if pid not in candidates:
                candidates[pid] = prow
                lexical_scores[pid] = 0.0
                sources[pid] = {MatchSource.IS_A_PARENT}
            else:
                sources[pid].add(MatchSource.IS_A_PARENT)
            convergence_counts[pid] = max(convergence_counts.get(pid, 0), conv)

        # Optionally rerank candidate pool on-the-fly with embedding-v1
        vector_scores: Dict[str, float] = {}
        if rerank_with_embeddings and candidates:
            candidate_ids = list(candidates.keys())
            missing_ids = (
                [cid for cid in candidate_ids if cid not in candidate_vec_cache]
                if candidate_vec_cache is not None
                else candidate_ids
            )
            attrs_by_id: Dict[str, List[str]] = {}
            if missing_ids:
                attr_rows = await self.client.execute_query(
                    _HYDRATE_CONCEPT_ATTRS_CYPHER,
                    {"concept_ids": missing_ids, "test_id": test_id},
                )
                attrs_by_id = {str(r["id"]): [a for a in (r.get("attrs") or []) if a] for r in attr_rows}

            if precomputed_query_vec is not None:
                query_vec: Optional[List[float]] = precomputed_query_vec
                if missing_ids:
                    missing_texts = [
                        _build_candidate_embedding_text(candidates[cid], attrs_by_id.get(cid, []))
                        for cid in missing_ids
                    ]
                    missing_vecs = await self._safe_embed_texts(
                        missing_texts,
                        embedding_tag=embedding_tag,
                        warnings=warnings,
                    )
                    if missing_vecs is not None and len(missing_vecs) == len(missing_ids):
                        for cid, cvec in zip(missing_ids, missing_vecs):
                            if candidate_vec_cache is not None:
                                candidate_vec_cache[cid] = cvec
                            vector_scores[cid] = cosine_similarity(query_vec, cvec)
                            sources[cid].add(MatchSource.HYBRID)
                    else:
                        query_vec = None
                if query_vec is not None and candidate_vec_cache is not None:
                    for cid in candidate_ids:
                        if cid not in vector_scores and cid in candidate_vec_cache:
                            vector_scores[cid] = cosine_similarity(query_vec, candidate_vec_cache[cid])
                            sources[cid].add(MatchSource.HYBRID)
            else:
                candidate_texts = [
                    _build_candidate_embedding_text(candidates[cid], attrs_by_id.get(cid, [])) for cid in candidate_ids
                ]
                vectors = await self._safe_embed_texts(
                    [query_text, *candidate_texts],
                    embedding_tag=embedding_tag,
                    warnings=warnings,
                )
                if vectors is not None and len(vectors) == len(candidate_ids) + 1:
                    q_vec_single = vectors[0]
                    for cid, cand_vec in zip(candidate_ids, vectors[1:]):
                        if candidate_vec_cache is not None:
                            candidate_vec_cache[cid] = cand_vec
                        vector_scores[cid] = cosine_similarity(q_vec_single, cand_vec)
                        sources[cid].add(MatchSource.HYBRID)

        # Compute final fused scores
        hits: List[SearchHit] = []
        for cid, row in candidates.items():
            raw_lex = lexical_scores.get(cid, 0.0)
            norm_lex = (raw_lex / max_lex) if max_lex > 0 else 0.0
            conv_boost = min(convergence_counts.get(cid, 0) / 3.0, 1.0)
            std_boost = 0.03 if row.get("standard_concept") == "S" else 0.0
            vec_score = vector_scores.get(cid)

            if vec_score is not None:
                final_score = 0.62 * vec_score + 0.25 * norm_lex + 0.10 * conv_boost + std_boost
            else:
                final_score = 0.72 * norm_lex + 0.25 * conv_boost + std_boost

            hits.append(
                _build_concept_hit(
                    row,
                    score=round(final_score, 4),
                    match_sources=sorted(sources[cid], key=lambda s: s.value),
                    lexical_score=raw_lex if raw_lex > 0 else None,
                    vector_score=vec_score,
                )
            )

        hits.sort(key=lambda h: (-h.score, h.coding.display.lower()))
        return hits[:limit]

    async def _search_dicom(
        self,
        *,
        query_text: str,
        entity_types: Sequence[SearchEntityType],
        valid_only: bool,
        rerank_with_embeddings: bool,
        embedding_tag: str,
        limit: int,
        test_id: Optional[str],
        warnings: List[str],
        precomputed_query_vec: Optional[List[float]] = None,
    ) -> List[SearchHit]:
        lucene_query = build_lucene_query(query_text)
        labels = [et.value for et in entity_types]
        fetch_limit = max(limit * 3, 25)

        rows_by_id: Dict[str, Dict[str, Any]] = {}
        lex_ranks: Dict[str, int] = {}
        lex_scores: Dict[str, float] = {}
        vec_ranks: Dict[str, int] = {}
        vec_scores: Dict[str, float] = {}

        # 1) Lexical retrieval over DICOM full-text indexes
        if lucene_query:
            if SearchEntityType.DICOM_ELEMENT in entity_types:
                elem_rows = await self.client.execute_query(
                    _DICOM_ELEMENT_FULLTEXT_CYPHER,
                    {
                        "lucene_query": lucene_query,
                        "valid_only": valid_only,
                        "limit": fetch_limit,
                        "test_id": test_id,
                    },
                )
                for rank, row in enumerate(elem_rows, start=1):
                    nid = str(row["id"])
                    rows_by_id[nid] = row
                    lex_ranks[nid] = rank
                    lex_scores[nid] = float(row.get("lexical_score") or 0.0)

            struct_labels = [
                lbl for lbl in labels if lbl in {"DicomCIOD", "DicomModule", "DicomMacro", "DicomSOPClass"}
            ]
            if struct_labels:
                struct_rows = await self.client.execute_query(
                    _DICOM_STRUCTURE_FULLTEXT_CYPHER,
                    {
                        "lucene_query": lucene_query,
                        "labels": struct_labels,
                        "limit": fetch_limit,
                        "test_id": test_id,
                    },
                )
                for rank, row in enumerate(struct_rows, start=1):
                    nid = str(row["id"])
                    rows_by_id.setdefault(nid, row)
                    if nid not in lex_ranks or rank < lex_ranks[nid]:
                        lex_ranks[nid] = rank
                        lex_scores[nid] = float(row.get("lexical_score") or 0.0)

        # 2) Precomputed HNSW Vector Index retrieval (embedding_vector_idx)
        if rerank_with_embeddings:
            query_vec = precomputed_query_vec
            if query_vec is None:
                vectors = await self._safe_embed_texts([query_text], embedding_tag=embedding_tag, warnings=warnings)
                if vectors is not None and vectors:
                    query_vec = vectors[0]
            if query_vec is not None:
                vec_rows = await self.client.execute_query(
                    _DICOM_VECTOR_SEARCH_CYPHER,
                    {
                        "fetch_k": max(limit * 6, 48),
                        "query_vec": query_vec,
                        "model_tag": embedding_tag,
                        "labels": labels,
                        "valid_only": valid_only,
                        "limit": fetch_limit,
                        "test_id": test_id,
                    },
                )
                for rank, row in enumerate(vec_rows, start=1):
                    nid = str(row["id"])
                    rows_by_id.setdefault(nid, row)
                    vec_ranks[nid] = rank
                    vec_scores[nid] = float(row.get("vector_score") or 0.0)

        if not rows_by_id:
            return []

        max_lex = max(lex_scores.values(), default=1.0) or 1.0
        hits: List[SearchHit] = []
        for nid, row in rows_by_id.items():
            raw_lex = lex_scores.get(nid)
            v_score = vec_scores.get(nid)
            norm_lex = (raw_lex / max_lex) if raw_lex is not None else 0.0

            match_sources: List[MatchSource] = []
            if raw_lex is not None:
                match_sources.append(MatchSource.FULLTEXT)
            if v_score is not None:
                match_sources.append(MatchSource.VECTOR)
            if raw_lex is not None and v_score is not None:
                match_sources.append(MatchSource.HYBRID)

            # Scale score to [0, 1] comparable with Clinical hits
            if v_score is not None and raw_lex is not None:
                rrf_bonus = (1.0 / (20 + lex_ranks[nid])) + (1.0 / (20 + vec_ranks[nid]))
                score = 0.65 * v_score + 0.25 * norm_lex + min(rrf_bonus * 2.0, 0.10)
            elif v_score is not None:
                score = 0.85 * v_score
            else:
                score = 0.72 * norm_lex

            hits.append(
                _build_dicom_hit(
                    row,
                    score=round(score, 4),
                    match_sources=match_sources,
                    lexical_score=raw_lex,
                    vector_score=v_score,
                )
            )

        hits.sort(key=lambda h: (-h.score, h.coding.display.lower()))
        return hits[:limit]

    async def _rerank_related_with_embeddings(
        self,
        *,
        modifier_text: str,
        related: List[RelatedConceptHit],
        embedding_tag: str,
        warnings: List[str],
    ) -> List[RelatedConceptHit]:
        texts = [
            f"{r.properties.get('vocabulary', '')} {r.properties.get('domain', '')}: {r.coding.display} "
            f"({r.relationship_id}) {' '.join(r.shared_axes)}".strip()
            for r in related
        ]
        vectors = await self._safe_embed_texts([modifier_text, *texts], embedding_tag=embedding_tag, warnings=warnings)
        if vectors is None or len(vectors) != len(related) + 1:
            return related

        q_vec = vectors[0]
        reranked: List[RelatedConceptHit] = []
        for item, vec in zip(related, vectors[1:]):
            sim = cosine_similarity(q_vec, vec)
            blended = round(0.70 * sim + 0.30 * item.score, 4)
            reranked.append(item.model_copy(update={"score": blended}))
        reranked.sort(key=lambda item: (-item.score, item.coding.display.lower()))
        return reranked

    async def _safe_embed_texts(
        self,
        texts: Sequence[str],
        *,
        embedding_tag: str,
        warnings: List[str],
    ) -> Optional[List[List[float]]]:
        try:
            embed_cfg = await self.llm_config_service.get_llm_config_by_tag(
                embedding_tag,
                group_code=DEFAULT_GROUP_CODE,
            )
            if embed_cfg is None:
                msg = f"Embedding config '{embedding_tag}' not found; falling back to lexical/graph ranking."
                if msg not in warnings:
                    warnings.append(msg)
                return None
            return await self.embedding_service.compute_vectors(texts, embed_cfg)
        except Exception as exc:
            logger.warning("search_embedding_fallback", tag=embedding_tag, error=str(exc))
            msg = f"Embedding provider unavailable ({exc.__class__.__name__}); falling back to lexical/graph ranking."
            if msg not in warnings:
                warnings.append(msg)
            return None

    async def _attach_related_to_hits(
        self,
        hits: Sequence[SearchHit],
        *,
        direction: RelationDirection,
        include_shared_attributes: bool,
        valid_only: bool,
        related_limit: int,
        test_id: Optional[str],
    ) -> None:
        for hit in hits:
            if hit.related:
                continue
            if hit.entity_type == SearchEntityType.CONCEPT and hit.properties.get("vocabulary"):
                hit.related = await self.find_related_concepts(
                    str(hit.properties["vocabulary"]),
                    hit.coding.code,
                    direction=direction,
                    include_shared_attributes=include_shared_attributes,
                    valid_only=valid_only,
                    limit=related_limit,
                    test_id=test_id,
                )
            elif hit.entity_type == SearchEntityType.DICOM_ELEMENT:
                refreshed = await self.lookup_dicom(
                    hit.coding.code,
                    include_related=True,
                    related_limit=related_limit,
                    test_id=test_id,
                )
                if refreshed is not None:
                    hit.related = refreshed.related

    @staticmethod
    def _merge_hits(
        clinical_hits: Sequence[SearchHit],
        dicom_hits: Sequence[SearchHit],
        *,
        limit: int,
    ) -> List[SearchHit]:
        merged = list(clinical_hits) + list(dicom_hits)
        merged.sort(key=lambda h: (-h.score, h.coding.display.lower()))
        return merged[:limit]

    @staticmethod
    def _row_to_related_hit(
        row: Dict[str, Any],
        direction: RelationDirection,
        reason: MatchSource,
        *,
        score_override: Optional[float] = None,
        relationship_id: Optional[str] = None,
        shared_axes: Optional[List[str]] = None,
    ) -> RelatedConceptHit:
        system_uri = str(row.get("system_uri") or "")
        code = str(row.get("code") or "")
        display = str(row.get("display") or "")
        macro = str(row.get("coding_macro") or f"{system_uri}|{code}|{display}")
        props: Dict[str, Any] = {
            "vocabulary": row.get("vocabulary_id"),
            "domain": row.get("domain_id"),
            "concept_class": row.get("concept_class_id"),
            "standard": row.get("standard_concept"),
            "deprecated": row.get("invalid_reason") is not None,
        }
        if row.get("equivalence"):
            props["equivalence"] = row["equivalence"]
        if row.get("hops") is not None:
            props["hops"] = row["hops"]
        return RelatedConceptHit(
            id=str(row["id"]),
            entity_type=SearchEntityType.CONCEPT,
            direction=direction,
            relation_reason=reason,
            relationship_id=relationship_id or str(row.get("relationship_id") or reason.value),
            shared_axes=shared_axes or [],
            score=score_override if score_override is not None else _RELATION_BASE_SCORE.get(reason, 0.70),
            coding=FHIRCoding(system=system_uri, code=code, display=display),
            macro=macro,
            properties={k: v for k, v in props.items() if v is not None},
        )


search_service = SearchService()

__all__ = [
    "SearchService",
    "build_lucene_query",
    "cosine_similarity",
    "escape_lucene_term",
    "search_service",
]
