"""Live Neo4j integration tests for SearchService and ClinicalNLUService.

Follows AGENTS.md testing rules:
- Cleans up test_id-scoped synthetic records BEFORE running (not after) so post-test graph
  state remains inspectable.
- Runs read-only smoke checks against production data (test_id IS NULL) without mutating it.
"""

from __future__ import annotations

import asyncio

import pytest

from gosmart_dictionary_service.db.neo4j import Neo4jClient
from gosmart_dictionary_service.model.search import (
    MatchSource,
    NLUIntent,
    RelationDirection,
    SearchEntityType,
)
from gosmart_dictionary_service.service.chat_service import ChatRequest, DictionaryChatService
from gosmart_dictionary_service.service.search_service import SearchService

TEST_ID = "test_search_service_neo4j"


def test_search_service_isolated_graph_operations() -> None:
    """Verify lookup, forward/reverse/shared-axis related aggregation, crosswalk, expand, subsumes, and NLU alerts."""

    async def _run() -> None:
        client = Neo4jClient()
        if not await client.verify_connectivity():
            pytest.skip("Neo4j not reachable")

        try:
            # 1) Clean up prior test_id fixtures BEFORE test execution (AGENTS.md §4)
            await client.execute_query(
                "MATCH (n {test_id: $tid}) DETACH DELETE n",
                {"tid": TEST_ID},
            )

            # 2) Seed minimal isolated ontology subgraph with test_id
            await client.execute_query(
                """
                CREATE (cs_snomed:CodeSystem {
                    id: 'test-cs-snomed', vocabulary_id: 'SNOMED', system_uri: 'http://snomed.info/sct',
                    name: 'SNOMED Test', version: 'test-1', test_id: $tid
                })
                CREATE (cs_icd:CodeSystem {
                    id: 'test-cs-icd', vocabulary_id: 'ICD10CM', system_uri: 'http://hl7.org/fhir/sid/icd-10-cm',
                    name: 'ICD10CM Test', version: 'test-1', test_id: $tid
                })
                CREATE (cs_loinc:CodeSystem {
                    id: 'test-cs-loinc', vocabulary_id: 'LOINC', system_uri: 'http://loinc.org',
                    name: 'LOINC Test', version: 'test-1', test_id: $tid
                })
                CREATE (c_parent:Concept {
                    id: 'test-c-pneumonia', vocabulary_id: 'SNOMED', code: 'T-233604007',
                    display: 'Test Pneumonia', system_uri: 'http://snomed.info/sct',
                    coding_macro: 'http://snomed.info/sct|T-233604007|Test Pneumonia',
                    domain_id: 'Condition', concept_class_id: 'Disorder', standard_concept: 'S',
                    is_deprecated: false, synonyms: ['Lung infection'], test_id: $tid
                })
                CREATE (c_child:Concept {
                    id: 'test-c-bact-pneumonia', vocabulary_id: 'SNOMED', code: 'T-53084003',
                    display: 'Test Bacterial pneumonia', system_uri: 'http://snomed.info/sct',
                    coding_macro: 'http://snomed.info/sct|T-53084003|Test Bacterial pneumonia',
                    domain_id: 'Condition', concept_class_id: 'Disorder', standard_concept: 'S',
                    is_deprecated: false, synonyms: [], test_id: $tid
                })
                CREATE (c_icd:Concept {
                    id: 'test-c-icd-j189', vocabulary_id: 'ICD10CM', code: 'T-J18.9',
                    display: 'Test Pneumonia unspecified', system_uri: 'http://hl7.org/fhir/sid/icd-10-cm',
                    coding_macro: 'http://hl7.org/fhir/sid/icd-10-cm|T-J18.9|Test Pneumonia unspecified',
                    domain_id: 'Condition', concept_class_id: '4-char billing code',
                    is_deprecated: false, synonyms: [], test_id: $tid
                })
                CREATE (c_old:Concept {
                    id: 'test-c-retired', vocabulary_id: 'SNOMED', code: 'T-OLD-1',
                    display: 'Test Old Pneumonia', system_uri: 'http://snomed.info/sct',
                    coding_macro: 'http://snomed.info/sct|T-OLD-1|Test Old Pneumonia',
                    domain_id: 'Condition', invalid_reason: 'D', is_deprecated: true, synonyms: [], test_id: $tid
                })
                CREATE (c_loinc1:Concept {
                    id: 'test-c-loinc-1', vocabulary_id: 'LOINC', code: 'T-87279-6',
                    display: 'Test CT Chest Screening', system_uri: 'http://loinc.org',
                    coding_macro: 'http://loinc.org|T-87279-6|Test CT Chest Screening',
                    domain_id: 'Procedure', standard_concept: 'S', is_deprecated: false, synonyms: [], test_id: $tid
                })
                CREATE (c_loinc2:Concept {
                    id: 'test-c-loinc-2', vocabulary_id: 'LOINC', code: 'T-24627-2',
                    display: 'Test CT Chest Diagnostic', system_uri: 'http://loinc.org',
                    coding_macro: 'http://loinc.org|T-24627-2|Test CT Chest Diagnostic',
                    domain_id: 'Procedure', standard_concept: 'S', is_deprecated: false, synonyms: [], test_id: $tid
                })
                CREATE (c_axis_ct:Concept {
                    id: 'test-c-axis-ct', vocabulary_id: 'LOINC', code: 'T-LP6207-7',
                    display: 'Test Computed tomography', system_uri: 'http://loinc.org',
                    domain_id: 'Observation', is_deprecated: false, synonyms: [], test_id: $tid
                })
                CREATE (c_axis_chest:Concept {
                    id: 'test-c-axis-chest', vocabulary_id: 'LOINC', code: 'T-LP7138-3',
                    display: 'Test Chest', system_uri: 'http://loinc.org',
                    domain_id: 'Observation', is_deprecated: false, synonyms: [], test_id: $tid
                })
                CREATE (c_parent)-[:BELONGS_TO_SYSTEM {id: 'test-r-1', test_id: $tid}]->(cs_snomed)
                CREATE (c_child)-[:BELONGS_TO_SYSTEM {id: 'test-r-2', test_id: $tid}]->(cs_snomed)
                CREATE (c_icd)-[:BELONGS_TO_SYSTEM {id: 'test-r-3', test_id: $tid}]->(cs_icd)
                CREATE (c_old)-[:BELONGS_TO_SYSTEM {id: 'test-r-4', test_id: $tid}]->(cs_snomed)
                CREATE (c_loinc1)-[:BELONGS_TO_SYSTEM {id: 'test-r-5', test_id: $tid}]->(cs_loinc)
                CREATE (c_loinc2)-[:BELONGS_TO_SYSTEM {id: 'test-r-6', test_id: $tid}]->(cs_loinc)
                CREATE (c_child)-[:IS_A {id: 'test-r-isa', test_id: $tid}]->(c_parent)
                CREATE (c_icd)-[:MAPS_TO {id: 'test-r-map', relationship_id: 'Maps to', test_id: $tid}]->(c_parent)
                CREATE (c_old)-[:REPLACED_BY {
                    id: 'test-r-rep', relationship_id: 'Concept replaced by', test_id: $tid
                }]->(c_parent)
                CREATE (c_loinc1)-[:HAS_ATTRIBUTE {
                    id: 'test-r-a1', relationship_id: 'Has method', test_id: $tid
                }]->(c_axis_ct)
                CREATE (c_loinc1)-[:HAS_ATTRIBUTE {
                    id: 'test-r-a2', relationship_id: 'Has system', test_id: $tid
                }]->(c_axis_chest)
                CREATE (c_loinc2)-[:HAS_ATTRIBUTE {
                    id: 'test-r-a3', relationship_id: 'Has method', test_id: $tid
                }]->(c_axis_ct)
                CREATE (c_loinc2)-[:HAS_ATTRIBUTE {
                    id: 'test-r-a4', relationship_id: 'Has system', test_id: $tid
                }]->(c_axis_chest)
                """,
                {"tid": TEST_ID},
            )

            svc = SearchService(client=client)

            # Direct lookup + deprecated auto-healing
            retired = await svc.lookup_code("SNOMED", "T-OLD-1", test_id=TEST_ID)
            assert retired is not None
            assert retired.properties["deprecated"] is True
            assert any(
                r.relation_reason == MatchSource.REPLACED_BY and r.coding.code == "T-233604007" for r in retired.related
            )

            # Forward + reverse + 2-hop shared-axis aggregation on LOINC T-87279-6
            related_loinc = await svc.find_related_concepts(
                "LOINC",
                "T-87279-6",
                direction=RelationDirection.BOTH,
                include_shared_attributes=True,
                test_id=TEST_ID,
            )
            codes_by_reason = {(r.relation_reason, r.coding.code) for r in related_loinc}
            assert (MatchSource.HAS_ATTRIBUTE, "T-LP6207-7") in codes_by_reason
            assert (MatchSource.HAS_ATTRIBUTE, "T-LP7138-3") in codes_by_reason
            assert (MatchSource.SHARED_ATTRIBUTE, "T-24627-2") in codes_by_reason

            # Subtype expansion & FHIR $subsumes
            descendants = await svc.expand_descendants("SNOMED", "T-233604007", max_depth=2, test_id=TEST_ID)
            assert [d.coding.code for d in descendants] == ["T-53084003"]
            assert await svc.subsumes("SNOMED", "T-233604007", "SNOMED", "T-53084003", test_id=TEST_ID) == "subsumes"
            assert await svc.subsumes("SNOMED", "T-53084003", "SNOMED", "T-233604007", test_id=TEST_ID) == "subsumed-by"
            assert await svc.subsumes("SNOMED", "T-233604007", "SNOMED", "T-233604007", test_id=TEST_ID) == "equivalent"

            # Crosswalk ICD10CM -> SNOMED
            mapped = await svc.crosswalk("ICD10CM", "T-J18.9", target_vocabularies=["SNOMED"], test_id=TEST_ID)
            assert len(mapped) == 1
            assert mapped[0].coding.code == "T-J18.9" or mapped[0].coding.code == "T-233604007"
            assert mapped[0].coding.code == "T-233604007"

            # Unified search() routing for "Find something related to LOINC T-87279-6"
            resp = await svc.search(
                "Find something related to LOINC T-87279-6",
                rerank_with_embeddings=False,
                test_id=TEST_ID,
            )
            assert resp.nlu.intent == NLUIntent.RELATED_LOOKUP
            assert resp.total_hits == 1
            assert resp.hits[0].coding.code == "T-87279-6"
            assert any(r.coding.code == "T-24627-2" for r in resp.hits[0].related)

            # Dynamic schema alert on unloaded RxNorm against isolated test_id graph
            rx_resp = await svc.search("RxNorm 1191", rerank_with_embeddings=False, test_id=TEST_ID)
            assert rx_resp.total_hits == 0
            assert rx_resp.nlu.unsupported_vocabularies == ["RxNorm"]
            assert any("RxNorm" in w and "ICD10CM, LOINC, SNOMED" in w for w in rx_resp.warnings)

            # Inspect entity neighborhood (powers /explore drill-down & future Graph UI)
            nb = await svc.inspect_entity(vocabulary_id="LOINC", code="T-87279-6", test_id=TEST_ID)
            assert nb is not None
            assert nb.seed.coding.code == "T-87279-6"
            assert len(nb.nodes) >= 4
            assert len(nb.edges) >= 3

            # Grounded DictionaryChatService end-to-end against isolated test_id graph
            chat_svc = DictionaryChatService(
                client=client,
                search_service=svc,
                generate_fn=lambda _sys, prompt, cfg: f"Grounded[{cfg.model}]: " + prompt.splitlines()[0],
            )
            chat_resp = await chat_svc.chat(
                ChatRequest(
                    message="Find something related to LOINC T-87279-6",
                    include_context=True,
                    include_related=True,
                    test_id=TEST_ID,
                )
            )
            assert chat_resp.nlu is not None
            assert chat_resp.nlu.intent == "related_lookup"
            assert len(chat_resp.citations) == 1
            assert chat_resp.citations[0].identifier == "LOINC:T-87279-6"
            assert chat_resp.citations[0].drilldown_url is not None
            assert any(r.identifier == "LOINC:T-24627-2" for r in chat_resp.citations[0].related)
        finally:
            await client.close()

    asyncio.run(_run())


def test_search_service_production_readonly_smoke() -> None:
    """Read-only smoke test against production Neo4j dictionary (skips if empty or offline)."""

    async def _run() -> None:
        client = Neo4jClient()
        if not await client.verify_connectivity():
            pytest.skip("Neo4j not reachable")

        try:
            svc = SearchService(client=client)
            seed = await svc.lookup_code("LOINC", "87279-6")
            if seed is None:
                pytest.skip("Production clinical terms not loaded in Neo4j")

            # 1) Related concepts for LOINC 87279-6 (forward + reverse + 2-hop shared axes)
            rel_resp = await svc.search(
                "Find something related to LOINC 87279-6",
                rerank_with_embeddings=False,
                related_limit=15,
            )
            assert rel_resp.nlu.intent == NLUIntent.RELATED_LOOKUP
            assert rel_resp.total_hits == 1
            assert len(rel_resp.hits[0].related) >= 5
            assert any(r.relation_reason == MatchSource.SHARED_ATTRIBUTE for r in rel_resp.hits[0].related)

            # 2) Direct code query ("LOINC 87279-6") -> 1.0 deterministic hit + reverse-lookup discovery < 1.0
            disc_resp = await svc.search(
                "LOINC 87279-6",
                rerank_with_embeddings=False,
                limit=8,
            )
            assert disc_resp.hits[0].coding.code == "87279-6"
            assert disc_resp.hits[0].score == 1.0
            assert disc_resp.nlu.rewritten_query == seed.coding.display
            assert len(disc_resp.hits) >= 2
            assert all(h.score <= 0.85 for h in disc_resp.hits[1:])

            # 3) Clinical free-text search with 1-hop [:IS_A] parent convergence
            pneu_resp = await svc.search(
                "acute pneumonia",
                entity_types=[SearchEntityType.CONCEPT],
                rerank_with_embeddings=False,
                limit=10,
            )
            assert pneu_resp.total_hits > 0
            returned_codes = {h.coding.code for h in pneu_resp.hits}
            assert "233604007" in returned_codes  # SNOMED Pneumonia surfaced via parent convergence!

            # 4) Subsumption check: Myocardial infarction subsumes Acute myocardial infarction
            outcome = await svc.subsumes("SNOMED", "22298006", "SNOMED", "57054005")
            assert outcome == "subsumes"
        finally:
            await client.close()

    asyncio.run(_run())
