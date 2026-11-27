// =============================================================================
// GoSmart Health Dictionary Service — Clinical Terminology & Ontology Schema
// =============================================================================

// -----------------------------------------------------------------------------
// 1. Clinical Terminology Node Uniqueness Constraints (Deterministic UUIDv5 id)
// -----------------------------------------------------------------------------
CREATE CONSTRAINT code_system_id_unique IF NOT EXISTS FOR (n:CodeSystem) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE;

// -----------------------------------------------------------------------------
// 2. Domain Lookup & Test Isolation Property Indexes
// -----------------------------------------------------------------------------
CREATE INDEX code_system_vocabulary_id_idx IF NOT EXISTS FOR (n:CodeSystem) ON (n.vocabulary_id);
CREATE INDEX code_system_system_uri_idx IF NOT EXISTS FOR (n:CodeSystem) ON (n.system_uri);
CREATE INDEX code_system_test_id_idx IF NOT EXISTS FOR (n:CodeSystem) ON (n.test_id);

CREATE INDEX concept_athena_id_idx IF NOT EXISTS FOR (n:Concept) ON (n.athena_concept_id);
CREATE INDEX concept_vocab_code_idx IF NOT EXISTS FOR (n:Concept) ON (n.vocabulary_id, n.code);
CREATE INDEX concept_coding_key_idx IF NOT EXISTS FOR (n:Concept) ON (n.coding_key);
CREATE INDEX concept_vocabulary_id_idx IF NOT EXISTS FOR (n:Concept) ON (n.vocabulary_id);
CREATE INDEX concept_domain_id_idx IF NOT EXISTS FOR (n:Concept) ON (n.domain_id);
CREATE INDEX concept_standard_concept_idx IF NOT EXISTS FOR (n:Concept) ON (n.standard_concept);
CREATE INDEX concept_is_deprecated_idx IF NOT EXISTS FOR (n:Concept) ON (n.is_deprecated);
CREATE INDEX concept_test_id_idx IF NOT EXISTS FOR (n:Concept) ON (n.test_id);

// -----------------------------------------------------------------------------
// 3. Clinical Ontology Relationship Indexes (Positive & NOT_* Lifecycle Edges)
// -----------------------------------------------------------------------------
CREATE INDEX belongs_to_system_id_idx IF NOT EXISTS FOR ()-[r:BELONGS_TO_SYSTEM]-() ON (r.id);
CREATE INDEX not_belongs_to_system_id_idx IF NOT EXISTS FOR ()-[r:NOT_BELONGS_TO_SYSTEM]-() ON (r.id);

CREATE INDEX is_a_id_idx IF NOT EXISTS FOR ()-[r:IS_A]-() ON (r.id);
CREATE INDEX not_is_a_id_idx IF NOT EXISTS FOR ()-[r:NOT_IS_A]-() ON (r.id);

CREATE INDEX maps_to_id_idx IF NOT EXISTS FOR ()-[r:MAPS_TO]-() ON (r.id);
CREATE INDEX not_maps_to_id_idx IF NOT EXISTS FOR ()-[r:NOT_MAPS_TO]-() ON (r.id);

CREATE INDEX has_attribute_id_idx IF NOT EXISTS FOR ()-[r:HAS_ATTRIBUTE]-() ON (r.id);
CREATE INDEX not_has_attribute_id_idx IF NOT EXISTS FOR ()-[r:NOT_HAS_ATTRIBUTE]-() ON (r.id);

CREATE INDEX replaced_by_id_idx IF NOT EXISTS FOR ()-[r:REPLACED_BY]-() ON (r.id);
CREATE INDEX not_replaced_by_id_idx IF NOT EXISTS FOR ()-[r:NOT_REPLACED_BY]-() ON (r.id);

// -----------------------------------------------------------------------------
// 4. Full-Text (Lucene) Index for Stage 1 Lexical Concept Lookup
// -----------------------------------------------------------------------------
CREATE FULLTEXT INDEX concept_fulltext_idx IF NOT EXISTS
FOR (n:Concept)
ON EACH [n.display, n.code, n.synonyms];
