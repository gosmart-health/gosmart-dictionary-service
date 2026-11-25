// =============================================================================
// GoSmart Health Dictionary Service — DICOM Dictionary Subgraph Schema
// =============================================================================

// -----------------------------------------------------------------------------
// 1. DICOM Node Uniqueness Constraints (Deterministic UUIDv5 id)
// -----------------------------------------------------------------------------
CREATE CONSTRAINT dicom_element_id_unique IF NOT EXISTS FOR (n:DicomElement) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT dicom_ciod_id_unique IF NOT EXISTS FOR (n:DicomCIOD) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT dicom_module_id_unique IF NOT EXISTS FOR (n:DicomModule) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT dicom_macro_id_unique IF NOT EXISTS FOR (n:DicomMacro) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT dicom_sop_class_id_unique IF NOT EXISTS FOR (n:DicomSOPClass) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT dicom_conf_profile_id_unique IF NOT EXISTS FOR (n:DicomConfidentialityProfile) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT dicom_std_ref_id_unique IF NOT EXISTS FOR (n:DicomStandardReference) REQUIRE n.id IS UNIQUE;

// -----------------------------------------------------------------------------
// 2. Domain Lookup & Test Isolation Property Indexes
// -----------------------------------------------------------------------------
CREATE INDEX dicom_element_tag_hex_idx IF NOT EXISTS FOR (n:DicomElement) ON (n.tag_hex);
CREATE INDEX dicom_element_tag_idx IF NOT EXISTS FOR (n:DicomElement) ON (n.tag);
CREATE INDEX dicom_element_keyword_idx IF NOT EXISTS FOR (n:DicomElement) ON (n.keyword);
CREATE INDEX dicom_element_coding_key_idx IF NOT EXISTS FOR (n:DicomElement) ON (n.coding_key);
CREATE INDEX dicom_element_test_id_idx IF NOT EXISTS FOR (n:DicomElement) ON (n.test_id);

CREATE INDEX dicom_ciod_slug_idx IF NOT EXISTS FOR (n:DicomCIOD) ON (n.ciod_id);
CREATE INDEX dicom_ciod_name_idx IF NOT EXISTS FOR (n:DicomCIOD) ON (n.name);
CREATE INDEX dicom_ciod_test_id_idx IF NOT EXISTS FOR (n:DicomCIOD) ON (n.test_id);

CREATE INDEX dicom_module_slug_idx IF NOT EXISTS FOR (n:DicomModule) ON (n.module_id);
CREATE INDEX dicom_module_test_id_idx IF NOT EXISTS FOR (n:DicomModule) ON (n.test_id);

CREATE INDEX dicom_macro_slug_idx IF NOT EXISTS FOR (n:DicomMacro) ON (n.macro_id);
CREATE INDEX dicom_macro_test_id_idx IF NOT EXISTS FOR (n:DicomMacro) ON (n.test_id);

CREATE INDEX dicom_sop_class_uid_idx IF NOT EXISTS FOR (n:DicomSOPClass) ON (n.sop_class_uid);
CREATE INDEX dicom_sop_class_test_id_idx IF NOT EXISTS FOR (n:DicomSOPClass) ON (n.test_id);

CREATE INDEX dicom_conf_profile_attr_id_idx IF NOT EXISTS FOR (n:DicomConfidentialityProfile) ON (n.attribute_id);
CREATE INDEX dicom_conf_profile_test_id_idx IF NOT EXISTS FOR (n:DicomConfidentialityProfile) ON (n.test_id);

CREATE INDEX dicom_std_ref_url_idx IF NOT EXISTS FOR (n:DicomStandardReference) ON (n.source_url);
CREATE INDEX dicom_std_ref_test_id_idx IF NOT EXISTS FOR (n:DicomStandardReference) ON (n.test_id);

// -----------------------------------------------------------------------------
// 3. DICOM Relationship Indexes (Positive & NOT_* Lifecycle Edges)
// -----------------------------------------------------------------------------
CREATE INDEX implements_ciod_id_idx IF NOT EXISTS FOR ()-[r:IMPLEMENTS_CIOD]-() ON (r.id);
CREATE INDEX not_implements_ciod_id_idx IF NOT EXISTS FOR ()-[r:NOT_IMPLEMENTS_CIOD]-() ON (r.id);

CREATE INDEX has_module_id_idx IF NOT EXISTS FOR ()-[r:HAS_MODULE]-() ON (r.id);
CREATE INDEX not_has_module_id_idx IF NOT EXISTS FOR ()-[r:NOT_HAS_MODULE]-() ON (r.id);

CREATE INDEX has_func_group_macro_id_idx IF NOT EXISTS FOR ()-[r:HAS_FUNC_GROUP_MACRO]-() ON (r.id);
CREATE INDEX not_has_func_group_macro_id_idx IF NOT EXISTS FOR ()-[r:NOT_HAS_FUNC_GROUP_MACRO]-() ON (r.id);

CREATE INDEX uses_element_id_idx IF NOT EXISTS FOR ()-[r:USES_ELEMENT]-() ON (r.id);
CREATE INDEX not_uses_element_id_idx IF NOT EXISTS FOR ()-[r:NOT_USES_ELEMENT]-() ON (r.id);

CREATE INDEX has_conf_profile_id_idx IF NOT EXISTS FOR ()-[r:HAS_CONFIDENTIALITY_PROFILE]-() ON (r.id);
CREATE INDEX not_has_conf_profile_id_idx IF NOT EXISTS FOR ()-[r:NOT_HAS_CONFIDENTIALITY_PROFILE]-() ON (r.id);

CREATE INDEX references_standard_id_idx IF NOT EXISTS FOR ()-[r:REFERENCES_STANDARD]-() ON (r.id);
CREATE INDEX not_references_standard_id_idx IF NOT EXISTS FOR ()-[r:NOT_REFERENCES_STANDARD]-() ON (r.id);

// -----------------------------------------------------------------------------
// 4. Full-Text (Lucene) Indexes for Stage 1 Lexical Lookup
// -----------------------------------------------------------------------------
CREATE FULLTEXT INDEX dicom_element_fulltext_idx IF NOT EXISTS
FOR (n:DicomElement)
ON EACH [n.name, n.keyword, n.tag, n.tag_hex, n.description];

CREATE FULLTEXT INDEX dicom_structure_fulltext_idx IF NOT EXISTS
FOR (n:DicomCIOD|DicomModule|DicomMacro|DicomSOPClass)
ON EACH [n.name, n.description];
