// =============================================================================
// GoSmart Health Dictionary Service — System, Auth, Group, LLM_Config & Audit Schema
// =============================================================================

// -----------------------------------------------------------------------------
// 1. System, Auth, Group, LLM_Config & AuditLog Uniqueness Constraints
// -----------------------------------------------------------------------------
CREATE CONSTRAINT system_sys_id_unique IF NOT EXISTS FOR (sys:System) REQUIRE sys.id IS UNIQUE;
CREATE CONSTRAINT user_sys_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT user_email_unique IF NOT EXISTS FOR (u:User) REQUIRE u.primary_email IS UNIQUE;
CREATE CONSTRAINT identity_sys_id_unique IF NOT EXISTS FOR (i:Identity) REQUIRE i.id IS UNIQUE;
CREATE CONSTRAINT authenticator_sys_id_unique IF NOT EXISTS FOR (a:Authenticator) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT group_sys_id_unique IF NOT EXISTS FOR (g:Group) REQUIRE g.id IS UNIQUE;
CREATE CONSTRAINT group_code_unique IF NOT EXISTS FOR (g:Group) REQUIRE g.code IS UNIQUE;
CREATE CONSTRAINT llm_config_sys_id_unique IF NOT EXISTS FOR (l:LLM_Config) REQUIRE l.id IS UNIQUE;
CREATE CONSTRAINT audit_log_sys_id_unique IF NOT EXISTS FOR (a:AuditLog) REQUIRE a.id IS UNIQUE;

// -----------------------------------------------------------------------------
// 2. Lookup & Test Isolation Property Indexes
// -----------------------------------------------------------------------------
CREATE INDEX system_test_id_idx IF NOT EXISTS FOR (sys:System) ON (sys.test_id);
CREATE INDEX user_test_id_idx IF NOT EXISTS FOR (u:User) ON (u.test_id);
CREATE INDEX user_verification_token_hash_idx IF NOT EXISTS FOR (u:User) ON (u.verification_token_hash);
CREATE INDEX identity_test_id_idx IF NOT EXISTS FOR (i:Identity) ON (i.test_id);
CREATE INDEX authenticator_secret_hash_idx IF NOT EXISTS FOR (a:Authenticator) ON (a.secret_hash);
CREATE INDEX authenticator_test_id_idx IF NOT EXISTS FOR (a:Authenticator) ON (a.test_id);
CREATE INDEX group_name_idx IF NOT EXISTS FOR (g:Group) ON (g.name);
CREATE INDEX group_test_id_idx IF NOT EXISTS FOR (g:Group) ON (g.test_id);
CREATE INDEX llm_config_tag_idx IF NOT EXISTS FOR (l:LLM_Config) ON (l.tag);
CREATE INDEX llm_config_provider_idx IF NOT EXISTS FOR (l:LLM_Config) ON (l.provider);
CREATE INDEX llm_config_test_id_idx IF NOT EXISTS FOR (l:LLM_Config) ON (l.test_id);

// -----------------------------------------------------------------------------
// 3. Group & LLM_Config Relationship Indexes (Positive & Negated Lifecycle)
// -----------------------------------------------------------------------------
CREATE INDEX belongs_to_id_idx IF NOT EXISTS FOR ()-[r:BELONGS_TO]-() ON (r.id);
CREATE INDEX not_belongs_to_id_idx IF NOT EXISTS FOR ()-[r:NOT_BELONGS_TO]-() ON (r.id);
CREATE INDEX has_llm_config_id_idx IF NOT EXISTS FOR ()-[r:HAS_LLM_CONFIG]-() ON (r.id);
CREATE INDEX not_has_llm_config_id_idx IF NOT EXISTS FOR ()-[r:NOT_HAS_LLM_CONFIG]-() ON (r.id);
