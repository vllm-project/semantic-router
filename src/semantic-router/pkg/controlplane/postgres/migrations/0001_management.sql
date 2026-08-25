-- vLLM Semantic Router durable Management schema.
-- This single forward-only baseline is applied only by the explicit migrator.

-- Access identities, policies, quotas, and API keys

CREATE TABLE access_namespaces (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  quota_partition_id TEXT NOT NULL UNIQUE,
  billing_currency TEXT NOT NULL CHECK (billing_currency ~ '^[A-Z]{3}$'),
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  runtime_epoch BIGINT NOT NULL DEFAULT 1 CHECK (runtime_epoch > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (id, quota_partition_id)
);

CREATE TABLE access_subjects (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  id UUID NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('user', 'team', 'api_key')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, id),
  UNIQUE (id),
  UNIQUE (namespace_id, id, kind)
);

CREATE TABLE access_users (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  email TEXT NOT NULL,
  display_name TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  deleted_at TIMESTAMPTZ,
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, email),
  FOREIGN KEY (namespace_id, id) REFERENCES access_subjects(namespace_id, id)
    DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE access_teams (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  deleted_at TIMESTAMPTZ,
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, name),
  FOREIGN KEY (namespace_id, id) REFERENCES access_subjects(namespace_id, id)
    DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE access_team_memberships (
  namespace_id UUID NOT NULL,
  team_id UUID NOT NULL,
  user_id UUID NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('member', 'admin')),
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, team_id, user_id),
  FOREIGN KEY (namespace_id, team_id) REFERENCES access_teams(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, user_id) REFERENCES access_users(namespace_id, id) ON DELETE RESTRICT
);

CREATE TABLE access_policies (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL CHECK (status IN ('draft', 'active', 'disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, name)
);

CREATE TABLE access_policy_grants (
  policy_id UUID NOT NULL REFERENCES access_policies(id) ON DELETE RESTRICT,
  resource_type TEXT NOT NULL CHECK (resource_type IN ('entrypoint', 'model')),
  resource_id TEXT NOT NULL,
  permission TEXT NOT NULL CHECK (permission IN ('discover', 'invoke')),
  effect TEXT NOT NULL DEFAULT 'allow' CHECK (effect IN ('allow', 'deny')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (policy_id, resource_type, resource_id, permission, effect)
);

CREATE TABLE rate_limit_policies (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL CHECK (status IN ('draft', 'active', 'disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, name)
);

CREATE TABLE rate_limit_rules (
  id UUID PRIMARY KEY,
  policy_id UUID NOT NULL REFERENCES rate_limit_policies(id) ON DELETE RESTRICT,
  metric TEXT NOT NULL,
  algorithm TEXT NOT NULL,
  limit_value NUMERIC(42,0),
  window_seconds BIGINT,
  calendar_period TEXT,
  timezone TEXT,
  bucket_capacity NUMERIC(42,0),
  refill_amount NUMERIC(42,0),
  refill_period_milliseconds BIGINT,
  gcra_emission_interval_microseconds BIGINT,
  gcra_burst_tolerance BIGINT,
  accounting TEXT NOT NULL CHECK (accounting IN ('request', 'response_actual')),
  enforcement TEXT NOT NULL CHECK (enforcement IN ('enforce', 'shadow')),
  ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (policy_id, ordinal),
  CHECK (
    (algorithm = 'sliding_log' AND limit_value > 0 AND window_seconds > 0
      AND calendar_period IS NULL AND timezone IS NULL AND bucket_capacity IS NULL
      AND refill_amount IS NULL AND refill_period_milliseconds IS NULL
      AND gcra_emission_interval_microseconds IS NULL AND gcra_burst_tolerance IS NULL)
    OR
    (algorithm = 'calendar_window' AND limit_value > 0
      AND calendar_period IN ('day', 'month') AND timezone IS NOT NULL
      AND window_seconds IS NULL AND bucket_capacity IS NULL AND refill_amount IS NULL
      AND refill_period_milliseconds IS NULL AND gcra_emission_interval_microseconds IS NULL
      AND gcra_burst_tolerance IS NULL)
    OR
    (algorithm = 'token_bucket' AND bucket_capacity > 0 AND refill_amount > 0
      AND refill_period_milliseconds > 0 AND limit_value IS NULL AND window_seconds IS NULL
      AND calendar_period IS NULL AND timezone IS NULL
      AND gcra_emission_interval_microseconds IS NULL AND gcra_burst_tolerance IS NULL)
    OR
    (algorithm = 'gcra' AND gcra_emission_interval_microseconds > 0
      AND gcra_burst_tolerance >= 0 AND limit_value IS NULL AND window_seconds IS NULL
      AND calendar_period IS NULL AND timezone IS NULL AND bucket_capacity IS NULL
      AND refill_amount IS NULL AND refill_period_milliseconds IS NULL)
    OR
    (algorithm = 'concurrency' AND limit_value > 0 AND window_seconds IS NULL
      AND calendar_period IS NULL AND timezone IS NULL AND bucket_capacity IS NULL
      AND refill_amount IS NULL AND refill_period_milliseconds IS NULL
      AND gcra_emission_interval_microseconds IS NULL AND gcra_burst_tolerance IS NULL)
  ),
  CHECK (
    (metric = 'requests' AND accounting = 'request' AND algorithm <> 'concurrency')
    OR (metric IN ('input_tokens','output_tokens','total_tokens','served_input_tokens',
                   'served_output_tokens','served_total_tokens','cost')
        AND accounting = 'response_actual' AND algorithm IN ('sliding_log','calendar_window'))
    OR (metric = 'concurrent_requests' AND accounting = 'request' AND algorithm = 'concurrency')
  )
);

CREATE TABLE access_api_keys (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  name TEXT NOT NULL,
  owner_user_id UUID,
  owner_team_id UUID,
  context_team_id UUID,
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  expires_at TIMESTAMPTZ,
  policy_epoch BIGINT NOT NULL DEFAULT 1 CHECK (policy_epoch > 0),
  delegation_epoch BIGINT NOT NULL DEFAULT 1 CHECK (delegation_epoch > 0),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  last_used_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  deleted_at TIMESTAMPTZ,
  CHECK (num_nonnulls(owner_user_id, owner_team_id) = 1),
  CHECK (owner_team_id IS NULL OR context_team_id IS NULL),
  UNIQUE (namespace_id, id),
  FOREIGN KEY (namespace_id, id) REFERENCES access_subjects(namespace_id, id)
    DEFERRABLE INITIALLY DEFERRED,
  FOREIGN KEY (namespace_id, owner_user_id) REFERENCES access_users(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, owner_team_id) REFERENCES access_teams(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, context_team_id) REFERENCES access_teams(namespace_id, id) ON DELETE RESTRICT
);

CREATE TABLE access_api_key_credentials (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  api_key_id UUID NOT NULL,
  kid TEXT NOT NULL UNIQUE CHECK (kid ~ '^[A-Za-z0-9_-]{12,96}$'),
  secret_hmac BYTEA NOT NULL,
  pepper_version TEXT NOT NULL,
  secret_ciphertext BYTEA,
  ciphertext_nonce BYTEA,
  kek_version TEXT,
  status TEXT NOT NULL CHECK (status IN ('active', 'retiring', 'revoked')),
  not_before TIMESTAMPTZ NOT NULL,
  expires_at TIMESTAMPTZ,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK ((secret_ciphertext IS NULL) = (ciphertext_nonce IS NULL)),
  CHECK ((secret_ciphertext IS NULL) = (kek_version IS NULL)),
  UNIQUE (namespace_id, id),
  FOREIGN KEY (namespace_id, api_key_id) REFERENCES access_api_keys(namespace_id, id) ON DELETE RESTRICT
);

CREATE UNIQUE INDEX access_api_key_one_active_credential
  ON access_api_key_credentials(namespace_id, api_key_id)
  WHERE status = 'active';

CREATE TABLE access_policy_bindings (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  policy_id UUID NOT NULL,
  subject_id UUID NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, policy_id, subject_id),
  FOREIGN KEY (namespace_id, policy_id) REFERENCES access_policies(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, subject_id) REFERENCES access_subjects(namespace_id, id) ON DELETE RESTRICT
);

CREATE TABLE rate_limit_bindings (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  policy_id UUID NOT NULL,
  subject_id UUID NOT NULL,
  binding_mode TEXT NOT NULL CHECK (binding_mode IN ('allocation', 'hard_cap')),
  quota_partition_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, id),
  FOREIGN KEY (namespace_id, policy_id) REFERENCES rate_limit_policies(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, subject_id) REFERENCES access_subjects(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, quota_partition_id) REFERENCES access_namespaces(id, quota_partition_id) ON DELETE RESTRICT
);

CREATE UNIQUE INDEX rate_limit_one_active_allocation
  ON rate_limit_bindings(namespace_id, subject_id)
  WHERE binding_mode = 'allocation' AND status = 'active';

CREATE TABLE routing_claim_schemas (
  namespace_id UUID PRIMARY KEY REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  definitions JSONB NOT NULL DEFAULT '{}'::jsonb,
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE routing_subject_claims (
  namespace_id UUID NOT NULL,
  subject_id UUID NOT NULL,
  claim_name TEXT NOT NULL,
  value JSONB NOT NULL,
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, subject_id, claim_name),
  FOREIGN KEY (namespace_id, subject_id) REFERENCES access_subjects(namespace_id, id) ON DELETE RESTRICT
);

CREATE OR REPLACE FUNCTION access_assert_subject_kind() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE expected_kind TEXT;
BEGIN
  expected_kind := TG_ARGV[0];
  IF NOT EXISTS (
    SELECT 1 FROM access_subjects s
    WHERE s.namespace_id = NEW.namespace_id AND s.id = NEW.id AND s.kind = expected_kind
  ) THEN
    RAISE EXCEPTION 'subject % in namespace % must have kind %', NEW.id, NEW.namespace_id, expected_kind;
  END IF;
  RETURN NEW;
END $$;

CREATE CONSTRAINT TRIGGER access_users_subject_kind
  AFTER INSERT OR UPDATE OF namespace_id, id ON access_users
  DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION access_assert_subject_kind('user');
CREATE CONSTRAINT TRIGGER access_teams_subject_kind
  AFTER INSERT OR UPDATE OF namespace_id, id ON access_teams
  DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION access_assert_subject_kind('team');
CREATE CONSTRAINT TRIGGER access_api_keys_subject_kind
  AFTER INSERT OR UPDATE OF namespace_id, id ON access_api_keys
  DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION access_assert_subject_kind('api_key');

CREATE OR REPLACE FUNCTION access_assert_key_context_membership() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.owner_user_id IS NOT NULL AND NEW.context_team_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM access_team_memberships m
    WHERE m.namespace_id = NEW.namespace_id AND m.team_id = NEW.context_team_id
      AND m.user_id = NEW.owner_user_id AND m.status = 'active'
  ) THEN
    RAISE EXCEPTION 'key context team requires active owner membership';
  END IF;
  RETURN NEW;
END $$;

CREATE CONSTRAINT TRIGGER access_api_key_context_membership
  AFTER INSERT OR UPDATE OF namespace_id, owner_user_id, context_team_id ON access_api_keys
  DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION access_assert_key_context_membership();

CREATE INDEX access_users_page_idx ON access_users(namespace_id, created_at DESC, id);
CREATE INDEX access_teams_page_idx ON access_teams(namespace_id, created_at DESC, id);
CREATE INDEX access_memberships_user_idx ON access_team_memberships(namespace_id, user_id, status, team_id);
CREATE INDEX access_memberships_team_idx ON access_team_memberships(namespace_id, team_id, status, user_id);
CREATE UNIQUE INDEX access_users_normalized_email_uq ON access_users(namespace_id, lower(email));
CREATE UNIQUE INDEX access_teams_normalized_name_uq ON access_teams(namespace_id, lower(name));
CREATE INDEX access_memberships_user_page_idx
  ON access_team_memberships(namespace_id, user_id, created_at DESC, team_id);
CREATE INDEX access_memberships_team_page_idx
  ON access_team_memberships(namespace_id, team_id, created_at DESC, user_id);
CREATE INDEX access_api_keys_owner_user_idx ON access_api_keys(namespace_id, owner_user_id, status, id);
CREATE INDEX access_api_keys_owner_team_idx ON access_api_keys(namespace_id, owner_team_id, status, id);
CREATE INDEX access_credentials_key_idx ON access_api_key_credentials(namespace_id, api_key_id, status, created_at DESC);
CREATE INDEX access_policies_page_idx
  ON access_policies(namespace_id, status, created_at DESC, id);
CREATE INDEX rate_limit_policies_page_idx
  ON rate_limit_policies(namespace_id, status, created_at DESC, id);
CREATE INDEX access_policy_bindings_page_idx
  ON access_policy_bindings(namespace_id, created_at DESC, id);
CREATE INDEX access_policy_bindings_policy_page_idx
  ON access_policy_bindings(namespace_id, policy_id, created_at DESC, id);
CREATE INDEX access_policy_bindings_subject_idx ON access_policy_bindings(namespace_id, subject_id, status, policy_id);
CREATE INDEX rate_limit_bindings_page_idx
  ON rate_limit_bindings(namespace_id, created_at DESC, id);
CREATE INDEX rate_limit_bindings_policy_page_idx
  ON rate_limit_bindings(namespace_id, policy_id, created_at DESC, id);
CREATE INDEX rate_limit_bindings_subject_idx ON rate_limit_bindings(namespace_id, subject_id, status, binding_mode, policy_id);

-- Management identity and managed routing resources

CREATE TABLE management_principals (
  id UUID PRIMARY KEY,
  issuer TEXT NOT NULL,
  subject TEXT NOT NULL,
  display_name TEXT NOT NULL,
  verified_email TEXT,
  attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (issuer, subject)
);

CREATE TABLE management_roles (
  id UUID PRIMARY KEY,
  namespace_id UUID REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  name TEXT NOT NULL,
  display_name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  permissions JSONB NOT NULL,
  permissions_digest BYTEA NOT NULL CHECK (octet_length(permissions_digest) = 32),
  builtin BOOLEAN NOT NULL DEFAULT FALSE,
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE NULLS NOT DISTINCT (namespace_id, name)
);

INSERT INTO management_roles
  (id, namespace_id, name, display_name, description, permissions,
   permissions_digest, builtin, status, revision)
VALUES
  ('10000000-0000-5000-8000-000000000001', NULL, 'cluster_admin', 'Cluster administrator',
   'Installation-wide identity and namespace administration.',
   '["audit.read","cluster.manage","cluster.read","health.read","identity_issuer.manage","identity_issuer.read","management_role.manage","management_role.read","namespace.manage","namespace.read","principal.manage","principal.read","role_binding.manage","role_binding.read","service_account.manage","service_account.read"]'::jsonb,
   decode('086307b27da2234547a903e183133ccc7613c917b9d54eebaded598c0ef1181f','hex'), TRUE, 'active', 1),
  ('10000000-0000-5000-8000-000000000002', NULL, 'platform_admin', 'Platform administrator',
   'Namespace-wide platform, access, routing, and observability administration.',
   '["access_policy.manage","access_policy.read","agent.manage","agent.read","agent.use","audit.read","delegation.manage","delegation.use","evaluation.run","health.read","invitation.manage","invitation.read","key.manage","key.read","log.read","log_payload.read","management_role.manage","management_role.read","membership.manage","namespace.manage","namespace.read","onboarding.manage","operation.manage","operation.read","principal_directory.read","principal_link.manage","principal_link.read","provider_catalog.read","provider_credential.manage","provider_credential.read","provider_credential.use","quota.read","quota.reconcile","rate_policy.manage","rate_policy.read","role_binding.manage","role_binding.read","routing.manage","routing.publish","routing.read","routing_context.manage","routing_context.read","service_account.manage","service_account.read","team.manage","team.read","tool.invoke","tool.manage","tool.read","usage.internal_dimensions.read","usage.read","user.manage","user.read"]'::jsonb,
   decode('e27404421cc93ba8b220ec599cc5a4a702242ae7f5bfb40da19f08c756a63ff2','hex'), TRUE, 'active', 1),
  ('10000000-0000-5000-8000-000000000003', NULL, 'operator', 'Operator',
   'Routing and runtime operations without identity administration.',
   '["access_policy.read","agent.manage","agent.read","agent.use","evaluation.run","health.read","log.read","log_payload.read","namespace.read","operation.manage","operation.read","provider_catalog.read","provider_credential.manage","provider_credential.read","provider_credential.use","quota.read","rate_policy.read","routing.manage","routing.publish","routing.read","routing_context.manage","routing_context.read","tool.invoke","tool.manage","tool.read","usage.internal_dimensions.read","usage.read"]'::jsonb,
   decode('ea55e1af54baf2f9e9894cf93f34df433b155cf7ea37af9dbec590c835759243','hex'), TRUE, 'active', 1),
  ('10000000-0000-5000-8000-000000000004', NULL, 'access_admin', 'Access administrator',
   'Identity links, consumers, keys, policy, quota, and scoped usage administration.',
   '["access_policy.manage","access_policy.read","audit.read","delegation.manage","delegation.use","invitation.manage","invitation.read","key.manage","key.read","membership.manage","namespace.read","onboarding.manage","operation.manage","operation.read","principal_directory.read","principal_link.manage","principal_link.read","quota.read","quota.reconcile","rate_policy.manage","rate_policy.read","routing.read","routing_context.manage","routing_context.read","team.manage","team.read","usage.read","user.manage","user.read"]'::jsonb,
   decode('a68c15103665e6341843627901809d8c9f8f2f5baffbcfd2a403daedf8faebdf','hex'), TRUE, 'active', 1),
  ('10000000-0000-5000-8000-000000000005', NULL, 'credential_revealer', 'Credential revealer',
   'Narrow, deliberately assigned inference-key reveal authority.',
   '["key.read","key.reveal"]'::jsonb,
   decode('3f190c38e61dd2603f48143bacb3840ef851a3cb87c0d2a7a2f7d418249f7f51','hex'), TRUE, 'active', 1),
  ('10000000-0000-5000-8000-000000000006', NULL, 'analyst', 'Analyst',
   'Scoped usage, logs, audit, and consumer metadata.',
   '["audit.read","key.read","log.read","namespace.read","quota.read","team.read","usage.read","user.read"]'::jsonb,
   decode('6721d4cc75a93b9da9f57bb270b74bd6c67490b5861129b02a9ca1055f4b85de','hex'), TRUE, 'active', 1),
  ('10000000-0000-5000-8000-000000000007', NULL, 'viewer', 'Viewer',
   'Read-only routing configuration and provider catalog.',
   '["agent.read","provider_catalog.read","routing.read","tool.read"]'::jsonb,
   decode('457a9204a91594a24e10ce7ab98b16fe61ec569104e7f25b9fadfe5e78f08ceb','hex'), TRUE, 'active', 1),
  ('10000000-0000-5000-8000-000000000008', NULL, 'consumer', 'Consumer',
   'User-scoped inference access and read-only account visibility.',
   '["access_policy.read","agent.read","agent.use","delegation.use","key.read","operation.read","quota.read","rate_policy.read","routing_context.read","team.read","tool.invoke","tool.read","usage.read","user.read"]'::jsonb,
   decode('42f87d9c0231abac6d6f5256f4c40d5d5789f9c9d8739c264785d0bf58560fd6','hex'), TRUE, 'active', 1);

CREATE TABLE management_role_bindings (
  id UUID PRIMARY KEY,
  principal_id UUID NOT NULL REFERENCES management_principals(id) ON DELETE RESTRICT,
  role_id UUID NOT NULL REFERENCES management_roles(id) ON DELETE RESTRICT,
  scope_kind TEXT NOT NULL CHECK (scope_kind IN ('cluster','namespace','team','user','resource')),
  namespace_id UUID REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  resource_type TEXT,
  resource_id TEXT CHECK (
    resource_id IS NULL OR resource_id ~ '^(?:[a-z][a-z0-9_-]{2,127}|[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$'
  ),
  delegation_ceiling JSONB NOT NULL DEFAULT '[]'::jsonb,
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK (
    (scope_kind = 'cluster' AND namespace_id IS NULL AND resource_type IS NULL AND resource_id IS NULL)
    OR (scope_kind = 'namespace' AND namespace_id IS NOT NULL AND resource_type IS NULL AND resource_id IS NULL)
    OR (scope_kind IN ('team','user') AND namespace_id IS NOT NULL AND resource_type IS NULL AND resource_id IS NOT NULL)
    OR (scope_kind = 'resource' AND namespace_id IS NOT NULL AND resource_type IS NOT NULL AND resource_id IS NOT NULL)
  )
);

CREATE TABLE management_installation_state (
  singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
  bootstrap_consumed_at TIMESTAMPTZ,
  bootstrap_principal_id UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  bootstrap_idempotency_hmac_version TEXT CHECK (
    bootstrap_idempotency_hmac_version IS NULL
    OR bootstrap_idempotency_hmac_version ~ '^[A-Za-z0-9][A-Za-z0-9-]{0,63}$'
  ),
  bootstrap_idempotency_key_digest BYTEA CHECK (
    bootstrap_idempotency_key_digest IS NULL
    OR octet_length(bootstrap_idempotency_key_digest) = 32
  ),
  bootstrap_request_digest BYTEA CHECK (
    bootstrap_request_digest IS NULL OR octet_length(bootstrap_request_digest) = 32
  ),
  bootstrap_response_ciphertext BYTEA,
  bootstrap_response_nonce BYTEA CHECK (
    bootstrap_response_nonce IS NULL OR octet_length(bootstrap_response_nonce) = 12
  ),
  bootstrap_response_kek_version TEXT CHECK (
    bootstrap_response_kek_version IS NULL
    OR bootstrap_response_kek_version ~ '^[A-Za-z0-9][A-Za-z0-9-]{0,63}$'
  ),
  bootstrap_response_status INTEGER CHECK (
    bootstrap_response_status IS NULL OR bootstrap_response_status BETWEEN 200 AND 299
  ),
  bootstrap_result_expires_at TIMESTAMPTZ,
  bootstrap_result_delivered_at TIMESTAMPTZ,
  seed_version BIGINT NOT NULL CHECK (seed_version > 0),
  recovery_consumed_at TIMESTAMPTZ,
  recovery_token_digest BYTEA CHECK (
    recovery_token_digest IS NULL OR octet_length(recovery_token_digest) = 32
  ),
  recovery_idempotency_hmac_version TEXT CHECK (
    recovery_idempotency_hmac_version IS NULL
    OR recovery_idempotency_hmac_version ~ '^[A-Za-z0-9][A-Za-z0-9-]{0,63}$'
  ),
  recovery_nonce_hmac BYTEA CHECK (
    recovery_nonce_hmac IS NULL OR octet_length(recovery_nonce_hmac) = 32
  ),
  recovery_request_digest BYTEA CHECK (
    recovery_request_digest IS NULL OR octet_length(recovery_request_digest) = 32
  ),
  recovery_principal_id UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  recovery_binding_id UUID REFERENCES management_role_bindings(id) ON DELETE RESTRICT,
  recovery_receipt JSONB,
  recovery_result_expires_at TIMESTAMPTZ,
  receipt JSONB NOT NULL DEFAULT '{}'::jsonb,
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK (
    (bootstrap_consumed_at IS NULL
      AND bootstrap_principal_id IS NULL
      AND bootstrap_idempotency_hmac_version IS NULL
      AND bootstrap_idempotency_key_digest IS NULL
      AND bootstrap_request_digest IS NULL
      AND bootstrap_response_ciphertext IS NULL
      AND bootstrap_response_nonce IS NULL
      AND bootstrap_response_kek_version IS NULL
      AND bootstrap_response_status IS NULL
      AND bootstrap_result_expires_at IS NULL
      AND bootstrap_result_delivered_at IS NULL)
    OR
    (bootstrap_consumed_at IS NOT NULL
      AND bootstrap_principal_id IS NOT NULL
      AND bootstrap_idempotency_hmac_version IS NOT NULL
      AND bootstrap_idempotency_key_digest IS NOT NULL
      AND bootstrap_request_digest IS NOT NULL
      AND bootstrap_response_status IS NOT NULL
      AND bootstrap_result_expires_at > bootstrap_consumed_at
      AND (bootstrap_result_delivered_at IS NULL
        OR bootstrap_result_delivered_at >= bootstrap_consumed_at))
  ),
  CHECK (
    num_nonnulls(bootstrap_response_ciphertext, bootstrap_response_nonce,
                 bootstrap_response_kek_version) IN (0, 3)
  ),
  CHECK (
    num_nonnulls(
      recovery_consumed_at, recovery_token_digest,
      recovery_idempotency_hmac_version, recovery_nonce_hmac,
      recovery_request_digest, recovery_principal_id, recovery_binding_id,
      recovery_receipt, recovery_result_expires_at
    ) IN (0, 9)
  ),
  CHECK (
    recovery_consumed_at IS NULL
    OR recovery_result_expires_at > recovery_consumed_at
  )
);

INSERT INTO management_installation_state
  (singleton, bootstrap_consumed_at, bootstrap_principal_id, seed_version,
   receipt, revision)
VALUES (TRUE, NULL, NULL, 1, '{}'::jsonb, 1);

CREATE TABLE management_session_policy (
  singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
  access_token_ttl_seconds BIGINT NOT NULL CHECK (access_token_ttl_seconds > 0),
  session_ttl_seconds BIGINT NOT NULL CHECK (session_ttl_seconds > 0),
  max_active_sessions INTEGER NOT NULL CHECK (max_active_sessions > 0),
  action_requirements JSONB NOT NULL,
  seed_version BIGINT NOT NULL CHECK (seed_version > 0),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

INSERT INTO management_session_policy
  (singleton, access_token_ttl_seconds, session_ttl_seconds,
   max_active_sessions, action_requirements, seed_version, revision)
VALUES (
  TRUE, 900, 28800, 5,
  '{"cluster_sensitive":{"any_of":[{"kind":"human","human":{"minimum_aal":"aal2","accepted_amr":[],"max_authentication_age_seconds":900}},{"kind":"workload","workload":{"minimum_workload_class":"workload_strong","max_source_age_seconds":2592000}}]}}'::jsonb,
  1, 1
);

CREATE TABLE management_security_policies (
  namespace_id UUID PRIMARY KEY REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  action_requirements JSONB NOT NULL,
  seed_version BIGINT NOT NULL CHECK (seed_version > 0),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE trusted_identity_issuers (
  id UUID PRIMARY KEY,
  issuer TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL CHECK (kind IN ('oidc','jwt')),
  discovery_url TEXT,
  jwks_url TEXT,
  audiences JSONB NOT NULL,
  claim_mapping JSONB NOT NULL,
  assurance_mapping JSONB NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active','disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK (num_nonnulls(discovery_url, jwks_url) = 1)
);

CREATE TABLE management_mtls_mappings (
  id UUID PRIMARY KEY,
  matcher_kind TEXT NOT NULL CHECK (matcher_kind IN ('spiffe_id','san_uri','san_dns','subject_dn_sha256')),
  matcher_value TEXT NOT NULL,
  principal_id UUID NOT NULL REFERENCES management_principals(id) ON DELETE RESTRICT,
  workload_class TEXT NOT NULL CHECK (workload_class IN ('workload_standard','workload_strong')),
  source_assured_at TIMESTAMPTZ NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active','disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (matcher_kind, matcher_value)
);

CREATE TABLE management_sessions (
  id UUID PRIMARY KEY,
  principal_id UUID NOT NULL REFERENCES management_principals(id) ON DELETE RESTRICT,
  issuer_session_id TEXT,
  token_id TEXT NOT NULL UNIQUE,
  audience TEXT NOT NULL,
  auth_source_kind TEXT NOT NULL CHECK (auth_source_kind IN ('issuer','service_credential','mtls')),
  auth_source_id UUID,
  evidence_kind TEXT NOT NULL CHECK (evidence_kind IN ('human','workload')),
  assurance JSONB NOT NULL,
  authenticated_at TIMESTAMPTZ NOT NULL,
  source_assured_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active','revoked','expired')),
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- PostgreSQL-backed Management authentication coordination is used when the
-- durable Management store is configured without a shared runtime store.
-- Challenges are one-time values; the digest keeps the bearer nonce out of
-- durable state. Explicit barriers cover short mutation fences while the
-- identity tables remain the authoritative source for durable revocations.
CREATE TABLE management_exchange_challenges (
  id UUID PRIMARY KEY,
  issuer_id UUID NOT NULL REFERENCES trusted_identity_issuers(id) ON DELETE CASCADE,
  nonce_digest BYTEA NOT NULL CHECK (octet_length(nonce_digest) = 32),
  rate_identity_digest BYTEA NOT NULL CHECK (octet_length(rate_identity_digest) = 32),
  expires_at TIMESTAMPTZ NOT NULL,
  consumed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK (consumed_at IS NULL OR consumed_at >= created_at)
);

CREATE INDEX management_exchange_challenges_rate_idx
  ON management_exchange_challenges(rate_identity_digest, expires_at);

CREATE TABLE management_revocation_barriers (
  barrier_kind TEXT NOT NULL CHECK (barrier_kind IN (
    'cluster_session_policy','namespace_security_policy','management_session',
    'management_principal','authentication_source'
  )),
  barrier_id TEXT NOT NULL CHECK (octet_length(barrier_id) BETWEEN 1 AND 256),
  installed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (barrier_kind, barrier_id)
);

CREATE TABLE management_principal_user_links (
  principal_id UUID NOT NULL REFERENCES management_principals(id) ON DELETE RESTRICT,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  user_id UUID NOT NULL,
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (principal_id, namespace_id),
  UNIQUE (namespace_id, user_id, principal_id),
  FOREIGN KEY (namespace_id, user_id) REFERENCES access_users(namespace_id, id) ON DELETE RESTRICT
);

CREATE TABLE management_service_accounts (
  id UUID PRIMARY KEY,
  principal_id UUID NOT NULL UNIQUE REFERENCES management_principals(id) ON DELETE RESTRICT,
  owner_scope TEXT NOT NULL CHECK (owner_scope IN ('cluster','namespace')),
  namespace_id UUID REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  status TEXT NOT NULL CHECK (status IN ('active','disabled')),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK ((owner_scope = 'cluster') = (namespace_id IS NULL))
);

CREATE TABLE management_service_account_credentials (
  id UUID PRIMARY KEY,
  service_account_id UUID NOT NULL REFERENCES management_service_accounts(id) ON DELETE RESTRICT,
  public_id TEXT NOT NULL UNIQUE,
  secret_hmac BYTEA NOT NULL,
  pepper_version TEXT NOT NULL,
  workload_class TEXT NOT NULL CHECK (workload_class IN ('workload_standard','workload_strong')),
  source_assured_at TIMESTAMPTZ NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active','retiring','revoked')),
  not_before TIMESTAMPTZ NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE self_service_policies (
  namespace_id UUID PRIMARY KEY REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  max_keys_per_user INTEGER NOT NULL DEFAULT 0 CHECK (max_keys_per_user >= 0),
  max_delegated_sessions INTEGER NOT NULL DEFAULT 0 CHECK (max_delegated_sessions >= 0),
  delegated_session_ttl_seconds BIGINT NOT NULL DEFAULT 900 CHECK (delegated_session_ttl_seconds > 0),
  allow_team_key_delegation BOOLEAN NOT NULL DEFAULT FALSE,
  automatic_first_key BOOLEAN NOT NULL DEFAULT FALSE,
  team_admin_capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
  default_access_policy_id UUID REFERENCES access_policies(id) ON DELETE RESTRICT,
  default_rate_limit_policy_id UUID REFERENCES rate_limit_policies(id) ON DELETE RESTRICT,
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  seed_version BIGINT NOT NULL CHECK (seed_version > 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE management_invitations (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  created_by_principal_id UUID NOT NULL REFERENCES management_principals(id) ON DELETE RESTRICT,
  expected_issuer TEXT NOT NULL,
  expected_subject TEXT,
  expected_email TEXT,
  display_name TEXT NOT NULL,
  token_hmac BYTEA NOT NULL,
  pepper_version TEXT NOT NULL,
  grants JSONB NOT NULL,
  team_id UUID,
  team_role TEXT CHECK (team_role IN ('member','admin')),
  pinned_access_policy_id UUID REFERENCES access_policies(id) ON DELETE RESTRICT,
  pinned_access_policy_revision BIGINT,
  pinned_rate_limit_policy_id UUID REFERENCES rate_limit_policies(id) ON DELETE RESTRICT,
  pinned_rate_limit_policy_revision BIGINT,
  expires_at TIMESTAMPTZ NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending','accepted','expired','revoked')),
  accepted_principal_id UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  accepted_user_id UUID,
  accepted_management_session_id UUID REFERENCES management_sessions(id) ON DELETE RESTRICT,
  accepted_auth_source_kind TEXT,
  accepted_auth_source_id TEXT,
  accepted_evidence_kind TEXT,
  accepted_at TIMESTAMPTZ,
  acceptance_response_ciphertext BYTEA,
  acceptance_response_nonce BYTEA CHECK (
    acceptance_response_nonce IS NULL OR octet_length(acceptance_response_nonce) = 12
  ),
  acceptance_response_kek_version TEXT CHECK (
    acceptance_response_kek_version IS NULL
    OR acceptance_response_kek_version ~ '^[A-Za-z0-9][A-Za-z0-9-]{0,63}$'
  ),
  acceptance_result_expires_at TIMESTAMPTZ,
  acceptance_result_delivered_at TIMESTAMPTZ,
  acceptance_result_erased_at TIMESTAMPTZ,
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK (expected_subject IS NOT NULL OR expected_email IS NOT NULL),
  CHECK (
    (team_id IS NULL
      AND team_role IS NULL
      AND pinned_access_policy_id IS NOT NULL
      AND pinned_access_policy_revision > 0
      AND pinned_rate_limit_policy_id IS NOT NULL
      AND pinned_rate_limit_policy_revision > 0)
    OR
    (team_id IS NOT NULL
      AND team_role IS NOT NULL
      AND pinned_access_policy_id IS NULL
      AND pinned_access_policy_revision IS NULL
      AND pinned_rate_limit_policy_id IS NULL
      AND pinned_rate_limit_policy_revision IS NULL)
  ),
  CHECK (
    (status <> 'accepted'
      AND accepted_principal_id IS NULL
      AND accepted_user_id IS NULL
      AND accepted_management_session_id IS NULL
      AND accepted_auth_source_kind IS NULL
      AND accepted_auth_source_id IS NULL
      AND accepted_evidence_kind IS NULL
      AND accepted_at IS NULL
      AND acceptance_response_ciphertext IS NULL
      AND acceptance_response_nonce IS NULL
      AND acceptance_response_kek_version IS NULL
      AND acceptance_result_expires_at IS NULL
      AND acceptance_result_delivered_at IS NULL
      AND acceptance_result_erased_at IS NULL)
    OR
    (status = 'accepted'
      AND accepted_principal_id IS NOT NULL
      AND accepted_user_id IS NOT NULL
      AND accepted_management_session_id IS NOT NULL
      AND accepted_auth_source_kind IS NOT NULL
      AND accepted_auth_source_id IS NOT NULL
      AND accepted_evidence_kind IN ('human','workload')
      AND accepted_at IS NOT NULL
      AND acceptance_result_expires_at > accepted_at
      AND (acceptance_result_delivered_at IS NULL
        OR (acceptance_result_delivered_at >= accepted_at
          AND acceptance_result_delivered_at < acceptance_result_expires_at))
      AND (
        (acceptance_response_ciphertext IS NOT NULL
          AND acceptance_response_nonce IS NOT NULL
          AND acceptance_response_kek_version IS NOT NULL
          AND acceptance_result_erased_at IS NULL)
        OR
        (acceptance_response_ciphertext IS NULL
          AND acceptance_response_nonce IS NULL
          AND acceptance_response_kek_version IS NULL
          AND acceptance_result_erased_at >= acceptance_result_expires_at)
      ))
  ),
  FOREIGN KEY (namespace_id, team_id) REFERENCES access_teams(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, accepted_user_id) REFERENCES access_users(namespace_id, id) ON DELETE RESTRICT
);

CREATE TABLE delegated_inference_sessions (
  id UUID PRIMARY KEY,
  public_id TEXT NOT NULL UNIQUE,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  management_session_id UUID NOT NULL REFERENCES management_sessions(id) ON DELETE RESTRICT,
  principal_id UUID NOT NULL REFERENCES management_principals(id) ON DELETE RESTRICT,
  api_key_id UUID NOT NULL,
  delegation_epoch BIGINT NOT NULL CHECK (delegation_epoch > 0),
  user_id UUID NOT NULL,
  team_id UUID,
  token_hmac BYTEA NOT NULL,
  pepper_version TEXT NOT NULL,
  audience TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active','revoked','expired')),
  not_before TIMESTAMPTZ NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  revoked_at TIMESTAMPTZ,
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, id),
  FOREIGN KEY (namespace_id, api_key_id) REFERENCES access_api_keys(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, user_id) REFERENCES access_users(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, team_id) REFERENCES access_teams(namespace_id, id) ON DELETE RESTRICT
);

CREATE TABLE provider_credentials (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  name TEXT NOT NULL,
  provider_id TEXT NOT NULL,
  credential_mode TEXT NOT NULL CHECK (credential_mode IN ('optional','required')),
  credential_adapter_id TEXT NOT NULL,
  provider_catalog_revision TEXT NOT NULL CHECK (provider_catalog_revision ~ '^sha256:[a-f0-9]{64}$'),
  normalized_origin TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active','disabled','deleted')),
  active_version_id UUID,
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  deleted_at TIMESTAMPTZ,
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, name),
  CHECK ((status IN ('active','disabled')) = (active_version_id IS NOT NULL)),
  CHECK ((status = 'deleted') = (deleted_at IS NOT NULL))
);

CREATE TABLE provider_credential_versions (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  provider_credential_id UUID NOT NULL,
  secret_ciphertext BYTEA,
  ciphertext_nonce BYTEA,
  kek_version TEXT,
  status TEXT NOT NULL CHECK (status IN ('active','retiring','revoked')),
  not_before TIMESTAMPTZ NOT NULL,
  expires_at TIMESTAMPTZ,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, provider_credential_id, id),
  CHECK (expires_at IS NULL OR expires_at > not_before),
  CHECK (
    (status = 'active' AND revoked_at IS NULL
      AND secret_ciphertext IS NOT NULL AND ciphertext_nonce IS NOT NULL AND kek_version IS NOT NULL)
    OR (status = 'retiring' AND expires_at IS NOT NULL AND revoked_at IS NULL
      AND secret_ciphertext IS NOT NULL AND ciphertext_nonce IS NOT NULL AND kek_version IS NOT NULL)
    OR (status = 'revoked' AND revoked_at IS NOT NULL
      AND secret_ciphertext IS NULL AND ciphertext_nonce IS NULL AND kek_version IS NULL)
  ),
  FOREIGN KEY (namespace_id, provider_credential_id) REFERENCES provider_credentials(namespace_id, id) ON DELETE RESTRICT
);

ALTER TABLE provider_credentials
  ADD CONSTRAINT provider_credentials_active_version_fk
  FOREIGN KEY (namespace_id, id, active_version_id)
  REFERENCES provider_credential_versions(namespace_id, provider_credential_id, id)
  DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE routing_models (
  id TEXT PRIMARY KEY CHECK (id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  name TEXT NOT NULL,
  aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
  status TEXT NOT NULL CHECK (status IN ('draft','active','disabled','deleted')),
  current_revision BIGINT NOT NULL CHECK (current_revision > 0),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  deleted_at TIMESTAMPTZ,
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, name)
);

CREATE TABLE routing_model_revisions (
  model_id TEXT NOT NULL REFERENCES routing_models(id) ON DELETE RESTRICT,
  revision BIGINT NOT NULL CHECK (revision > 0),
  provider_catalog_revision TEXT NOT NULL CHECK (provider_catalog_revision ~ '^sha256:[a-f0-9]{64}$'),
  name TEXT NOT NULL,
  aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
  capabilities JSONB NOT NULL,
  reasoning JSONB NOT NULL DEFAULT '{}'::jsonb,
  loras JSONB NOT NULL DEFAULT '[]'::jsonb,
  execution JSONB NOT NULL,
  pricing JSONB NOT NULL,
  content_digest BYTEA NOT NULL,
  created_by UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (model_id, revision),
  UNIQUE (model_id, content_digest)
);

CREATE TABLE routing_model_backends (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  model_id TEXT NOT NULL,
  model_revision BIGINT NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
  provider_id TEXT NOT NULL,
  wire_format TEXT NOT NULL,
  normalized_origin TEXT NOT NULL,
  provider_model_id TEXT NOT NULL,
  provider_credential_id UUID,
  connection JSONB NOT NULL DEFAULT '{}'::jsonb,
  weight NUMERIC(20,9) NOT NULL DEFAULT 1 CHECK (weight > 0),
  FOREIGN KEY (namespace_id, model_id) REFERENCES routing_models(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (model_id, model_revision) REFERENCES routing_model_revisions(model_id, revision) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, provider_credential_id) REFERENCES provider_credentials(namespace_id, id) ON DELETE RESTRICT,
  UNIQUE (model_id, model_revision, ordinal)
);

ALTER TABLE routing_models
  ADD CONSTRAINT routing_models_current_revision_fk
  FOREIGN KEY (id, current_revision)
  REFERENCES routing_model_revisions(model_id, revision)
  DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE routing_recipes (
  id TEXT PRIMARY KEY CHECK (id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL CHECK (status IN ('draft','active','disabled','deleted')),
  current_revision BIGINT NOT NULL CHECK (current_revision > 0),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  deleted_at TIMESTAMPTZ,
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, name)
);

CREATE TABLE routing_recipe_revisions (
  recipe_id TEXT NOT NULL REFERENCES routing_recipes(id) ON DELETE RESTRICT,
  revision BIGINT NOT NULL CHECK (revision > 0),
  name TEXT NOT NULL,
  document JSONB NOT NULL,
  content_digest BYTEA NOT NULL,
  created_by UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (recipe_id, revision),
  UNIQUE (recipe_id, content_digest)
);

CREATE TABLE routing_recipe_decisions (
  recipe_id TEXT NOT NULL,
  recipe_revision BIGINT NOT NULL,
  decision_id TEXT NOT NULL CHECK (decision_id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  name TEXT NOT NULL,
  dispatch_cardinality TEXT NOT NULL CHECK (dispatch_cardinality IN ('single','multi')),
  ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
  capabilities JSONB NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (recipe_id, recipe_revision, decision_id),
  UNIQUE (recipe_id, recipe_revision, ordinal),
  FOREIGN KEY (recipe_id, recipe_revision) REFERENCES routing_recipe_revisions(recipe_id, revision) ON DELETE RESTRICT
);

ALTER TABLE routing_recipes
  ADD CONSTRAINT routing_recipes_current_revision_fk
  FOREIGN KEY (id, current_revision)
  REFERENCES routing_recipe_revisions(recipe_id, revision)
  DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE routing_entrypoints (
  id TEXT PRIMARY KEY CHECK (id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  name TEXT NOT NULL,
  aliases JSONB NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('draft','active','disabled','deleted')),
  current_revision BIGINT NOT NULL CHECK (current_revision > 0),
  published_revision BIGINT CHECK (published_revision > 0),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  deleted_at TIMESTAMPTZ,
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, name),
  CHECK ((status = 'active') = (published_revision IS NOT NULL))
);

CREATE TABLE routing_entrypoint_revisions (
  entrypoint_id TEXT NOT NULL REFERENCES routing_entrypoints(id) ON DELETE RESTRICT,
  revision BIGINT NOT NULL CHECK (revision > 0),
  name TEXT NOT NULL,
  aliases JSONB NOT NULL,
  content_digest BYTEA NOT NULL,
  created_by UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (entrypoint_id, revision),
  UNIQUE (entrypoint_id, content_digest)
);

CREATE TABLE routing_entrypoint_rules (
  id TEXT NOT NULL CHECK (id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  entrypoint_id TEXT NOT NULL,
  entrypoint_revision BIGINT NOT NULL,
  name TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
  matchers JSONB NOT NULL DEFAULT '[]'::jsonb,
  recipe_id TEXT NOT NULL,
  recipe_revision BIGINT NOT NULL,
  PRIMARY KEY (entrypoint_id, entrypoint_revision, id),
  FOREIGN KEY (entrypoint_id, entrypoint_revision)
    REFERENCES routing_entrypoint_revisions(entrypoint_id, revision) ON DELETE RESTRICT,
  FOREIGN KEY (recipe_id, recipe_revision) REFERENCES routing_recipe_revisions(recipe_id, revision) ON DELETE RESTRICT,
  UNIQUE (entrypoint_id, entrypoint_revision, ordinal)
);

CREATE TABLE routing_decision_assignments (
  entrypoint_id TEXT NOT NULL,
  entrypoint_revision BIGINT NOT NULL,
  rule_id TEXT NOT NULL,
  recipe_id TEXT NOT NULL,
  recipe_revision BIGINT NOT NULL,
  decision_id TEXT NOT NULL CHECK (decision_id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  fallback_strategy TEXT CHECK (fallback_strategy = 'priority'),
  fallback_on JSONB,
  PRIMARY KEY (entrypoint_id, entrypoint_revision, rule_id, decision_id),
  FOREIGN KEY (entrypoint_id, entrypoint_revision, rule_id)
    REFERENCES routing_entrypoint_rules(entrypoint_id, entrypoint_revision, id) ON DELETE RESTRICT,
  FOREIGN KEY (recipe_id, recipe_revision, decision_id)
    REFERENCES routing_recipe_decisions(recipe_id, recipe_revision, decision_id) ON DELETE RESTRICT,
  CHECK ((fallback_strategy IS NULL AND fallback_on IS NULL)
    OR (fallback_strategy = 'priority' AND jsonb_typeof(fallback_on) = 'array'
      AND jsonb_array_length(fallback_on) BETWEEN 1 AND 3))
);

CREATE TABLE routing_assignment_models (
  entrypoint_id TEXT NOT NULL,
  entrypoint_revision BIGINT NOT NULL,
  rule_id TEXT NOT NULL,
  decision_id TEXT NOT NULL CHECK (decision_id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
  model_id TEXT NOT NULL,
  model_revision BIGINT NOT NULL,
  priority SMALLINT NOT NULL DEFAULT 0 CHECK (priority BETWEEN 0 AND 31),
  weight NUMERIC(20,9) NOT NULL DEFAULT 1 CHECK (weight > 0),
  lora_name TEXT,
  reasoning JSONB,
  PRIMARY KEY (entrypoint_id, entrypoint_revision, rule_id, decision_id, ordinal),
  FOREIGN KEY (entrypoint_id, entrypoint_revision, rule_id, decision_id)
    REFERENCES routing_decision_assignments(entrypoint_id, entrypoint_revision, rule_id, decision_id) ON DELETE RESTRICT,
  FOREIGN KEY (model_id, model_revision) REFERENCES routing_model_revisions(model_id, revision) ON DELETE RESTRICT
);

ALTER TABLE routing_entrypoints
  ADD CONSTRAINT routing_entrypoints_current_revision_fk
  FOREIGN KEY (id, current_revision)
  REFERENCES routing_entrypoint_revisions(entrypoint_id, revision)
  DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE routing_entrypoints
  ADD CONSTRAINT routing_entrypoints_published_revision_fk
  FOREIGN KEY (id, published_revision)
  REFERENCES routing_entrypoint_revisions(entrypoint_id, revision)
  DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE routing_snapshots (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  routing_revision BIGINT NOT NULL CHECK (routing_revision > 0),
  content_digest BYTEA NOT NULL,
  compiled_blob BYTEA NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('staged','active','failed','retired')),
  failure_reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  activated_at TIMESTAMPTZ,
  PRIMARY KEY (namespace_id, routing_revision),
  UNIQUE (namespace_id, content_digest)
);

CREATE TABLE routing_snapshot_members (
  namespace_id UUID NOT NULL,
  routing_revision BIGINT NOT NULL,
  resource_type TEXT NOT NULL CHECK (resource_type IN ('model','recipe','entrypoint')),
  resource_id TEXT NOT NULL CHECK (resource_id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  resource_revision BIGINT NOT NULL,
  PRIMARY KEY (namespace_id, routing_revision, resource_type, resource_id),
  FOREIGN KEY (namespace_id, routing_revision) REFERENCES routing_snapshots(namespace_id, routing_revision) ON DELETE RESTRICT
);

-- Durable routing publication coordination. PostgreSQL is the publication
-- authority when no shared runtime store is configured. Notifications only
-- wake replicas; leases, immutable payloads, acknowledgements, and heads are
-- durable rows and remain recoverable by polling after a disconnect.
CREATE TABLE routing_publications (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  desired_revision BIGINT NOT NULL CHECK (desired_revision > 0),
  publication_id TEXT NOT NULL CHECK (octet_length(publication_id) BETWEEN 1 AND 256),
  quota_partition_id TEXT NOT NULL CHECK (octet_length(quota_partition_id) BETWEEN 1 AND 256),
  runtime_epoch BIGINT NOT NULL CHECK (runtime_epoch > 0),
  publication_digest TEXT NOT NULL CHECK (publication_digest ~ '^[a-f0-9]{64}$'),
  manifest_digest TEXT NOT NULL CHECK (manifest_digest ~ '^[a-f0-9]{64}$'),
  routing_digest TEXT NOT NULL CHECK (routing_digest ~ '^[a-f0-9]{64}$'),
  state TEXT NOT NULL CHECK (state IN ('prepared','staged','validated','active','applied','finalized')),
  restrictive BOOLEAN NOT NULL DEFAULT FALSE CHECK (restrictive = FALSE),
  publication_blob BYTEA NOT NULL CHECK (octet_length(publication_blob) > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  activated_at TIMESTAMPTZ,
  PRIMARY KEY (namespace_id, desired_revision),
  UNIQUE (namespace_id, publication_id),
  FOREIGN KEY (namespace_id, desired_revision)
    REFERENCES routing_snapshots(namespace_id, routing_revision) ON DELETE RESTRICT
);

CREATE TABLE routing_publication_heads (
  namespace_id UUID PRIMARY KEY REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  quota_partition_id TEXT NOT NULL CHECK (octet_length(quota_partition_id) BETWEEN 1 AND 256),
  active_publication_id TEXT,
  active_revision BIGINT CHECK (active_revision > 0),
  candidate_publication_id TEXT,
  candidate_revision BIGINT CHECK (candidate_revision > 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK ((active_publication_id IS NULL) = (active_revision IS NULL)),
  CHECK ((candidate_publication_id IS NULL) = (candidate_revision IS NULL))
);

CREATE TABLE routing_fleet_replicas (
  replica_id TEXT PRIMARY KEY CHECK (octet_length(replica_id) BETWEEN 1 AND 256),
  lease_expires_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE routing_replica_leases (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  replica_id TEXT NOT NULL CHECK (octet_length(replica_id) BETWEEN 1 AND 256),
  runtime_epoch BIGINT NOT NULL CHECK (runtime_epoch > 0),
  access_publication_id TEXT,
  routing_publication_id TEXT,
  lease_expires_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, replica_id),
  CHECK ((access_publication_id IS NULL) = (routing_publication_id IS NULL)),
  CHECK (access_publication_id IS NULL OR access_publication_id = routing_publication_id)
);

CREATE TABLE routing_publication_required_replicas (
  namespace_id UUID NOT NULL,
  publication_id TEXT NOT NULL,
  replica_id TEXT NOT NULL CHECK (octet_length(replica_id) BETWEEN 1 AND 256),
  required_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, publication_id, replica_id),
  FOREIGN KEY (namespace_id, publication_id)
    REFERENCES routing_publications(namespace_id, publication_id) ON DELETE CASCADE
);

CREATE TABLE routing_publication_acknowledgements (
  namespace_id UUID NOT NULL,
  publication_id TEXT NOT NULL,
  replica_id TEXT NOT NULL CHECK (octet_length(replica_id) BETWEEN 1 AND 256),
  kind TEXT NOT NULL CHECK (kind IN ('barrier','routing')),
  publication_digest TEXT NOT NULL CHECK (publication_digest ~ '^[a-f0-9]{64}$'),
  acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, publication_id, replica_id, kind),
  FOREIGN KEY (namespace_id, publication_id, replica_id)
    REFERENCES routing_publication_required_replicas(namespace_id, publication_id, replica_id) ON DELETE CASCADE
);

CREATE INDEX routing_fleet_replicas_lease_idx ON routing_fleet_replicas(lease_expires_at, replica_id);
CREATE INDEX routing_replica_leases_expiry_idx ON routing_replica_leases(namespace_id, lease_expires_at, replica_id);
CREATE INDEX routing_publications_state_idx ON routing_publications(namespace_id, state, desired_revision DESC);

-- Router-native Agent resources

CREATE TABLE agent_skills (
  id UUID PRIMARY KEY,
  namespace_id UUID REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  builtin BOOLEAN NOT NULL DEFAULT FALSE,
  status TEXT NOT NULL CHECK (status IN ('active','disabled','deleted')),
  current_revision BIGINT NOT NULL CHECK (current_revision > 0),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  deleted_at TIMESTAMPTZ,
  UNIQUE NULLS NOT DISTINCT (namespace_id, id),
  UNIQUE NULLS NOT DISTINCT (namespace_id, name),
  CHECK (builtin = (namespace_id IS NULL)),
  CHECK ((status = 'deleted') = (deleted_at IS NOT NULL))
);

CREATE TABLE agent_skill_revisions (
  skill_id UUID NOT NULL REFERENCES agent_skills(id) ON DELETE RESTRICT,
  namespace_id UUID,
  revision BIGINT NOT NULL CHECK (revision > 0),
  instructions TEXT NOT NULL CHECK (octet_length(instructions) BETWEEN 1 AND 262144),
  required_tools JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(required_tools) = 'array'),
  minimum_capabilities JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(minimum_capabilities) = 'array'),
  content_digest BYTEA NOT NULL CHECK (octet_length(content_digest) = 32),
  created_by UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (skill_id, revision),
  UNIQUE NULLS NOT DISTINCT (namespace_id, skill_id, revision),
  UNIQUE (skill_id, content_digest),
  FOREIGN KEY (namespace_id, skill_id)
    REFERENCES agent_skills(namespace_id, id) ON DELETE RESTRICT
);

ALTER TABLE agent_skills
  ADD CONSTRAINT agent_skills_current_revision_fk
  FOREIGN KEY (id, current_revision)
  REFERENCES agent_skill_revisions(skill_id, revision)
  DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE agent_tool_sources (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  source_kind TEXT NOT NULL CHECK (source_kind IN ('remote','builtin_integration')),
  status TEXT NOT NULL CHECK (status IN ('active','disabled','deleted')),
  current_revision BIGINT NOT NULL CHECK (current_revision > 0),
  approved_discovery_digest BYTEA CHECK (
    approved_discovery_digest IS NULL OR octet_length(approved_discovery_digest) = 32
  ),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  deleted_at TIMESTAMPTZ,
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, name),
  CHECK ((status = 'deleted') = (deleted_at IS NOT NULL))
);

CREATE TABLE agent_tool_credentials (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  name TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active','disabled','deleted')),
  active_version_id UUID,
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  deleted_at TIMESTAMPTZ,
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, name),
  CHECK ((status = 'deleted') = (active_version_id IS NULL)),
  CHECK ((status = 'deleted') = (deleted_at IS NOT NULL))
);

CREATE TABLE agent_tool_credential_versions (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  credential_id UUID NOT NULL,
  secret_ciphertext BYTEA,
  ciphertext_nonce BYTEA,
  kek_version TEXT,
  status TEXT NOT NULL CHECK (status IN ('active','retiring','revoked')),
  not_before TIMESTAMPTZ NOT NULL,
  expires_at TIMESTAMPTZ,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, credential_id, id),
  FOREIGN KEY (namespace_id, credential_id)
    REFERENCES agent_tool_credentials(namespace_id, id) ON DELETE RESTRICT,
  CHECK (expires_at IS NULL OR expires_at > not_before),
  CHECK (
    (status = 'active' AND revoked_at IS NULL
      AND secret_ciphertext IS NOT NULL AND ciphertext_nonce IS NOT NULL AND kek_version IS NOT NULL)
    OR (status = 'retiring' AND expires_at IS NOT NULL AND revoked_at IS NULL
      AND secret_ciphertext IS NOT NULL AND ciphertext_nonce IS NOT NULL AND kek_version IS NOT NULL)
    OR (status = 'revoked' AND revoked_at IS NOT NULL
      AND secret_ciphertext IS NULL AND ciphertext_nonce IS NULL AND kek_version IS NULL)
  )
);

ALTER TABLE agent_tool_credentials
  ADD CONSTRAINT agent_tool_credentials_active_version_fk
  FOREIGN KEY (namespace_id, id, active_version_id)
  REFERENCES agent_tool_credential_versions(namespace_id, credential_id, id)
  DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE agent_tool_source_revisions (
  source_id UUID NOT NULL REFERENCES agent_tool_sources(id) ON DELETE RESTRICT,
  namespace_id UUID NOT NULL,
  revision BIGINT NOT NULL CHECK (revision > 0),
  transport TEXT NOT NULL CHECK (transport = 'streamable_http'),
  endpoint TEXT NOT NULL,
  credential_id UUID,
  egress_policy JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(egress_policy) = 'object'),
  discovered_tools JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(discovered_tools) = 'array'),
  discovery_digest BYTEA CHECK (discovery_digest IS NULL OR octet_length(discovery_digest) = 32),
  content_digest BYTEA NOT NULL CHECK (octet_length(content_digest) = 32),
  created_by UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (source_id, revision),
  UNIQUE (namespace_id, source_id, revision),
  UNIQUE (source_id, content_digest),
  FOREIGN KEY (namespace_id, source_id) REFERENCES agent_tool_sources(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, credential_id) REFERENCES agent_tool_credentials(namespace_id, id) ON DELETE RESTRICT
);

CREATE TABLE agent_tool_registry_revisions (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  registry_revision TEXT NOT NULL CHECK (registry_revision ~ '^sha256:[a-f0-9]{64}$'),
  manifest JSONB NOT NULL CHECK (jsonb_typeof(manifest) = 'object'),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  expires_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (namespace_id, registry_revision),
  CHECK (expires_at > created_at)
);

ALTER TABLE agent_tool_sources
  ADD CONSTRAINT agent_tool_sources_current_revision_fk
  FOREIGN KEY (id, current_revision)
  REFERENCES agent_tool_source_revisions(source_id, revision)
  DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE agent_profiles (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL CHECK (status IN ('active','disabled','deleted')),
  current_revision BIGINT NOT NULL CHECK (current_revision > 0),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  deleted_at TIMESTAMPTZ,
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, name),
  CHECK ((status = 'deleted') = (deleted_at IS NOT NULL))
);

CREATE TABLE agent_profile_revisions (
  profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE RESTRICT,
  namespace_id UUID NOT NULL,
  revision BIGINT NOT NULL CHECK (revision > 0),
  target_model_id TEXT,
  target_entrypoint_id TEXT,
  minimum_target_capabilities JSONB NOT NULL DEFAULT '[]'::jsonb
    CHECK (jsonb_typeof(minimum_target_capabilities) = 'array'),
  supported_modes JSONB NOT NULL DEFAULT '["builder","chat"]'::jsonb
    CHECK (
      jsonb_typeof(supported_modes) = 'array'
      AND supported_modes IN ('["builder"]'::jsonb, '["chat"]'::jsonb, '["builder","chat"]'::jsonb)
    ),
  tool_policy JSONB NOT NULL CHECK (jsonb_typeof(tool_policy) = 'object'),
  approval_policy TEXT NOT NULL CHECK (approval_policy = 'required'),
  maximum_turn_seconds BIGINT NOT NULL CHECK (maximum_turn_seconds BETWEEN 10 AND 86400),
  maximum_tool_steps INTEGER NOT NULL CHECK (maximum_tool_steps BETWEEN 1 AND 256),
  context_token_budget BIGINT NOT NULL CHECK (context_token_budget BETWEEN 1024 AND 1048576),
  content_digest BYTEA NOT NULL CHECK (octet_length(content_digest) = 32),
  created_by UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (profile_id, revision),
  UNIQUE (namespace_id, profile_id, revision),
  UNIQUE (profile_id, content_digest),
  FOREIGN KEY (namespace_id, profile_id) REFERENCES agent_profiles(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, target_model_id) REFERENCES routing_models(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, target_entrypoint_id) REFERENCES routing_entrypoints(namespace_id, id) ON DELETE RESTRICT,
  CHECK (num_nonnulls(target_model_id, target_entrypoint_id) <= 1)
);

CREATE TABLE agent_profile_skills (
  namespace_id UUID NOT NULL,
  profile_id UUID NOT NULL,
  profile_revision BIGINT NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
  skill_id UUID NOT NULL,
  skill_namespace_id UUID,
  skill_revision BIGINT NOT NULL CHECK (skill_revision > 0),
  PRIMARY KEY (profile_id, profile_revision, ordinal),
  UNIQUE (profile_id, profile_revision, skill_id),
  FOREIGN KEY (namespace_id, profile_id, profile_revision)
    REFERENCES agent_profile_revisions(namespace_id, profile_id, revision) ON DELETE RESTRICT,
  FOREIGN KEY (skill_namespace_id, skill_id, skill_revision)
    REFERENCES agent_skill_revisions(namespace_id, skill_id, revision) ON DELETE RESTRICT,
  FOREIGN KEY (skill_id, skill_revision)
    REFERENCES agent_skill_revisions(skill_id, revision) ON DELETE RESTRICT,
  CHECK (skill_namespace_id IS NULL OR skill_namespace_id = namespace_id)
);

ALTER TABLE agent_profiles
  ADD CONSTRAINT agent_profiles_current_revision_fk
  FOREIGN KEY (id, current_revision)
  REFERENCES agent_profile_revisions(profile_id, revision)
  DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE agent_profile_defaults (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  mode TEXT NOT NULL CHECK (mode IN ('chat','builder')),
  profile_id UUID NOT NULL,
  profile_revision BIGINT NOT NULL CHECK (profile_revision > 0),
  updated_by UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, mode),
  FOREIGN KEY (namespace_id, profile_id, profile_revision)
    REFERENCES agent_profile_revisions(namespace_id, profile_id, revision) ON DELETE RESTRICT
);

CREATE TABLE agent_sessions (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  owner_principal_id UUID NOT NULL REFERENCES management_principals(id) ON DELETE RESTRICT,
  effective_user_id UUID,
  effective_team_id UUID,
  delegated_inference_session_id UUID NOT NULL,
  profile_id UUID NOT NULL,
  profile_revision BIGINT NOT NULL CHECK (profile_revision > 0),
  target_model_id TEXT,
  target_entrypoint_id TEXT,
  target_public_id TEXT NOT NULL CHECK (
    octet_length(target_public_id) BETWEEN 1 AND 256
    AND target_public_id !~ '[[:space:][:cntrl:]]'
  ),
  authority_digest TEXT NOT NULL CHECK (authority_digest ~ '^sha256:[a-f0-9]{64}$'),
  mode TEXT NOT NULL CHECK (mode IN ('chat','builder')),
  title TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL CHECK (status IN ('active','closed','deleted')),
  next_sequence BIGINT NOT NULL DEFAULT 1 CHECK (next_sequence > 0),
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  closed_at TIMESTAMPTZ,
  deleted_at TIMESTAMPTZ,
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, delegated_inference_session_id),
  FOREIGN KEY (namespace_id, effective_user_id) REFERENCES access_users(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, effective_team_id) REFERENCES access_teams(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, delegated_inference_session_id)
    REFERENCES delegated_inference_sessions(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, profile_id, profile_revision)
    REFERENCES agent_profile_revisions(namespace_id, profile_id, revision) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, target_model_id) REFERENCES routing_models(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, target_entrypoint_id) REFERENCES routing_entrypoints(namespace_id, id) ON DELETE RESTRICT,
  CHECK (num_nonnulls(target_model_id, target_entrypoint_id) = 1),
  CHECK ((status IN ('closed','deleted')) = (closed_at IS NOT NULL)),
  CHECK ((status = 'deleted') = (deleted_at IS NOT NULL))
);

-- The delegated credential backing an Agent Session is internal execution
-- state. It is encrypted and is never included in Management responses or
-- persisted Agent events.
CREATE TABLE agent_session_inference_credentials (
  namespace_id UUID NOT NULL,
  session_id UUID NOT NULL,
  delegated_inference_session_id UUID NOT NULL,
  secret_ciphertext BYTEA NOT NULL,
  ciphertext_nonce BYTEA NOT NULL CHECK (octet_length(ciphertext_nonce) = 12),
  kek_version TEXT NOT NULL CHECK (kek_version ~ '^[A-Za-z0-9][A-Za-z0-9-]{0,63}$'),
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, session_id),
  UNIQUE (namespace_id, delegated_inference_session_id),
  FOREIGN KEY (namespace_id, session_id)
    REFERENCES agent_sessions(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, delegated_inference_session_id)
    REFERENCES delegated_inference_sessions(namespace_id, id) ON DELETE RESTRICT,
  CHECK (expires_at > created_at)
);

CREATE TABLE agent_turns (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  session_id UUID NOT NULL,
  ordinal BIGINT NOT NULL CHECK (ordinal > 0),
  actor_principal_id UUID NOT NULL REFERENCES management_principals(id) ON DELETE RESTRICT,
  idempotency_hmac_version TEXT NOT NULL CHECK (idempotency_hmac_version ~ '^[A-Za-z0-9][A-Za-z0-9-]{0,63}$'),
  idempotency_key_digest BYTEA NOT NULL CHECK (octet_length(idempotency_key_digest) = 32),
  request_digest BYTEA NOT NULL CHECK (octet_length(request_digest) = 32),
  input JSONB NOT NULL CHECK (jsonb_typeof(input) = 'object'),
  status TEXT NOT NULL CHECK (status IN ('queued','running','waiting_approval','completed','failed','cancelled')),
  registry_revision TEXT CHECK (registry_revision IS NULL OR registry_revision ~ '^sha256:[a-f0-9]{64}$'),
  fence BIGINT NOT NULL DEFAULT 0 CHECK (fence >= 0),
  worker_id TEXT,
  lease_expires_at TIMESTAMPTZ,
  cancel_requested_at TIMESTAMPTZ,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  failure_code TEXT,
  failure_message TEXT,
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, session_id, ordinal),
  UNIQUE (namespace_id, session_id, id),
  UNIQUE (namespace_id, session_id, actor_principal_id, idempotency_hmac_version, idempotency_key_digest),
  FOREIGN KEY (namespace_id, session_id) REFERENCES agent_sessions(namespace_id, id) ON DELETE RESTRICT,
  CHECK ((worker_id IS NULL) = (lease_expires_at IS NULL)),
  CHECK ((status IN ('completed','failed','cancelled')) = (completed_at IS NOT NULL)),
  CHECK (status <> 'failed' OR failure_code IS NOT NULL)
);

CREATE TABLE agent_events (
  namespace_id UUID NOT NULL,
  session_id UUID NOT NULL,
  sequence BIGINT NOT NULL CHECK (sequence > 0),
  turn_id UUID NOT NULL,
  origin TEXT NOT NULL CHECK (origin IN ('control','worker')),
  fence BIGINT CHECK (fence IS NULL OR fence > 0),
  event_type TEXT NOT NULL CHECK (event_type IN (
    'user_input','assistant_delta','model_step_summary','tool_request','tool_result','progress',
    'context_checkpoint','approval_request','approval_result','cancellation','terminal'
  )),
  payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (session_id, sequence),
  FOREIGN KEY (namespace_id, session_id) REFERENCES agent_sessions(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, session_id, turn_id)
    REFERENCES agent_turns(namespace_id, session_id, id) ON DELETE RESTRICT,
  CHECK ((origin = 'worker') = (fence IS NOT NULL)),
  CHECK (origin <> 'worker' OR event_type IN (
    'assistant_delta','model_step_summary','tool_request','tool_result','progress',
    'context_checkpoint','approval_request','terminal'
  )),
  CHECK (origin <> 'control' OR event_type IN (
    'user_input','approval_result','cancellation','terminal','context_checkpoint'
  ))
);

-- A model step is staged before the delegated public inference request and is
-- committed together with its complete semantic output and context
-- checkpoint. An expired worker may replay a completed step, but a started
-- step from an older fence is an unknown outcome and is never invoked twice.
CREATE TABLE agent_model_steps (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  session_id UUID NOT NULL,
  turn_id UUID NOT NULL,
  ordinal BIGINT NOT NULL CHECK (ordinal > 0),
  fence BIGINT NOT NULL CHECK (fence > 0),
  registry_revision TEXT NOT NULL CHECK (registry_revision ~ '^sha256:[a-f0-9]{64}$'),
  request_digest BYTEA NOT NULL CHECK (octet_length(request_digest) = 32),
  status TEXT NOT NULL CHECK (status IN ('started','completed','unknown')),
  stop_reason TEXT CHECK (stop_reason IS NULL OR stop_reason IN (
    'end_turn','max_tokens','stop_sequence','tool_call','content_filter'
  )),
  output_digest BYTEA CHECK (output_digest IS NULL OR octet_length(output_digest) = 32),
  started_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  completed_at TIMESTAMPTZ,
  UNIQUE (namespace_id, session_id, turn_id, ordinal),
  UNIQUE (namespace_id, session_id, turn_id, id),
  FOREIGN KEY (namespace_id, session_id) REFERENCES agent_sessions(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, session_id, turn_id)
    REFERENCES agent_turns(namespace_id, session_id, id) ON DELETE RESTRICT,
  CHECK ((status = 'started') = (completed_at IS NULL)),
  CHECK ((status = 'completed') = (stop_reason IS NOT NULL AND output_digest IS NOT NULL))
);

CREATE TABLE agent_artifacts (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  session_id UUID NOT NULL,
  turn_id UUID NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('probe','evaluation','topology','diff','tool_result')),
  media_type TEXT NOT NULL,
  content BYTEA NOT NULL CHECK (octet_length(content) BETWEEN 1 AND 16777216),
  content_digest BYTEA NOT NULL CHECK (octet_length(content_digest) = 32),
  safe_preview JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(safe_preview) = 'object'),
  access_scope JSONB NOT NULL CHECK (jsonb_typeof(access_scope) = 'object'),
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, session_id, content_digest),
  FOREIGN KEY (namespace_id, session_id) REFERENCES agent_sessions(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, session_id, turn_id)
    REFERENCES agent_turns(namespace_id, session_id, id) ON DELETE RESTRICT,
  CHECK (expires_at > created_at)
);

CREATE TABLE agent_tool_invocations (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  session_id UUID NOT NULL,
  turn_id UUID NOT NULL,
  fence BIGINT NOT NULL CHECK (fence > 0),
  registry_revision TEXT NOT NULL CHECK (registry_revision ~ '^sha256:[a-f0-9]{64}$'),
  tool_name TEXT NOT NULL CHECK (tool_name ~ '^[a-z][a-z0-9_.-]{2,127}$'),
  credential_version_id UUID,
  input_digest BYTEA NOT NULL CHECK (octet_length(input_digest) = 32),
  input JSONB NOT NULL CHECK (jsonb_typeof(input) = 'object'),
  idempotency TEXT NOT NULL CHECK (idempotency IN ('none','invocation')),
  classification TEXT NOT NULL CHECK (classification IN ('read','write','execute')),
  status TEXT NOT NULL CHECK (status IN ('started','completed','failed','unknown')),
  result JSONB,
  artifact_id UUID,
  error_code TEXT,
  started_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  completed_at TIMESTAMPTZ,
  UNIQUE (namespace_id, turn_id, id),
  FOREIGN KEY (namespace_id, session_id) REFERENCES agent_sessions(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, session_id, turn_id)
    REFERENCES agent_turns(namespace_id, session_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, artifact_id) REFERENCES agent_artifacts(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, credential_version_id)
    REFERENCES agent_tool_credential_versions(namespace_id, id) ON DELETE RESTRICT,
  CHECK ((status = 'started') = (completed_at IS NULL)),
  CHECK (num_nonnulls(result, artifact_id) <= 1),
  CHECK (status <> 'failed' OR error_code IS NOT NULL)
);

CREATE TABLE agent_context_checkpoints (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  session_id UUID NOT NULL,
  turn_id UUID NOT NULL,
  through_sequence BIGINT NOT NULL CHECK (through_sequence > 0),
  format_version INTEGER NOT NULL DEFAULT 1 CHECK (format_version > 0),
  summary TEXT NOT NULL CHECK (octet_length(summary) BETWEEN 1 AND 1048576),
  unresolved_goals JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(unresolved_goals) = 'array'),
  resource_references JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(resource_references) = 'array'),
  tool_result_references JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(tool_result_references) = 'array'),
  decisions JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(decisions) = 'array'),
  state JSONB NOT NULL CHECK (
    jsonb_typeof(state) = 'object' AND octet_length(state::text) BETWEEN 2 AND 1048576
  ),
  content_digest BYTEA NOT NULL CHECK (octet_length(content_digest) = 32),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, session_id, through_sequence),
  FOREIGN KEY (namespace_id, session_id) REFERENCES agent_sessions(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, session_id, turn_id)
    REFERENCES agent_turns(namespace_id, session_id, id) ON DELETE RESTRICT
);

CREATE TABLE agent_publication_plans (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL,
  session_id UUID NOT NULL,
  turn_id UUID NOT NULL,
  recipe_id TEXT NOT NULL,
  recipe_content_revision BIGINT NOT NULL CHECK (recipe_content_revision > 0),
  recipe_resource_revision BIGINT NOT NULL CHECK (recipe_resource_revision > 0),
  entrypoint_id TEXT NOT NULL,
  entrypoint_content_revision BIGINT NOT NULL CHECK (entrypoint_content_revision > 0),
  entrypoint_resource_revision BIGINT NOT NULL CHECK (entrypoint_resource_revision > 0),
  catalog_revision TEXT NOT NULL CHECK (catalog_revision ~ '^sha256:[a-f0-9]{64}$'),
  exact_diff JSONB NOT NULL CHECK (jsonb_typeof(exact_diff) = 'object'),
  diagnostics JSONB NOT NULL CHECK (jsonb_typeof(diagnostics) = 'array'),
  gate_results JSONB NOT NULL CHECK (jsonb_typeof(gate_results) = 'array'),
  plan_digest BYTEA NOT NULL CHECK (octet_length(plan_digest) = 32),
  status TEXT NOT NULL CHECK (status IN ('ready','publishing','committed','expired','invalidated','failed')),
  expires_at TIMESTAMPTZ NOT NULL,
  committed_by UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  committed_operation_id UUID,
  committed_at TIMESTAMPTZ,
  failure_code TEXT,
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, plan_digest),
  FOREIGN KEY (namespace_id, session_id) REFERENCES agent_sessions(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, session_id, turn_id)
    REFERENCES agent_turns(namespace_id, session_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, recipe_id) REFERENCES routing_recipes(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (recipe_id, recipe_content_revision) REFERENCES routing_recipe_revisions(recipe_id, revision) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, entrypoint_id) REFERENCES routing_entrypoints(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (entrypoint_id, entrypoint_content_revision) REFERENCES routing_entrypoint_revisions(entrypoint_id, revision) ON DELETE RESTRICT,
  CHECK (expires_at > created_at),
  CHECK ((status = 'committed') = (committed_at IS NOT NULL)),
  CHECK ((status IN ('publishing','committed')) = (committed_by IS NOT NULL)),
  CHECK ((status = 'committed') = (committed_operation_id IS NOT NULL)),
  CHECK (status <> 'failed' OR failure_code IS NOT NULL)
);

CREATE UNIQUE INDEX agent_publication_plans_turn_idx
  ON agent_publication_plans(namespace_id, session_id, turn_id);

CREATE INDEX agent_profiles_page_idx
  ON agent_profiles(namespace_id, created_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_profiles_name_search_idx
  ON agent_profiles(namespace_id, lower(name) text_pattern_ops, created_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_profiles_description_search_idx
  ON agent_profiles(namespace_id, lower(description) text_pattern_ops, created_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_skills_page_idx
  ON agent_skills(namespace_id, created_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_skills_name_search_idx
  ON agent_skills(lower(name) text_pattern_ops, created_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_skills_description_search_idx
  ON agent_skills(lower(description) text_pattern_ops, created_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_tool_credentials_page_idx
  ON agent_tool_credentials(namespace_id, created_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_tool_credentials_name_search_idx
  ON agent_tool_credentials(namespace_id, lower(name) text_pattern_ops, created_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_tool_sources_page_idx
  ON agent_tool_sources(namespace_id, created_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_tool_sources_name_search_idx
  ON agent_tool_sources(namespace_id, lower(name) text_pattern_ops, created_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_tool_sources_description_search_idx
  ON agent_tool_sources(namespace_id, lower(description) text_pattern_ops, created_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_tool_registry_retention_idx
  ON agent_tool_registry_revisions(expires_at, namespace_id, registry_revision);
CREATE INDEX agent_sessions_owner_page_idx
  ON agent_sessions(namespace_id, owner_principal_id, updated_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_sessions_page_idx
  ON agent_sessions(namespace_id, updated_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_sessions_title_search_idx
  ON agent_sessions(namespace_id, lower(title) text_pattern_ops, updated_at DESC, id DESC)
  WHERE status <> 'deleted';
CREATE INDEX agent_turns_queued_claim_idx
  ON agent_turns(created_at, id) WHERE status = 'queued';
CREATE INDEX agent_turns_expired_claim_idx
  ON agent_turns(lease_expires_at, created_at, id) WHERE status = 'running';
CREATE INDEX agent_turns_session_page_idx
  ON agent_turns(namespace_id, session_id, ordinal DESC, id);
CREATE UNIQUE INDEX agent_turns_one_nonterminal_per_session_idx
  ON agent_turns(namespace_id, session_id)
  WHERE status IN ('queued','running','waiting_approval');
CREATE INDEX agent_events_resume_idx
  ON agent_events(namespace_id, session_id, sequence);
CREATE INDEX agent_artifacts_retention_idx
  ON agent_artifacts(expires_at, namespace_id, id);
CREATE INDEX agent_tool_invocations_turn_idx
  ON agent_tool_invocations(namespace_id, turn_id, started_at, id);
CREATE INDEX agent_model_steps_turn_idx
  ON agent_model_steps(namespace_id, session_id, turn_id, ordinal);
CREATE INDEX agent_checkpoints_resume_idx
  ON agent_context_checkpoints(namespace_id, session_id, through_sequence DESC);
CREATE INDEX agent_publication_plans_ready_idx
  ON agent_publication_plans(namespace_id, expires_at, id) WHERE status = 'ready';

CREATE INDEX management_role_bindings_principal_idx ON management_role_bindings(principal_id, status, scope_kind, namespace_id);
CREATE INDEX management_principals_directory_name_idx
  ON management_principals (lower(display_name) text_pattern_ops, id);
CREATE INDEX management_principals_directory_email_idx
  ON management_principals (lower(verified_email) text_pattern_ops, id)
  WHERE verified_email IS NOT NULL;
CREATE INDEX management_sessions_principal_idx ON management_sessions(principal_id, status, expires_at DESC, id);
CREATE INDEX management_sessions_source_idx ON management_sessions(auth_source_kind, auth_source_id, status, id);
CREATE INDEX management_invitations_namespace_idx ON management_invitations(namespace_id, status, expires_at, id);
CREATE INDEX delegated_sessions_key_idx ON delegated_inference_sessions(namespace_id, api_key_id, status, expires_at, id);
CREATE INDEX provider_credentials_namespace_idx ON provider_credentials(namespace_id, status, id);
CREATE INDEX routing_models_page_idx ON routing_models(namespace_id, status, created_at DESC, id);
CREATE INDEX routing_recipes_page_idx ON routing_recipes(namespace_id, status, created_at DESC, id);
CREATE INDEX routing_entrypoints_page_idx ON routing_entrypoints(namespace_id, status, created_at DESC, id);

-- Publication, usage ledger, and audit

CREATE TABLE policy_revisions (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  revision BIGINT NOT NULL CHECK (revision > 0),
  runtime_epoch BIGINT NOT NULL CHECK (runtime_epoch > 0),
  reason TEXT NOT NULL,
  actor_principal_id UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, revision)
);

CREATE TABLE policy_outbox (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  desired_revision BIGINT NOT NULL CHECK (desired_revision > 0),
  aggregate_type TEXT NOT NULL,
  aggregate_id TEXT NOT NULL CHECK (
    aggregate_id ~ '^(?:[a-z][a-z0-9_-]{2,127}|[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$'
  ),
  operation TEXT NOT NULL,
  payload JSONB NOT NULL,
  state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending','processing','applied','failed')),
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
  available_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  locked_by TEXT,
  locked_at TIMESTAMPTZ,
  applied_at TIMESTAMPTZ,
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, desired_revision, aggregate_type, aggregate_id)
);

-- PostgreSQL notifications are only a low-latency wake-up. Replicas still
-- reconcile from durable publication state on a bounded poll interval, so a
-- disconnect or lost notification cannot lose a routing revision. Because
-- PostgreSQL delivers NOTIFY only after commit, this wake-up is atomic with
-- the desired revision and outbox row that caused it.
CREATE FUNCTION notify_routing_desired_state() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  PERFORM pg_notify('vllm_sr_routing_publication', NEW.namespace_id::text);
  RETURN NEW;
END;
$$;

CREATE TRIGGER policy_outbox_notify_routing_desired_state
AFTER INSERT ON policy_outbox
FOR EACH ROW EXECUTE FUNCTION notify_routing_desired_state();

CREATE TABLE projector_watermarks (
  projector TEXT NOT NULL,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  desired_revision BIGINT NOT NULL DEFAULT 0 CHECK (desired_revision >= 0),
  applied_revision BIGINT NOT NULL DEFAULT 0 CHECK (applied_revision >= 0),
  runtime_epoch BIGINT NOT NULL CHECK (runtime_epoch > 0),
  last_error TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (projector, namespace_id),
  CHECK (applied_revision <= desired_revision)
);

CREATE TABLE management_operations (
  id UUID PRIMARY KEY,
  namespace_id UUID REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  kind TEXT NOT NULL,
  origin_principal_id UUID NOT NULL REFERENCES management_principals(id) ON DELETE RESTRICT,
  actor_chain JSONB NOT NULL DEFAULT '[]'::jsonb,
  request_digest BYTEA NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('pending','running','succeeded','partially_succeeded','failed','cancelled')),
  progress_completed BIGINT NOT NULL DEFAULT 0 CHECK (progress_completed >= 0),
  progress_total BIGINT NOT NULL DEFAULT 0 CHECK (progress_total >= 0),
  target_scope JSONB NOT NULL,
  target_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
  desired_revision BIGINT,
  publication_revision BIGINT,
  applied_revision BIGINT,
  item_errors JSONB NOT NULL DEFAULT '[]'::jsonb,
  cancelled_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK (progress_completed <= progress_total)
);

ALTER TABLE agent_publication_plans
  ADD CONSTRAINT agent_publication_plans_operation_fk
  FOREIGN KEY (committed_operation_id) REFERENCES management_operations(id) ON DELETE RESTRICT;

CREATE TABLE management_idempotency (
  scope_kind TEXT NOT NULL CHECK (scope_kind IN ('cluster','namespace')),
  namespace_id UUID REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  principal_id UUID NOT NULL REFERENCES management_principals(id) ON DELETE RESTRICT,
  endpoint TEXT NOT NULL CHECK (
    char_length(endpoint) BETWEEN 2 AND 512
    AND endpoint ~ '^/[!-~]+$'
    AND position('//' IN endpoint) = 0
  ),
  hmac_version TEXT NOT NULL CHECK (hmac_version ~ '^[A-Za-z0-9][A-Za-z0-9-]{0,63}$'),
  idempotency_key_digest BYTEA NOT NULL CHECK (octet_length(idempotency_key_digest) = 32),
  request_digest BYTEA NOT NULL CHECK (octet_length(request_digest) = 32),
  operation_id UUID REFERENCES management_operations(id) ON DELETE RESTRICT,
  resource_type TEXT CHECK (resource_type IS NULL OR resource_type ~ '^[a-z][a-z0-9._-]{0,127}$'),
  resource_id TEXT CHECK (
    resource_id IS NULL
    OR (char_length(resource_id) BETWEEN 1 AND 512 AND resource_id ~ '^[!-~]+$')
  ),
  resource_revision BIGINT CHECK (resource_revision > 0),
  desired_revision BIGINT CHECK (desired_revision > 0),
  response_status INTEGER NOT NULL CHECK (response_status BETWEEN 200 AND 299),
  secret_response_ciphertext BYTEA,
  secret_response_nonce BYTEA,
  response_kek_version TEXT CHECK (
    response_kek_version IS NULL
    OR response_kek_version ~ '^[A-Za-z0-9][A-Za-z0-9-]{0,63}$'
  ),
  secret_response_expires_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK (
    (scope_kind = 'cluster' AND namespace_id IS NULL)
    OR (scope_kind = 'namespace' AND namespace_id IS NOT NULL)
  ),
  CHECK (expires_at > created_at),
  CHECK (
    num_nonnulls(secret_response_ciphertext, secret_response_nonce,
                 response_kek_version, secret_response_expires_at) IN (0, 4)
  ),
  CHECK (
    secret_response_expires_at IS NULL
    OR (secret_response_expires_at > created_at AND secret_response_expires_at <= expires_at)
  ),
  CHECK (
    (operation_id IS NOT NULL
      AND resource_type IS NULL AND resource_id IS NULL
      AND resource_revision IS NULL)
    OR
    (operation_id IS NULL
      AND resource_type IS NOT NULL AND resource_id IS NOT NULL
      AND resource_revision IS NOT NULL AND desired_revision IS NULL)
  ),
  CHECK (secret_response_ciphertext IS NULL OR operation_id IS NULL)
);

CREATE UNIQUE INDEX management_idempotency_cluster_identity_uq
  ON management_idempotency
    (principal_id, endpoint, hmac_version, idempotency_key_digest)
  WHERE scope_kind = 'cluster';

CREATE UNIQUE INDEX management_idempotency_namespace_identity_uq
  ON management_idempotency
    (namespace_id, principal_id, endpoint, hmac_version, idempotency_key_digest)
  WHERE scope_kind = 'namespace';

CREATE INDEX management_idempotency_expiry_hmac_idx
  ON management_idempotency (expires_at, hmac_version);

CREATE TABLE usage_settlements (
  namespace_id UUID NOT NULL,
  admission_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('unknown','settled','waived')),
  canonical_usage_digest BYTEA,
  reconciliation_id UUID,
  revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0),
  settled_at TIMESTAMPTZ,
  event_partition_date DATE NOT NULL,
  event_retained BOOLEAN NOT NULL DEFAULT TRUE,
  raw_retired_at TIMESTAMPTZ,
  PRIMARY KEY (namespace_id, admission_id),
  FOREIGN KEY (namespace_id) REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  CHECK ((event_retained AND raw_retired_at IS NULL)
    OR (NOT event_retained AND raw_retired_at IS NOT NULL))
) PARTITION BY HASH (namespace_id, admission_id);

CREATE TABLE usage_settlements_p0 PARTITION OF usage_settlements FOR VALUES WITH (MODULUS 8, REMAINDER 0);
CREATE TABLE usage_settlements_p1 PARTITION OF usage_settlements FOR VALUES WITH (MODULUS 8, REMAINDER 1);
CREATE TABLE usage_settlements_p2 PARTITION OF usage_settlements FOR VALUES WITH (MODULUS 8, REMAINDER 2);
CREATE TABLE usage_settlements_p3 PARTITION OF usage_settlements FOR VALUES WITH (MODULUS 8, REMAINDER 3);
CREATE TABLE usage_settlements_p4 PARTITION OF usage_settlements FOR VALUES WITH (MODULUS 8, REMAINDER 4);
CREATE TABLE usage_settlements_p5 PARTITION OF usage_settlements FOR VALUES WITH (MODULUS 8, REMAINDER 5);
CREATE TABLE usage_settlements_p6 PARTITION OF usage_settlements FOR VALUES WITH (MODULUS 8, REMAINDER 6);
CREATE TABLE usage_settlements_p7 PARTITION OF usage_settlements FOR VALUES WITH (MODULUS 8, REMAINDER 7);

CREATE TABLE usage_events (
  namespace_id UUID NOT NULL,
  admission_id TEXT NOT NULL,
  event_date DATE NOT NULL,
  event_id UUID NOT NULL,
  event_kind TEXT NOT NULL CHECK (event_kind IN ('actual','unknown','correction','waiver')),
  external_request_id TEXT,
  protocol TEXT NOT NULL,
  path TEXT NOT NULL,
  api_key_id UUID,
  credential_id UUID,
  user_id UUID,
  team_id UUID,
  entrypoint_id TEXT CHECK (entrypoint_id IS NULL OR entrypoint_id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  entrypoint_rule_id TEXT CHECK (entrypoint_rule_id IS NULL OR entrypoint_rule_id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  recipe_id TEXT CHECK (recipe_id IS NULL OR recipe_id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  routing_revision BIGINT,
  status_code INTEGER NOT NULL,
  error_code TEXT,
  input_tokens NUMERIC(42,0) NOT NULL DEFAULT 0,
  output_tokens NUMERIC(42,0) NOT NULL DEFAULT 0,
  total_tokens NUMERIC(42,0) NOT NULL DEFAULT 0,
  served_input_tokens NUMERIC(42,0) NOT NULL DEFAULT 0,
  served_output_tokens NUMERIC(42,0) NOT NULL DEFAULT 0,
  served_total_tokens NUMERIC(42,0) NOT NULL DEFAULT 0,
  latency_ms BIGINT NOT NULL DEFAULT 0,
  ttft_ms BIGINT,
  usage_state TEXT NOT NULL CHECK (usage_state IN ('known_zero','known_actual','unknown')),
  costs JSONB NOT NULL DEFAULT '[]'::jsonb,
  request_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  payload_ciphertext BYTEA,
  payload_nonce BYTEA,
  payload_kek_version TEXT,
  occurred_at TIMESTAMPTZ NOT NULL,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, event_date, event_id),
  FOREIGN KEY (namespace_id, admission_id) REFERENCES usage_settlements(namespace_id, admission_id) ON DELETE RESTRICT,
  CHECK (total_tokens = input_tokens + output_tokens),
  CHECK (served_total_tokens = served_input_tokens + served_output_tokens),
  CHECK ((payload_ciphertext IS NULL) = (payload_nonce IS NULL)),
  CHECK ((payload_ciphertext IS NULL) = (payload_kek_version IS NULL))
) PARTITION BY RANGE (event_date);

CREATE TABLE usage_dispatches (
  namespace_id UUID NOT NULL,
  event_date DATE NOT NULL,
  event_id UUID NOT NULL,
  admission_id TEXT NOT NULL,
  dispatch_id TEXT NOT NULL,
  parent_dispatch_id TEXT,
  dispatch_ordinal INTEGER NOT NULL CHECK (dispatch_ordinal >= 0),
  attempt_count INTEGER NOT NULL CHECK (attempt_count > 0),
  dispatch_type TEXT NOT NULL,
  logical_model_id TEXT CHECK (
    logical_model_id IS NULL OR (
      char_length(logical_model_id) BETWEEN 1 AND 256
      AND logical_model_id ~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$'
    )
  ),
  model_revision BIGINT,
  backend_id UUID,
  provider_id TEXT,
  provider_model_id TEXT,
  pricing_revision BIGINT,
  input_tokens NUMERIC(42,0) NOT NULL DEFAULT 0,
  cache_read_tokens NUMERIC(42,0) NOT NULL DEFAULT 0,
  cache_write_tokens NUMERIC(42,0) NOT NULL DEFAULT 0,
  output_tokens NUMERIC(42,0) NOT NULL DEFAULT 0,
  usage_state TEXT NOT NULL CHECK (usage_state IN ('known_zero','known_actual','unknown')),
  cost_numerator NUMERIC(42,0),
  currency TEXT CHECK (currency IS NULL OR currency ~ '^[A-Z]{3}$'),
  evidence_digest BYTEA,
  started_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ,
  PRIMARY KEY (namespace_id, event_date, event_id, dispatch_id),
  FOREIGN KEY (namespace_id, event_date, event_id) REFERENCES usage_events(namespace_id, event_date, event_id) ON DELETE RESTRICT,
  UNIQUE (namespace_id, event_date, admission_id, dispatch_id),
  UNIQUE (namespace_id, event_date, event_id, dispatch_id, admission_id)
) PARTITION BY RANGE (event_date);

CREATE TABLE usage_dispatch_attempts (
  namespace_id UUID NOT NULL,
  event_date DATE NOT NULL,
  event_id UUID NOT NULL,
  dispatch_id TEXT NOT NULL,
  admission_id TEXT NOT NULL,
  attempt_id TEXT NOT NULL,
  attempt_ordinal INTEGER NOT NULL CHECK (attempt_ordinal >= 0),
  backend_id UUID,
  provider_id TEXT,
  state TEXT NOT NULL CHECK (state IN ('known_zero','known_actual','unknown')),
  status_code INTEGER CHECK (status_code IS NULL OR status_code BETWEEN 100 AND 599),
  error_code TEXT,
  started_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (namespace_id, event_date, event_id, dispatch_id, attempt_id),
  UNIQUE (namespace_id, event_date, event_id, dispatch_id, attempt_ordinal),
  FOREIGN KEY (namespace_id, event_date, event_id, dispatch_id, admission_id)
    REFERENCES usage_dispatches(namespace_id, event_date, event_id, dispatch_id, admission_id) ON DELETE RESTRICT,
  CHECK (completed_at >= started_at)
) PARTITION BY RANGE (event_date);

CREATE TABLE unknown_usage_fences (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  admission_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  evidence JSONB NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('open','reconciling','resolved')),
  etag_revision BIGINT NOT NULL DEFAULT 1 CHECK (etag_revision > 0),
  reconciliation_id UUID,
  reconciliation_strategy TEXT CHECK (reconciliation_strategy IN ('actual','conservative_debit','waive')),
  reconciliation_actor_id UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  reconciliation_reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  resolved_at TIMESTAMPTZ,
  UNIQUE (namespace_id, admission_id),
  FOREIGN KEY (namespace_id, admission_id) REFERENCES usage_settlements(namespace_id, admission_id) ON DELETE RESTRICT
);

CREATE TABLE unknown_usage_fence_bindings (
  fence_id UUID NOT NULL REFERENCES unknown_usage_fences(id) ON DELETE RESTRICT,
  binding_id UUID NOT NULL REFERENCES rate_limit_bindings(id) ON DELETE RESTRICT,
  rule_id UUID NOT NULL REFERENCES rate_limit_rules(id) ON DELETE RESTRICT,
  admission_limit NUMERIC(42,0),
  maximum_debit NUMERIC(42,0),
  PRIMARY KEY (fence_id, binding_id, rule_id)
);

CREATE TABLE access_audit_events (
  id UUID PRIMARY KEY,
  namespace_id UUID REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  desired_revision BIGINT,
  chain_sequence BIGINT,
  actor_principal_id UUID REFERENCES management_principals(id) ON DELETE RESTRICT,
  actor_chain JSONB NOT NULL DEFAULT '[]'::jsonb,
  action TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  resource_id TEXT CHECK (
    resource_id IS NULL OR resource_id ~ '^(?:[a-z][a-z0-9_-]{2,127}|[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$'
  ),
  request_id TEXT NOT NULL,
  source_ip INET,
  outcome TEXT NOT NULL CHECK (outcome IN ('allowed','denied','failed')),
  reason TEXT NOT NULL,
  before_revision BIGINT,
  after_revision BIGINT,
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  previous_hash BYTEA,
  event_hash BYTEA NOT NULL CHECK (octet_length(event_hash) = 32),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, desired_revision),
  UNIQUE (namespace_id, chain_sequence),
  UNIQUE (namespace_id, id),
  FOREIGN KEY (namespace_id, desired_revision)
    REFERENCES policy_revisions(namespace_id, revision) ON DELETE RESTRICT,
  CHECK ((namespace_id IS NULL) = (chain_sequence IS NULL)),
  CHECK (desired_revision IS NULL OR namespace_id IS NOT NULL),
  CHECK (
    namespace_id IS NULL
    OR (chain_sequence = 1 AND previous_hash IS NULL)
    OR (chain_sequence > 1 AND octet_length(previous_hash) = 32)
  )
);

CREATE TABLE access_audit_heads (
  namespace_id UUID PRIMARY KEY REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  last_event_id UUID,
  last_hash BYTEA CHECK (last_hash IS NULL OR octet_length(last_hash) = 32),
  event_count BIGINT NOT NULL DEFAULT 0 CHECK (event_count >= 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK (
    (event_count = 0 AND last_event_id IS NULL AND last_hash IS NULL)
    OR (event_count > 0 AND last_event_id IS NOT NULL AND octet_length(last_hash) = 32)
  ),
  FOREIGN KEY (namespace_id, last_event_id)
    REFERENCES access_audit_events(namespace_id, id) ON DELETE RESTRICT
    DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE usage_rollup_1m (
  namespace_id UUID NOT NULL,
  bucket_start TIMESTAMPTZ NOT NULL,
  view TEXT NOT NULL CHECK (view IN ('request','dispatch')),
  dimensions JSONB NOT NULL,
  dimensions_digest BYTEA NOT NULL,
  requests NUMERIC(42,0) NOT NULL,
  successful_requests NUMERIC(42,0) NOT NULL,
  input_tokens NUMERIC(42,0) NOT NULL,
  output_tokens NUMERIC(42,0) NOT NULL,
  costs JSONB NOT NULL DEFAULT '[]'::jsonb,
  incomplete_dispatches NUMERIC(42,0) NOT NULL DEFAULT 0,
  ledger_watermark TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, bucket_start, view, dimensions_digest),
  FOREIGN KEY (namespace_id) REFERENCES access_namespaces(id) ON DELETE RESTRICT
);

CREATE TABLE usage_rollup_1h (LIKE usage_rollup_1m INCLUDING ALL);
CREATE TABLE usage_rollup_1d (LIKE usage_rollup_1m INCLUDING ALL);

ALTER TABLE usage_rollup_1h
  ADD FOREIGN KEY (namespace_id) REFERENCES access_namespaces(id) ON DELETE RESTRICT;
ALTER TABLE usage_rollup_1d
  ADD FOREIGN KEY (namespace_id) REFERENCES access_namespaces(id) ON DELETE RESTRICT;

CREATE INDEX policy_outbox_pending_idx ON policy_outbox(state, available_at, created_at, id);
CREATE INDEX management_operations_actor_idx ON management_operations(origin_principal_id, created_at DESC, id);
CREATE INDEX management_operations_namespace_idx ON management_operations(namespace_id, created_at DESC, id);
CREATE INDEX usage_events_time_idx ON usage_events(namespace_id, occurred_at DESC, event_id);
CREATE INDEX usage_events_admission_idx ON usage_events(namespace_id, admission_id);
CREATE INDEX usage_events_external_request_idx
  ON usage_events(namespace_id, external_request_id, occurred_at DESC, event_id)
  WHERE external_request_id IS NOT NULL;
CREATE INDEX usage_events_api_key_idx ON usage_events(namespace_id, api_key_id, occurred_at DESC, event_id);
CREATE INDEX usage_events_user_idx ON usage_events(namespace_id, user_id, occurred_at DESC, event_id);
CREATE INDEX usage_events_team_idx ON usage_events(namespace_id, team_id, occurred_at DESC, event_id);
CREATE INDEX usage_events_entrypoint_idx ON usage_events(namespace_id, entrypoint_id, occurred_at DESC, event_id);
CREATE INDEX usage_events_recipe_idx ON usage_events(namespace_id, recipe_id, occurred_at DESC, event_id);
CREATE INDEX usage_events_status_idx ON usage_events(namespace_id, status_code, occurred_at DESC, event_id);
CREATE INDEX usage_dispatches_model_idx ON usage_dispatches(namespace_id, logical_model_id, started_at DESC, dispatch_id);
CREATE INDEX usage_dispatches_backend_idx ON usage_dispatches(namespace_id, backend_id, started_at DESC, dispatch_id);
CREATE INDEX usage_dispatches_provider_idx ON usage_dispatches(namespace_id, provider_id, started_at DESC, dispatch_id);
CREATE INDEX usage_dispatch_attempts_admission_idx ON usage_dispatch_attempts(namespace_id, admission_id, dispatch_id, attempt_ordinal);
CREATE INDEX usage_dispatch_attempts_timeline_idx ON usage_dispatch_attempts(namespace_id, started_at DESC, attempt_id);
CREATE INDEX usage_dispatch_attempts_state_idx ON usage_dispatch_attempts(namespace_id, state, started_at DESC, attempt_id);
CREATE INDEX usage_rollup_1m_view_time_idx ON usage_rollup_1m(namespace_id, view, bucket_start);
CREATE INDEX usage_rollup_1h_view_time_idx ON usage_rollup_1h(namespace_id, view, bucket_start);
CREATE INDEX usage_rollup_1d_view_time_idx ON usage_rollup_1d(namespace_id, view, bucket_start);
CREATE INDEX unknown_fences_state_idx ON unknown_usage_fences(namespace_id, state, created_at, id);
CREATE INDEX audit_namespace_idx ON access_audit_events(namespace_id, created_at DESC, id);
CREATE INDEX audit_actor_idx ON access_audit_events(actor_principal_id, created_at DESC, id);
CREATE INDEX audit_resource_idx ON access_audit_events(namespace_id, resource_type, resource_id, created_at DESC, id);
CREATE INDEX audit_action_idx ON access_audit_events(namespace_id, action, outcome, created_at DESC, id);

-- Provider catalog

CREATE TABLE provider_catalog_revisions (
  revision TEXT PRIMARY KEY CHECK (revision ~ '^sha256:[a-f0-9]{64}$'),
  snapshot_bytes BYTEA NOT NULL CHECK (octet_length(snapshot_bytes) BETWEEN 1 AND 67108864),
  snapshot_digest BYTEA NOT NULL CHECK (octet_length(snapshot_digest) = 32),
  integration_references JSONB NOT NULL CHECK (jsonb_typeof(integration_references) = 'array'),
  catalog JSONB NOT NULL CHECK (jsonb_typeof(catalog) = 'object'),
  required_wire_formats JSONB NOT NULL CHECK (jsonb_typeof(required_wire_formats) = 'array'),
  required_credential_adapters JSONB NOT NULL CHECK (jsonb_typeof(required_credential_adapters) = 'array'),
  required_discovery_adapters JSONB NOT NULL CHECK (jsonb_typeof(required_discovery_adapters) = 'array'),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE provider_catalog_state (
  singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
  desired_revision TEXT REFERENCES provider_catalog_revisions(revision) ON DELETE RESTRICT,
  active_revision TEXT REFERENCES provider_catalog_revisions(revision) ON DELETE RESTRICT,
  generation BIGINT NOT NULL DEFAULT 1 CHECK (generation > 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK (desired_revision IS NOT NULL OR active_revision IS NULL)
);

INSERT INTO provider_catalog_state(singleton, generation) VALUES (TRUE, 1);

CREATE TABLE provider_catalog_required_rollout_groups (
  generation BIGINT NOT NULL CHECK (generation > 1),
  revision TEXT NOT NULL REFERENCES provider_catalog_revisions(revision) ON DELETE RESTRICT,
  plane TEXT NOT NULL CHECK (plane IN ('control','data')),
  rollout_group TEXT NOT NULL CHECK (rollout_group ~ '^[a-z][a-z0-9._-]{0,127}$'),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (generation, plane, rollout_group)
);

CREATE INDEX provider_catalog_required_rollout_groups_revision_idx
  ON provider_catalog_required_rollout_groups(revision, generation, plane, rollout_group);

CREATE TABLE provider_catalog_replica_acks (
  revision TEXT NOT NULL REFERENCES provider_catalog_revisions(revision) ON DELETE CASCADE,
  plane TEXT NOT NULL CHECK (plane IN ('control','data')),
  rollout_group TEXT NOT NULL CHECK (rollout_group ~ '^[a-z][a-z0-9._-]{0,127}$'),
  replica_id TEXT NOT NULL CHECK (
    replica_id = btrim(replica_id)
    AND char_length(replica_id) BETWEEN 1 AND 256
  ),
  capability_digest BYTEA NOT NULL CHECK (octet_length(capability_digest) = 32),
  status TEXT NOT NULL CHECK (status IN ('compatible','incompatible')),
  reason TEXT NOT NULL DEFAULT '' CHECK (char_length(reason) <= 1024),
  acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  lease_expires_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (revision, plane, rollout_group, replica_id),
  CHECK (lease_expires_at > acknowledged_at)
);

CREATE INDEX provider_catalog_replica_acks_lease_idx
  ON provider_catalog_replica_acks(revision, plane, rollout_group, status, lease_expires_at);

CREATE FUNCTION reject_provider_catalog_revision_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  RAISE EXCEPTION 'provider catalog revisions are immutable' USING ERRCODE = '55000';
END;
$$;

CREATE TRIGGER provider_catalog_revisions_immutable
BEFORE UPDATE OR DELETE ON provider_catalog_revisions
FOR EACH ROW EXECUTE FUNCTION reject_provider_catalog_revision_mutation();

ALTER TABLE provider_credentials
  ADD CONSTRAINT provider_credentials_catalog_revision_fk
  FOREIGN KEY (provider_catalog_revision)
  REFERENCES provider_catalog_revisions(revision)
  ON DELETE RESTRICT;

ALTER TABLE routing_model_revisions
  ADD CONSTRAINT routing_model_revisions_catalog_revision_fk
  FOREIGN KEY (provider_catalog_revision)
  REFERENCES provider_catalog_revisions(revision)
  ON DELETE RESTRICT;

-- Policy bulk operations

-- Policy bulk jobs deliberately persist typed, secret-free work items instead
-- of opaque task payloads.  A worker can therefore validate, authorize, and
-- execute every item through the ordinary policy-management domain path.
ALTER TABLE management_operations
  ADD COLUMN version BIGINT NOT NULL DEFAULT 1 CHECK (version > 0),
  ADD COLUMN completed_at TIMESTAMPTZ,
  ADD CONSTRAINT management_operations_id_namespace_uq UNIQUE (id, namespace_id);

CREATE FUNCTION management_operation_increment_version() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.version := OLD.version + 1;
  RETURN NEW;
END
$$;

CREATE TRIGGER management_operation_version
  BEFORE UPDATE ON management_operations
  FOR EACH ROW EXECUTE FUNCTION management_operation_increment_version();

CREATE TABLE policy_bulk_operation_contexts (
  operation_id UUID PRIMARY KEY REFERENCES management_operations(id) ON DELETE RESTRICT,
  request_id TEXT NOT NULL CHECK (char_length(request_id) BETWEEN 1 AND 256),
  source_ip INET,
  expires_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE policy_bulk_operation_items (
  operation_id UUID NOT NULL,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  item_id UUID NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
  item_kind TEXT NOT NULL CHECK (item_kind IN ('access_policy_binding','rate_limit_binding')),
  access_policy_id UUID,
  rate_policy_id UUID,
  inline_policy_name TEXT,
  inline_policy_description TEXT,
  subject_id UUID NOT NULL,
  subject_kind TEXT NOT NULL CHECK (subject_kind IN ('user','team','api_key')),
  binding_mode TEXT CHECK (binding_mode IN ('allocation','hard_cap')),
  state TEXT NOT NULL DEFAULT 'pending'
    CHECK (state IN ('pending','running','succeeded','failed','cancelled')),
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
  available_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  lease_owner TEXT,
  lease_token UUID,
  lease_expires_at TIMESTAMPTZ,
  result_binding_id UUID,
  result_policy_id UUID,
  error_code TEXT,
  error_reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  finished_at TIMESTAMPTZ,
  PRIMARY KEY (operation_id, item_id),
  UNIQUE (operation_id, ordinal),
  FOREIGN KEY (operation_id, namespace_id)
    REFERENCES management_operations(id, namespace_id) ON DELETE RESTRICT,
  CHECK (
    (item_kind = 'access_policy_binding'
      AND access_policy_id IS NOT NULL AND rate_policy_id IS NULL
      AND inline_policy_name IS NULL AND inline_policy_description IS NULL
      AND binding_mode IS NULL)
    OR
    (item_kind = 'rate_limit_binding'
      AND access_policy_id IS NULL AND binding_mode IS NOT NULL
      AND ((rate_policy_id IS NOT NULL AND inline_policy_name IS NULL
            AND inline_policy_description IS NULL)
        OR (rate_policy_id IS NULL AND inline_policy_name IS NOT NULL
            AND inline_policy_description IS NOT NULL)))
  ),
  CHECK (
    (state = 'running' AND lease_owner IS NOT NULL AND lease_token IS NOT NULL
      AND lease_expires_at IS NOT NULL AND finished_at IS NULL)
    OR
    (state <> 'running' AND lease_owner IS NULL AND lease_token IS NULL
      AND lease_expires_at IS NULL)
  ),
  CHECK ((state IN ('succeeded','failed','cancelled')) = (finished_at IS NOT NULL)),
  CHECK ((state = 'failed') = (error_code IS NOT NULL AND error_reason IS NOT NULL))
);

CREATE TABLE policy_bulk_rate_rules (
  operation_id UUID NOT NULL,
  item_id UUID NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
  rule_id UUID,
  metric TEXT NOT NULL,
  algorithm TEXT NOT NULL,
  limit_value TEXT CHECK (limit_value IS NULL OR limit_value ~ '^(0|[1-9][0-9]*)(\.[0-9]{1,15})?$'),
  window_nanoseconds BIGINT,
  calendar_period TEXT,
  timezone TEXT,
  bucket_capacity TEXT CHECK (bucket_capacity IS NULL OR bucket_capacity ~ '^[1-9][0-9]{0,41}$'),
  refill_amount TEXT CHECK (refill_amount IS NULL OR refill_amount ~ '^[1-9][0-9]{0,41}$'),
  refill_period_nanoseconds BIGINT,
  gcra_emission_interval_nanoseconds BIGINT,
  gcra_burst_tolerance BIGINT,
  accounting TEXT NOT NULL,
  enforcement TEXT NOT NULL,
  PRIMARY KEY (operation_id, item_id, ordinal),
  FOREIGN KEY (operation_id, item_id)
    REFERENCES policy_bulk_operation_items(operation_id, item_id) ON DELETE RESTRICT
);

CREATE INDEX policy_bulk_items_claim_idx
  ON policy_bulk_operation_items(state, available_at, lease_expires_at, operation_id, ordinal);
CREATE INDEX policy_bulk_items_operation_state_idx
  ON policy_bulk_operation_items(operation_id, state, ordinal);
CREATE INDEX policy_bulk_context_expiry_idx
  ON policy_bulk_operation_contexts(expires_at, operation_id);
CREATE INDEX policy_bulk_operations_page_idx
  ON management_operations(namespace_id, created_at DESC, id ASC)
  WHERE kind IN ('access_policy_bindings.bulk_apply','rate_limit_bindings.bulk_apply');
CREATE INDEX policy_bulk_operations_kind_page_idx
  ON management_operations(namespace_id, kind, created_at DESC, id ASC)
  WHERE kind IN ('access_policy_bindings.bulk_apply','rate_limit_bindings.bulk_apply');
CREATE INDEX policy_bulk_operations_state_page_idx
  ON management_operations(namespace_id, state, created_at DESC, id ASC)
  WHERE kind IN ('access_policy_bindings.bulk_apply','rate_limit_bindings.bulk_apply');
CREATE INDEX policy_bulk_operations_origin_page_idx
  ON management_operations(namespace_id, origin_principal_id, created_at DESC, id ASC)
  WHERE kind IN ('access_policy_bindings.bulk_apply','rate_limit_bindings.bulk_apply');

CREATE INDEX delegated_sessions_self_idx
  ON delegated_inference_sessions(namespace_id, principal_id, created_at DESC, id);

CREATE INDEX delegated_sessions_active_user_idx
  ON delegated_inference_sessions(namespace_id, user_id, status, expires_at);

-- Back-channel logout replay protection

CREATE TABLE management_backchannel_logout_replays (
  issuer_id UUID NOT NULL,
  token_id_digest BYTEA NOT NULL CHECK (octet_length(token_id_digest) = 32),
  claims_digest BYTEA NOT NULL CHECK (octet_length(claims_digest) = 32),
  expires_at TIMESTAMPTZ NOT NULL,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (issuer_id, token_id_digest)
);

CREATE INDEX management_backchannel_logout_replays_expiry_idx
  ON management_backchannel_logout_replays(expires_at);

-- Unknown-usage reconciliation

ALTER TABLE unknown_usage_fence_bindings
  ADD COLUMN metric TEXT NOT NULL,
  ADD COLUMN algorithm TEXT NOT NULL,
  ADD COLUMN enforcement TEXT NOT NULL,
  ADD COLUMN window_seconds BIGINT,
  ADD COLUMN calendar_period TEXT,
  ADD COLUMN timezone TEXT,
  ADD COLUMN currency TEXT,
  ADD COLUMN unknown_dispatch_count NUMERIC(42,0) NOT NULL,
  ADD COLUMN counter_incomplete_count NUMERIC(42,0) NOT NULL,
  ADD CONSTRAINT unknown_fence_binding_metric_ck CHECK (
    metric IN ('input_tokens','output_tokens','total_tokens','served_input_tokens',
               'served_output_tokens','served_total_tokens','cost')
  ),
  ADD CONSTRAINT unknown_fence_binding_algorithm_ck CHECK (
    (algorithm = 'sliding_log' AND window_seconds > 0
      AND calendar_period IS NULL AND timezone IS NULL)
    OR
    (algorithm = 'calendar_window' AND window_seconds IS NULL
      AND calendar_period IN ('day','month') AND timezone IS NOT NULL)
  ),
  ADD CONSTRAINT unknown_fence_binding_enforcement_ck CHECK (
    enforcement IN ('enforce','shadow')
  ),
  ADD CONSTRAINT unknown_fence_binding_currency_ck CHECK (
    (metric = 'cost' AND currency ~ '^[A-Z]{3}$')
    OR (metric <> 'cost' AND currency IS NULL)
  ),
  ADD CONSTRAINT unknown_fence_binding_dispatches_ck CHECK (
    unknown_dispatch_count > 0 AND counter_incomplete_count >= unknown_dispatch_count
  );

INSERT INTO management_security_policies
  (namespace_id,action_requirements,seed_version,revision)
SELECT id,
  '{"unknown_usage_fence.waive":{"any_of":[{"kind":"human","human":{"minimum_aal":"aal2","accepted_amr":[],"max_authentication_age_seconds":900}},{"kind":"workload","workload":{"minimum_workload_class":"workload_strong","max_source_age_seconds":2592000}}]}}'::jsonb,
  1,1
FROM access_namespaces
ON CONFLICT (namespace_id) DO UPDATE
SET action_requirements = management_security_policies.action_requirements
    || EXCLUDED.action_requirements,
    seed_version = GREATEST(management_security_policies.seed_version, EXCLUDED.seed_version),
    revision = management_security_policies.revision + 1,
    updated_at = clock_timestamp();

ALTER TABLE usage_events
  ADD COLUMN reconciliation_id UUID,
  ADD COLUMN reconciliation_strategy TEXT,
  ADD COLUMN corrects_event_id UUID,
  ADD COLUMN incomplete_dispatch_delta NUMERIC(42,0) NOT NULL DEFAULT 0,
  ADD CONSTRAINT usage_event_reconciliation_shape_ck CHECK (
    (event_kind IN ('actual','unknown')
      AND reconciliation_id IS NULL AND reconciliation_strategy IS NULL
      AND corrects_event_id IS NULL AND incomplete_dispatch_delta = 0)
    OR
    (event_kind IN ('correction','waiver')
      AND reconciliation_id IS NOT NULL
      AND reconciliation_strategy IN ('actual','conservative_debit','waive')
      AND corrects_event_id IS NOT NULL AND incomplete_dispatch_delta < 0)
  );

ALTER TABLE usage_dispatches
  ADD COLUMN corrects_dispatch_id TEXT,
  ADD CONSTRAINT usage_dispatch_correction_shape_ck CHECK (
    (corrects_dispatch_id IS NULL)
    OR (usage_state IN ('known_zero','known_actual'))
  );

CREATE UNIQUE INDEX usage_events_reconciliation_uq
  ON usage_events(namespace_id, event_date, reconciliation_id)
  WHERE reconciliation_id IS NOT NULL;
CREATE INDEX usage_dispatches_correction_idx
  ON usage_dispatches(namespace_id, admission_id, corrects_dispatch_id)
  WHERE corrects_dispatch_id IS NOT NULL;

CREATE TABLE unknown_usage_reconciliation_plans (
  reconciliation_id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  fence_id UUID NOT NULL UNIQUE REFERENCES unknown_usage_fences(id) ON DELETE RESTRICT,
  operation_id UUID NOT NULL UNIQUE REFERENCES management_operations(id) ON DELETE RESTRICT,
  strategy TEXT NOT NULL CHECK (strategy IN ('actual','conservative_debit','waive')),
  plan_digest BYTEA NOT NULL CHECK (octet_length(plan_digest) = 32),
  plan_payload JSONB NOT NULL,
  phase TEXT NOT NULL DEFAULT 'runtime_pending'
    CHECK (phase IN ('runtime_pending','runtime_applied','ledger_applied','completed')),
  runtime_stream_id TEXT,
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
  available_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  lease_owner TEXT,
  lease_token UUID,
  lease_expires_at TIMESTAMPTZ,
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  completed_at TIMESTAMPTZ,
  CHECK (jsonb_typeof(plan_payload) = 'object'),
  CHECK (
    (lease_owner IS NULL AND lease_token IS NULL AND lease_expires_at IS NULL)
    OR (lease_owner IS NOT NULL AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)
  ),
  CHECK ((phase = 'runtime_pending' AND runtime_stream_id IS NULL)
    OR (phase <> 'runtime_pending' AND runtime_stream_id IS NOT NULL)),
  CHECK ((phase = 'completed') = (completed_at IS NOT NULL))
);

CREATE INDEX unknown_usage_reconciliation_claim_idx
  ON unknown_usage_reconciliation_plans(phase, available_at, created_at, reconciliation_id)
  WHERE phase <> 'completed';

CREATE OR REPLACE FUNCTION protect_unknown_usage_reconciliation_plan()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.reconciliation_id IS DISTINCT FROM OLD.reconciliation_id
     OR NEW.namespace_id IS DISTINCT FROM OLD.namespace_id
     OR NEW.fence_id IS DISTINCT FROM OLD.fence_id
     OR NEW.operation_id IS DISTINCT FROM OLD.operation_id
     OR NEW.strategy IS DISTINCT FROM OLD.strategy
     OR NEW.plan_digest IS DISTINCT FROM OLD.plan_digest
     OR NEW.plan_payload IS DISTINCT FROM OLD.plan_payload
     OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
    RAISE EXCEPTION 'unknown usage reconciliation plans are immutable';
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER protect_unknown_usage_reconciliation_plan_update
BEFORE UPDATE ON unknown_usage_reconciliation_plans
FOR EACH ROW EXECUTE FUNCTION protect_unknown_usage_reconciliation_plan();

CREATE OR REPLACE FUNCTION protect_fenced_rate_limit_binding()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE candidate rate_limit_bindings%ROWTYPE;
BEGIN
  candidate := OLD;
  IF EXISTS (
    SELECT 1
    FROM unknown_usage_fence_bindings binding_fence
    JOIN unknown_usage_fences fence ON fence.id = binding_fence.fence_id
    WHERE binding_fence.binding_id = candidate.id
      AND fence.namespace_id = candidate.namespace_id
      AND fence.state IN ('open','reconciling')
  ) AND (
    TG_OP = 'DELETE'
    OR NEW.namespace_id IS DISTINCT FROM OLD.namespace_id
    OR NEW.policy_id IS DISTINCT FROM OLD.policy_id
    OR NEW.subject_id IS DISTINCT FROM OLD.subject_id
    OR NEW.binding_mode IS DISTINCT FROM OLD.binding_mode
    OR NEW.quota_partition_id IS DISTINCT FROM OLD.quota_partition_id
    OR NEW.status IS DISTINCT FROM OLD.status
  ) THEN
    RAISE EXCEPTION 'rate-limit binding has unresolved unknown usage';
  END IF;
  IF TG_OP = 'DELETE' THEN
    RETURN OLD;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER protect_fenced_rate_limit_binding_mutation
BEFORE UPDATE OR DELETE ON rate_limit_bindings
FOR EACH ROW EXECUTE FUNCTION protect_fenced_rate_limit_binding();

CREATE OR REPLACE FUNCTION protect_fenced_rate_limit_rule()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE candidate rate_limit_rules%ROWTYPE;
BEGIN
  candidate := OLD;
  IF EXISTS (
    SELECT 1
    FROM unknown_usage_fence_bindings binding_fence
    JOIN unknown_usage_fences fence ON fence.id = binding_fence.fence_id
    WHERE binding_fence.rule_id = candidate.id
      AND fence.state IN ('open','reconciling')
  ) AND (
    TG_OP = 'DELETE'
    OR NEW.policy_id IS DISTINCT FROM OLD.policy_id
    OR NEW.metric IS DISTINCT FROM OLD.metric
    OR NEW.algorithm IS DISTINCT FROM OLD.algorithm
    OR NEW.window_seconds IS DISTINCT FROM OLD.window_seconds
    OR NEW.calendar_period IS DISTINCT FROM OLD.calendar_period
    OR NEW.timezone IS DISTINCT FROM OLD.timezone
    OR NEW.bucket_capacity IS DISTINCT FROM OLD.bucket_capacity
    OR NEW.refill_amount IS DISTINCT FROM OLD.refill_amount
    OR NEW.refill_period_milliseconds IS DISTINCT FROM OLD.refill_period_milliseconds
    OR NEW.gcra_emission_interval_microseconds IS DISTINCT FROM OLD.gcra_emission_interval_microseconds
    OR NEW.gcra_burst_tolerance IS DISTINCT FROM OLD.gcra_burst_tolerance
    OR NEW.accounting IS DISTINCT FROM OLD.accounting
    OR NEW.enforcement IS DISTINCT FROM OLD.enforcement
  ) THEN
    RAISE EXCEPTION 'rate-limit rule has unresolved unknown usage';
  END IF;
  IF TG_OP = 'DELETE' THEN
    RETURN OLD;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER protect_fenced_rate_limit_rule_mutation
BEFORE UPDATE OR DELETE ON rate_limit_rules
FOR EACH ROW EXECUTE FUNCTION protect_fenced_rate_limit_rule();

-- Inference replay and outcome feedback

CREATE TABLE inference_replays (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  replay_id TEXT NOT NULL CHECK (
    char_length(replay_id) BETWEEN 1 AND 256
    AND replay_id ~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$'
  ),
  api_key_id UUID NOT NULL REFERENCES access_api_keys(id) ON DELETE RESTRICT,
  user_id UUID REFERENCES access_subjects(id) ON DELETE RESTRICT,
  team_id UUID REFERENCES access_subjects(id) ON DELETE RESTRICT,
  event_date DATE NOT NULL,
  event_id UUID NOT NULL,
  routing_context JSONB NOT NULL CHECK (jsonb_typeof(routing_context) = 'object'),
  served_models JSONB NOT NULL CHECK (jsonb_typeof(served_models) = 'array'),
  created_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (namespace_id, replay_id),
  UNIQUE (namespace_id, event_date, event_id),
  FOREIGN KEY (namespace_id, event_date, event_id)
    REFERENCES usage_events(namespace_id, event_date, event_id) ON DELETE RESTRICT
);

CREATE INDEX inference_replays_logical_key_idx
  ON inference_replays (namespace_id, api_key_id, created_at DESC);

CREATE TABLE inference_outcome_projection_heads (
  namespace_id UUID PRIMARY KEY REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  desired_revision BIGINT NOT NULL DEFAULT 0 CHECK (desired_revision >= 0),
  applied_revision BIGINT NOT NULL DEFAULT 0 CHECK (applied_revision >= 0),
  applied_digest BYTEA CHECK (
    applied_digest IS NULL OR octet_length(applied_digest) = 32
  ),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK (applied_revision <= desired_revision),
  CHECK ((applied_revision = 0) = (applied_digest IS NULL))
);

CREATE TABLE inference_outcomes (
  id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  replay_id TEXT NOT NULL,
  api_key_id UUID NOT NULL REFERENCES access_api_keys(id) ON DELETE RESTRICT,
  user_id UUID REFERENCES access_subjects(id) ON DELETE RESTRICT,
  team_id UUID REFERENCES access_subjects(id) ON DELETE RESTRICT,
  source TEXT NOT NULL CHECK (source IN ('api_key','delegated_inference_session')),
  target TEXT NOT NULL CHECK (target IN ('model','route','policy','stability','provider','router')),
  target_ref TEXT CHECK (
    target_ref IS NULL OR char_length(target_ref) BETWEEN 1 AND 512
  ),
  target_revision BIGINT CHECK (target_revision > 0),
  target_model_id TEXT CHECK (
    target_model_id IS NULL OR (
      char_length(target_model_id) BETWEEN 1 AND 256
      AND target_model_id ~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$'
    )
  ),
  target_model_name TEXT CHECK (
    target_model_name IS NULL OR char_length(target_model_name) BETWEEN 1 AND 512
  ),
  verdict TEXT NOT NULL CHECK (verdict IN ('good_fit','underpowered','overprovisioned','failed')),
  reason TEXT CHECK (
    reason IS NULL OR char_length(reason) <= 2048
  ),
  score NUMERIC(10,9) CHECK (score >= 0 AND score <= 1),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
  request_digest BYTEA NOT NULL CHECK (octet_length(request_digest) = 32),
  projection_revision BIGINT NOT NULL CHECK (projection_revision > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (namespace_id, id),
  UNIQUE (namespace_id, projection_revision),
  FOREIGN KEY (namespace_id, replay_id)
    REFERENCES inference_replays(namespace_id, replay_id) ON DELETE RESTRICT,
  CHECK (
    (target = 'model' AND target_ref IS NOT NULL AND target_revision IS NOT NULL
      AND target_model_id IS NOT NULL AND target_model_name IS NOT NULL)
    OR
    (target <> 'model' AND target_revision IS NULL
      AND target_model_id IS NULL AND target_model_name IS NULL)
  )
);

CREATE INDEX inference_outcomes_replay_idx
  ON inference_outcomes (namespace_id, replay_id, created_at DESC);
CREATE INDEX inference_outcomes_model_projection_idx
  ON inference_outcomes (
    namespace_id, target_model_id, target_revision, projection_revision
  ) WHERE target = 'model';

CREATE TABLE inference_outcome_idempotency (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  api_key_id UUID NOT NULL REFERENCES access_api_keys(id) ON DELETE RESTRICT,
  replay_id TEXT NOT NULL,
  idempotency_digest BYTEA NOT NULL CHECK (octet_length(idempotency_digest) = 32),
  request_digest BYTEA NOT NULL CHECK (octet_length(request_digest) = 32),
  receipt_id UUID NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, api_key_id, replay_id, idempotency_digest),
  FOREIGN KEY (namespace_id, replay_id)
    REFERENCES inference_replays(namespace_id, replay_id) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, receipt_id)
    REFERENCES inference_outcomes(namespace_id, id)
    ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX inference_outcome_idempotency_receipt_idx
  ON inference_outcome_idempotency (namespace_id, receipt_id);

CREATE TABLE inference_outcome_projection_outbox (
  outcome_id UUID PRIMARY KEY,
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  desired_revision BIGINT NOT NULL CHECK (desired_revision > 0),
  state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending','staged','applied')),
  available_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  applied_at TIMESTAMPTZ,
  UNIQUE (namespace_id, desired_revision),
  FOREIGN KEY (namespace_id, outcome_id)
    REFERENCES inference_outcomes(namespace_id, id) ON DELETE RESTRICT
);

CREATE INDEX inference_outcome_projection_outbox_pending_idx
  ON inference_outcome_projection_outbox (available_at, namespace_id, desired_revision)
  WHERE state <> 'applied';

CREATE TABLE inference_outcome_projection_snapshots (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  revision BIGINT NOT NULL CHECK (revision > 0),
  snapshot JSONB NOT NULL CHECK (jsonb_typeof(snapshot) = 'object'),
  snapshot_digest BYTEA NOT NULL CHECK (octet_length(snapshot_digest) = 32),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, revision)
);

-- Built-in Recipe distribution

-- Built-in Recipes are immutable Management resources installed from the
-- Router distribution. The release identity is content-addressed and scoped
-- by Namespace; an upgraded distribution creates sibling Recipes rather than
-- mutating user state or existing Entrypoint references.

CREATE TABLE routing_recipe_distributions (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  distribution_id TEXT NOT NULL CHECK (distribution_id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  distribution_version TEXT NOT NULL CHECK (
    distribution_version ~ '^[A-Za-z0-9][A-Za-z0-9.-]{0,63}$'
  ),
  asset_digest BYTEA NOT NULL CHECK (octet_length(asset_digest) = 32),
  recipe_count INTEGER NOT NULL CHECK (recipe_count BETWEEN 1 AND 64),
  installed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, distribution_id, distribution_version)
);

CREATE TABLE routing_recipe_provenance (
  namespace_id UUID NOT NULL,
  recipe_id TEXT NOT NULL,
  recipe_revision BIGINT NOT NULL CHECK (recipe_revision > 0),
  distribution_id TEXT NOT NULL,
  distribution_version TEXT NOT NULL,
  source_recipe_id TEXT NOT NULL CHECK (source_recipe_id ~ '^[a-z][a-z0-9_-]{2,127}$'),
  source_recipe_revision BIGINT NOT NULL CHECK (source_recipe_revision > 0),
  asset_digest BYTEA NOT NULL CHECK (octet_length(asset_digest) = 32),
  recipe_digest BYTEA NOT NULL CHECK (octet_length(recipe_digest) = 32),
  installed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (namespace_id, recipe_id),
  UNIQUE (namespace_id, distribution_id, distribution_version, source_recipe_id),
  FOREIGN KEY (namespace_id, recipe_id)
    REFERENCES routing_recipes(namespace_id, id) ON DELETE RESTRICT,
  FOREIGN KEY (recipe_id, recipe_revision)
    REFERENCES routing_recipe_revisions(recipe_id, revision) ON DELETE RESTRICT,
  FOREIGN KEY (namespace_id, distribution_id, distribution_version)
    REFERENCES routing_recipe_distributions(namespace_id, distribution_id, distribution_version)
    ON DELETE RESTRICT
);

CREATE INDEX routing_recipe_provenance_distribution_idx
  ON routing_recipe_provenance (
    namespace_id, distribution_id, distribution_version, source_recipe_id
  );

-- Usage timing rollups

-- Add mergeable request/dispatch timing summaries to every usage grain.
-- Histograms use the fixed bucket contract owned by usageledger; keeping the
-- counts additive lets hour/day queries stay on rollups instead of scanning
-- raw events.
DO $$
DECLARE
  table_name TEXT;
BEGIN
  FOREACH table_name IN ARRAY ARRAY['usage_rollup_1m', 'usage_rollup_1h', 'usage_rollup_1d']
  LOOP
    EXECUTE format('ALTER TABLE %I
      ADD COLUMN latency_count NUMERIC(42,0) NOT NULL DEFAULT 0,
      ADD COLUMN latency_sum_ms NUMERIC(42,0) NOT NULL DEFAULT 0,
      ADD COLUMN latency_histogram JSONB NOT NULL DEFAULT ''[]''::jsonb,
      ADD COLUMN ttft_count NUMERIC(42,0) NOT NULL DEFAULT 0,
      ADD COLUMN ttft_sum_ms NUMERIC(42,0) NOT NULL DEFAULT 0,
      ADD COLUMN ttft_histogram JSONB NOT NULL DEFAULT ''[]''::jsonb', table_name);
  END LOOP;
END $$;

-- Management statistics indexes

CREATE INDEX access_users_statistics_idx
  ON access_users(namespace_id, id) WHERE deleted_at IS NULL;
CREATE INDEX access_teams_statistics_idx
  ON access_teams(namespace_id, id) WHERE deleted_at IS NULL;
CREATE INDEX access_api_keys_statistics_active_idx
  ON access_api_keys(namespace_id, expires_at, id)
  INCLUDE (owner_user_id, owner_team_id)
  WHERE deleted_at IS NULL AND status = 'active';
CREATE INDEX rate_limit_policies_statistics_active_idx
  ON rate_limit_policies(namespace_id, id) WHERE status = 'active';

-- Routing model metadata

ALTER TABLE routing_model_revisions
  ADD COLUMN param_size TEXT NOT NULL DEFAULT '',
  ADD COLUMN context_window_size BIGINT NOT NULL DEFAULT 0
    CHECK (context_window_size BETWEEN 0 AND 100000000),
  ADD COLUMN description TEXT NOT NULL DEFAULT '',
  ADD COLUMN quality_score DOUBLE PRECISION NOT NULL DEFAULT 0
    CHECK (quality_score >= 0 AND quality_score <= 1),
  ADD COLUMN modality TEXT NOT NULL DEFAULT '',
  ADD COLUMN tags JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE routing_recipe_revisions
  ADD COLUMN description TEXT NOT NULL DEFAULT '';

-- Management collection search

-- Bounded Management API search is literal, case-insensitive prefix matching.
-- Keep each public identity field independently indexable so PostgreSQL can
-- combine selective matches without inspecting credential material.
CREATE INDEX access_users_email_search_idx
  ON access_users(namespace_id, lower(email) text_pattern_ops);
CREATE INDEX access_users_display_name_search_idx
  ON access_users(namespace_id, lower(display_name) text_pattern_ops);
CREATE INDEX access_users_public_id_search_idx
  ON access_users(namespace_id, (id::text) text_pattern_ops);

CREATE INDEX access_teams_name_search_idx
  ON access_teams(namespace_id, lower(name) text_pattern_ops);
CREATE INDEX access_teams_public_id_search_idx
  ON access_teams(namespace_id, (id::text) text_pattern_ops);

CREATE INDEX access_api_keys_name_search_idx
  ON access_api_keys(namespace_id, lower(name) text_pattern_ops);
CREATE INDEX access_api_keys_public_id_search_idx
  ON access_api_keys(namespace_id, (id::text) text_pattern_ops);

CREATE INDEX access_policies_name_search_idx
  ON access_policies(namespace_id, lower(name) text_pattern_ops);
CREATE INDEX access_policies_public_id_search_idx
  ON access_policies(namespace_id, (id::text) text_pattern_ops);

CREATE INDEX rate_limit_policies_name_search_idx
  ON rate_limit_policies(namespace_id, lower(name) text_pattern_ops);
CREATE INDEX rate_limit_policies_public_id_search_idx
  ON rate_limit_policies(namespace_id, (id::text) text_pattern_ops);

-- Built-in role invariants

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM management_roles
    WHERE id = '10000000-0000-5000-8000-000000000002'
      AND namespace_id IS NULL
      AND name = 'platform_admin'
      AND builtin = TRUE
      AND status = 'active'
      AND revision = 1
      AND permissions = '["access_policy.manage","access_policy.read","agent.manage","agent.read","agent.use","audit.read","delegation.manage","delegation.use","evaluation.run","health.read","invitation.manage","invitation.read","key.manage","key.read","log.read","log_payload.read","management_role.manage","management_role.read","membership.manage","namespace.manage","namespace.read","onboarding.manage","operation.manage","operation.read","principal_directory.read","principal_link.manage","principal_link.read","provider_catalog.read","provider_credential.manage","provider_credential.read","provider_credential.use","quota.read","quota.reconcile","rate_policy.manage","rate_policy.read","role_binding.manage","role_binding.read","routing.manage","routing.publish","routing.read","routing_context.manage","routing_context.read","service_account.manage","service_account.read","team.manage","team.read","tool.invoke","tool.manage","tool.read","usage.internal_dimensions.read","usage.read","user.manage","user.read"]'::jsonb
      AND permissions_digest = decode('e27404421cc93ba8b220ec599cc5a4a702242ae7f5bfb40da19f08c756a63ff2','hex')
  ) THEN
    RAISE EXCEPTION 'platform_admin built-in role seed does not match its immutable contract';
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM management_roles
    WHERE id = '10000000-0000-5000-8000-000000000003'
      AND namespace_id IS NULL
      AND name = 'operator'
      AND builtin = TRUE
      AND status = 'active'
      AND revision = 1
      AND permissions = '["access_policy.read","agent.manage","agent.read","agent.use","evaluation.run","health.read","log.read","log_payload.read","namespace.read","operation.manage","operation.read","provider_catalog.read","provider_credential.manage","provider_credential.read","provider_credential.use","quota.read","rate_policy.read","routing.manage","routing.publish","routing.read","routing_context.manage","routing_context.read","tool.invoke","tool.manage","tool.read","usage.internal_dimensions.read","usage.read"]'::jsonb
      AND permissions_digest = decode('ea55e1af54baf2f9e9894cf93f34df433b155cf7ea37af9dbec590c835759243','hex')
  ) THEN
    RAISE EXCEPTION 'operator built-in role seed does not match its immutable contract';
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM management_roles
    WHERE id = '10000000-0000-5000-8000-000000000007'
      AND namespace_id IS NULL
      AND name = 'viewer'
      AND builtin = TRUE
      AND status = 'active'
      AND revision = 1
      AND permissions = '["agent.read","provider_catalog.read","routing.read","tool.read"]'::jsonb
      AND permissions_digest = decode('457a9204a91594a24e10ce7ab98b16fe61ec569104e7f25b9fadfe5e78f08ceb','hex')
  ) THEN
    RAISE EXCEPTION 'viewer built-in role seed does not match its least-privilege contract';
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM management_roles
    WHERE id = '10000000-0000-5000-8000-000000000008'
      AND namespace_id IS NULL
      AND name = 'consumer'
      AND builtin = TRUE
      AND status = 'active'
      AND revision = 1
      AND permissions = '["access_policy.read","agent.read","agent.use","delegation.use","key.read","operation.read","quota.read","rate_policy.read","routing_context.read","team.read","tool.invoke","tool.read","usage.read","user.read"]'::jsonb
      AND permissions_digest = decode('42f87d9c0231abac6d6f5256f4c40d5d5789f9c9d8739c264785d0bf58560fd6','hex')
  ) THEN
    RAISE EXCEPTION 'consumer built-in role seed does not match its least-privilege contract';
  END IF;
END
$$;

-- Routing collection search

-- Managed routing collection search is a bounded, case-insensitive literal
-- prefix match over public names and stable resource IDs.
CREATE INDEX routing_models_name_search_idx
  ON routing_models(namespace_id, lower(name) text_pattern_ops);
CREATE INDEX routing_models_id_search_idx
  ON routing_models(namespace_id, id text_pattern_ops);

CREATE INDEX routing_recipes_name_search_idx
  ON routing_recipes(namespace_id, lower(name) text_pattern_ops);
CREATE INDEX routing_recipes_id_search_idx
  ON routing_recipes(namespace_id, id text_pattern_ops);

CREATE INDEX routing_entrypoints_name_search_idx
  ON routing_entrypoints(namespace_id, lower(name) text_pattern_ops);
CREATE INDEX routing_entrypoints_id_search_idx
  ON routing_entrypoints(namespace_id, id text_pattern_ops);

-- Usage partition lifecycle

CREATE TABLE usage_partition_months (
  month_start DATE PRIMARY KEY CHECK (month_start = date_trunc('month', month_start)::date),
  month_end DATE NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('active','retired')),
  event_partition TEXT NOT NULL UNIQUE CHECK (event_partition ~ '^usage_events_[0-9]{6}$'),
  dispatch_partition TEXT NOT NULL UNIQUE CHECK (dispatch_partition ~ '^usage_dispatches_[0-9]{6}$'),
  attempt_partition TEXT NOT NULL UNIQUE CHECK (attempt_partition ~ '^usage_dispatch_attempts_[0-9]{6}$'),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  last_checked_at TIMESTAMPTZ,
  retired_at TIMESTAMPTZ,
  CHECK (month_end = (month_start + INTERVAL '1 month')::date),
  CHECK ((state = 'active' AND retired_at IS NULL)
    OR (state = 'retired' AND retired_at IS NOT NULL))
);

CREATE TABLE usage_rollup_dirty_minutes (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  event_partition_date DATE NOT NULL,
  bucket_start TIMESTAMPTZ NOT NULL,
  ledger_watermark TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (namespace_id, event_partition_date, bucket_start),
  CHECK (bucket_start = date_trunc('minute', bucket_start AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')
);

CREATE TABLE usage_rollup_dirty_hours (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  event_partition_date DATE NOT NULL,
  bucket_start TIMESTAMPTZ NOT NULL,
  ledger_watermark TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (namespace_id, event_partition_date, bucket_start),
  CHECK (bucket_start = date_trunc('hour', bucket_start AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')
);

CREATE TABLE usage_rollup_dirty_days (
  namespace_id UUID NOT NULL REFERENCES access_namespaces(id) ON DELETE RESTRICT,
  event_partition_date DATE NOT NULL,
  bucket_start TIMESTAMPTZ NOT NULL,
  ledger_watermark TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (namespace_id, event_partition_date, bucket_start),
  CHECK (bucket_start = date_trunc('day', bucket_start AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')
);

CREATE OR REPLACE FUNCTION ensure_usage_month_partition(target_date DATE)
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
  start_date DATE := date_trunc('month', target_date)::date;
  end_date DATE := (date_trunc('month', target_date) + INTERVAL '1 month')::date;
  suffix TEXT := to_char(start_date, 'YYYYMM');
  event_name TEXT := 'usage_events_' || suffix;
  dispatch_name TEXT := 'usage_dispatches_' || suffix;
  attempt_name TEXT := 'usage_dispatch_attempts_' || suffix;
  current_state TEXT;
  current_event TEXT;
  current_dispatch TEXT;
  current_attempt TEXT;
BEGIN
  IF target_date IS NULL THEN
    RAISE EXCEPTION 'usage partition date is required';
  END IF;

  SELECT state, event_partition, dispatch_partition, attempt_partition
  INTO current_state, current_event, current_dispatch, current_attempt
  FROM usage_partition_months
  WHERE month_start = start_date;

  IF current_state = 'active' THEN
    IF to_regclass(current_event) IS NULL OR to_regclass(current_dispatch) IS NULL
       OR to_regclass(current_attempt) IS NULL THEN
      RAISE EXCEPTION 'active usage partition registry entry for % is incomplete', start_date;
    END IF;
    RETURN;
  END IF;

  PERFORM pg_advisory_xact_lock(hashtextextended('vllm-sr/usage-partition/' || suffix, 0));

  SELECT state, event_partition, dispatch_partition, attempt_partition
  INTO current_state, current_event, current_dispatch, current_attempt
  FROM usage_partition_months
  WHERE month_start = start_date
  FOR UPDATE;

  IF current_state = 'active' THEN
    IF to_regclass(current_event) IS NULL OR to_regclass(current_dispatch) IS NULL
       OR to_regclass(current_attempt) IS NULL THEN
      RAISE EXCEPTION 'active usage partition registry entry for % is incomplete', start_date;
    END IF;
    RETURN;
  END IF;

  EXECUTE format(
    'CREATE TABLE %I PARTITION OF usage_events FOR VALUES FROM (%L) TO (%L)',
    event_name, start_date, end_date
  );
  EXECUTE format(
    'CREATE TABLE %I PARTITION OF usage_dispatches FOR VALUES FROM (%L) TO (%L)',
    dispatch_name, start_date, end_date
  );
  EXECUTE format(
    'CREATE TABLE %I PARTITION OF usage_dispatch_attempts FOR VALUES FROM (%L) TO (%L)',
    attempt_name, start_date, end_date
  );

  INSERT INTO usage_partition_months (
    month_start, month_end, state, event_partition, dispatch_partition, attempt_partition
  ) VALUES (start_date, end_date, 'active', event_name, dispatch_name, attempt_name)
  ON CONFLICT (month_start) DO UPDATE
  SET state = 'active', event_partition = EXCLUDED.event_partition,
      dispatch_partition = EXCLUDED.dispatch_partition,
      attempt_partition = EXCLUDED.attempt_partition, created_at = clock_timestamp(),
      last_checked_at = NULL, retired_at = NULL;
END;
$$;

CREATE OR REPLACE FUNCTION mark_usage_event_rollup_dirty()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE
  minute_start TIMESTAMPTZ :=
    date_trunc('minute', NEW.occurred_at AT TIME ZONE 'UTC') AT TIME ZONE 'UTC';
BEGIN
  INSERT INTO usage_rollup_dirty_minutes(
    namespace_id, event_partition_date, bucket_start, ledger_watermark
  ) VALUES (NEW.namespace_id, NEW.event_date, minute_start, NEW.ingested_at)
  ON CONFLICT (namespace_id, event_partition_date, bucket_start) DO UPDATE
  SET ledger_watermark = GREATEST(
    usage_rollup_dirty_minutes.ledger_watermark,
    EXCLUDED.ledger_watermark
  );
  RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION mark_usage_dispatch_rollup_dirty()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE
  minute_start TIMESTAMPTZ :=
    date_trunc('minute', NEW.started_at AT TIME ZONE 'UTC') AT TIME ZONE 'UTC';
  watermark TIMESTAMPTZ;
BEGIN
  SELECT ingested_at INTO STRICT watermark
  FROM usage_events
  WHERE namespace_id = NEW.namespace_id
    AND event_date = NEW.event_date
    AND event_id = NEW.event_id;

  INSERT INTO usage_rollup_dirty_minutes(
    namespace_id, event_partition_date, bucket_start, ledger_watermark
  ) VALUES (NEW.namespace_id, NEW.event_date, minute_start, watermark)
  ON CONFLICT (namespace_id, event_partition_date, bucket_start) DO UPDATE
  SET ledger_watermark = GREATEST(
    usage_rollup_dirty_minutes.ledger_watermark,
    EXCLUDED.ledger_watermark
  );
  RETURN NEW;
END;
$$;

CREATE TRIGGER mark_usage_event_rollup_dirty_after_insert
AFTER INSERT ON usage_events
FOR EACH ROW EXECUTE FUNCTION mark_usage_event_rollup_dirty();

CREATE TRIGGER mark_usage_dispatch_rollup_dirty_after_insert
AFTER INSERT ON usage_dispatches
FOR EACH ROW EXECUTE FUNCTION mark_usage_dispatch_rollup_dirty();

CREATE INDEX usage_settlements_partition_retention_idx
  ON usage_settlements(event_partition_date, event_retained);
CREATE INDEX usage_partition_months_active_idx
  ON usage_partition_months(month_end, month_start) WHERE state = 'active';
CREATE INDEX usage_rollup_dirty_minutes_scan_idx
  ON usage_rollup_dirty_minutes(namespace_id, bucket_start, ledger_watermark, event_partition_date);
CREATE INDEX usage_rollup_dirty_hours_scan_idx
  ON usage_rollup_dirty_hours(namespace_id, bucket_start, ledger_watermark, event_partition_date);
CREATE INDEX usage_rollup_dirty_days_scan_idx
  ON usage_rollup_dirty_days(namespace_id, bucket_start, ledger_watermark, event_partition_date);

CREATE INDEX usage_rollup_dirty_minutes_retention_idx
  ON usage_rollup_dirty_minutes(event_partition_date, namespace_id, bucket_start);
CREATE INDEX usage_rollup_dirty_hours_retention_idx
  ON usage_rollup_dirty_hours(event_partition_date, namespace_id, bucket_start);
CREATE INDEX usage_rollup_dirty_days_retention_idx
  ON usage_rollup_dirty_days(event_partition_date, namespace_id, bucket_start);

SELECT ensure_usage_month_partition(current_date);
