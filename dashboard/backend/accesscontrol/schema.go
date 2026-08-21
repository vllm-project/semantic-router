package accesscontrol

const schema = `
CREATE TABLE IF NOT EXISTS access_users (
  id TEXT PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS access_teams (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  description TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS access_team_members (
  team_id TEXT NOT NULL REFERENCES access_teams(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL REFERENCES access_users(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (team_id, user_id)
);
CREATE TABLE IF NOT EXISTS access_budgets (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  scope_type TEXT NOT NULL CHECK (scope_type IN ('global', 'user', 'team', 'key')),
  scope_id TEXT NOT NULL DEFAULT '',
  rpm BIGINT NOT NULL DEFAULT 0 CHECK (rpm >= 0),
  tpm BIGINT NOT NULL DEFAULT 0 CHECK (tpm >= 0),
  daily_tokens BIGINT NOT NULL DEFAULT 0 CHECK (daily_tokens >= 0),
  enabled BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (scope_type, scope_id)
);
CREATE TABLE IF NOT EXISTS access_api_keys (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  prefix TEXT NOT NULL,
  digest TEXT NOT NULL UNIQUE,
  secret_ciphertext TEXT NOT NULL,
  user_id TEXT REFERENCES access_users(id) ON DELETE RESTRICT,
  team_id TEXT REFERENCES access_teams(id) ON DELETE RESTRICT,
  budget_id TEXT REFERENCES access_budgets(id) ON DELETE SET NULL,
  status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
  expires_at TIMESTAMPTZ,
  last_used_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK ((user_id IS NOT NULL) <> (team_id IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS access_groups (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  description TEXT NOT NULL DEFAULT '',
  model_patterns JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS access_group_bindings (
  group_id TEXT NOT NULL REFERENCES access_groups(id) ON DELETE CASCADE,
  subject_type TEXT NOT NULL CHECK (subject_type IN ('user', 'team', 'key')),
  subject_id TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (group_id, subject_type, subject_id)
);
CREATE TABLE IF NOT EXISTS access_usage_events (
  id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL UNIQUE,
  key_id TEXT NOT NULL,
  user_id TEXT NOT NULL DEFAULT '',
  team_id TEXT NOT NULL DEFAULT '',
  model TEXT NOT NULL,
  status_code INTEGER NOT NULL,
  prompt_tokens BIGINT NOT NULL DEFAULT 0,
  completion_tokens BIGINT NOT NULL DEFAULT 0,
  total_tokens BIGINT NOT NULL DEFAULT 0,
  latency_ms BIGINT NOT NULL DEFAULT 0,
  ttft_ms BIGINT NOT NULL DEFAULT 0,
  error_code TEXT NOT NULL DEFAULT '',
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS access_audit_events (
  id TEXT PRIMARY KEY,
  actor_id TEXT NOT NULL DEFAULT '',
  actor_email TEXT NOT NULL DEFAULT '',
  action TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  resource_id TEXT NOT NULL DEFAULT '',
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_access_keys_owner ON access_api_keys(user_id, team_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_access_team_members_user ON access_team_members(user_id);
CREATE INDEX IF NOT EXISTS idx_access_keys_status ON access_api_keys(status, expires_at);
CREATE INDEX IF NOT EXISTS idx_access_keys_budget ON access_api_keys(budget_id);
CREATE INDEX IF NOT EXISTS idx_access_group_bindings_subject ON access_group_bindings(subject_type, subject_id);
CREATE INDEX IF NOT EXISTS idx_access_usage_created ON access_usage_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_access_usage_subject ON access_usage_events(user_id, team_id, key_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_access_usage_user_created ON access_usage_events(user_id, created_at DESC, id);
CREATE INDEX IF NOT EXISTS idx_access_usage_team_created ON access_usage_events(team_id, created_at DESC, id);
CREATE INDEX IF NOT EXISTS idx_access_usage_key_created ON access_usage_events(key_id, created_at DESC, id);
CREATE INDEX IF NOT EXISTS idx_access_usage_status_created ON access_usage_events(status_code, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_access_usage_model ON access_usage_events(model, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_access_audit_created ON access_audit_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_access_audit_actor_created ON access_audit_events(actor_id, created_at DESC);
`
