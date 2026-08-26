package postgres

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

var immutableMigrationSHA256 = map[string]string{
	"0001_management.sql":                       "8bb1fba4c42abbd66058d89a8d1b97c171ad87ffb99025ef15219902ae3ce9fb",
	"0002_access_audit_actor_chain.sql":         "47527b2825b8b7467743baab73c7e8efa4c29d04b7699e98b66849e2c1730786",
	"0003_issuer_logout_tombstones.sql":         "db4d183eada857e4c7d45bd259b99fc78eacf927e6ae03597608f0594b334463",
	"0004_agent_event_summary_usage_lookup.sql": "54e6a37ac2fe7e73bf5890e37e00563b661b2c3622aa46fdc83acc6d9db88696",
}

func TestEmbeddedMigrationsAreOrderedAndCoverAuthorities(t *testing.T) {
	migrations, err := Migrations()
	if err != nil {
		t.Fatal(err)
	}
	if len(migrations) == 0 {
		t.Fatal("embedded Management migrations are empty")
	}
	if len(migrations) != len(immutableMigrationSHA256) {
		t.Fatalf("migration count = %d, immutable digest manifest count = %d",
			len(migrations), len(immutableMigrationSHA256))
	}
	if migrations[0].Version != 1 || migrations[0].Name != "0001_management.sql" {
		t.Fatalf("baseline migration = (%d, %q), want (1, %q)",
			migrations[0].Version, migrations[0].Name, "0001_management.sql")
	}
	if expected := sha256.Sum256([]byte(migrations[0].SQL)); migrations[0].Digest != expected {
		t.Fatal("baseline migration digest does not cover its exact embedded SQL")
	}
	var schema strings.Builder
	for i, migration := range migrations {
		if i > 0 && migrations[i-1].Version >= migration.Version {
			t.Fatalf("migration versions are not strictly increasing: %d then %d", migrations[i-1].Version, migration.Version)
		}
		wantDigest, exists := immutableMigrationSHA256[migration.Name]
		if !exists {
			t.Fatalf("migration %q is missing from the immutable digest manifest", migration.Name)
		}
		if got := hex.EncodeToString(migration.Digest[:]); got != wantDigest {
			t.Fatalf("immutable migration %q digest = %q, want %q; add a new migration instead of editing history",
				migration.Name, got, wantDigest)
		}
		schema.WriteString(migration.SQL)
	}
	assertMigrationTables(t, schema.String())
	assertMigrationContracts(t, schema.String())
}

func assertMigrationTables(t *testing.T, schema string) {
	t.Helper()
	for _, table := range []string{
		"access_namespaces", "access_subjects", "access_api_keys",
		"access_api_key_credentials", "access_policies", "rate_limit_policies",
		"management_principals", "management_sessions", "management_invitations",
		"management_backchannel_logout_replays", "management_exchange_challenges",
		"management_revocation_barriers", "management_issuer_logout_tombstones",
		"provider_credentials", "routing_models", "routing_recipes",
		"routing_recipe_distributions", "routing_recipe_provenance",
		"routing_entrypoints", "routing_snapshots", "routing_publications",
		"routing_publication_heads", "routing_fleet_replicas", "routing_replica_leases",
		"routing_publication_required_replicas", "routing_publication_acknowledgements", "policy_outbox",
		"management_idempotency",
		"provider_catalog_revisions", "provider_catalog_state", "provider_catalog_required_rollout_groups",
		"provider_catalog_replica_acks",
		"usage_settlements", "usage_events", "usage_dispatches", "usage_dispatch_attempts",
		"usage_partition_months", "usage_rollup_dirty_minutes", "usage_rollup_dirty_hours",
		"usage_rollup_dirty_days",
		"unknown_usage_fences", "access_audit_events", "access_audit_heads",
		"inference_replays", "inference_outcomes", "inference_outcome_idempotency",
		"inference_outcome_projection_heads", "inference_outcome_projection_outbox",
		"inference_outcome_projection_snapshots",
	} {
		if !strings.Contains(schema, "CREATE TABLE "+table) {
			t.Errorf("embedded migrations do not create %s", table)
		}
	}
}

func assertMigrationContracts(t *testing.T, schema string) {
	t.Helper()
	for _, contract := range []string{
		"resource_id TEXT NOT NULL",
		"gcra_burst_tolerance BIGINT",
		"desired_revision BIGINT",
		"chain_sequence BIGINT",
		"octet_length(event_hash) = 32",
		"UNIQUE (namespace_id, id)",
		"CHECK (desired_revision IS NULL OR namespace_id IS NOT NULL)",
		"FOREIGN KEY (namespace_id, last_event_id)",
		"view TEXT NOT NULL CHECK (view IN ('request','dispatch'))",
		"REFERENCES usage_dispatches(namespace_id, event_date, event_id, dispatch_id, admission_id)",
		"usage_events_api_key_idx",
		"usage_events_admission_idx",
		"event_retained BOOLEAN NOT NULL DEFAULT TRUE",
		"CREATE OR REPLACE FUNCTION ensure_usage_month_partition",
		"PRIMARY KEY (namespace_id, event_partition_date, bucket_start)",
		"wire_format TEXT NOT NULL",
		"provider_catalog_revision TEXT NOT NULL",
		"UNIQUE (namespace_id, distribution_id, distribution_version, source_recipe_id)",
		"recipe_digest BYTEA NOT NULL CHECK (octet_length(recipe_digest) = 32)",
		"PRIMARY KEY (namespace_id, replay_id)",
		"PRIMARY KEY (namespace_id, api_key_id, replay_id, idempotency_digest)",
		"FOREIGN KEY (namespace_id, replay_id)",
		"REFERENCES inference_replays(namespace_id, replay_id)",
		"octet_length(idempotency_digest) = 32",
		"octet_length(snapshot_digest) = 32",
		"claims_digest BYTEA NOT NULL CHECK (octet_length(claims_digest) = 32)",
		"selector_kind TEXT NOT NULL CHECK (selector_kind IN ('sid','subject'))",
		"PRIMARY KEY (issuer_id, selector_kind, selector_digest)",
		"credential_mode TEXT NOT NULL CHECK (credential_mode IN ('optional','required'))",
		"management_idempotency_cluster_identity_uq",
		"management_idempotency_namespace_identity_uq",
		"provider_catalog_revision TEXT NOT NULL",
		"access_users_statistics_idx",
		"access_teams_statistics_idx",
		"access_api_keys_statistics_active_idx",
		"rate_limit_policies_statistics_active_idx",
		"access_users_email_search_idx",
		"access_users_display_name_search_idx",
		"access_users_public_id_search_idx",
		"access_teams_name_search_idx",
		"access_teams_public_id_search_idx",
		"access_api_keys_name_search_idx",
		"access_api_keys_public_id_search_idx",
		"access_policies_name_search_idx",
		"access_policies_public_id_search_idx",
		"rate_limit_policies_name_search_idx",
		"rate_limit_policies_public_id_search_idx",
		"agent_profiles_name_search_idx",
		"agent_profiles_description_search_idx",
		"agent_skills_name_search_idx",
		"agent_skills_description_search_idx",
		"agent_tool_credentials_name_search_idx",
		"agent_tool_sources_name_search_idx",
		"agent_tool_sources_description_search_idx",
		"agent_sessions_page_idx",
		"agent_sessions_title_search_idx",
		"'assistant_delta','model_step_summary','tool_request'",
		"CONSTRAINT management_operations_id_namespace_uq UNIQUE (id, namespace_id)",
		"CONSTRAINT usage_event_reconciliation_shape_ck CHECK",
		"CONSTRAINT usage_dispatch_correction_shape_ck CHECK",
		"CONSTRAINT unknown_fence_binding_metric_ck CHECK",
		"latency_histogram JSONB NOT NULL DEFAULT ''[]''::jsonb",
		"param_size TEXT NOT NULL DEFAULT ''",
		"usage_events_external_request_idx",
		"access_scope JSONB NOT NULL CHECK (jsonb_typeof(access_scope) = 'object')",
		"policy_outbox_notify_routing_desired_state",
		"pg_notify('vllm_sr_routing_publication', NEW.namespace_id::text)",
	} {
		if !strings.Contains(schema, contract) {
			t.Errorf("embedded migrations do not contain contract %q", contract)
		}
	}
}

func TestMigrationVersionRejectsInvalidNames(t *testing.T) {
	for _, name := range []string{"migration.sql", "0_zero.sql", "-1_bad.sql", "x_bad.sql"} {
		if _, err := migrationVersion(name); err == nil {
			t.Errorf("migrationVersion(%q) unexpectedly succeeded", name)
		}
	}
}

func TestAppliedMigrationsMustMatchEmbeddedIdentityAndPrefix(t *testing.T) {
	migrations := []Migration{
		migrationFixture(1, "0001_baseline.sql", "CREATE TABLE baseline(id BIGINT PRIMARY KEY);"),
		migrationFixture(2, "0002_forward.sql", "CREATE TABLE forward(id BIGINT PRIMARY KEY);"),
	}
	first := appliedMigrationFixture(migrations[0])
	second := appliedMigrationFixture(migrations[1])
	renamed := first
	renamed.Name = "0001_preview.sql"
	unknown := first
	unknown.Name = "0003_future.sql"
	for name, test := range map[string]struct {
		applied map[int64]appliedMigration
		wantErr bool
	}{
		"empty":          {applied: map[int64]appliedMigration{}},
		"prefix":         {applied: map[int64]appliedMigration{1: first}},
		"complete":       {applied: map[int64]appliedMigration{1: first, 2: second}},
		"renamed":        {applied: map[int64]appliedMigration{1: renamed}, wantErr: true},
		"unknown":        {applied: map[int64]appliedMigration{3: unknown}, wantErr: true},
		"missing prefix": {applied: map[int64]appliedMigration{2: second}, wantErr: true},
	} {
		t.Run(name, func(t *testing.T) {
			err := validateAppliedMigrations(migrations, test.applied)
			if (err != nil) != test.wantErr {
				t.Fatalf("validateAppliedMigrations() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
	tampered := appliedMigrationFixture(migrations[0])
	tampered.Digest[0] ^= 0xff
	if err := validateAppliedMigrations(
		migrations,
		map[int64]appliedMigration{1: tampered},
	); err == nil || !strings.Contains(err.Error(), "content digest") {
		t.Fatalf("tampered migration digest error = %v", err)
	}
	missingDigest := appliedMigration{Name: migrations[0].Name}
	if err := validateAppliedMigrations(
		migrations,
		map[int64]appliedMigration{migrations[0].Version: missingDigest},
	); err == nil || !strings.Contains(err.Error(), "content digest") {
		t.Fatalf("missing migration digest error = %v", err)
	}
}

func migrationFixture(version int64, name, sql string) Migration {
	return Migration{Version: version, Name: name, Digest: sha256.Sum256([]byte(sql)), SQL: sql}
}

func appliedMigrationFixture(migration Migration) appliedMigration {
	return appliedMigration{Name: migration.Name, Digest: bytes.Clone(migration.Digest[:])}
}

func TestBaselineSeedsLeastPrivilegeBuiltInRoles(t *testing.T) {
	migrations, err := Migrations()
	if err != nil {
		t.Fatal(err)
	}
	if len(migrations) == 0 {
		t.Fatal("embedded Management migrations are empty")
	}
	baseline := migrations[0].SQL
	for _, contract := range []string{
		"10000000-0000-5000-8000-000000000007",
		"10000000-0000-5000-8000-000000000008",
		"viewer built-in role seed does not match its least-privilege contract",
		"consumer built-in role seed does not match its least-privilege contract",
	} {
		if !strings.Contains(baseline, contract) {
			t.Fatalf("Management baseline does not contain %q", contract)
		}
	}
	if strings.Contains(baseline, "UPDATE management_roles") {
		t.Fatal("Management baseline contains an unreleased built-in role correction")
	}
	if strings.Contains(baseline, "preview usage schema") {
		t.Fatal("Management baseline contains a preview-schema upgrade guard")
	}
	for _, forwardContract := range []string{
		"model_step_summary",
		"usage_events_external_request_idx",
	} {
		if strings.Contains(baseline, forwardContract) {
			t.Fatalf("Management baseline contains forward contract %q", forwardContract)
		}
	}
}

func TestMigrationsApplyToPostgreSQL(t *testing.T) {
	databaseURL := migrationTestDatabaseURL(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	db, schemaName := isolatedMigrationDatabase(t, ctx, databaseURL)
	if err := (Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatalf("apply migrations: %v", err)
	}
	if err := (Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatalf("reapply migrations: %v", err)
	}
	assertColumnType(t, ctx, db, schemaName, "access_policy_grants", "resource_id", "text")
	assertColumnType(t, ctx, db, schemaName, "rate_limit_rules", "gcra_burst_tolerance", "bigint")
	assertColumnType(t, ctx, db, schemaName, "access_audit_events", "event_hash", "bytea")
	assertColumnType(t, ctx, db, schemaName, "access_audit_events", "reason", "text")
	assertColumnType(t, ctx, db, schemaName, "routing_model_backends", "wire_format", "text")
	assertColumnType(t, ctx, db, schemaName, "routing_model_revisions", "provider_catalog_revision", "text")
	assertColumnType(t, ctx, db, schemaName, "routing_recipe_distributions", "asset_digest", "bytea")
	assertColumnType(t, ctx, db, schemaName, "routing_recipe_provenance", "recipe_digest", "bytea")
	assertColumnType(t, ctx, db, schemaName, "provider_catalog_revisions", "snapshot_bytes", "bytea")
	assertColumnType(t, ctx, db, schemaName, "provider_catalog_revisions", "snapshot_digest", "bytea")
	assertColumnType(t, ctx, db, schemaName, "provider_credentials", "credential_mode", "text")
	assertColumnType(t, ctx, db, schemaName, "management_idempotency", "resource_id", "text")
	assertColumnType(t, ctx, db, schemaName, "management_idempotency", "hmac_version", "text")
	assertColumnType(t, ctx, db, schemaName, "management_installation_state", "bootstrap_idempotency_hmac_version", "text")
	assertColumnType(t, ctx, db, schemaName, "management_installation_state", "bootstrap_response_ciphertext", "bytea")
	assertColumnType(t, ctx, db, schemaName, "management_installation_state", "recovery_token_digest", "bytea")
	assertColumnType(t, ctx, db, schemaName, "management_installation_state", "recovery_request_digest", "bytea")
	assertColumnType(t, ctx, db, schemaName, "management_installation_state", "recovery_receipt", "jsonb")
	assertColumnType(t, ctx, db, schemaName, "agent_artifacts", "access_scope", "jsonb")
	assertColumnType(t, ctx, db, schemaName, "inference_replays", "served_models", "jsonb")
	assertColumnType(t, ctx, db, schemaName, "inference_outcomes", "request_digest", "bytea")
	assertColumnType(t, ctx, db, schemaName, "inference_outcome_projection_heads", "desired_revision", "bigint")
	assertColumnType(t, ctx, db, schemaName, "inference_outcome_projection_snapshots", "snapshot", "jsonb")
	for _, table := range []string{"usage_events", "usage_dispatches", "usage_dispatch_attempts"} {
		var kind string
		if err := db.QueryRowContext(ctx, `SELECT relkind::text FROM pg_class
WHERE oid=to_regclass($1)`, table).Scan(&kind); err != nil {
			t.Fatalf("read %s relation kind: %v", table, err)
		}
		if kind != "p" {
			t.Fatalf("%s relation kind = %q, want partitioned table", table, kind)
		}
	}
	var currentPartitionCount int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM usage_partition_months
WHERE month_start=date_trunc('month',current_date)::date AND state='active'`).Scan(&currentPartitionCount); err != nil {
		t.Fatalf("read current usage partition registry: %v", err)
	}
	if currentPartitionCount != 1 {
		t.Fatalf("current usage partition count = %d, want 1", currentPartitionCount)
	}
	assertManagementIdentitySeeds(t, ctx, db)
	var migrationCount int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM router_management_schema_migrations`).Scan(&migrationCount); err != nil {
		t.Fatalf("read migration ledger: %v", err)
	}
	migrations, err := Migrations()
	if err != nil {
		t.Fatal(err)
	}
	if migrationCount != len(migrations) {
		t.Fatalf("migration ledger count = %d, want %d", migrationCount, len(migrations))
	}
	var appliedName string
	var appliedDigest []byte
	if err := db.QueryRowContext(ctx, `SELECT name,content_digest
FROM router_management_schema_migrations WHERE version=$1`, migrations[0].Version).Scan(
		&appliedName, &appliedDigest,
	); err != nil {
		t.Fatalf("read migration ledger digest: %v", err)
	}
	if appliedName != migrations[0].Name || !bytes.Equal(appliedDigest, migrations[0].Digest[:]) {
		t.Fatalf("migration ledger identity = (%q,%x), want (%q,%x)",
			appliedName, appliedDigest, migrations[0].Name, migrations[0].Digest)
	}
}

func TestMigratorRejectsIncompleteLedgerSchema(t *testing.T) {
	databaseURL := migrationTestDatabaseURL(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	db, schemaName := isolatedMigrationDatabase(t, ctx, databaseURL)
	if _, err := db.ExecContext(ctx, `
CREATE TABLE router_management_schema_migrations (
  version BIGINT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
)`); err != nil {
		t.Fatalf("create incomplete migration ledger: %v", err)
	}

	if err := (Migrator{DB: db}).Apply(ctx); err == nil {
		t.Fatal("Migrator accepted a ledger without immutable content digests")
	}
	var digestColumnCount int
	if err := db.QueryRowContext(ctx, `
SELECT count(*)
FROM information_schema.columns
WHERE table_schema=$1
  AND table_name='router_management_schema_migrations'
  AND column_name='content_digest'`, schemaName).Scan(&digestColumnCount); err != nil {
		t.Fatalf("inspect incomplete migration ledger: %v", err)
	}
	if digestColumnCount != 0 {
		t.Fatal("Migrator rewrote an incomplete migration ledger")
	}
}

func assertManagementIdentitySeeds(t *testing.T, ctx context.Context, db *sql.DB) {
	t.Helper()
	var roles, distinctIDs, validDigests int
	if err := db.QueryRowContext(ctx, `SELECT count(*), count(DISTINCT id),
 count(*) FILTER (WHERE builtin AND status='active' AND namespace_id IS NULL
   AND revision=1 AND octet_length(permissions_digest)=32)
FROM management_roles`).Scan(&roles, &distinctIDs, &validDigests); err != nil {
		t.Fatalf("read built-in Management role seed: %v", err)
	}
	if roles != 8 || distinctIDs != 8 || validDigests != 8 {
		t.Fatalf("built-in Management role seed = (%d,%d,%d), want (8,8,8)", roles, distinctIDs, validDigests)
	}
	for name, contract := range map[string]struct {
		permissions []string
		digest      string
	}{
		"viewer": {
			permissions: []string{"agent.read", "provider_catalog.read", "routing.read", "tool.read"},
			digest:      "457a9204a91594a24e10ce7ab98b16fe61ec569104e7f25b9fadfe5e78f08ceb",
		},
		"consumer": {
			permissions: []string{"access_policy.read", "agent.read", "agent.use", "delegation.use", "key.read", "operation.read", "quota.read", "rate_policy.read", "routing_context.read", "team.read", "tool.invoke", "tool.read", "usage.read", "user.read"},
			digest:      "42f87d9c0231abac6d6f5256f4c40d5d5789f9c9d8739c264785d0bf58560fd6",
		},
	} {
		var permissionJSON []byte
		var digest string
		if err := db.QueryRowContext(ctx, `SELECT permissions,encode(permissions_digest,'hex')
FROM management_roles WHERE name=$1 AND builtin=TRUE`, name).Scan(&permissionJSON, &digest); err != nil {
			t.Fatalf("read %s built-in Management role: %v", name, err)
		}
		var permissions []string
		if err := json.Unmarshal(permissionJSON, &permissions); err != nil {
			t.Fatalf("decode %s built-in Management permissions: %v", name, err)
		}
		if strings.Join(permissions, "\x00") != strings.Join(contract.permissions, "\x00") || digest != contract.digest {
			t.Fatalf("%s built-in Management role = permissions %v digest %q", name, permissions, digest)
		}
	}
	var installationVersion, installationRevision, sessionVersion, sessionRevision int64
	var installationCount, sessionCount int
	if err := db.QueryRowContext(ctx, `SELECT
 (SELECT count(*) FROM management_installation_state WHERE singleton=TRUE),
 (SELECT seed_version FROM management_installation_state WHERE singleton=TRUE),
 (SELECT revision FROM management_installation_state WHERE singleton=TRUE),
 (SELECT count(*) FROM management_session_policy WHERE singleton=TRUE),
 (SELECT seed_version FROM management_session_policy WHERE singleton=TRUE),
 (SELECT revision FROM management_session_policy WHERE singleton=TRUE)`).Scan(
		&installationCount, &installationVersion, &installationRevision,
		&sessionCount, &sessionVersion, &sessionRevision,
	); err != nil {
		t.Fatalf("read Management singleton seeds: %v", err)
	}
	if installationCount != 1 || installationVersion != 1 || installationRevision != 1 ||
		sessionCount != 1 || sessionVersion != 1 || sessionRevision != 1 {
		t.Fatalf("invalid Management singleton seeds: installation=(%d,%d,%d) session=(%d,%d,%d)",
			installationCount, installationVersion, installationRevision,
			sessionCount, sessionVersion, sessionRevision)
	}
}

func migrationTestDatabaseURL(t *testing.T) string {
	t.Helper()
	databaseURL := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if databaseURL == "" {
		databaseURL = os.Getenv("VLLM_SR_ACCESS_CONTROL_TEST_DATABASE_URL")
	}
	if databaseURL == "" {
		t.Skip("PostgreSQL migration test database is not configured")
	}
	return databaseURL
}

func isolatedMigrationDatabase(
	t *testing.T,
	ctx context.Context,
	databaseURL string,
) (*sql.DB, string) {
	t.Helper()
	adminDB, isolatedMigrationDatabaseErr := sql.Open("postgres", databaseURL)
	if isolatedMigrationDatabaseErr != nil {
		t.Fatal(isolatedMigrationDatabaseErr)
	}
	t.Cleanup(func() { _ = adminDB.Close() })
	if err := adminDB.PingContext(ctx); err != nil {
		t.Fatalf("ping PostgreSQL: %v", err)
	}
	schemaName := "vsr_migration_test_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := adminDB.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schemaName)); err != nil {
		t.Fatalf("create isolated migration schema: %v", err)
	}
	t.Cleanup(func() {
		cleanupContext, cleanupCancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cleanupCancel()
		_, _ = adminDB.ExecContext(cleanupContext, "DROP SCHEMA "+pq.QuoteIdentifier(schemaName)+" CASCADE")
	})

	scopedURL, isolatedMigrationDatabaseErr := postgresURLWithSearchPath(databaseURL, schemaName)
	if isolatedMigrationDatabaseErr != nil {
		t.Fatal(isolatedMigrationDatabaseErr)
	}
	db, isolatedMigrationDatabaseErr := sql.Open("postgres", scopedURL)
	if isolatedMigrationDatabaseErr != nil {
		t.Fatal(isolatedMigrationDatabaseErr)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db, schemaName
}

func postgresURLWithSearchPath(databaseURL, schemaName string) (string, error) {
	if !strings.Contains(databaseURL, "://") {
		return databaseURL + " search_path=" + schemaName, nil
	}
	parsed, err := url.Parse(databaseURL)
	if err != nil {
		return "", err
	}
	query := parsed.Query()
	query.Set("search_path", schemaName)
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}

func assertColumnType(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	schemaName string,
	tableName string,
	columnName string,
	want string,
) {
	t.Helper()
	var got string
	if err := db.QueryRowContext(ctx, `SELECT data_type
FROM information_schema.columns
WHERE table_schema = $1 AND table_name = $2 AND column_name = $3`,
		schemaName, tableName, columnName,
	).Scan(&got); err != nil {
		t.Fatalf("read %s.%s column type: %v", tableName, columnName, err)
	}
	if got != want {
		t.Fatalf("%s.%s type = %q, want %q", tableName, columnName, got, want)
	}
}
