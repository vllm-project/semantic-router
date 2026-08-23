package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

func TestEmbeddedMigrationsAreOrderedAndCoverAuthorities(t *testing.T) {
	migrations, err := Migrations()
	if err != nil {
		t.Fatal(err)
	}
	if len(migrations) != 1 {
		t.Fatalf("migration count = %d, want one clean v0.4 baseline", len(migrations))
	}
	if migrations[0].Version != 1 || migrations[0].Name != "0001_v04_control_plane.sql" {
		t.Fatalf("baseline migration = (%d, %q), want (1, %q)",
			migrations[0].Version, migrations[0].Name, "0001_v04_control_plane.sql")
	}
	var schema strings.Builder
	for i, migration := range migrations {
		if i > 0 && migrations[i-1].Version >= migration.Version {
			t.Fatalf("migration versions are not strictly increasing: %d then %d", migrations[i-1].Version, migration.Version)
		}
		schema.WriteString(migration.SQL)
	}
	for _, table := range []string{
		"access_namespaces", "access_subjects", "access_api_keys",
		"access_api_key_credentials", "access_policies", "rate_limit_policies",
		"management_principals", "management_sessions", "management_invitations",
		"management_backchannel_logout_replays",
		"provider_credentials", "routing_models", "routing_recipes",
		"routing_recipe_distributions", "routing_recipe_provenance",
		"routing_entrypoints", "routing_snapshots", "policy_outbox",
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
		if !strings.Contains(schema.String(), "CREATE TABLE "+table) {
			t.Errorf("embedded migrations do not create %s", table)
		}
	}
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
		"access_scope JSONB NOT NULL CHECK (jsonb_typeof(access_scope) = 'object')",
	} {
		if !strings.Contains(schema.String(), contract) {
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

func TestAppliedMigrationsMustMatchEmbeddedPrefix(t *testing.T) {
	migrations := []Migration{
		{Version: 1, Name: "0001_baseline.sql"},
		{Version: 2, Name: "0002_forward.sql"},
	}
	for name, test := range map[string]struct {
		applied map[int64]string
		wantErr bool
	}{
		"empty":          {applied: map[int64]string{}},
		"prefix":         {applied: map[int64]string{1: "0001_baseline.sql"}},
		"complete":       {applied: map[int64]string{1: "0001_baseline.sql", 2: "0002_forward.sql"}},
		"renamed":        {applied: map[int64]string{1: "0001_preview.sql"}, wantErr: true},
		"unknown":        {applied: map[int64]string{3: "0003_future.sql"}, wantErr: true},
		"missing prefix": {applied: map[int64]string{2: "0002_forward.sql"}, wantErr: true},
	} {
		t.Run(name, func(t *testing.T) {
			err := validateAppliedMigrations(migrations, test.applied)
			if (err != nil) != test.wantErr {
				t.Fatalf("validateAppliedMigrations() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}

func TestBaselineSeedsLeastPrivilegeBuiltInRoles(t *testing.T) {
	migrations, err := Migrations()
	if err != nil {
		t.Fatal(err)
	}
	if len(migrations) != 1 {
		t.Fatalf("migration count = %d, want one baseline", len(migrations))
	}
	baseline := migrations[0].SQL
	for _, contract := range []string{
		"10000000-0000-5000-8000-000000000007",
		"10000000-0000-5000-8000-000000000008",
		"viewer built-in role seed does not match its least-privilege contract",
		"consumer built-in role seed does not match its least-privilege contract",
	} {
		if !strings.Contains(baseline, contract) {
			t.Fatalf("v0.4 baseline does not contain %q", contract)
		}
	}
	if strings.Contains(baseline, "UPDATE management_roles") {
		t.Fatal("v0.4 baseline contains an unreleased built-in role correction")
	}
	if strings.Contains(baseline, "preview usage schema") {
		t.Fatal("v0.4 baseline contains a preview-schema upgrade guard")
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
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM router_control_plane_schema_migrations`).Scan(&migrationCount); err != nil {
		t.Fatalf("read migration ledger: %v", err)
	}
	migrations, err := Migrations()
	if err != nil {
		t.Fatal(err)
	}
	if migrationCount != len(migrations) {
		t.Fatalf("migration ledger count = %d, want %d", migrationCount, len(migrations))
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
			permissions: []string{"provider_catalog.read", "routing.read"},
			digest:      "26ad6974c2b80418b7df50daa1de9582b922bff92b44c2f2f7acc0d98fd6cab0",
		},
		"consumer": {
			permissions: []string{"access_policy.read", "delegation.use", "key.read", "operation.read", "quota.read", "rate_policy.read", "routing_context.read", "team.read", "usage.read", "user.read"},
			digest:      "a5b59e087f29c4a510d034d9ca44565805482ca1b4c477fb0d96b3919eb4c1fd",
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
