package accesspublisher

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
)

func TestPostgresDesiredReaderCompilesCompleteAccessAndRoutingProjection(t *testing.T) {
	db := postgresIntegrationDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	namespaceID, partition, keyID, entrypointID := insertCompleteDesiredState(t, ctx, db)
	reader, err := NewPostgresDesiredStateReader(db)
	if err != nil {
		t.Fatal(err)
	}
	state, err := reader.LoadDesiredState(ctx, namespaceID, 1)
	if err != nil {
		t.Fatal(err)
	}
	if state.Namespace.QuotaPartitionID != accesscontrol.QuotaPartitionID(partition) || len(state.Keys) != 1 ||
		len(state.Credentials) != 1 || len(state.ProviderCredentials) != 1 ||
		len(state.ProviderCredentials[0].Versions) != 1 || len(state.Routing.Models) != 1 || len(state.Routing.Recipes) != 1 ||
		len(state.Routing.Entrypoints) != 1 {
		t.Fatalf("desired state is incomplete: %+v", state)
	}
	if state.Routing.Entrypoints[0].Revision != 1 {
		t.Fatalf("entrypoint immutable revision = %d, want 1", state.Routing.Entrypoints[0].Revision)
	}
	publication, err := Compile(state)
	if err != nil {
		t.Fatal(err)
	}
	if len(publication.Access) != 1 || publication.Access[0].KeyID != keyID || len(publication.Credentials) != 1 ||
		len(publication.ProviderCredentials) != 1 {
		t.Fatalf("compiled access publication = %+v", publication.Access)
	}
	providerCredentialID := state.ProviderCredentials[0].Credential.ID
	if _, exists := publication.Manifest.ProviderCredentials[providerCredentialID]; !exists {
		t.Fatalf("compiled publication lacks provider credential %s", providerCredentialID)
	}
	if _, exists := publication.Routing.ResourceDigests[routingResourceKey("entrypoint", entrypointID)]; !exists {
		t.Fatalf("compiled routing publication lacks entrypoint %s", entrypointID)
	}
	insertRevision(t, ctx, db, namespaceID, 2, 11)
	if _, err := reader.LoadDesiredState(ctx, namespaceID, 1); !errors.Is(err, ErrSuperseded) {
		t.Fatalf("stale desired state load = %v", err)
	}
}

func TestPostgresOutboxIsTransactionalCoalescedAndRevisionFenced(t *testing.T) {
	db := postgresIntegrationDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	namespaceID := uuid.NewString()
	partition := "partition-" + uuid.NewString()
	if _, err := db.ExecContext(ctx, `INSERT INTO access_namespaces
  (id, name, quota_partition_id, billing_currency, status, revision, runtime_epoch)
VALUES ($1, $2, $3, 'USD', 'active', 1, 7)`, namespaceID, "namespace-"+namespaceID, partition); err != nil {
		t.Fatal(err)
	}
	insertRevision(t, ctx, db, namespaceID, 1, 7)

	tx, testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr := db.BeginTx(ctx, nil)
	if testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr != nil {
		t.Fatal(testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr)
	}
	row1 := uuid.NewString()
	insertOutbox(t, ctx, tx, row1, namespaceID, 1)
	store, testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr := NewPostgresStore(db, PostgresStoreOptions{Projector: "access-publisher-it"})
	if testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr != nil {
		t.Fatal(testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr)
	}
	if _, err := store.ClaimLatest(ctx, "worker-a", 10*time.Second); !errors.Is(err, ErrNoWork) {
		_ = tx.Rollback()
		t.Fatalf("uncommitted outbox was visible: %v", err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	for revision := int64(2); revision <= 3; revision++ {
		insertRevision(t, ctx, db, namespaceID, revision, 7)
		insertOutbox(t, ctx, db, uuid.NewString(), namespaceID, revision)
	}
	batch, testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr := store.ClaimLatest(ctx, "worker-a", 10*time.Second)
	if testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr != nil {
		t.Fatal(testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr)
	}
	if batch.DesiredRevision != 3 || len(batch.RowIDs) != 3 || batch.QuotaPartition != partition {
		t.Fatalf("coalesced batch = %+v", batch)
	}
	publication := postgresPublication(t, namespaceID, partition, batch.DesiredRevision, batch.RuntimeEpoch)
	if err := store.RecordStaged(ctx, batch, publication); err != nil {
		t.Fatal(err)
	}
	activated := 0
	if err := store.WithRevisionFence(ctx, batch, func(context.Context) error {
		activated++
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if activated != 1 {
		t.Fatalf("activation callback count = %d", activated)
	}
	var appliedRows int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM policy_outbox
WHERE namespace_id = $1 AND state = 'applied'`, namespaceID).Scan(&appliedRows); err != nil {
		t.Fatal(err)
	}
	if appliedRows != 3 {
		t.Fatalf("applied outbox rows = %d", appliedRows)
	}
	applied, testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr := store.Applied(ctx, namespaceID)
	if testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr != nil {
		t.Fatal(testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr)
	}
	if applied.DesiredRevision != 3 || applied.RuntimeEpoch != 7 || applied.QuotaPartition != partition || applied.RoutingDigest == "" {
		t.Fatalf("applied state = %+v", applied)
	}

	insertRevision(t, ctx, db, namespaceID, 4, 7)
	insertOutbox(t, ctx, db, uuid.NewString(), namespaceID, 4)
	stale, testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr := store.ClaimLatest(ctx, "worker-a", 10*time.Second)
	if testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr != nil {
		t.Fatal(testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr)
	}
	insertRevision(t, ctx, db, namespaceID, 5, 7)
	insertOutbox(t, ctx, db, uuid.NewString(), namespaceID, 5)
	called := false
	testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr = store.WithRevisionFence(ctx, stale, func(context.Context) error {
		called = true
		return nil
	})
	if !errors.Is(testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr, ErrSuperseded) || called {
		t.Fatalf("stale revision fence = %v, callback=%t", testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr, called)
	}
	if err := store.Release(ctx, stale, testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr, 0); err != nil {
		t.Fatal(err)
	}
	latest, testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr := store.ClaimLatest(ctx, "worker-b", 10*time.Second)
	if testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr != nil {
		t.Fatal(testPostgresOutboxIsTransactionalCoalescedAndRevisionFencedErr)
	}
	if latest.DesiredRevision != 5 || len(latest.RowIDs) != 2 {
		t.Fatalf("latest coalesced batch after supersede = %+v", latest)
	}
}

func TestPostgresLaterFullRevisionSupersedesFailedOutboxLineage(t *testing.T) {
	db := postgresIntegrationDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	namespaceID := uuid.NewString()
	partition := "partition-" + uuid.NewString()
	if _, err := db.ExecContext(ctx, `INSERT INTO access_namespaces
  (id, name, quota_partition_id, billing_currency, status, revision, runtime_epoch)
VALUES ($1, $2, $3, 'USD', 'active', 1, 7)`, namespaceID, "namespace-"+namespaceID, partition); err != nil {
		t.Fatal(err)
	}
	store, testPostgresLaterFullRevisionSupersedesFailedOutboxLineageErr := NewPostgresStore(db, PostgresStoreOptions{Projector: "access-publisher-failed-lineage-it"})
	if testPostgresLaterFullRevisionSupersedesFailedOutboxLineageErr != nil {
		t.Fatal(testPostgresLaterFullRevisionSupersedesFailedOutboxLineageErr)
	}
	firstID := uuid.NewString()
	insertRevision(t, ctx, db, namespaceID, 1, 7)
	insertOutbox(t, ctx, db, firstID, namespaceID, 1)
	failed, testPostgresLaterFullRevisionSupersedesFailedOutboxLineageErr := store.ClaimLatest(ctx, "worker-a", 10*time.Second)
	if testPostgresLaterFullRevisionSupersedesFailedOutboxLineageErr != nil {
		t.Fatal(testPostgresLaterFullRevisionSupersedesFailedOutboxLineageErr)
	}
	if err := store.Fail(ctx, failed, ErrStagedCorrupt); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ClaimLatest(ctx, "worker-b", 10*time.Second); !errors.Is(err, ErrNoWork) {
		t.Fatalf("failed revision retried without a newer desired revision: %v", err)
	}

	secondID := uuid.NewString()
	insertRevision(t, ctx, db, namespaceID, 2, 7)
	insertOutbox(t, ctx, db, secondID, namespaceID, 2)
	repair, testPostgresLaterFullRevisionSupersedesFailedOutboxLineageErr := store.ClaimLatest(ctx, "worker-b", 10*time.Second)
	if testPostgresLaterFullRevisionSupersedesFailedOutboxLineageErr != nil {
		t.Fatal(testPostgresLaterFullRevisionSupersedesFailedOutboxLineageErr)
	}
	if repair.DesiredRevision != 2 || len(repair.RowIDs) != 2 {
		t.Fatalf("repair batch did not retain failed revision lineage: %+v", repair)
	}
	publication := postgresPublication(t, namespaceID, partition, repair.DesiredRevision, repair.RuntimeEpoch)
	if err := store.RecordStaged(ctx, repair, publication); err != nil {
		t.Fatal(err)
	}
	if err := store.WithRevisionFence(ctx, repair, func(context.Context) error { return nil }); err != nil {
		t.Fatal(err)
	}
	var applied int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM policy_outbox
WHERE id = ANY($1) AND state = 'applied'`, pq.Array([]string{firstID, secondID})).Scan(&applied); err != nil {
		t.Fatal(err)
	}
	if applied != 2 {
		t.Fatalf("applied lineage rows = %d, want 2", applied)
	}
}

func insertCompleteDesiredState(t testing.TB, ctx context.Context, db *sql.DB) (string, string, string, string) {
	t.Helper()
	ids := map[string]string{}
	for _, name := range []string{
		"namespace", "user", "team", "key", "credential", "access_policy", "access_binding",
		"rate_policy", "rate_rule", "rate_binding", "provider_credential", "provider_version",
		"unreferenced_provider_credential",
		"model", "backend", "recipe", "decision",
		"entrypoint", "rule",
	} {
		ids[name] = uuid.NewString()
	}
	ids["model"] = "model_chat"
	ids["recipe"] = "recipe_chat"
	ids["decision"] = "decision_chat"
	ids["entrypoint"] = "entrypoint_chat"
	ids["rule"] = "rule_chat"
	partition := "partition-" + uuid.NewString()
	now := fixtureTime
	statements := completeDesiredStateStatements(ids, partition, now)
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback() }()
	for index, statement := range statements {
		if _, err := tx.ExecContext(ctx, statement.query, statement.args...); err != nil {
			t.Fatalf("insert complete desired state statement %d: %v", index, err)
		}
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	return ids["namespace"], partition, ids["key"], ids["entrypoint"]
}

type desiredStateStatement struct {
	query string
	args  []any
}

func completeDesiredStateStatements(ids map[string]string, partition string, now time.Time) []desiredStateStatement {
	statements := completeAccessStateStatements(ids, partition, now)
	statements = append(statements, completeProviderStateStatements(ids, now)...)
	return append(statements, completeRoutingStateStatements(ids, now)...)
}

func completeAccessStateStatements(ids map[string]string, partition string, now time.Time) []desiredStateStatement {
	return []desiredStateStatement{
		{`INSERT INTO access_namespaces
  (id, name, quota_partition_id, billing_currency, status, revision, runtime_epoch, created_at, updated_at)
VALUES ($1, $2, $3, 'USD', 'active', 1, 11, $4, $4)`, []any{ids["namespace"], "namespace-" + ids["namespace"], partition, now}},
		{`INSERT INTO access_subjects(namespace_id, id, kind, created_at) VALUES
  ($1, $2, 'user', $5), ($1, $3, 'team', $5), ($1, $4, 'api_key', $5)`, []any{ids["namespace"], ids["user"], ids["team"], ids["key"], now}},
		{`INSERT INTO access_users(id, namespace_id, email, display_name, status, revision, created_at, updated_at)
VALUES ($1, $2, 'publisher@example.com', 'Publisher', 'active', 1, $3, $3)`, []any{ids["user"], ids["namespace"], now}},
		{`INSERT INTO access_teams(id, namespace_id, name, status, revision, created_at, updated_at)
VALUES ($1, $2, 'Publisher team', 'active', 1, $3, $3)`, []any{ids["team"], ids["namespace"], now}},
		{`INSERT INTO access_team_memberships
  (namespace_id, team_id, user_id, role, status, revision, created_at, updated_at)
VALUES ($1, $2, $3, 'member', 'active', 1, $4, $4)`, []any{ids["namespace"], ids["team"], ids["user"], now}},
		{`INSERT INTO access_policies(id, namespace_id, name, status, revision, created_at, updated_at)
VALUES ($1, $2, 'Publisher access', 'active', 1, $3, $3)`, []any{ids["access_policy"], ids["namespace"], now}},
		{`INSERT INTO access_policy_grants
  (policy_id, resource_type, resource_id, permission, effect, created_at)
VALUES ($1, 'entrypoint', $2, 'invoke', 'allow', $3),
       ($1, 'entrypoint', $2, 'discover', 'allow', $3)`, []any{ids["access_policy"], ids["entrypoint"], now}},
		{`INSERT INTO rate_limit_policies(id, namespace_id, name, status, revision, created_at, updated_at)
VALUES ($1, $2, 'Publisher quota', 'active', 1, $3, $3)`, []any{ids["rate_policy"], ids["namespace"], now}},
		{`INSERT INTO rate_limit_rules
  (id, policy_id, metric, algorithm, limit_value, window_seconds, accounting, enforcement, ordinal, created_at)
VALUES ($1, $2, 'requests', 'sliding_log', 100, 60, 'request', 'enforce', 0, $3)`, []any{ids["rate_rule"], ids["rate_policy"], now}},
		{`INSERT INTO access_api_keys
  (id, namespace_id, name, owner_user_id, context_team_id, status, policy_epoch, delegation_epoch, revision, created_at, updated_at)
VALUES ($1, $2, 'Publisher key', $3, $4, 'active', 1, 1, 1, $5, $5)`, []any{ids["key"], ids["namespace"], ids["user"], ids["team"], now}},
		{`INSERT INTO access_api_key_credentials
  (id, namespace_id, api_key_id, kid, secret_hmac, pepper_version, status, not_before, created_at)
VALUES ($1, $2, $3, 'publisherkid0001', $4, 'pepper-1', 'active', $5, $5)`, []any{ids["credential"], ids["namespace"], ids["key"], bytes.Repeat([]byte{0x2a}, 32), now}},
		{`INSERT INTO access_policy_bindings
  (id, namespace_id, policy_id, subject_id, status, revision, created_at, updated_at)
VALUES ($1, $2, $3, $4, 'active', 1, $5, $5)`, []any{ids["access_binding"], ids["namespace"], ids["access_policy"], ids["user"], now}},
		{`INSERT INTO rate_limit_bindings
  (id, namespace_id, policy_id, subject_id, binding_mode, quota_partition_id, status, revision, created_at, updated_at)
VALUES ($1, $2, $3, $4, 'allocation', $5, 'active', 1, $6, $6)`, []any{ids["rate_binding"], ids["namespace"], ids["rate_policy"], ids["team"], partition, now}},
	}
}

func completeProviderStateStatements(ids map[string]string, now time.Time) []desiredStateStatement {
	return []desiredStateStatement{
		{`INSERT INTO provider_catalog_revisions
  (revision, snapshot_bytes, snapshot_digest, integration_references, catalog, required_wire_formats,
   required_credential_adapters, required_discovery_adapters)
VALUES ('sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  convert_to('{}', 'UTF8'), decode('44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a', 'hex'),
  '[]'::jsonb, '{}'::jsonb, '["openai.chat.v1"]'::jsonb, '["bearer"]'::jsonb, '[]'::jsonb)`, nil},
		{
			`INSERT INTO provider_credentials
  (id, namespace_id, name, provider_id, credential_mode, credential_adapter_id,
   provider_catalog_revision, normalized_origin, status, active_version_id, revision, created_at, updated_at)
VALUES ($1, $2, 'Publisher provider credential', 'openai-compatible', 'required', 'bearer',
  'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  'https://models.example/v1', 'active', $3, 1, $4, $4)`,
			[]any{ids["provider_credential"], ids["namespace"], ids["provider_version"], now},
		},
		{
			`INSERT INTO provider_credential_versions
  (id, namespace_id, provider_credential_id, secret_ciphertext, ciphertext_nonce,
   kek_version, status, not_before, created_at)
VALUES ($1, $2, $3, $4, $5, 'provider-kek-v1', 'active', $6, $6)`,
			[]any{
				ids["provider_version"], ids["namespace"], ids["provider_credential"],
				bytes.Repeat([]byte{0x44}, 48), bytes.Repeat([]byte{0x55}, 12), now,
			},
		},
		{
			`INSERT INTO provider_credentials
  (id, namespace_id, name, provider_id, credential_mode, credential_adapter_id,
   provider_catalog_revision, normalized_origin, status, revision, created_at, updated_at, deleted_at)
VALUES ($1, $2, 'Unreferenced provider credential', 'openai-compatible', 'required', 'bearer',
  'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  'https://unused.example/v1', 'deleted', 1, $3, $3, $3)`,
			[]any{ids["unreferenced_provider_credential"], ids["namespace"], now},
		},
	}
}

func completeRoutingStateStatements(ids map[string]string, now time.Time) []desiredStateStatement {
	return []desiredStateStatement{
		{`INSERT INTO routing_models
  (id, namespace_id, name, aliases, status, current_revision, revision, created_at, updated_at)
VALUES ($1, $2, 'local/chat', '["local/chat"]'::jsonb, 'active', 1, 1, $3, $3)`, []any{ids["model"], ids["namespace"], now}},
		{`INSERT INTO routing_model_revisions
  (model_id, revision, provider_catalog_revision, name, aliases, capabilities, reasoning, loras, execution, pricing, content_digest, created_at)
VALUES ($1, 1, 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  'local/chat', '["local/chat"]'::jsonb, '["text"]'::jsonb, '{}'::jsonb, '[]'::jsonb,
  '{"maxRetries":2,"requestTimeout":"30s","streamTimeout":"60s"}'::jsonb,
  '{"inputCostPerMillionTokens":"0.25","outputCostPerMillionTokens":"1","cacheReadCostPerMillionTokens":"0.25","cacheWriteCostPerMillionTokens":"0.25"}'::jsonb,
  $2, $3)`, []any{ids["model"], bytes.Repeat([]byte{0x11}, 32), now}},
		{
			`INSERT INTO routing_model_backends
  (id, namespace_id, model_id, model_revision, ordinal, provider_id, wire_format,
   normalized_origin, provider_model_id, connection, provider_credential_id, weight)
VALUES ($1, $2, $3, 1, 0, 'openai-compatible', 'openai.chat.v1',
  'https://models.example/v1', 'chat', '{"path":"/chat/completions"}'::jsonb, $4, 1)`,
			[]any{ids["backend"], ids["namespace"], ids["model"], ids["provider_credential"]},
		},
		{`INSERT INTO routing_recipes
  (id, namespace_id, name, status, current_revision, revision, created_at, updated_at)
VALUES ($1, $2, 'Chat', 'active', 1, 1, $3, $3)`, []any{ids["recipe"], ids["namespace"], now}},
		{`INSERT INTO routing_recipe_revisions(recipe_id, revision, name, document, content_digest, created_at)
VALUES ($1, 1, 'Chat', '{"signals":[],"decisions":[]}'::jsonb, $2, $3)`, []any{ids["recipe"], bytes.Repeat([]byte{0x22}, 32), now}},
		{`INSERT INTO routing_recipe_decisions
  (recipe_id, recipe_revision, decision_id, name, dispatch_cardinality, ordinal, capabilities)
VALUES ($1, 1, $2, 'Chat', 'single', 0, '{}'::jsonb)`, []any{ids["recipe"], ids["decision"]}},
		{`INSERT INTO routing_entrypoints
	  (id, namespace_id, name, aliases, status, current_revision, published_revision, revision, created_at, updated_at)
VALUES ($1, $2, 'Chat', '["vllm-sr/chat"]'::jsonb, 'active', 1, 1, 6, $3, $3)`, []any{ids["entrypoint"], ids["namespace"], now}},
		{`INSERT INTO routing_entrypoint_revisions
  (entrypoint_id, revision, name, aliases, content_digest, created_at)
VALUES ($1, 1, 'Chat', '["vllm-sr/chat"]'::jsonb, $2, $3)`, []any{ids["entrypoint"], bytes.Repeat([]byte{0x33}, 32), now}},
		{`INSERT INTO routing_entrypoint_rules
	  (id, entrypoint_id, entrypoint_revision, name, ordinal, matchers, recipe_id, recipe_revision)
VALUES ($1, $2, 1, 'Chat', 0, '[]'::jsonb, $3, 1)`, []any{ids["rule"], ids["entrypoint"], ids["recipe"]}},
		{`INSERT INTO routing_decision_assignments
  (entrypoint_id, entrypoint_revision, rule_id, recipe_id, recipe_revision, decision_id)
VALUES ($1, 1, $2, $3, 1, $4)`, []any{ids["entrypoint"], ids["rule"], ids["recipe"], ids["decision"]}},
		{`INSERT INTO routing_assignment_models
  (entrypoint_id, entrypoint_revision, rule_id, decision_id, ordinal, model_id, model_revision, priority, weight)
VALUES ($1, 1, $2, $3, 0, $4, 1, 0, 1)`, []any{ids["entrypoint"], ids["rule"], ids["decision"], ids["model"]}},
		{`INSERT INTO policy_revisions(namespace_id, revision, runtime_epoch, reason, created_at)
VALUES ($1, 1, 11, 'complete desired state', $2)`, []any{ids["namespace"], now}},
	}
}

type sqlExecutor interface {
	ExecContext(context.Context, string, ...any) (sql.Result, error)
}

func insertRevision(t testing.TB, ctx context.Context, db sqlExecutor, namespaceID string, revision, epoch int64) {
	t.Helper()
	if _, err := db.ExecContext(ctx, `INSERT INTO policy_revisions
  (namespace_id, revision, runtime_epoch, reason) VALUES ($1, $2, $3, 'integration test')`,
		namespaceID, revision, epoch); err != nil {
		t.Fatal(err)
	}
}

func insertOutbox(t testing.TB, ctx context.Context, db sqlExecutor, id, namespaceID string, revision int64) {
	t.Helper()
	if _, err := db.ExecContext(ctx, `INSERT INTO policy_outbox
  (id, namespace_id, desired_revision, aggregate_type, aggregate_id, operation, payload)
VALUES ($1, $2, $3, 'namespace', $4, 'publish', '{}'::jsonb)`, id, namespaceID, revision, namespaceID); err != nil {
		t.Fatal(err)
	}
}

func postgresPublication(t testing.TB, namespaceID, partition string, revision, epoch uint64) Publication {
	t.Helper()
	state := validDesiredState(revision, "100")
	state.Namespace.ID = accessNamespaceID(namespaceID)
	state.Namespace.QuotaPartitionID = accessQuotaPartitionID(partition)
	state.Namespace.RuntimeEpoch = epoch
	state.Keys = nil
	state.Credentials = nil
	state.Routing.NamespaceID = namespaceID
	modelID, recipeID, entrypointID := "model_chat", "recipe_chat", "entrypoint_chat"
	state.Routing.Models[0].ID = modelID
	state.Routing.Recipes[0].ID = recipeID
	state.Routing.Entrypoints[0].ID = entrypointID
	state.Routing.Entrypoints[0].Rules[0].RecipeID = recipeID
	assignmentSet := state.Routing.Entrypoints[0].Rules[0].Assignments["decision-chat"]
	assignmentSet.Models[0].ModelID = modelID
	state.Routing.Entrypoints[0].Rules[0].Assignments["decision-chat"] = assignmentSet
	publication, err := Compile(state)
	if err != nil {
		t.Fatalf("compile PostgreSQL publication: %v", err)
	}
	return publication
}

func accessNamespaceID(value string) accesscontrol.NamespaceID {
	return accesscontrol.NamespaceID(value)
}

func accessQuotaPartitionID(value string) accesscontrol.QuotaPartitionID {
	return accesscontrol.QuotaPartitionID(value)
}

func postgresIntegrationDatabase(t *testing.T) *sql.DB {
	t.Helper()
	dsn := os.Getenv("ACCESSPUBLISHER_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("ACCESSPUBLISHER_POSTGRES_DSN is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	t.Cleanup(cancel)
	admin, postgresIntegrationDatabaseErr := sql.Open("postgres", dsn)
	if postgresIntegrationDatabaseErr != nil {
		t.Fatal(postgresIntegrationDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatalf("ping PostgreSQL: %v", err)
	}
	schema := "access_publisher_it_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 15*time.Second)
		defer stop()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	scopedDSN, postgresIntegrationDatabaseErr := dsnWithSearchPath(dsn, schema)
	if postgresIntegrationDatabaseErr != nil {
		t.Fatal(postgresIntegrationDatabaseErr)
	}
	db, postgresIntegrationDatabaseErr := sql.Open("postgres", scopedDSN)
	if postgresIntegrationDatabaseErr != nil {
		t.Fatal(postgresIntegrationDatabaseErr)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := (controlpostgres.Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatalf("apply control-plane migrations: %v", err)
	}
	return db
}

func dsnWithSearchPath(dsn, schema string) (string, error) {
	parsed, err := url.Parse(dsn)
	if err != nil {
		return "", err
	}
	if parsed.Scheme != "postgres" && parsed.Scheme != "postgresql" {
		return "", fmt.Errorf("PostgreSQL DSN must be a URL")
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}
