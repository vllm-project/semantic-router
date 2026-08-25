package postgres

import (
	"context"
	"database/sql"
	"errors"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type coordinatorTestEnvironment struct {
	ctx           context.Context
	first, second *sql.DB
}

type lockOutcome struct {
	result   managementcommand.StoredResult
	replayed bool
	err      error
}

func TestCoordinatorSerializesReplicasAndRollbackDoesNotConsumeKey(t *testing.T) {
	environment := newCoordinatorTestEnvironment(t)
	environment.assertCrossReplicaReplay(t)
	environment.assertIdempotencyConflict(t)
	environment.assertRollbackDoesNotConsumeKey(t)
	codec := environment.assertCrossVersionReplay(t)
	environment.assertHMACVersionReadiness(t, codec)
	environment.assertStoredDigestsHideInput(t)
}

func newCoordinatorTestEnvironment(t *testing.T) coordinatorTestEnvironment {
	t.Helper()
	dsn := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if dsn == "" {
		dsn = os.Getenv("VLLM_SR_ACCESS_CONTROL_TEST_DATABASE_URL")
	}
	if dsn == "" {
		t.Skip("PostgreSQL Management command test database is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	first, second := isolatedCommandDatabases(t, ctx, dsn)
	if _, err := first.ExecContext(ctx, `CREATE TABLE management_idempotency (
  scope_kind TEXT NOT NULL CHECK (scope_kind IN ('cluster','namespace')),
  namespace_id UUID,
  principal_id UUID NOT NULL,
  endpoint TEXT NOT NULL,
  hmac_version TEXT NOT NULL,
  idempotency_key_digest BYTEA NOT NULL,
  request_digest BYTEA NOT NULL,
  operation_id UUID,
  resource_type TEXT,
  resource_id TEXT,
  resource_revision BIGINT,
  desired_revision BIGINT,
  response_status INTEGER NOT NULL,
  secret_response_ciphertext BYTEA,
  secret_response_nonce BYTEA,
  response_kek_version TEXT,
  secret_response_expires_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK ((scope_kind = 'cluster' AND namespace_id IS NULL) OR
         (scope_kind = 'namespace' AND namespace_id IS NOT NULL))
)`); err != nil {
		t.Fatal(err)
	}
	if _, err := first.ExecContext(ctx, `CREATE UNIQUE INDEX management_idempotency_cluster_identity_uq
  ON management_idempotency (principal_id, endpoint, hmac_version, idempotency_key_digest)
  WHERE scope_kind = 'cluster';
CREATE UNIQUE INDEX management_idempotency_namespace_identity_uq
  ON management_idempotency (namespace_id, principal_id, endpoint, hmac_version, idempotency_key_digest)
  WHERE scope_kind = 'namespace'`); err != nil {
		t.Fatal(err)
	}
	return coordinatorTestEnvironment{ctx: ctx, first: first, second: second}
}

func (environment coordinatorTestEnvironment) assertCrossReplicaReplay(t *testing.T) {
	t.Helper()
	command, _ := testCommand(t, []byte(`{"secret":"never-persist"}`), "cross-replica-0123456789")
	firstTx, err := environment.first.BeginTx(environment.ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, replayed, lockErr := Lock(environment.ctx, firstTx, command); lockErr != nil || replayed {
		t.Fatalf("first replica Lock() replayed=%t err=%v", replayed, lockErr)
	}
	started := make(chan struct{})
	finished := lockAndCommitAsync(environment.ctx, environment.second, command, started)
	<-started
	if completeErr := CompleteResource(environment.ctx, firstTx, command, managementcommand.ResourceResult{
		ResourceType: "provider_credential", ResourceID: testResourceID,
		ResourceRevision: 1, ResponseStatus: 201,
	}); completeErr != nil {
		_ = firstTx.Rollback()
		t.Fatal(completeErr)
	}
	if commitErr := firstTx.Commit(); commitErr != nil {
		t.Fatal(commitErr)
	}
	outcome := <-finished
	if outcome.err != nil || !outcome.replayed || outcome.result.Resource == nil ||
		outcome.result.Resource.ResourceRevision != 1 {
		t.Fatalf("second replica outcome = %#v", outcome)
	}
}

func lockAndCommitAsync(
	ctx context.Context,
	database *sql.DB,
	command managementcommand.Command,
	started chan<- struct{},
) <-chan lockOutcome {
	finished := make(chan lockOutcome, 1)
	go func() {
		transaction, err := database.BeginTx(ctx, nil)
		if err != nil {
			finished <- lockOutcome{err: err}
			return
		}
		if started != nil {
			close(started)
		}
		result, replayed, lockErr := Lock(ctx, transaction, command)
		if lockErr == nil {
			lockErr = transaction.Commit()
		} else {
			_ = transaction.Rollback()
		}
		finished <- lockOutcome{result: result, replayed: replayed, err: lockErr}
	}()
	return finished
}

func (environment coordinatorTestEnvironment) assertIdempotencyConflict(t *testing.T) {
	t.Helper()
	codec := newCoordinatorTestCodec(t, "command-v1", map[string][]byte{
		"command-v1": []byte(strings.Repeat("k", 32)),
	})
	conflict, err := codec.Bind(
		managementcommand.NamespaceCommandScope(testNamespaceID), testPrincipalID,
		"/management/v1/provider-credentials", "cross-replica-0123456789",
		[]byte(`{"secret":"different"}`), time.Now().UTC(), time.Now().UTC().Add(time.Hour),
	)
	if err != nil {
		t.Fatal(err)
	}
	transaction, err := environment.second.BeginTx(environment.ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, lockErr := Lock(environment.ctx, transaction, conflict); !errors.Is(lockErr, managementcommand.ErrConflict) {
		_ = transaction.Rollback()
		t.Fatalf("conflicting command error = %v", lockErr)
	}
	_ = transaction.Rollback()
}

func (environment coordinatorTestEnvironment) assertRollbackDoesNotConsumeKey(t *testing.T) {
	t.Helper()
	command, _ := testCommand(t, []byte(`{"name":"rollback"}`), "rollback-key-0123456789")
	transaction, err := environment.first.BeginTx(environment.ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, replayed, lockErr := Lock(environment.ctx, transaction, command); lockErr != nil || replayed {
		t.Fatalf("rollback winner Lock() replayed=%t err=%v", replayed, lockErr)
	}
	if rollbackErr := transaction.Rollback(); rollbackErr != nil {
		t.Fatal(rollbackErr)
	}
	retry, err := environment.second.BeginTx(environment.ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, replayed, lockErr := Lock(environment.ctx, retry, command); lockErr != nil || replayed {
		_ = retry.Rollback()
		t.Fatalf("post-rollback Lock() replayed=%t err=%v", replayed, lockErr)
	}
	_ = retry.Rollback()
}

func (environment coordinatorTestEnvironment) assertCrossVersionReplay(t *testing.T) *managementcommand.Codec {
	t.Helper()
	keys := map[string][]byte{
		"command-v1": []byte(strings.Repeat("o", 32)),
		"command-v2": []byte(strings.Repeat("n", 32)),
	}
	oldCodec := newCoordinatorTestCodec(t, "command-v1", keys)
	newCodec := newCoordinatorTestCodec(t, "command-v2", keys)
	now := time.Now().UTC()
	oldCommand := bindRotatedCommand(t, oldCodec, now)
	newCommand := bindRotatedCommand(t, newCodec, now)
	if oldCommand.AdvisoryLockKey() != newCommand.AdvisoryLockKey() {
		t.Fatal("cross-version commands did not share a stable advisory identity")
	}
	transaction, err := environment.first.BeginTx(environment.ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, replayed, lockErr := Lock(environment.ctx, transaction, oldCommand); lockErr != nil || replayed {
		t.Fatalf("old-version winner Lock() replayed=%t err=%v", replayed, lockErr)
	}
	finished := lockAndCommitAsync(environment.ctx, environment.second, newCommand, nil)
	if completeErr := CompleteResource(environment.ctx, transaction, oldCommand, managementcommand.ResourceResult{
		ResourceType: "provider_credential", ResourceID: testResourceID,
		ResourceRevision: 2, ResponseStatus: 201,
	}); completeErr != nil {
		_ = transaction.Rollback()
		t.Fatal(completeErr)
	}
	if commitErr := transaction.Commit(); commitErr != nil {
		t.Fatal(commitErr)
	}
	outcome := <-finished
	if outcome.err != nil || !outcome.replayed || outcome.result.Resource == nil ||
		outcome.result.Resource.ResourceRevision != 2 {
		t.Fatalf("cross-version outcome = %#v", outcome)
	}
	return newCodec
}

func bindRotatedCommand(
	t *testing.T,
	codec *managementcommand.Codec,
	now time.Time,
) managementcommand.Command {
	t.Helper()
	command, err := codec.Bind(
		managementcommand.NamespaceCommandScope(testNamespaceID), testPrincipalID,
		"/management/v1/provider-credentials", "cross-version-0123456789",
		[]byte(`{"name":"rotated"}`), now, now.Add(time.Hour),
	)
	if err != nil {
		t.Fatal(err)
	}
	return command
}

func newCoordinatorTestCodec(t *testing.T, activeVersion string, keys map[string][]byte) *managementcommand.Codec {
	t.Helper()
	codec, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: activeVersion,
		Keys:          keys,
	})
	if err != nil {
		t.Fatal(err)
	}
	return codec
}

func (environment coordinatorTestEnvironment) assertHMACVersionReadiness(t *testing.T, codec *managementcommand.Codec) {
	t.Helper()
	if err := ValidateReferencedHMACVersions(environment.ctx, environment.first, codec); err != nil {
		t.Fatalf("retained-version readiness = %v", err)
	}
	if _, err := environment.first.ExecContext(environment.ctx, `INSERT INTO management_idempotency
  (scope_kind, namespace_id, principal_id, endpoint, hmac_version, idempotency_key_digest,
   request_digest, resource_type, resource_id, resource_revision, response_status, expires_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)`,
		string(managementcommand.ScopeNamespace), testNamespaceID, testPrincipalID,
		"/management/v1/missing-version", "retired-v0", []byte(strings.Repeat("d", 32)),
		[]byte(strings.Repeat("r", 32)), "provider_credential", testResourceID, 3, 201,
		time.Now().UTC().Add(time.Hour)); err != nil {
		t.Fatal(err)
	}
	if err := ValidateReferencedHMACVersions(environment.ctx, environment.first, codec); !errors.Is(err, managementcommand.ErrHMACVersionUnavailable) {
		t.Fatalf("missing-version readiness = %v", err)
	}
	if _, err := environment.first.ExecContext(environment.ctx, `UPDATE management_idempotency
SET expires_at = clock_timestamp() - interval '1 second' WHERE hmac_version = 'retired-v0'`); err != nil {
		t.Fatal(err)
	}
	if err := ValidateReferencedHMACVersions(environment.ctx, environment.first, codec); err != nil {
		t.Fatalf("post-retention readiness = %v", err)
	}
}

func (environment coordinatorTestEnvironment) assertStoredDigestsHideInput(t *testing.T) {
	t.Helper()
	var keyDigest, requestDigest string
	if err := environment.first.QueryRowContext(environment.ctx, `SELECT encode(idempotency_key_digest, 'hex'), encode(request_digest, 'hex')
FROM management_idempotency LIMIT 1`).Scan(&keyDigest, &requestDigest); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(keyDigest, "cross-replica") || strings.Contains(requestDigest, "never-persist") {
		t.Fatal("stored command exposed raw idempotency or request material")
	}
}

func isolatedCommandDatabases(t *testing.T, ctx context.Context, dsn string) (*sql.DB, *sql.DB) {
	t.Helper()
	admin, isolatedCommandDatabasesErr := sql.Open("postgres", dsn)
	if isolatedCommandDatabasesErr != nil {
		t.Fatal(isolatedCommandDatabasesErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	schema := "vsr_command_test_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	scoped, isolatedCommandDatabasesErr := commandDSNWithSearchPath(dsn, schema)
	if isolatedCommandDatabasesErr != nil {
		t.Fatal(isolatedCommandDatabasesErr)
	}
	open := func() *sql.DB {
		db, openErr := sql.Open("postgres", scoped)
		if openErr != nil {
			t.Fatal(openErr)
		}
		t.Cleanup(func() { _ = db.Close() })
		return db
	}
	return open(), open()
}

func commandDSNWithSearchPath(dsn, schema string) (string, error) {
	if !strings.Contains(dsn, "://") {
		return dsn + " search_path=" + schema, nil
	}
	parsed, err := url.Parse(dsn)
	if err != nil {
		return "", err
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}
